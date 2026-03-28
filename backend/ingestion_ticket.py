import base64
import os
from io import BytesIO
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv
load_dotenv(Path(__file__).with_name(".env"))
MODEL_NAME = os.getenv("GEMINI_MODEL", os.getenv("OLLAMA_MODEL", "gemini-3.1-flash-lite-preview"))
MAX_IMAGE_SIZE = (1024, 1024)

def get_model() -> ChatGoogleGenerativeAI:
    # LangChain docs: GOOGLE_API_KEY is primary, GEMINI_API_KEY is fallback.
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=api_key,
        # Gemini 3.1 supports thinking_budget; 0 keeps responses concise for ticket triage.
        thinking_budget=0,
    )
def get_output_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "structured_problem": {"type": "string"},
            "related_issues": {"type": "array", "items": {"type": "string"}},
            "image_evidence": {"type": "array", "items": {"type": "string"}},
            "impact_assessment": {"type": "string"},
            "preliminary_assessment": {"type": "string"},
        },
        "required": [
            "structured_problem",
            "related_issues",
            "image_evidence",
            "impact_assessment",
            "preliminary_assessment",
        ],
        "additionalProperties": False,
    }


def encode_image_bytes(image_bytes: bytes) -> str:
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            rgb_image = img.convert("RGB")
            rgb_image.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)

            output = BytesIO()
            rgb_image.save(output, format="JPEG", quality=92)
            return base64.b64encode(output.getvalue()).decode("utf-8")
    except UnidentifiedImageError as exc:
        raise ValueError("One of the uploaded files is not a valid image.") from exc


def build_ticket_prompt(ticket_data: dict[str, str], image_count: int) -> str:
    return (
        "You are a product support intake analyst. Your job is to identify what is wrong, using both the issue description and screenshots.\n\n"
        "TICKET CONTEXT (provided form data):\n"
        f"- Request ID: {ticket_data.get('requestId', 'Unknown')}\n"
        f"- Submitter Email: {ticket_data.get('userEmail', 'Unknown')}\n"
        f"- Request Type: {ticket_data.get('requestType', 'Unknown')}\n"
        f"- Routing Path: {ticket_data['primaryChoice']}\n"
        f"- Submission Type: {ticket_data['reviewType']}\n"
        f"- Issue Description: {ticket_data['issueDescription']}\n"
        f"- Supporting Images: {image_count}\n\n"
        "YOUR TASK:\n"
        "1. Use the Issue Description as the primary complaint to investigate.\n"
        "2. Correlate screenshots with the complaint and identify likely UI/UX/functionality failure points.\n"
        "3. Use only factual visual evidence from images (labels, broken layouts, missing controls, error banners, clipped text, disabled actions, etc.).\n"
        "4. Do NOT evaluate submission policy quality (e.g., whether Request ID text looks professional) unless explicitly requested by the user.\n"
        "5. If images are present, image_evidence must not be empty.\n"
        "6. If image evidence is unclear, explicitly state what is visible and what cannot be confirmed.\n"
        "Provide concise, evidence-based RCA output.\n"
    )


def build_multimodal_message(ticket_data: dict[str, str], encoded_images: list[str]) -> HumanMessage:
    content = [{"type": "text", "text": build_ticket_prompt(ticket_data, len(encoded_images))}]

    for encoded_image in encoded_images:
        content.append(
            {
                "type": "image",
                "base64": encoded_image,
                "mime_type": "image/jpeg",
            }
        )

    return HumanMessage(content=content)


async def analyze_ticket(ticket_data: dict[str, str], image_bytes_list: list[bytes]) -> dict:
    encoded_images = [encode_image_bytes(image_bytes) for image_bytes in image_bytes_list if image_bytes]
    message = build_multimodal_message(ticket_data, encoded_images)

    structured_model = get_model().with_structured_output(
        schema=get_output_schema(),
        method="json_schema",
    )
    parsed_output = await structured_model.ainvoke([message])

    return {
        "model": MODEL_NAME,
        "imageCount": len(encoded_images),
        "structured": parsed_output,
        "rawOutput": str(parsed_output),
    }