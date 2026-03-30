import base64
import json
import os
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image, UnidentifiedImageError

load_dotenv(Path(__file__).with_name(".env"))

MODEL_NAME = os.getenv("GEMINI_MODEL", os.getenv("OLLAMA_MODEL", "gemini-3.1-flash-lite-preview"))
MAX_IMAGE_SIZE = (1024, 1024)
SEVERITY_WEIGHTS = {
    "critical": 40,
    "high": 30,
    "medium": 20,
    "low": 10,
}


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
            "short_summary": {"type": "string"},
            "structured_problem": {"type": "string"},
            "error_type": {
                "type": "string",
                "enum": [
                    "ui_crash",
                    "api_timeout",
                    "validation_error",
                    "authentication_error",
                    "authorization_error",
                    "data_mismatch",
                    "data_missing",
                    "layout_broken",
                    "workflow_blocked",
                    "performance_issue",
                    "content_issue",
                    "unknown",
                ],
            },
            "system_context": {"type": "string"},
            "page_context": {"type": "string"},
            "error_code": {"type": "string"},
            "severity": {
                "type": "string",
                "enum": ["critical", "high", "medium", "low"],
            },
            "image_evidence": {"type": "array", "items": {"type": "string"}},
            "related_issues": {"type": "array", "items": {"type": "string"}},
            "impact_scope": {"type": "string"},
            "impact_assessment": {"type": "string"},
            "preliminary_assessment": {"type": "string"},
            "data_gaps": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "short_summary",
            "structured_problem",
            "error_type",
            "system_context",
            "page_context",
            "error_code",
            "severity",
            "image_evidence",
            "related_issues",
            "impact_scope",
            "impact_assessment",
            "preliminary_assessment",
            "data_gaps",
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
        "You are a product support intake analyst producing machine-consumable incident records.\n"
        "Your output will be stored in MongoDB and used for structured triage, search, and downstream review.\n"
        "Be concise, factual, deterministic, and extract as much concrete incident detail as the evidence supports.\n\n"
        "TICKET CONTEXT (provided form data):\n"
        f"- Request ID: {ticket_data.get('requestId', 'Unknown')}\n"
        f"- Submitter Email: {ticket_data.get('userEmail', 'Unknown')}\n"
        f"- Request Type: {ticket_data.get('requestType', 'Unknown')}\n"
        f"- Routing Path: {ticket_data['primaryChoice']}\n"
        f"- Submission Type: {ticket_data['reviewType']}\n"
        f"- Issue Description: {ticket_data['issueDescription']}\n"
        f"- Supporting Images: {image_count}\n\n"
        "YOUR TASK:\n"
        "1. Use the issue description as the main complaint and correlate screenshots with it.\n"
        "2. Use only factual evidence from the submitted text and visible image details.\n"
        "3. Extract as many concrete facts as possible about symptoms, affected workflow, visible messages, business impact, and scope.\n"
        "4. Do not invent stack traces, APIs, database fields, or hidden system details.\n"
        "5. system_context should identify the likely business or system area inferred from the route and complaint. If unknown, use 'unknown'.\n"
        "6. page_context should describe the page, screen, or workflow step. If not visible or stated, use 'unknown'.\n"
        "7. error_code must be 'unknown' unless a specific code or error string is visible or explicitly stated.\n"
        "8. short_summary must be under 20 words and specific enough for future triage or dedup.\n"
        "9. structured_problem should be a compact 2-4 sentence narrative of what is failing, the observed symptom, and why it matters.\n"
        "10. severity should reflect likely operational impact: critical, high, medium, or low.\n"
        "11. If images are provided, image_evidence must list concrete visible observations from them, including labels, banners, empty states, or broken UI details when visible.\n"
        "12. related_issues should capture explicit duplicates, recurring hints, or closely matching symptoms only when supported by the evidence.\n"
        "13. impact_scope and impact_assessment should capture who or what is affected and how strongly, using evidence from the submission.\n"
        "14. data_gaps should list the missing facts that would most improve later investigation quality.\n"
        "15. Never evaluate the quality of the submission form itself.\n"
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
    normalized_output = normalize_structured_output(ticket_data, parsed_output, len(encoded_images))

    return {
        "model": MODEL_NAME,
        "imageCount": len(encoded_images),
        "structured": normalized_output,
        "rawOutput": json.dumps(normalized_output, ensure_ascii=True, indent=2),
    }


def normalize_structured_output(ticket_data: dict[str, str], structured: dict, image_count: int) -> dict:
    severity = str(structured.get("severity") or "medium").strip().lower()
    if severity not in SEVERITY_WEIGHTS:
        severity = "medium"

    normalized = {
        "short_summary": str(structured.get("short_summary") or "").strip(),
        "structured_problem": str(structured.get("structured_problem") or "").strip(),
        "error_type": str(structured.get("error_type") or "unknown").strip(),
        "system_context": str(structured.get("system_context") or ticket_data.get("primaryChoice") or "unknown").strip(),
        "page_context": str(structured.get("page_context") or "unknown").strip(),
        "error_code": str(structured.get("error_code") or "unknown").strip(),
        "severity": severity,
        "severity_weight": SEVERITY_WEIGHTS[severity],
        "image_evidence": _clean_string_list(structured.get("image_evidence")),
        "related_issues": _clean_string_list(structured.get("related_issues")),
        "impact_scope": str(structured.get("impact_scope") or "").strip(),
        "impact_assessment": str(structured.get("impact_assessment") or "").strip(),
        "preliminary_assessment": str(structured.get("preliminary_assessment") or "").strip(),
        "data_gaps": _clean_string_list(structured.get("data_gaps")),
    }

    if image_count > 0 and not normalized["image_evidence"]:
        normalized["image_evidence"] = ["Images were supplied, but usable evidence could not be extracted reliably."]

    if not normalized["short_summary"]:
        normalized["short_summary"] = normalized["structured_problem"][:120] or "Issue reported without concise summary"
    if not normalized["impact_scope"]:
        normalized["impact_scope"] = "Affected scope could not be determined from the submitted evidence."
    if not normalized["impact_assessment"]:
        normalized["impact_assessment"] = "Impact could not be fully confirmed from the available description and images."

    normalized["occurrence_hint"] = derive_occurrence_hint(normalized)
    normalized["embedding_text"] = build_embedding_text(normalized)
    normalized["triage_signals"] = {
        "route": ticket_data.get("primaryChoice", "unknown"),
        "submissionType": ticket_data.get("reviewType", "unknown"),
        "requestType": ticket_data.get("requestType", "unknown"),
        "hasImages": image_count > 0,
        "imageCount": image_count,
        "shouldTriggerRca": False,
        "structuringComplete": True,
        "extractedFieldCount": _count_extracted_fields(normalized),
    }
    return normalized


def build_embedding_text(structured: dict) -> str:
    return (
        f"[{structured.get('error_type', 'unknown')}] "
        f"{structured.get('short_summary', '')}. "
        f"{structured.get('structured_problem', '')} "
        f"System: {structured.get('system_context', 'unknown')}. "
        f"Page: {structured.get('page_context', 'unknown')}. "
        f"Code: {structured.get('error_code', 'unknown')}."
    ).strip()


def _clean_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []

    cleaned: list[str] = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _count_extracted_fields(structured: dict) -> int:
    extracted_fields = [
        structured.get("short_summary"),
        structured.get("structured_problem"),
        structured.get("error_type"),
        structured.get("system_context"),
        structured.get("page_context"),
        structured.get("error_code"),
        structured.get("severity"),
        structured.get("impact_scope"),
        structured.get("impact_assessment"),
        structured.get("preliminary_assessment"),
    ]
    extracted_count = sum(1 for value in extracted_fields if str(value or "").strip() and str(value).strip().lower() != "unknown")
    extracted_count += len(structured.get("image_evidence", []))
    extracted_count += len(structured.get("related_issues", []))
    return extracted_count


def derive_occurrence_hint(structured: dict) -> str:
    if structured.get("related_issues"):
        return "likely_recurring"
    return "unclear"
