import asyncio

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pymongo.errors import DuplicateKeyError

from backend.db import (
    append_ticket_status_event,
    ensure_ticket_indexes,
    find_ticket_by_request_id,
    insert_ticket_document,
    ping_mongo,
    update_ticket_fields,
)
from backend.ingestion_ticket import analyze_ticket
from backend.s3_upload import (
    get_s3_client,
    upload_issue_photos,
    upload_json_artifact,
    upload_log_artifact,
)

app = FastAPI(
    title="JUBISmartPV Ticket Ingestion API",
    version="1.0.0",
    description="Accepts ticket text and multiple images, then runs multimodal ingestion analysis.",
)

# Allow frontend calls from local development origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_checks() -> None:
    # Validate data stores and ensure indexes before serving requests.
    await asyncio.to_thread(ping_mongo)
    await asyncio.to_thread(ensure_ticket_indexes)
    await asyncio.to_thread(get_s3_client)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


def _default_storage_summary(route: str) -> dict[str, object]:
    return {
        "route": route,
        "imageCount": 0,
        "imageObjects": [],
        "inputArtifact": None,
        "outputArtifact": None,
        "logArtifacts": [],
    }


def _build_artifact_urls(storage_summary: dict[str, object]) -> dict[str, list[str]]:
    image_payload_urls = [
        obj.get("objectUrl")
        for obj in storage_summary.get("imageObjects", [])
        if isinstance(obj, dict) and obj.get("objectUrl")
    ]
    input_urls = [storage_summary["inputArtifact"].get("objectUrl")] if isinstance(storage_summary.get("inputArtifact"), dict) and storage_summary["inputArtifact"].get("objectUrl") else []
    output_urls = [storage_summary["outputArtifact"].get("objectUrl")] if isinstance(storage_summary.get("outputArtifact"), dict) and storage_summary["outputArtifact"].get("objectUrl") else []
    log_urls = [
        obj.get("objectUrl")
        for obj in storage_summary.get("logArtifacts", [])
        if isinstance(obj, dict) and obj.get("objectUrl")
    ]
    return {
        "problems": image_payload_urls,
        "input": input_urls,
        "output": output_urls,
        "logs": log_urls,
    }


async def process_ticket_background(
    request_id: str,
    ticket_payload: dict[str, str],
    image_bytes_list: list[bytes],
    storage_summary: dict[str, object],
) -> None:
    route = ticket_payload["primaryChoice"]

    try:
        await asyncio.to_thread(
            append_ticket_status_event,
            request_id,
            "processing",
            "analyzing",
            "Running AI analysis",
        )
        analysis = await analyze_ticket(ticket_payload, image_bytes_list)

        uploaded_count = int(storage_summary.get("imageCount", 0))
        if uploaded_count != len(image_bytes_list):
            raise RuntimeError("Image upload count mismatch")
        if not analysis.get("structured"):
            raise RuntimeError("AI analysis did not return structured output")

        await asyncio.to_thread(
            append_ticket_status_event,
            request_id,
            "processing",
            "saving_output",
            "Saving output artifact",
        )
        output_payload = {
            "requestId": request_id,
            "model": analysis.get("model"),
            "imageCount": analysis.get("imageCount", 0),
            "structured": analysis.get("structured", {}),
            "rawOutput": analysis.get("rawOutput", ""),
        }
        output_artifact = await asyncio.to_thread(
            upload_json_artifact,
            route,
            "output",
            request_id,
            "response.json",
            output_payload,
        )
        storage_summary["outputArtifact"] = output_artifact

        success_log = await asyncio.to_thread(
            upload_log_artifact,
            route,
            request_id,
            "Submission processed successfully",
        )
        storage_summary["logArtifacts"] = [success_log]
        artifact_urls = _build_artifact_urls(storage_summary)
        image_payload_urls = artifact_urls["problems"]

        await asyncio.to_thread(
            update_ticket_fields,
            request_id,
            {
                "status": "completed",
                "currentStep": "completed",
                "statusMessage": "Submission processed successfully",
                "receivedImageCount": len(image_bytes_list),
                "storage": storage_summary,
                "imagePayloadUrls": image_payload_urls,
                "artifactUrls": artifact_urls,
                "ai": {
                    "model": analysis.get("model"),
                    "imageCount": analysis.get("imageCount", 0),
                    "summary": analysis.get("structured", {}),
                    "rawOutput": analysis.get("rawOutput", ""),
                },
            },
        )
        await asyncio.to_thread(
            append_ticket_status_event,
            request_id,
            "completed",
            "completed",
            "Submission processed successfully",
        )
    except Exception as exc:
        try:
            error_log = await asyncio.to_thread(
                upload_log_artifact,
                route,
                request_id,
                f"Processing failure: {exc}",
            )
            storage_summary["logArtifacts"] = [error_log]
        except Exception:
            pass
        artifact_urls = _build_artifact_urls(storage_summary)
        image_payload_urls = artifact_urls["problems"]
        await asyncio.to_thread(
            update_ticket_fields,
            request_id,
            {
                "status": "failed",
                "currentStep": "failed",
                "statusMessage": "Submission processing failed",
                "receivedImageCount": len(image_bytes_list),
                "storage": storage_summary,
                "imagePayloadUrls": image_payload_urls,
                "artifactUrls": artifact_urls,
                "error": {
                    "type": "processing",
                    "message": str(exc),
                },
            },
        )
        await asyncio.to_thread(
            append_ticket_status_event,
            request_id,
            "failed",
            "failed",
            f"Submission processing failed: {exc}",
        )


@app.post("/api/tickets/ingest")
async def ingest_ticket(
    background_tasks: BackgroundTasks,
    requestId: str = Form(...),
    userEmail: str = Form(...),
    requestType: str = Form(...),
    issueDescription: str = Form(...),
    primaryChoice: str = Form(...),
    reviewType: str = Form(...),
    issuePhotos: list[UploadFile] | None = File(None),
) -> dict:
    request_id = requestId.strip()
    user_email = userEmail.strip()
    request_type = requestType.strip()
    issue_description = issueDescription.strip()
    route = primaryChoice.strip()
    submission_type = reviewType.strip()

    if not request_id:
        raise HTTPException(status_code=400, detail="requestId is required")
    if not user_email or "@" not in user_email:
        raise HTTPException(status_code=400, detail="A valid userEmail is required")
    if not issue_description:
        raise HTTPException(status_code=400, detail="issueDescription is required")
    if route not in {"JDI", "JGL"}:
        raise HTTPException(status_code=400, detail="primaryChoice must be JDI or JGL")
    if submission_type not in {"PSUR", "PADER", "Literature Review"}:
        raise HTTPException(status_code=400, detail="reviewType is invalid")

    image_bytes_list: list[bytes] = []
    file_payloads: list[dict[str, object]] = []
    if issuePhotos:
        for image_file in issuePhotos:
            content = await image_file.read()
            if content:
                file_payloads.append(
                    {
                        "filename": image_file.filename or "image.bin",
                        "contentType": image_file.content_type,
                        "content": content,
                    }
                )
                image_bytes_list.append(content)

    ticket_payload = {
        "requestId": request_id,
        "userEmail": user_email,
        "requestType": request_type,
        "issueDescription": issue_description,
        "primaryChoice": route,
        "reviewType": submission_type,
    }

    initial_storage_summary = _default_storage_summary(route)

    initial_ticket_document = {
        **ticket_payload,
        "status": "processing",
        "currentStep": "received",
        "statusMessage": "Request received and queued",
        "form": ticket_payload,
        "receivedImageCount": len(image_bytes_list),
        "storage": initial_storage_summary,
        "imagePayloadUrls": [],
        "artifactUrls": {
            "problems": [],
            "input": [],
            "output": [],
            "logs": [],
        },
        "statusHistory": [
            {
                "status": "processing",
                "step": "received",
                "message": "Request received and queued",
            }
        ],
    }

    try:
        ticket_id = await asyncio.to_thread(insert_ticket_document, initial_ticket_document)
    except DuplicateKeyError as exc:
        raise HTTPException(status_code=409, detail="A ticket with this requestId already exists") from exc

    # Complete initial persistence synchronously, then return processing response.
    storage_summary = _default_storage_summary(route)
    try:
        await asyncio.to_thread(
            append_ticket_status_event,
            request_id,
            "processing",
            "uploading_images",
            "Uploading images to storage",
        )
        storage_summary = await asyncio.to_thread(upload_issue_photos, route, request_id, file_payloads)

        await asyncio.to_thread(
            append_ticket_status_event,
            request_id,
            "processing",
            "saving_input",
            "Saving input artifact",
        )
        input_artifact = await asyncio.to_thread(
            upload_json_artifact,
            route,
            "input",
            request_id,
            "payload.json",
            ticket_payload,
        )
        storage_summary["inputArtifact"] = input_artifact

        initial_artifact_urls = _build_artifact_urls(storage_summary)
        await asyncio.to_thread(
            update_ticket_fields,
            request_id,
            {
                "status": "processing",
                "currentStep": "analyzing",
                "statusMessage": "Initial artifacts saved. Analysis started.",
                "storage": storage_summary,
                "imagePayloadUrls": initial_artifact_urls["problems"],
                "artifactUrls": initial_artifact_urls,
            },
        )
        await asyncio.to_thread(
            append_ticket_status_event,
            request_id,
            "processing",
            "analyzing",
            "Initial artifacts saved. Analysis started.",
        )
    except Exception as exc:
        try:
            error_log = await asyncio.to_thread(
                upload_log_artifact,
                route,
                request_id,
                f"Initial persistence failure: {exc}",
            )
            storage_summary["logArtifacts"] = [error_log]
        except Exception:
            pass
        await asyncio.to_thread(
            update_ticket_fields,
            request_id,
            {
                "status": "failed",
                "currentStep": "failed",
                "statusMessage": "Initial persistence failed",
                "storage": storage_summary,
                "artifactUrls": _build_artifact_urls(storage_summary),
                "error": {
                    "type": "initial_persistence",
                    "message": str(exc),
                },
            },
        )
        await asyncio.to_thread(
            append_ticket_status_event,
            request_id,
            "failed",
            "failed",
            f"Initial persistence failed: {exc}",
        )
        raise HTTPException(status_code=500, detail="Failed to save initial request artifacts") from exc

    background_tasks.add_task(
        process_ticket_background,
        request_id,
        ticket_payload,
        image_bytes_list,
        storage_summary,
    )

    return {
        "success": True,
        "message": "Request received. Processing started.",
        "ticketId": ticket_id,
        "requestId": request_id,
        "status": "processing",
    }
