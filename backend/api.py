import asyncio
import hashlib
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pymongo.errors import DuplicateKeyError

from backend.db import (
    append_ticket_status_event,
    ensure_ticket_indexes,
    find_ticket_by_request_id,
    get_tickets_collection,
    insert_ticket_document,
    ping_mongo,
    update_ticket_fields,
)
from backend.embedder import (
    build_embedding_record,
    build_failed_embedding_record,
    build_pending_embedding_record,
    build_vector_search_pipeline,
    embed_text,
)
from backend.ingestion_ticket import analyze_ticket
from backend.opencode_orchestrator import execute_rag_flow
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

logger = logging.getLogger(__name__)
RAG_LOOKBACK_DAYS = int(os.getenv("RAG_LOOKBACK_DAYS", "360"))
DEPENDENCY_STATUS: dict[str, dict[str, str]] = {
    "mongo": {"status": "unknown", "detail": "Not checked yet"},
    "s3": {"status": "unknown", "detail": "Not checked yet"},
}


class VectorSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(default=5, ge=1, le=25)
    review_type: str | None = None


@app.on_event("startup")
async def startup_checks() -> None:
    # Validate external dependencies, but do not abort the API process if one is unavailable.
    try:
        await asyncio.to_thread(ping_mongo)
        await asyncio.to_thread(ensure_ticket_indexes)
        DEPENDENCY_STATUS["mongo"] = {"status": "ok", "detail": "MongoDB reachable"}
    except Exception as exc:
        DEPENDENCY_STATUS["mongo"] = {"status": "error", "detail": str(exc)}
        logger.warning("MongoDB startup check failed: %s", exc)

    try:
        await asyncio.to_thread(get_s3_client)
        DEPENDENCY_STATUS["s3"] = {"status": "ok", "detail": "S3 client initialized"}
    except Exception as exc:
        DEPENDENCY_STATUS["s3"] = {"status": "error", "detail": str(exc)}
        logger.warning("S3 startup check failed: %s", exc)


@app.get("/health")
async def health_check() -> dict[str, object]:
    overall_status = "ok" if all(item["status"] == "ok" for item in DEPENDENCY_STATUS.values()) else "degraded"
    return {"status": overall_status, "dependencies": DEPENDENCY_STATUS}


def _require_dependency(name: str) -> None:
    dependency = DEPENDENCY_STATUS.get(name, {})
    if dependency.get("status") != "ok":
        raise HTTPException(status_code=503, detail=f"{name.upper()} unavailable: {dependency.get('detail', 'Unknown error')}")


@app.get("/api/tickets/{request_id}")
async def get_ticket(request_id: str) -> dict[str, object]:
    _require_dependency("mongo")
    ticket = await asyncio.to_thread(find_ticket_by_request_id, request_id.strip())
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    if ticket.get("_id") is not None:
        ticket["_id"] = str(ticket["_id"])
    return {"success": True, "ticket": jsonable_encoder(ticket)}


@app.get("/api/tickets/recent/list")
async def get_recent_tickets() -> dict[str, object]:
    _require_dependency("mongo")

    try:
        tickets = await asyncio.to_thread(
            lambda: list(
                get_tickets_collection()
                .find({}, {"embeddings": 0})
                .sort("createdAt", -1)
                .limit(5)
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch recent tickets: {exc}") from exc

    for ticket in tickets:
        if ticket.get("_id") is not None:
            ticket["_id"] = str(ticket["_id"])

    return {
        "success": True,
        "count": len(tickets),
        "tickets": jsonable_encoder(tickets),
    }


def _default_storage_summary(route: str) -> dict[str, object]:
    return {
        "route": route,
        "imageCount": 0,
        "imageObjects": [],
        "inputArtifact": None,
        "outputArtifact": None,
        "logArtifacts": [],
    }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _build_issue_fingerprint(ticket_payload: dict[str, str]) -> str:
    fingerprint_source = "|".join(
        [
            ticket_payload.get("primaryChoice", "").strip().lower(),
            ticket_payload.get("reviewType", "").strip().lower(),
            ticket_payload.get("requestType", "").strip().lower(),
            ticket_payload.get("issueDescription", "").strip().lower(),
        ]
    )
    return _digest_text(fingerprint_source)


def _normalize_review_type(value: str | None) -> str | None:
    if not value or not value.strip():
        return None

    normalized = value.strip().lower()
    review_type_map = {
        "psur": "PSUR",
        "pader": "PADER",
        "lit review": "Literature Review",
        "literature review": "Literature Review",
        "image studio": "Image Studio",
        "imagestudio": "Image Studio",
    }
    if normalized not in review_type_map:
        raise HTTPException(status_code=400, detail="review_type must be PSUR, PADER, Literature Review, or Image Studio")
    return review_type_map[normalized]


def _build_vector_search_filter(search_request: VectorSearchRequest) -> dict[str, Any]:
    filter_clauses: list[dict[str, Any]] = [
        {"embeddings.summary.status": "completed"},
        {"updatedAt": {"$gte": _utc_now() - timedelta(days=RAG_LOOKBACK_DAYS)}},
    ]

    normalized_review_type = _normalize_review_type(search_request.review_type)
    if normalized_review_type:
        filter_clauses.append({"embeddings.summary.metadata.reviewType": normalized_review_type})

    return {"$and": filter_clauses}


def _build_workflow_state() -> dict[str, object]:
    return {
        "rag": {
            "eligible": False,
            "status": "not_started",
            "decision": None,
            "evaluatedAt": None,
        },
        "dedup": {
            "eligible": False,
            "status": "not_started",
            "matchedRecordId": None,
            "originalRequestId": None,
            "evaluatedAt": None,
        },
        "rca": {
            "eligible": False,
            "status": "not_applicable",
            "queuedAt": None,
            "startedAt": None,
            "completedAt": None,
            "summary": None,
        },
    }


def _build_initial_ticket_document(
    ticket_payload: dict[str, str],
    received_image_count: int,
    storage_summary: dict[str, object],
) -> dict[str, object]:
    now = _utc_now()
    issue_description = ticket_payload["issueDescription"]
    route = ticket_payload["primaryChoice"]

    return {
        "requestId": ticket_payload["requestId"],
        "userEmail": ticket_payload["userEmail"],
        "requestType": ticket_payload["requestType"],
        "primaryChoice": route,
        "reviewType": ticket_payload["reviewType"],
        "status": "processing",
        "currentStep": "received",
        "statusMessage": "Request received and queued",
        "documentType": "incident_intake",
        "pipelineVersion": "v2",
        "form": ticket_payload,
        "intake": {
            "requestId": ticket_payload["requestId"],
            "submitterEmail": ticket_payload["userEmail"],
            "requestType": ticket_payload["requestType"],
            "routingPath": route,
            "submissionType": ticket_payload["reviewType"],
            "issueDescription": issue_description,
            "issueDescriptionHash": _digest_text(issue_description.strip()),
            "issueFingerprint": _build_issue_fingerprint(ticket_payload),
            "receivedImageCount": received_image_count,
            "submittedAt": now,
        },
        "storage": storage_summary,
        "imagePayloadUrls": [],
        "artifactUrls": {
            "problems": [],
            "input": [],
            "output": [],
            "logs": [],
        },
        "analysis": {
            "model": None,
            "imageCount": 0,
            "structured": None,
            "rawOutput": None,
            "embeddingText": None,
            "embeddingModel": None,
            "triageSignals": None,
            "analyzedAt": None,
        },
        "embeddings": {
            "summary": build_pending_embedding_record(ticket_payload),
        },
        "triage": {
            "summary": None,
            "errorType": None,
            "systemContext": route,
            "pageContext": None,
            "errorCode": None,
            "severity": None,
            "severityWeight": None,
            "impactScope": None,
            "impactAssessment": None,
            "preliminaryAssessment": None,
            "relatedIssues": [],
            "imageEvidence": [],
            "occurrenceHint": None,
            "dataGaps": [],
        },
        "workflow": _build_workflow_state(),
        "rca": {
            "status": "not_applicable",
            "eligible": False,
            "result": None,
        },
        "statusHistory": [
            {
                "status": "processing",
                "step": "received",
                "message": "Request received and queued",
                "at": now,
            }
        ],
    }


def _build_analysis_update(analysis: dict[str, object], storage_summary: dict[str, object], image_bytes_list: list[bytes]) -> dict[str, object]:
    structured = analysis.get("structured", {})
    triage_signals = structured.get("triage_signals", {}) if isinstance(structured, dict) else {}
    now = _utc_now()
    embedding_record = analysis.get("embedding", {}) if isinstance(analysis.get("embedding"), dict) else {}

    return {
        "status": "completed",
        "currentStep": "completed",
        "statusMessage": "Submission processed successfully",
        "receivedImageCount": len(image_bytes_list),
        "storage": storage_summary,
        "imagePayloadUrls": _build_artifact_urls(storage_summary)["problems"],
        "artifactUrls": _build_artifact_urls(storage_summary),
        "analysis": {
            "model": analysis.get("model"),
            "imageCount": analysis.get("imageCount", 0),
            "structured": structured,
            "rawOutput": analysis.get("rawOutput", ""),
            "embeddingText": structured.get("embedding_text") if isinstance(structured, dict) else None,
            "embeddingModel": embedding_record.get("model"),
            "triageSignals": triage_signals,
            "analyzedAt": now,
        },
        "embeddings": {
            "summary": {
                **embedding_record,
                "embeddedAt": now,
            }
        },
        "triage": {
            "summary": structured.get("short_summary") if isinstance(structured, dict) else None,
            "structuredProblem": structured.get("structured_problem") if isinstance(structured, dict) else None,
            "errorType": structured.get("error_type") if isinstance(structured, dict) else None,
            "systemContext": structured.get("system_context") if isinstance(structured, dict) else None,
            "pageContext": structured.get("page_context") if isinstance(structured, dict) else None,
            "errorCode": structured.get("error_code") if isinstance(structured, dict) else None,
            "severity": structured.get("severity") if isinstance(structured, dict) else None,
            "severityWeight": structured.get("severity_weight") if isinstance(structured, dict) else None,
            "impactScope": structured.get("impact_scope") if isinstance(structured, dict) else None,
            "impactAssessment": structured.get("impact_assessment") if isinstance(structured, dict) else None,
            "preliminaryAssessment": structured.get("preliminary_assessment") if isinstance(structured, dict) else None,
            "relatedIssues": structured.get("related_issues", []) if isinstance(structured, dict) else [],
            "imageEvidence": structured.get("image_evidence", []) if isinstance(structured, dict) else [],
            "occurrenceHint": structured.get("occurrence_hint") if isinstance(structured, dict) else None,
            "dataGaps": structured.get("data_gaps", []) if isinstance(structured, dict) else [],
        },
        "workflow": {
            "rag": {
                "eligible": True,
                "status": "pending",
                "decision": None,
                "evaluatedAt": None,
            },
            "dedup": {
                "eligible": False,
                "status": "waiting_for_rag",
                "matchedRecordId": None,
                "originalRequestId": None,
                "evaluatedAt": None,
            },
            "rca": {
                "eligible": False,
                "status": "waiting_for_rag",
                "queuedAt": None,
                "startedAt": None,
                "completedAt": None,
                "summary": "Structured record uploaded. Waiting for the RAG agent to decide reuse vs RCA.",
            },
        },
        "rca": {
            "status": "waiting_for_rag",
            "eligible": False,
            "result": None,
        },
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
    logger.info("Background processing started for requestId=%s route=%s", request_id, route)

    try:
        await asyncio.to_thread(
            append_ticket_status_event,
            request_id,
            "processing",
            "analyzing",
            "Running AI analysis",
        )
        analysis = await analyze_ticket(ticket_payload, image_bytes_list)
        logger.info("Structured analysis returned for requestId=%s", request_id)

        uploaded_count = int(storage_summary.get("imageCount", 0))
        if uploaded_count != len(image_bytes_list):
            raise RuntimeError("Image upload count mismatch")
        if not analysis.get("structured"):
            raise RuntimeError("AI analysis did not return structured output")

        structured = analysis.get("structured", {})
        if not isinstance(structured, dict):
            raise RuntimeError("AI analysis returned invalid structured output")

        try:
            logger.info("Generating summary embedding for requestId=%s", request_id)
            analysis["embedding"] = await asyncio.to_thread(build_embedding_record, ticket_payload, structured)
            logger.info("Embedding generated for requestId=%s", request_id)
        except Exception as embedding_exc:
            logger.exception("Embedding generation failed for requestId=%s", request_id)
            analysis["embedding"] = build_failed_embedding_record(ticket_payload, structured, embedding_exc)

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
            "embedding": analysis.get("embedding", {}),
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
        logger.info("Structured output artifacts saved for requestId=%s", request_id)

        await asyncio.to_thread(
            update_ticket_fields,
            request_id,
            _build_analysis_update(analysis, storage_summary, image_bytes_list),
        )
        logger.info("Ticket document updated after structuring for requestId=%s", request_id)
        await asyncio.to_thread(
            execute_rag_flow,
            request_id,
        )
        logger.info("RAG/OpenCode flow finished for requestId=%s", request_id)
    except Exception as exc:
        logger.exception("Background processing failed for requestId=%s", request_id)
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
    _require_dependency("mongo")
    _require_dependency("s3")

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
    if submission_type not in {"PSUR", "PADER", "Literature Review", "Image Studio"}:
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
    initial_ticket_document = _build_initial_ticket_document(
        ticket_payload,
        len(image_bytes_list),
        initial_storage_summary,
    )

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


@app.post("/api/tickets/search/vector")
async def search_tickets_vector(search_request: VectorSearchRequest) -> dict[str, object]:
    _require_dependency("mongo")
    metadata_filters = _build_vector_search_filter(search_request)

    try:
        query_vector = await asyncio.to_thread(embed_text, search_request.query)
        pipeline = build_vector_search_pipeline(
            query_vector,
            limit=search_request.limit,
            metadata_filters=metadata_filters or None,
        )
        results = await asyncio.to_thread(
            lambda: list(get_tickets_collection().aggregate(pipeline))
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Vector search failed: {exc}") from exc

    return {
        "success": True,
        "query": search_request.query,
        "filters": metadata_filters,
        "count": len(results),
        "results": jsonable_encoder(results),
    }
