from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).with_name(".env"))

EMBEDDING_MODEL_NAME = os.getenv("OPENROUTER_EMBEDDING_MODEL", "intfloat/e5-base-v2")
VECTOR_INDEX_NAME = os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME", "ticket_summary_vector_idx")
VECTOR_PATH = "embeddings.summary.vector"
DEFAULT_VECTOR_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))
DEFAULT_CANDIDATES_MULTIPLIER = int(os.getenv("VECTOR_SEARCH_CANDIDATE_MULTIPLIER", "15"))
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
OPENROUTER_TITLE = os.getenv("OPENROUTER_TITLE", "").strip()


@lru_cache(maxsize=1)
def get_openrouter_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY for embedding generation.")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def get_vector_index_name() -> str:
    return VECTOR_INDEX_NAME


def get_vector_dimensions() -> int:
    return DEFAULT_VECTOR_DIMENSIONS


def get_vector_index_definition(dimensions: int | None = None) -> dict[str, Any]:
    return {
        "name": VECTOR_INDEX_NAME,
        "type": "vectorSearch",
        "definition": {
            "fields": [
                {
                    "type": "vector",
                    "path": VECTOR_PATH,
                    "numDimensions": dimensions or DEFAULT_VECTOR_DIMENSIONS,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "embeddings.summary.metadata.reviewType"},
                {"type": "filter", "path": "embeddings.summary.status"},
                {"type": "filter", "path": "updatedAt"},
            ]
        },
    }


def embed_text(text: str) -> list[float]:
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Cannot generate an embedding from empty text.")
    extra_headers: dict[str, str] = {}
    if OPENROUTER_HTTP_REFERER:
        extra_headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
    if OPENROUTER_TITLE:
        extra_headers["X-OpenRouter-Title"] = OPENROUTER_TITLE

    response = get_openrouter_client().embeddings.create(
        model=EMBEDDING_MODEL_NAME,
        input=cleaned_text,
        encoding_format="float",
        extra_headers=extra_headers or None,
    )
    return list(response.data[0].embedding)


def build_embedding_metadata(ticket_payload: dict[str, str], structured: dict[str, Any]) -> dict[str, Any]:
    route = str(ticket_payload.get("primaryChoice") or "unknown").strip() or "unknown"
    request_type = str(ticket_payload.get("requestType") or "unknown").strip() or "unknown"
    review_type = str(ticket_payload.get("reviewType") or "unknown").strip() or "unknown"

    return {
        "requestId": str(ticket_payload.get("requestId") or "").strip(),
        "project": route,
        "route": route,
        "requestType": request_type,
        "reviewType": review_type,
        "errorType": str(structured.get("error_type") or "unknown").strip() or "unknown",
        "severity": str(structured.get("severity") or "unknown").strip() or "unknown",
        "systemContext": str(structured.get("system_context") or "unknown").strip() or "unknown",
        "pageContext": str(structured.get("page_context") or "unknown").strip() or "unknown",
        "documentType": "incident_intake",
    }


def build_embedding_record(ticket_payload: dict[str, str], structured: dict[str, Any]) -> dict[str, Any]:
    embedding_text = str(structured.get("embedding_text") or structured.get("short_summary") or "").strip()
    metadata = build_embedding_metadata(ticket_payload, structured)
    vector = embed_text(embedding_text)

    return {
        "text": embedding_text,
        "vector": vector,
        "dimensions": len(vector),
        "model": EMBEDDING_MODEL_NAME,
        "indexName": VECTOR_INDEX_NAME,
        "metadata": metadata,
        "embeddedAt": None,
        "error": None,
        "status": "completed",
    }


def build_pending_embedding_record(ticket_payload: dict[str, str]) -> dict[str, Any]:
    route = str(ticket_payload.get("primaryChoice") or "unknown").strip() or "unknown"
    return {
        "text": None,
        "vector": [],
        "dimensions": 0,
        "model": EMBEDDING_MODEL_NAME,
        "indexName": VECTOR_INDEX_NAME,
        "metadata": {
            "requestId": str(ticket_payload.get("requestId") or "").strip(),
            "project": route,
            "route": route,
            "requestType": str(ticket_payload.get("requestType") or "unknown").strip() or "unknown",
            "reviewType": str(ticket_payload.get("reviewType") or "unknown").strip() or "unknown",
            "documentType": "incident_intake",
        },
        "embeddedAt": None,
        "status": "pending",
        "error": None,
    }


def build_failed_embedding_record(ticket_payload: dict[str, str], structured: dict[str, Any], error: Exception) -> dict[str, Any]:
    failed_record = build_pending_embedding_record(ticket_payload)
    failed_record["text"] = str(structured.get("embedding_text") or structured.get("short_summary") or "").strip() or None
    failed_record["metadata"] = build_embedding_metadata(ticket_payload, structured)
    failed_record["status"] = "failed"
    failed_record["error"] = str(error)
    return failed_record


def build_vector_search_pipeline(
    query_vector: list[float],
    *,
    limit: int,
    num_candidates: int | None = None,
    metadata_filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    vector_stage: dict[str, Any] = {
        "index": VECTOR_INDEX_NAME,
        "path": VECTOR_PATH,
        "queryVector": query_vector,
        "limit": limit,
        "numCandidates": num_candidates or max(limit * DEFAULT_CANDIDATES_MULTIPLIER, limit),
    }
    if metadata_filters:
        vector_stage["filter"] = metadata_filters

    return [
        {"$vectorSearch": vector_stage},
        {
            "$project": {
                "_id": 0,
                "requestId": 1,
                "primaryChoice": 1,
                "requestType": 1,
                "reviewType": 1,
                "status": 1,
                "triage": 1,
                "analysis.embeddingText": 1,
                "embeddings.summary.metadata": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
