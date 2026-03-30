from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import certifi
from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError
from pymongo.operations import SearchIndexModel
from pymongo.server_api import ServerApi

from backend.embedder import get_vector_dimensions, get_vector_index_definition

load_dotenv(Path(__file__).with_name(".env"))

_MONGO_CLIENT: MongoClient | None = None


def _get_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip()
    return default


def _require_env(*names: str) -> str:
    value = _get_env(*names)
    if not value:
        raise ValueError(f"Missing required environment variable. Tried: {', '.join(names)}")
    return value


def get_mongo_client() -> MongoClient:
    global _MONGO_CLIENT
    if _MONGO_CLIENT is not None:
        return _MONGO_CLIENT

    uri = _require_env("uri", "MONGODB_URI", "MONGO_URI")
    _MONGO_CLIENT = MongoClient(
        uri,
        appname="mamba-ticketing",
        tls=True,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=int(_get_env("MONGO_SERVER_SELECTION_TIMEOUT_MS", default="8000") or "8000"),
        connectTimeoutMS=int(_get_env("MONGO_CONNECT_TIMEOUT_MS", default="8000") or "8000"),
        socketTimeoutMS=int(_get_env("MONGO_SOCKET_TIMEOUT_MS", default="8000") or "8000"),
        retryWrites=True,
        server_api=ServerApi(version="1", strict=True, deprecation_errors=True),
    )
    return _MONGO_CLIENT


def get_database_name() -> str:
    return _get_env("MONGODB_DATABASE", "MONGO_DB_NAME", default="ticketing") or "ticketing"


def get_tickets_collection() -> Collection:
    client = get_mongo_client()
    database = client[get_database_name()]
    return database["tickets"]


def ping_mongo() -> None:
    get_mongo_client().admin.command({"ping": 1})


def ensure_ticket_indexes() -> None:
    collection = get_tickets_collection()
    collection.create_index([("requestId", ASCENDING)], unique=True, name="request_id_unique")
    collection.create_index([("createdAt", DESCENDING)], name="created_at_desc")
    collection.create_index([("primaryChoice", ASCENDING), ("createdAt", DESCENDING)], name="route_created_desc")
    collection.create_index([("status", ASCENDING), ("updatedAt", DESCENDING)], name="status_updated_desc")
    collection.create_index([("currentStep", ASCENDING), ("updatedAt", DESCENDING)], name="step_updated_desc")
    collection.create_index([("triage.systemContext", ASCENDING), ("status", ASCENDING)], name="system_status_idx")
    collection.create_index([("triage.severity", ASCENDING), ("updatedAt", DESCENDING)], name="severity_updated_desc")
    collection.create_index([("rca.status", ASCENDING), ("updatedAt", DESCENDING)], name="rca_status_updated_desc")
    collection.create_index([("intake.issueFingerprint", ASCENDING)], name="issue_fingerprint_idx")
    collection.create_index([("embeddings.summary.status", ASCENDING), ("updatedAt", DESCENDING)], name="embedding_status_updated_desc")
    collection.create_index([("embeddings.summary.metadata.project", ASCENDING), ("updatedAt", DESCENDING)], name="embedding_project_updated_desc")
    collection.create_index([("embeddings.summary.metadata.requestType", ASCENDING), ("updatedAt", DESCENDING)], name="embedding_request_type_updated_desc")
    collection.create_index([("embeddings.summary.metadata.reviewType", ASCENDING), ("updatedAt", DESCENDING)], name="embedding_review_type_updated_desc")
    collection.create_index([("embeddings.summary.metadata.errorType", ASCENDING), ("updatedAt", DESCENDING)], name="embedding_error_type_updated_desc")
    collection.create_index([("embeddings.summary.metadata.severity", ASCENDING), ("updatedAt", DESCENDING)], name="embedding_severity_updated_desc")
    ensure_ticket_vector_index(collection)


def ensure_ticket_vector_index(collection: Collection | None = None) -> None:
    if collection is None:
        collection = get_tickets_collection()
    index_definition = get_vector_index_definition(get_vector_dimensions())

    try:
        existing_indexes = list(collection.list_search_indexes())
    except Exception:
        return

    if any(index.get("name") == index_definition["name"] for index in existing_indexes):
        return

    try:
        collection.create_search_index(
            SearchIndexModel(
                definition=index_definition["definition"],
                name=index_definition["name"],
                type=index_definition["type"],
            )
        )
    except Exception:
        # Search index management is Atlas-specific and may be unavailable in some environments.
        return


def find_ticket_by_request_id(request_id: str) -> dict | None:
    return get_tickets_collection().find_one({"requestId": request_id})


def insert_ticket_document(document: dict) -> str:
    now = datetime.now(timezone.utc)
    document.setdefault("createdAt", now)
    document["updatedAt"] = now

    result = get_tickets_collection().insert_one(document)
    return str(result.inserted_id)


def try_insert_failure_document(document: dict) -> str | None:
    try:
        return insert_ticket_document(document)
    except DuplicateKeyError:
        return None


def update_ticket_fields(request_id: str, fields: dict) -> None:
    fields = dict(fields)
    fields["updatedAt"] = datetime.now(timezone.utc)
    get_tickets_collection().update_one({"requestId": request_id}, {"$set": fields})


def append_ticket_status_event(request_id: str, status: str, step: str, message: str) -> None:
    now = datetime.now(timezone.utc)
    get_tickets_collection().update_one(
        {"requestId": request_id},
        {
            "$set": {
                "status": status,
                "currentStep": step,
                "statusMessage": message,
                "updatedAt": now,
            },
            "$push": {
                "statusHistory": {
                    "status": status,
                    "step": step,
                    "message": message,
                    "at": now,
                }
            },
        },
    )
