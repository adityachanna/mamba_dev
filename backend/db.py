from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError
from pymongo.server_api import ServerApi

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
