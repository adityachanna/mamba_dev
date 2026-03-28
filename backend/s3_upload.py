from __future__ import annotations

import mimetypes
import os
import re
from datetime import datetime, timezone
import json
from pathlib import Path

import boto3
from botocore.client import BaseClient
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

_ALLOWED_ROUTES = {"JDI", "JGL"}
_REQUEST_ID_SAFE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
_S3_CLIENT: BaseClient | None = None


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


def get_s3_client() -> BaseClient:
  global _S3_CLIENT

  if _S3_CLIENT is not None:
    return _S3_CLIENT

  endpoint_url = _require_env("S3_API", "R2_ENDPOINT")
  access_key = _require_env("S3_ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID")
  secret_key = _require_env("Secret_Access_Key", "AWS_SECRET_ACCESS_KEY")

  _S3_CLIENT = boto3.client(
    service_name="s3",
    endpoint_url=endpoint_url,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name="auto",
  )
  return _S3_CLIENT


def get_bucket_name() -> str:
  return _require_env("Bucket", "R2_BUCKET_NAME", "AWS_BUCKET_NAME")


def sanitize_request_id(request_id: str) -> str:
  normalized = _REQUEST_ID_SAFE_PATTERN.sub("_", request_id.strip())
  normalized = normalized.strip("._-")
  if not normalized:
    raise ValueError("requestId must contain at least one valid character.")
  return normalized


def build_prefix(route: str, category: str, request_id: str) -> str:
  route_value = route.strip().upper()
  if route_value not in _ALLOWED_ROUTES:
    raise ValueError("primaryChoice must be JDI or JGL")
  safe_request_id = sanitize_request_id(request_id)
  return f"{route_value}/{category}/{safe_request_id}/"


def _guess_content_type(file_name: str, provided_type: str | None) -> str:
  if provided_type:
    return provided_type
  guessed_type, _ = mimetypes.guess_type(file_name)
  return guessed_type or "application/octet-stream"


def _get_endpoint_url() -> str:
  return _require_env("S3_API", "R2_ENDPOINT").rstrip("/")


def _build_object_url(bucket_name: str, object_key: str) -> str:
  return f"{_get_endpoint_url()}/{bucket_name}/{object_key}"


def upload_json_artifact(
  route: str,
  category: str,
  request_id: str,
  filename: str,
  payload: dict[str, object],
) -> dict[str, object]:
  s3_client = get_s3_client()
  bucket_name = get_bucket_name()
  prefix = build_prefix(route, category, request_id)
  object_key = f"{prefix}{filename}"
  body = json.dumps(payload, ensure_ascii=True, indent=2).encode("utf-8")

  put_result = s3_client.put_object(
    Bucket=bucket_name,
    Key=object_key,
    Body=body,
    ContentType="application/json",
  )

  return {
    "bucket": bucket_name,
    "key": object_key,
    "fileName": filename,
    "contentType": "application/json",
    "size": len(body),
    "etag": put_result.get("ETag"),
    "objectUrl": _build_object_url(bucket_name, object_key),
  }


def upload_log_artifact(route: str, request_id: str, message: str) -> dict[str, object]:
  s3_client = get_s3_client()
  bucket_name = get_bucket_name()
  prefix = build_prefix(route, "logs", request_id)
  object_key = f"{prefix}app.log"
  timestamp = datetime.now(timezone.utc).isoformat()
  body_text = f"[{timestamp}] {message}\n"
  body = body_text.encode("utf-8")

  put_result = s3_client.put_object(
    Bucket=bucket_name,
    Key=object_key,
    Body=body,
    ContentType="text/plain",
  )

  return {
    "bucket": bucket_name,
    "key": object_key,
    "fileName": "app.log",
    "contentType": "text/plain",
    "size": len(body),
    "etag": put_result.get("ETag"),
    "objectUrl": _build_object_url(bucket_name, object_key),
  }


def upload_issue_photos(
  route: str,
  request_id: str,
  files: list[dict[str, object]],
) -> dict[str, object]:
  if not files:
    return {
      "bucket": get_bucket_name(),
      "route": route,
      "problemsPrefix": build_prefix(route, "problems", request_id),
      "imageCount": 0,
      "imageObjects": [],
    }

  s3_client = get_s3_client()
  bucket_name = get_bucket_name()
  problems_prefix = build_prefix(route, "problems", request_id)

  uploaded_objects: list[dict[str, object]] = []
  for index, file_info in enumerate(files, start=1):
    file_name = str(file_info.get("filename") or f"image_{index}.bin")
    content = file_info.get("content")
    if not isinstance(content, bytes) or not content:
      continue

    content_type = _guess_content_type(file_name, file_info.get("contentType") if isinstance(file_info.get("contentType"), str) else None)
    object_key = f"{problems_prefix}{index:03d}_{file_name}"

    put_result = s3_client.put_object(
      Bucket=bucket_name,
      Key=object_key,
      Body=content,
      ContentType=content_type,
    )

    uploaded_objects.append(
      {
        "bucket": bucket_name,
        "key": object_key,
        "fileName": file_name,
        "contentType": content_type,
        "size": len(content),
        "etag": put_result.get("ETag"),
        "objectUrl": _build_object_url(bucket_name, object_key),
      }
    )

  return {
    "bucket": bucket_name,
    "route": route,
    "problemsPrefix": problems_prefix,
    "imageCount": len(uploaded_objects),
    "imageObjects": uploaded_objects,
  }
