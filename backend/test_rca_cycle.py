"""
test_rca_cycle.py
=================
End-to-end smoke test for the OpenCode API RCA cycle.

Sends a dummy "Image Resizer not working" complaint through the full
execute_rag_flow() pipeline:
  1. Seeds a minimal ticket with structured analysis into MongoDB.
  2. Ensures the OpenCode server is running (auto-starts if needed).
  3. Runs execute_rag_flow() → RAG routing → OpenCode API RCA.
  4. Reads the ticket back from MongoDB and asserts the expected fields.
  5. Prints a clean summary of the report + context.

Usage (from repo root):
    .venv\\Scripts\\python.exe -m backend.test_rca_cycle
    # or with a custom request ID:
    .venv\\Scripts\\python.exe -m backend.test_rca_cycle --request-id YOUR_ID
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_rca_cycle")

# ──────────────────────────────────────────────────────────────────────────────
# Dummy complaint — "image resizing not working"
# This mirrors what the VLM structurer would produce for a real ticket.
# ──────────────────────────────────────────────────────────────────────────────
DUMMY_STRUCTURED = {
    "short_summary": "Image Resizer fails silently — output file is never produced.",
    "error_type": "silent_failure",
    "structured_problem": (
        "User uploaded a 4 MB JPEG and selected a target width of 800 px. "
        "After clicking Resize, the progress spinner appeared for ~3 seconds and then "
        "disappeared with no output file and no error message shown in the UI. "
        "The browser console shows a 500 response from POST /api/resize. "
        "The server log captured: 'TypeError: Cannot read properties of undefined (reading \"buffer\")' "
        "originating from the sharp integration layer."
    ),
    "enumerated_report": [
        "1. User uploads image via the Resizer tab.",
        "2. Frontend POSTs multipart/form-data to /api/resize with target dimensions.",
        "3. Backend receives the request but sharp throws TypeError on undefined .buffer.",
        "4. 500 is returned; frontend swallows the error — spinner disappears silently.",
        "5. No output file is stored; user sees a blank result area.",
    ],
    "image_evidence": [],
    "embedding_text": (
        "Image Studio Resizer silent 500 error sharp TypeError undefined buffer "
        "resize fails no output no error message"
    ),
    "error_class": "runtime_exception",
    "affected_feature": "Image Resizer",
    "severity": "high",
    "user_description": "Resizing is not working at all. I click Resize and nothing happens.",
}


def _seed_ticket(request_id: str) -> None:
    """Insert a minimal ticket into MongoDB that looks like a post-VLM record."""
    from backend.db import get_tickets_collection

    now = datetime.now(timezone.utc)
    doc = {
        "requestId": request_id,
        "reviewType": "Image Studio",
        "requestType": "error_report",
        "primaryChoice": "resize",
        "status": "processing",
        "currentStep": "rca_pending",
        "statusMessage": "Awaiting RCA.",
        "createdAt": now,
        "updatedAt": now,
        "storage": {
            "route": "resize",
            "imageCount": 1,
            "imageObjects": [],
            "inputArtifact": None,
            "outputArtifact": None,
            "logArtifacts": None,
        },
        "artifactUrls": {},
        "analysis": {
            "model": "gemini-2.0-flash",
            "imageCount": 1,
            "embeddingText": DUMMY_STRUCTURED["embedding_text"],
            "structured": DUMMY_STRUCTURED,
        },
        "embeddings": {
            "summary": {
                "status": "completed",
                "vector": [],          # empty — not needed for RCA path
                "metadata": {
                    "reviewType": "Image Studio",
                    "errorType": "silent_failure",
                    "severity": "high",
                },
            }
        },
        "workflow": {},
        "rca": {"status": "pending", "eligible": True, "result": None},
    }
    coll = get_tickets_collection()
    # Remove any leftover test doc with same ID
    coll.delete_one({"requestId": request_id})
    coll.insert_one(doc)
    logger.info("✅ Seeded test ticket requestId=%s", request_id)


def _ensure_server_running() -> bool:
    """Best-effort check that the OpenCode server is up before the flow starts."""
    import requests as req
    base = os.getenv("OPENCODE_BASE_URL", "http://localhost:4096")
    try:
        r = req.get(f"{base}/global/health", timeout=5)
        if r.status_code == 200:
            logger.info("✅ OpenCode server already healthy at %s", base)
            return True
    except Exception:
        pass

    logger.info("🚀 OpenCode server not detected — attempting auto-start…")
    try:
        from backend.initiate_openrouter import start_opencode_server
        start_opencode_server(
            port=int(os.getenv("OPENCODE_PORT", "4096")),
            hostname=os.getenv("OPENCODE_HOSTNAME", "127.0.0.1"),
            log_file=str(Path(__file__).resolve().parent.parent / "opencode_server.log"),
        )
    except Exception as e:
        logger.error("❌ Could not auto-start OpenCode server: %s", e)
        return False

    # Poll for up to 20 s
    for i in range(20):
        time.sleep(1)
        try:
            r = req.get(f"{base}/global/health", timeout=3)
            if r.status_code == 200:
                logger.info("✅ OpenCode server ready after %ds", i + 1)
                return True
        except Exception:
            pass

    logger.error("❌ OpenCode server did not become healthy within 20s")
    return False


def _print_result(result: dict) -> None:
    """Pretty-print the RCA result."""
    sep = "─" * 72
    status = result.get("status", "unknown")
    icon = "✅" if "completed" in status or status == "matched" else "❌"

    print(f"\n{sep}")
    print(f"  {icon}  RCA CYCLE RESULT  —  status: {status.upper()}")
    print(sep)
    print(f"  requestId : {result.get('requestId')}")
    print(f"  decision  : {(result.get('decision') or {}).get('flow', 'n/a')}")
    print(f"  sessionId : {result.get('opencodeSessionId', 'n/a')}")
    print(f"  exitCode  : {result.get('exitCode', 'n/a')}")
    print(f"  timedOut  : {result.get('timedOut', False)}")
    print(sep)

    report = result.get("report", "")
    if report:
        print("\n📄  REPORT (first 2000 chars):\n")
        print(report[:2000])
        if len(report) > 2000:
            print(f"\n… [{len(report) - 2000} more chars]")
    else:
        print("\n⚠️  No report text captured.")

    ctx = result.get("context") or {}
    if ctx.get("markdown"):
        md_preview = ctx["markdown"][:500]
        print(f"\n📋  CONTEXT MARKDOWN (first 500 chars):\n{md_preview}")

    print(f"\n{sep}\n")


def _assert_db_record(request_id: str) -> None:
    """Validate DB fields after the flow completes."""
    from backend.db import find_ticket_by_request_id

    ticket = find_ticket_by_request_id(request_id)
    assert ticket, f"Ticket {request_id} not found in DB after RCA"

    rca = ticket.get("rca") or {}
    result = rca.get("result") or {}

    assert rca.get("eligible") is True, "rca.eligible should be True"
    assert rca.get("status") in {"completed", "failed", "timed_out", "skipped_duplicate"}, \
        f"Unexpected rca.status: {rca.get('status')}"

    # For opencode_api path — must have report + context
    if result.get("source") == "opencode_api":
        assert "report" in result, "rca.result must contain 'report'"
        assert "context" in result, "rca.result must contain 'context'"
        assert isinstance(result["context"], dict), "'context' must be a dict"
        logger.info("✅ DB assertions passed (source=opencode_api)")
    elif result.get("source") == "vector_search":
        assert result.get("matchedRequestId"), "matchedRequestId required for vector_search source"
        logger.info("✅ DB assertions passed (source=vector_search / dedup match)")
    else:
        logger.warning("⚠️  Unexpected source=%s — skipping deep assertions", result.get("source"))


def run_test(request_id: str | None = None) -> int:
    request_id = request_id or f"test-rca-resize-{uuid.uuid4().hex[:8]}"
    logger.info("=" * 60)
    logger.info("  RCA CYCLE TEST  —  requestId=%s", request_id)
    logger.info("=" * 60)

    # 1. Seed the ticket
    _seed_ticket(request_id)

    # 2. Make sure OpenCode is running
    _ensure_server_running()

    # 3. Run the full RAG → RCA flow
    logger.info("▶  Running execute_rag_flow()…")
    from backend.opencode_orchestrator import execute_rag_flow, DEFAULT_REPO_DIR

    t0 = time.time()
    try:
        result = execute_rag_flow(request_id, repo_dir=DEFAULT_REPO_DIR)
    except Exception as exc:
        logger.error("❌ execute_rag_flow() raised an exception: %s", exc, exc_info=True)
        return 1

    elapsed = time.time() - t0
    logger.info("⏱  Flow completed in %.1fs", elapsed)

    # 4. Print result
    _print_result(result)

    # 5. Validate DB
    try:
        _assert_db_record(request_id)
    except AssertionError as ae:
        logger.error("❌ DB assertion failed: %s", ae)
        return 1

    status = result.get("status", "")
    success = "completed" in status or status == "matched"

    if success:
        logger.info("✅  TEST PASSED — requestId=%s  status=%s", request_id, status)
        return 0
    else:
        logger.error("❌  TEST FAILED — requestId=%s  status=%s", request_id, status)
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="RCA cycle smoke test — dummy image resize complaint.")
    parser.add_argument("--request-id", default=None, help="Use a specific requestId (auto-generated if omitted)")
    args = parser.parse_args()
    sys.exit(run_test(args.request_id))


if __name__ == "__main__":
    main()
