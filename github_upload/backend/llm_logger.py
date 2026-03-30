import json
import os
from datetime import datetime, timezone
from typing import Any
import logging

logger = logging.getLogger(__name__)

LLM_LOG_FILE = os.path.join(os.path.dirname(__file__), "llm_responses.log")


def log_llm_response(source: str, request_id: str | None, response: Any, prompt: Any = None) -> None:
    """Appends an LLM response to a shared local file."""
    try:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "requestId": request_id,
        }

        def _safe_serialize(obj: Any) -> Any:
            try:
                if hasattr(obj, "model_dump"):
                    return obj.model_dump()
                return json.loads(json.dumps(obj, default=str))
            except Exception:
                return str(obj)

        if prompt is not None:
            entry["prompt"] = _safe_serialize(prompt)
        entry["response"] = _safe_serialize(response)

        with open(LLM_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error("Failed to log LLM response: %s", e)
