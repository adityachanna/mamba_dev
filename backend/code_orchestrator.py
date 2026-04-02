import os
import json
import time
import requests
from typing import Dict, Any, Optional
from langchain_core.tools import tool

# ========================= CONFIG =========================
BASE_URL = os.getenv("OPENCODE_BASE_URL", "http://localhost:4096")
USERNAME = os.getenv("OPENCODE_SERVER_USERNAME", "opencode")
PASSWORD = os.getenv("OPENCODE_SERVER_PASSWORD")
SESSION_ID_ENV_VAR = "OPENCODE_SESSION_ID"
OPENCODE_PROVIDER_ID = os.getenv("OPENCODE_PROVIDER_ID", "anthropic")
OPENCODE_MODEL_ID = os.getenv("OPENCODE_MODEL_ID", "claude-3-5-sonnet-latest")

AUTH = (USERNAME, PASSWORD)


# ========================= HELPERS =========================
def _get_json(resp: requests.Response) -> Any:
    resp.raise_for_status()
    return resp.json()

# ========================= TOOL 2: Get or Create Session =========================
@tool
def opencode_get_or_create_session(title: str = "Error-RCA-Agent-Session") -> str:
    """Create a persistent session (or reuse existing one via env var)."""
    session_id = os.getenv(SESSION_ID_ENV_VAR)
    if session_id:
        try:
            r = requests.get(f"{BASE_URL}/session/{session_id}", auth=AUTH, timeout=5)
            if r.status_code == 200:
                return session_id
        except:
            pass

    payload = {"title": title}
    r = requests.post(f"{BASE_URL}/session", json=payload, auth=AUTH, timeout=10)
    data = _get_json(r)
    session_id = data.get("id") or data.get("sessionId")
    os.environ[SESSION_ID_ENV_VAR] = session_id
    return session_id


# ========================= TOOL 3: Init Session (loads repo + AGENTS.md) =========================
@tool
def opencode_init_session(
    provider_id: Optional[str] = None,
    model_id: Optional[str] = None,
) -> str:
    """Run /init so OpenCode understands your entire codebase."""
    session_id = opencode_get_or_create_session.invoke({})
    provider = provider_id or OPENCODE_PROVIDER_ID
    model = model_id or OPENCODE_MODEL_ID
    payload = {"providerID": provider, "modelID": model}
    requests.post(f"{BASE_URL}/session/{session_id}/init", json=payload, auth=AUTH, timeout=30)
    return f"Session {session_id} initialized with AGENTS.md"


# ========================= TOOL 4: Send Plan Message (RCA style) =========================
@tool
def opencode_send_plan_message(error_structured: dict) -> str:
    """Send to official Plan agent (per OpenCode docs)."""
    session_id = opencode_get_or_create_session.invoke({})
    opencode_init_session.invoke({})

    prompt = f"""You are the official Plan agent.
Use full codebase context from AGENTS.md.

Error data:
- Short summary: {error_structured.get('short_summary', '')}
- Error type: {error_structured.get('error_type', '')}
- Structured problem: {error_structured.get('structured_problem', '')}
- Enumerated report: {error_structured.get('enumerated_report', [])}
- Image evidence: {error_structured.get('image_evidence', [])}

TASK (Plan mode):
1. List exactly which files/functions are causing this error.
2. Step-by-step root cause hypothesis with evidence.
3. Concrete fix plan with suggested code changes."""

    payload = {
        "parts": [{"role": "user", "content": prompt}],
        "agent": "plan",          # ← Official Plan agent from OpenCode docs
    }

    r = requests.post(f"{BASE_URL}/session/{session_id}/message", json=payload, auth=AUTH, timeout=60)
    result = _get_json(r)

    # Extract final assistant response
    parts = result.get("parts", [])
    for p in reversed(parts):
        if p.get("role") == "assistant" or p.get("type") == "text":
            return p.get("content", str(p))
    return str(result)


# ========================= TOOL 5: EXPORT FULL CONTEXT (JSON + Markdown) =========================
@tool
def opencode_export_full_context(session_id: Optional[str] = None) -> Dict[str, str]:
    """
    Export complete session context.
    Returns:
      - 'json_export': JSON string ready for `opencode import exported.json`
      - 'markdown_export': Clean Markdown transcript + diffs
    """
    if not session_id:
        session_id = opencode_get_or_create_session.invoke({})

    # 1. Session details
    session_data = _get_json(requests.get(f"{BASE_URL}/session/{session_id}", auth=AUTH))

    # 2. All messages
    messages = _get_json(requests.get(f"{BASE_URL}/session/{session_id}/message", auth=AUTH))

    # 3. Diffs
    diff = _get_json(requests.get(f"{BASE_URL}/session/{session_id}/diff", auth=AUTH))

    # 4. Summary
    summary_payload = {"providerID": OPENCODE_PROVIDER_ID, "modelID": OPENCODE_MODEL_ID}
    requests.post(f"{BASE_URL}/session/{session_id}/summarize", json=summary_payload, auth=AUTH)

    # Combine into CLI-compatible export JSON
    export_json = {
        "info": session_data,
        "messages": messages,
        "diff": diff,
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Generate nice Markdown version
    md = f"# OpenCode Session Export\n\n**Session ID:** {session_id}\n**Exported:** {export_json['exported_at']}\n\n"
    md += "## Summary\n" + (str(summary_payload) if summary_payload else "No summary generated\n\n")
    md += "## Messages\n\n"
    for msg in messages:
        role = msg.get("info", {}).get("role", "unknown")
        content = msg.get("parts", [{}])[0].get("content", "")
        md += f"### {role.upper()}\n{content}\n\n"
    md += "## Diffs / File Changes\n" + json.dumps(diff, indent=2) + "\n"

    json_str = json.dumps(export_json, indent=2)

    # You can save these files automatically if you want
    # with open(f"opencode-export-{session_id[:8]}.json", "w") as f: f.write(json_str)
    # with open(f"opencode-export-{session_id[:8]}.md", "w") as f: f.write(md)

    return {
        "json_export": json_str,
        "markdown_export": md,
        "session_id": session_id,
        "message": "✅ Export ready! Save the JSON and run `opencode import filename.json` in CLI."
    }


# ========================= BONUS TOOLS (very useful) =========================
@tool
def opencode_list_sessions() -> str:
    """List all sessions."""
    data = _get_json(requests.get(f"{BASE_URL}/session", auth=AUTH))
    return json.dumps(data, indent=2)


@tool
def opencode_get_diff(session_id: Optional[str] = None) -> str:
    """Get file diffs for the current session."""
    if not session_id:
        session_id = opencode_get_or_create_session.invoke({})
    data = _get_json(requests.get(f"{BASE_URL}/session/{session_id}/diff", auth=AUTH))
    return json.dumps(data, indent=2)


# ========================= HOW TO USE IN YOUR LANGGRAPH PIPELINE ========================
# =============================================================================
# How to use in LangGraph / LangChain
# =============================================================================
# In your RCA node:
# rca_result = opencode_rca_plan_tool.invoke({
#     "error_structured": state["structured_error"],
#     "provider_id": "anthropic",
#     "model_id": "claude-3-5-sonnet-latest"
# })
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver  # or PostgresSaver for production

# ========================= STATE =========================
class RCAState(TypedDict):
    structured_error: dict                     # from Stage 1 VLM
    is_new_error: bool                         # from Stage 2 dedup
    rca_plan: Annotated[str, "The full RCA plan from OpenCode"]
    export_json: Annotated[str, "JSON ready for opencode import"]
    export_md: Annotated[str, "Human-readable Markdown"]
    final_error_record: dict                   # ready to save to DB


# ========================= RCA NODE (the core flow you asked for) =========================
def rca_node(state: RCAState) -> RCAState:
    """
    Full RCA flow:
    1. Get/create persistent session
    2. Init (loads codebase + AGENTS.md)
    3. Send structured error to Plan agent (or custom error-rca agent)
    4. Save the plan
    5. Export full context (JSON + MD)
    """
    error = state["structured_error"]

    # 1. Session
    session_id = opencode_get_or_create_session.invoke({"title": "Error-RCA-Agent-Session"})

    # 2. Init → AGENTS.md + full repo context
    opencode_init_session.invoke({
        "provider_id": OPENCODE_PROVIDER_ID,
        "model_id": OPENCODE_MODEL_ID,
    })

    # 3. Send to Plan Agent (official way per docs)
    plan_result = opencode_send_plan_message.invoke({"error_structured": error})
    # (Inside the tool we already set "agent": "plan" — see below)

    # 4. Save plan
    state["rca_plan"] = plan_result

    # 5. Export full context (JSON for CLI import + MD)
    export = opencode_export_full_context.invoke({"session_id": session_id})
    state["export_json"] = export["json_export"]
    state["export_md"] = export["markdown_export"]

    # 6. Build final record for your database
    state["final_error_record"] = {
        "structured_error": error,
        "rca_plan": plan_result,
        "opencode_session_id": session_id,
        "export_json": export["json_export"],
        "export_md": export["markdown_export"],
        "timestamp": "now",  # replace with actual timestamp
    }

    return state
def build_rca_graph(vlm_structurer_node, dedup_node):
    """Build RCA graph using caller-provided structurer and dedup nodes."""
    graph_builder = StateGraph(RCAState)
    graph_builder.add_node("structurer", vlm_structurer_node)
    graph_builder.add_node("dedup", dedup_node)
    graph_builder.add_node("rca", rca_node)

    def route_after_dedup(state: RCAState):
        return "rca" if state.get("is_new_error", True) else END

    graph_builder.add_conditional_edges("dedup", route_after_dedup, ["rca", END])
    graph_builder.add_edge("structurer", "dedup")
    graph_builder.set_entry_point("structurer")
    checkpointer = MemorySaver()
    return graph_builder.compile(checkpointer=checkpointer)


# Example usage:
# compiled_graph = build_rca_graph(vlm_structurer_node, dedup_node)
# result = compiled_graph.invoke({"structured_error": structured_error, "is_new_error": True})