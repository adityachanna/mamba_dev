#!/bin/bash
set -euo pipefail

REQUEST_ID="${1:-}"
REPO_URL="${2:-${TARGET_REPO_URL:-}}"
REPO_DIR_NAME="${REPO_DIR_NAME:-image}"
OPENCODE_BIN="${OPENCODE_BIN:-opencode}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${ROOT_DIR}/${REPO_DIR_NAME}"
LOCAL_INSTRUCTIONS="${REPO_DIR}/opencode_readonly.md"

if [[ -z "${REQUEST_ID}" ]]; then
  echo "Usage: $0 <request-id> [repo-url]"
  exit 1
fi

if [[ ! -d "${REPO_DIR}" ]]; then
  if [[ -z "${REPO_URL}" ]]; then
    echo "Repository directory '${REPO_DIR_NAME}' does not exist and no repo URL was provided."
    exit 1
  fi
  git clone "${REPO_URL}" "${REPO_DIR_NAME}"
fi

cd "${REPO_DIR}"

cat > "${LOCAL_INSTRUCTIONS}" <<'EOF'
You are running in read-only repository investigation mode.

Primary objective:
- Use the structured incident details provided in the OpenCode prompt to inspect the cloned repository and produce a deep planning report.

Hard rules:
- Do not edit, create, delete, or rename files.
- Do not run install, build, test, migration, formatter, or package-manager commands.
- Do not change git state.
- Only inspect source code, configuration, dependency manifests, and repository structure.

Investigation priorities:
- Find the most likely code paths related to the structured incident.
- Prefer concrete evidence over speculation.
- Trace request flow, UI flow, data transformation, validation, persistence, and error handling when relevant.
- Identify suspicious conditions, missing guards, weak assumptions, and places where the code could produce the observed behavior.
- If the report references a file, include the full repository-relative path and explain why it matters.

Required output:
- Executive summary
- Most likely root causes
- Relevant code paths and directories
- Specific suspicious code patterns or missing checks
- What information is still missing
- Validation plan
- Recommended implementation plan

Important style:
- Keep the report technical and specific.
- Prefer short evidence-backed bullets over vague prose.
EOF

cat > .opencode.json <<EOF
{
  "permission": "allow",
  "instructions": [
    "${LOCAL_INSTRUCTIONS}"
  ]
}
EOF

echo "Running vector dedup and OpenCode plan for request '${REQUEST_ID}' in '${REPO_DIR}'."

cd "${ROOT_DIR}"
uv run python -m backend.opencode_orchestrator \
  --request-id "${REQUEST_ID}" \
  --repo-dir "${REPO_DIR}" \
  --opencode-bin "${OPENCODE_BIN}"
