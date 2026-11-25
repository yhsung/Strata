#!/usr/bin/env bash
set -euo pipefail

STRATA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${STRATA_ROOT}/terminal/.venv/bin/python"
if [[ ! -x "$PY_BIN" ]]; then
  PY_BIN="$(command -v python3)"
fi

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

STRATA_PORT="${STRATA_PORT:-8095}"
STRATA_HOST="${STRATA_HOST:-127.0.0.1}"
STRATA_BIN="${STRATA_BIN:-${STRATA_ROOT}/engine/build-metal/strata-q36-metal}"
STRATA_TOKENIZER="${STRATA_TOKENIZER:-${STRATA_ROOT}/pack/qwen36-ud-q4ks/tokenizer}"
STRATA_EXPERT_PROFILE="${STRATA_EXPERT_PROFILE:-${STRATA_ROOT}/pack/qwen36-ud-q4ks/expert-profile.bin}"
STRATA_CTX="${STRATA_CTX:-8192}"
STRATA_ARENA_GB="${STRATA_ARENA_GB:-4}"
STRATA_MTP="${STRATA_MTP:-2}"
STRATA_PARALLEL="${STRATA_PARALLEL:-2}"
STRATA_STREAM="${STRATA_STREAM:-1}"

LOCAL_CFG="${STRATA_ROOT}/examples/strata_qwen36_config.local.json"
TEMP_CFG=""
cleanup() {
  if [[ -n "$TEMP_CFG" && -f "$TEMP_CFG" ]]; then
    rm -f "$TEMP_CFG"
  fi
}
trap cleanup EXIT

if [[ -n "${STRATA_CONFIG:-}" ]]; then
  CFG_PATH="$STRATA_CONFIG"
elif [[ -z "${QWEN36_GGUF:-}" && -f "$LOCAL_CFG" ]]; then
  CFG_PATH="$LOCAL_CFG"
else
  if [[ -z "${QWEN36_GGUF:-}" ]]; then
    cat >&2 <<'EOF'
Usage:
  QWEN36_GGUF=/path/to/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf ./examples/start_strata_qwen36.sh [--dry-run]

Optional environment variables:
  STRATA_CONFIG          Explicit config JSON path (overrides QWEN36_GGUF)
  STRATA_BIN             Engine binary (default: <repo>/engine/build-metal/strata-q36-metal)
  STRATA_TOKENIZER       Tokenizer path (default: <repo>/pack/qwen36-ud-q4ks/tokenizer)
  STRATA_EXPERT_PROFILE  Optional STRP profile path
  STRATA_PORT            Server port (default: 8095)
  STRATA_HOST            Bind host (default: 127.0.0.1)
  STRATA_CTX             Context window (default: 8192)
  STRATA_STREAM          1 to enable SSD ExpertStream (default: 1)
  STRATA_ARENA_GB        Stream expert arena GiB (default: 4)
  STRATA_MTP             MTP drafts count, 0 to disable (default: 2)
  STRATA_PARALLEL        Parallel batch slots (default: 2)
EOF
    exit 2
  fi
  if [[ ! -f "$QWEN36_GGUF" ]]; then
    echo "Error: QWEN36_GGUF not found: $QWEN36_GGUF" >&2
    exit 2
  fi
  TEMP_CFG="$(mktemp "${TMPDIR:-/tmp}/strata_qwen36_config.XXXXXX")"
  STRATA_ROOT="$STRATA_ROOT" \
  STRATA_BIN="$STRATA_BIN" \
  QWEN36_GGUF="$QWEN36_GGUF" \
  STRATA_TOKENIZER="$STRATA_TOKENIZER" \
  STRATA_EXPERT_PROFILE="$STRATA_EXPERT_PROFILE" \
  STRATA_HOST="$STRATA_HOST" \
  STRATA_CTX="$STRATA_CTX" \
  STRATA_STREAM="$STRATA_STREAM" \
  STRATA_ARENA_GB="$STRATA_ARENA_GB" \
  STRATA_MTP="$STRATA_MTP" \
  STRATA_PARALLEL="$STRATA_PARALLEL" \
  "$PY_BIN" - "$TEMP_CFG" <<'PY'
import json
import os
import sys

out_path = sys.argv[1]
args = ["--native", os.environ["QWEN36_GGUF"], "--max-context", os.environ["STRATA_CTX"]]
if os.environ.get("STRATA_STREAM", "1") != "0":
    args += ["--stream", "--arena-gb", os.environ["STRATA_ARENA_GB"]]
prof = os.environ.get("STRATA_EXPERT_PROFILE", "")
if prof and os.path.isfile(prof):
    args += ["--expert-profile", prof]
mtp = int(os.environ.get("STRATA_MTP", "0"))
if mtp > 0:
    args += ["--mtp", str(mtp)]

cfg = {
    "arch": "qwen35moe",
    "exe": os.environ["STRATA_BIN"],
    "args": args,
    "parallel": int(os.environ["STRATA_PARALLEL"]),
    "cwd": os.environ["STRATA_ROOT"],
    "tokenizer": os.environ["STRATA_TOKENIZER"],
    "model_name": "qwen3.6-35b-a3b",
    "reasoning_effort": "none",
    "mtp": mtp > 0,
    "host": os.environ["STRATA_HOST"],
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PY
  CFG_PATH="$TEMP_CFG"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "CFG_PATH=$CFG_PATH"
  cat "$CFG_PATH"
  echo "CMD: $PY_BIN $STRATA_ROOT/terminal/serve/server.py --engine strata --config $CFG_PATH --port $STRATA_PORT --no-api-key --offline"
  exit 0
fi

export PYTHONPATH="${STRATA_ROOT}/terminal:${STRATA_ROOT}/terminal/tools:${PYTHONPATH:-}"
exec "$PY_BIN" "${STRATA_ROOT}/terminal/serve/server.py" \
  --engine strata \
  --config "$CFG_PATH" \
  --port "$STRATA_PORT" \
  --no-api-key \
  --offline
