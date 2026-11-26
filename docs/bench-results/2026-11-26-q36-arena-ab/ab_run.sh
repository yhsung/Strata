#!/usr/bin/env bash
# One A/B cell, end to end: build a variant config from the canonical JSON,
# restart the server on that config, run ab_probe.py, stop the server.
#
# Usage:
#   ./ab_run.sh <arena_gb> <mtp_depth> <label> [canonical_config] [port]
#
# Example (reproduces the arena-16 row of runs.tsv):
#   ./ab_run.sh 16 2 verify-arena16
#
# All paths below are absolute, so run it from anywhere. It stops any Strata
# serve instance on the same port first, and always stops the server it started.
set -euo pipefail

ARENA="${1:?usage: ab_run.sh <arena_gb> <mtp_depth> <label> <max_context> [canonical_config] [port]}"
MTP="${2:?usage: ab_run.sh <arena_gb> <mtp_depth> <label> <max_context> [canonical_config] [port]}"
LABEL="${3:?usage: ab_run.sh <arena_gb> <mtp_depth> <label> <max_context> [canonical_config] [port]}"
CTX="${4:?usage: ab_run.sh <arena_gb> <mtp_depth> <label> <max_context> [canonical_config] [port]}"
CANON="${5:-/Volumes/Samsung970EVOPlus/dev-projects/codeact/strata_qwen36_config.json}"
PORT="${6:-8095}"
STRATA_ROOT="/Volumes/Samsung970EVOPlus/dev-projects/strata-q36"

TMP="$(mktemp /tmp/strata_ab_variant_XXXXXX.json)"
LOG="/tmp/strata_ab_${PORT}.log"
SRV=""
cleanup() {
  if [ -n "$SRV" ]; then kill "$SRV" 2>/dev/null || true; wait "$SRV" 2>/dev/null || true; fi
  rm -f "$TMP"
}
trap cleanup EXIT

# Variant config = canonical JSON with --arena-gb / --mtp / --max-context rewritten
# (the context is pinned per report; inheriting the live value mislabels reruns).
python3 - "$CANON" "$TMP" "$ARENA" "$MTP" "$CTX" <<'EOF'
import json, sys
src, dst, arena, mtp, ctx = sys.argv[1:6]
d = json.load(open(src))
a = list(d["args"])
a[a.index("--arena-gb") + 1] = arena
a[a.index("--mtp") + 1] = mtp
a[a.index("--max-context") + 1] = ctx
d["args"] = a
json.dump(d, open(dst, "w"), indent=2)
EOF

pkill -f "serve/server.py.*--port ${PORT}" 2>/dev/null || true
sleep 3

PYTHONPATH="$STRATA_ROOT/terminal:$STRATA_ROOT/terminal/tools" \
  "$STRATA_ROOT/terminal/.venv/bin/python" "$STRATA_ROOT/terminal/serve/server.py" \
  --engine strata --config "$TMP" --port "$PORT" --no-api-key --offline \
  >"$LOG" 2>&1 &
SRV=$!

ready=0
for _ in $(seq 1 60); do
  if curl -sf -m 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then ready=1; break; fi
  sleep 5
done
if [ "$ready" -ne 1 ]; then
  echo "server did not become ready; see $LOG" >&2
  kill "$SRV" 2>/dev/null || true
  exit 1
fi

python3 "$(dirname "$0")/ab_probe.py" --port "$PORT" --label "$LABEL"

