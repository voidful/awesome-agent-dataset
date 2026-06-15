#!/usr/bin/env bash
# Wait for the detached `agentds run` to finish, sanity-gate the output, then push.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
REPO="${1:-voidful/agent-sft}"
OUT="${2:-data/expansion}"
MIN_ROWS="${3:-50000}"

echo "[wait] waiting for 'agentds.cli run' to finish..."
while pgrep -f "agentds.cli run" >/dev/null 2>&1; do sleep 30; done
echo "[wait] run process exited."

if [ ! -f "$OUT/_report.json" ]; then
  echo "[abort] no $OUT/_report.json — run did not complete cleanly. NOT pushing."
  exit 1
fi

TOTAL=$($PY -c "import json;print(json.load(open('$OUT/_report.json'))['total_written'])")
echo "[gate] total_written=$TOTAL (min $MIN_ROWS)"
if [ "$TOTAL" -lt "$MIN_ROWS" ]; then
  echo "[abort] too few rows ($TOTAL) — likely a failure. NOT pushing. Inspect $OUT/_report.json."
  exit 1
fi

echo "[push] pushing $TOTAL rows to $REPO (public)..."
$PY -m agentds.cli push --repo "$REPO" --out "$OUT" --public
echo "[push] done."
