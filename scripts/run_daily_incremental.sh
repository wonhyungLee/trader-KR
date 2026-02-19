#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYBIN="python3"
if [ -x ".venv/bin/python" ]; then
  PYBIN=".venv/bin/python"
elif [ -x "myenv/bin/python" ]; then
  PYBIN="myenv/bin/python"
fi

export PYTHONUNBUFFERED=1
LOCK_FILE="$ROOT/data/daily_loader.lock"
mkdir -p "$(dirname "$LOCK_FILE")"

if [ -f "$LOCK_FILE" ]; then
  EXISTING_PID="$(tr -d '[:space:]' < "$LOCK_FILE" 2>/dev/null || true)"
  case "$EXISTING_PID" in
    ''|*[!0-9]*) EXISTING_PID=0 ;;
  esac
  if [ "$EXISTING_PID" -gt 0 ] && kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "[run_daily_incremental] skip: daily loader already running pid=$EXISTING_PID"
    exit 0
  fi
fi

echo "$$" > "$LOCK_FILE"
cleanup() { rm -f "$LOCK_FILE"; }
trap cleanup EXIT INT TERM

"$PYBIN" -u -m src.collectors.daily_loader
