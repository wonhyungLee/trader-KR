#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$ROOT" ]; then
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi
cd "$ROOT"
export PYTHONUNBUFFERED=1

PYBIN="python3"
if [ -x ".venv/bin/python" ]; then
  PYBIN=".venv/bin/python"
elif [ -x "myenv/bin/python" ]; then
  PYBIN="myenv/bin/python"
fi

INTERVAL_ARG=()
if [ -n "${WATCHDOG_INTERVAL_SEC:-}" ]; then
  INTERVAL_ARG=(--interval "$WATCHDOG_INTERVAL_SEC")
fi

$PYBIN -u -m src.utils.data_watchdog "${INTERVAL_ARG[@]}"
