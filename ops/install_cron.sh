#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cron_file="$ROOT/ops/cron.txt"
if [ ! -f "$cron_file" ]; then
  echo "cron.txt not found" >&2
  exit 1
fi

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

if crontab -l 2>/dev/null > "$tmp"; then
  awk '
    /# BEGIN BNF-K DATA/ {inblock=1; next}
    /# END BNF-K DATA/ {inblock=0; next}
    !inblock {print}
  ' "$tmp" > "${tmp}.clean"
  mv "${tmp}.clean" "$tmp"
else
  : > "$tmp"
fi

cat "$cron_file" >> "$tmp"
crontab "$tmp"

echo "Installed BNF-K cron jobs."
