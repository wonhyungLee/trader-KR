#!/usr/bin/env bash
set -euo pipefail
cron_file="/home/ubuntu/종목선별매매프로그램/ops/cron.txt"
if [ ! -f "$cron_file" ]; then
  echo "cron.txt not found" >&2
  exit 1
fi

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

crontab -l 2>/dev/null | awk '
  /# BEGIN BNF-K DATA/ {inblock=1; next}
  /# END BNF-K DATA/ {inblock=0; next}
  !inblock {print}
' > "$tmp"

cat "$cron_file" >> "$tmp"
crontab "$tmp"

echo "Installed BNF-K cron jobs."
