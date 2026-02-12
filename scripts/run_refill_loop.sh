#!/usr/bin/env bash
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$(dirname "$DIR")"
cd "$PARENT_DIR"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PARENT_DIR"

echo "Starting continuous refill loop at $(date)"

while true; do
    echo "----------------------------------------------------------------"
    echo "Launching refill process (Full Universe)..."
    
    # Removed --limit 50 to process all remaining stocks in one go
    ./myenv/bin/python -u -m src.collectors.refill_loader \
      --universe data/universe_kospi100.csv \
      --universe data/universe_kosdaq150.csv \
      --chunk-days 150 \
      --resume

    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Refill process finished successfully."
        # If finished successfully, it likely means all stocks are DONE.
        # We can stop the loop or sleep for a long time.
        echo "All done? Checking DB..."
        REMAINING=$(/usr/bin/sqlite3 data/market_data.db "SELECT count(*) FROM refill_progress WHERE status != 'DONE';")
        if [ "$REMAINING" -eq "0" ]; then
             echo "All stocks are DONE. Exiting loop."
             break
        fi
    else
        echo "Refill process crashed with code $EXIT_CODE."
    fi

    echo "Restarting in 10 seconds..."
    sleep 10
done