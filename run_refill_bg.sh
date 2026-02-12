#!/bin/bash
cd /home/dldnjsrk/remote-ubuntu-5/종목선별매매프로그램
/usr/bin/sqlite3 data/market_data.db "DELETE FROM refill_progress;"
./myenv/bin/python -u -m src.collectors.refill_loader \
  --universe data/universe_kospi100.csv \
  --universe data/universe_kosdaq150.csv \
  --chunk-days 150 \
  --notify-every 100 \
  > refill_debug.log 2>&1