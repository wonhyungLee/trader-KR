# Sector-Based Universe (KOSPI100 + KOSDAQ150)

This project uses a fixed universe of 250 tickers based on:
- `data/universe_kospi100.csv` (100)
- `data/universe_kosdaq150.csv` (150)

The sector/industry classification is stored in DB (`sector_map`) and exported to CSVs under `data/universe_sectors`.

## 1) Sector Classification Build

Run the sector classifier (cached by `updated_at`):

```bash
python -m src.collectors.sector_classifier --refresh-days 30
```

What it does:
- Reads the 250 `universe_members`.
- Calls KIS stock info APIs to fill `sector_map`.
- Skips symbols updated within the last N days.
- Generates sector CSVs:
  - `data/universe_sectors/KOSPI/{sector_name}.csv`
  - `data/universe_sectors/KOSDAQ/{sector_name}.csv`
  - `data/universe_sectors/UNKNOWN.csv` (missing sector)

**Rate-limit safety**
- Uses the existing `KISBroker` retry/backoff/sleep logic.
- Per-item sleep is configurable via `--sleep` or `kis.accuracy_item_sleep_sec`.

## 2) Universe CSV Update (Diff Tracking)

When the universe CSVs change, record diffs and update DB:

```bash
python -m src.collectors.universe_loader --record-diff
```

This will:
- Compare the new CSV snapshot with existing `universe_members`.
- Write diff rows into `universe_changes`.
- Update `universe_members` and keep the universe size at **250**.

## 3) Optional auto_rank Mode (Market Cap Top N)

You can re-generate the universe by market cap rank:

```bash
python -m src.collectors.universe_loader --auto-rank --top-kospi 100 --top-kosdaq 150
```

**Important limitation**
- This mode depends on `stock_info.marcap` being populated.
- The resulting list can **differ from the official KOSPI100/KOSDAQ150 index 구성**.

## 4) Data Freshness / Cache

Sector classification is cached in `sector_map`:
- Entries updated within `--refresh-days` are not re-fetched.
- The cache reduces API calls and prevents rate-limit issues.

## 5) Acceptance Criteria Checks

- `universe_members` is always **250**
- `sector_map.sector_name != NULL` should be **>= 95%**
- `data/universe_sectors` contains sector CSVs whose total code count is 250 (no duplicates / no missing)

Unknowns are exported to:
- `data/universe_sectors/UNKNOWN.csv`
