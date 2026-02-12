# BNF-K 데이터 파이프라인 가이드 (250 종목 유니버스)

이 문서는 유니버스(KOSPI 100 + KOSDAQ 150) 250종목에 대해 데이터 리필 및 증분 업데이트를 수행하는 절차를 설명합니다.

## 1. 개요
본 시스템은 오직 고정된 250개 종목만을 대상으로 하며, 모든 선별/매매/분석은 이 유니버스 안에서만 이루어집니다.

- **유니버스**: `data/universe_kospi100.csv` + `data/universe_kosdaq150.csv` (총 250개)
- **저장소**: SQLite (`data/market_data.db`)
- **수집원**: 한국투자증권(KIS) OpenAPI (REST)

## 2. 데이터 수집 단계 (초기 구축 시)

### Step 1: 유니버스 초기화
유니버스 멤버를 DB에 등록하고 `stock_info`를 250개로 고정합니다.
```bash
python -m src.collectors.universe_loader
```
* 결과: `universe_members` 테이블에 250개 적재, `stock_info` 250개로 초기화.

### Step 2: 히스토리 리필 (Refill)
각 종목의 과거 데이터부터 현재까지 빈틈없이 수집합니다. KIS 일봉 API의 응답 제한(100건)을 고려해 자동으로 시간축을 과거로 이동하며 호출합니다.
```bash
# 또는 ./run_refill.sh 실행
python -m src.collectors.refill_loader --source kis --resume
```
* **동작 원리**: 오늘부터 시작해 과거로 150일씩 chunk를 나누어 호출하며, 응답의 `min_date`를 다음 호출의 `end_date`로 설정하여 빈틈없이 채웁니다.
* **중단 재개**: `--resume` 옵션 사용 시 `refill_progress` 테이블을 참조해 마지막 `DONE`이 아닌 종목부터 이어서 수행합니다.

## 3. 일일 운영 (운영 단계)

### 증분 업데이트 (Daily Update)
장 마감 후(16:00 이후) 마지막 저장일 이후의 데이터만 수집합니다.
```bash
python -m src.collectors.daily_loader
```
* 결과: 각 종목의 `max(date) + 1`부터 `today`까지의 데이터만 효율적으로 수집.

## 4. 모니터링 및 시각화
수집 현황 및 엔진 동작 상태는 대시보드에서 확인할 수 있습니다.

1. **백엔드 실행**: `python server.py` (5000 포트)
2. **프론트엔드 실행**: `cd frontend && npm run dev` (5173 포트)
3. **확인 항목**:
   - **데이터 준비도**: 유니버스 250개 중 몇 개가 수집되었는지, 결측치는 없는지 표시.
   - **작업 로그 (Job Runs)**: 각 로더의 마지막 실행 시간과 성공 여부 표시.

## 5. 설정 및 유량 제한 (Rate Limit)
`config/settings.yaml`에서 KIS API 호출 간격을 조절할 수 있습니다.

```yaml
kis:
  env: "prod"              # prod(20건/초), paper(2건/초)
  rate_limit_sleep_sec: 0.1 # prod 권장 0.06~0.1, paper 권장 0.5~1.0
```

## 6. 주의 사항
- `data/market_data.db-wal` 및 `-shm` 파일은 SQLite의 쓰기 모드 파일이므로 삭제하지 마십시오.
- 유니버스 250개 외의 데이터는 수집되지 않으며, 선별 엔진에서도 제외됩니다.