# Refill / Backfill 가이드 (KOSPI 100 + KOSDAQ 150) — KIS 기간별시세로 “상장일 ~ 현재” 적재

이 문서는 **KOSPI 100 + KOSDAQ 150 유니버스**에 대해, 한국투자증권(KIS) Open API의 **기간별시세(일봉)**를 이용해
**각 종목의 상장일(가능한 한 가장 이른 시점)부터 오늘까지** `daily_price`를 “빈틈 없이(refill)” 적재하는 절차를 정리한 운영 문서입니다.

> 핵심: KIS 일봉은 **호출당 데이터 건수 제한**이 있으므로, **기간을 쪼개서 반복 호출**(chunking)로 누적 적재합니다.  
> 데이터가 “상장일”까지 항상 100% 내려오는지는 종목/시장/역사 데이터 제공 범위에 따라 달라질 수 있으니, 아래의 *검증/예외 처리*를 반드시 포함하세요.

---

## 0) 목표 / 범위

- 대상 유니버스: **KOSPI 100 + KOSDAQ 150**
- 데이터: **일봉 OHLCV + 거래대금(amount)** 중심
- 기간: **(가능한 한) 상장일 ~ 현재**
- 적재: SQLite `daily_price(code, date)` 기준 **UPSERT(중복 제거)**

> 전략/트레이딩 로직과 분리: refill은 “데이터 인프라 작업”이며, 운영 루프(close/open/sync/cancel)와 분리해 **1회 또는 주기적(예: 월 1회)으로 실행**합니다.

---

## 1) 사전 준비 체크리스트

### 1.1 KIS 자격증명(필수)
- `APP_KEY`, `APP_SECRET`
- (필요 시) 계좌 식별자: `CANO`, `ACNT_PRDT_CD`  
  - **시세 조회만**이라면 계좌 정보가 필수는 아닌 구성도 가능하지만, 프로젝트 구현체에 따라 공통 헤더/인증 로직 때문에 함께 쓰는 경우가 많습니다.

### 1.2 레이트리밋/백오프 기본값(권장)
- `rate_limit_sleep_sec`: **0.5s** (보수적 권장)
- 429/일시 오류 시: `0.5 → 1 → 2 → 4` 백오프 + 재시도 횟수 제한

### 1.3 DB 준비
- `daily_price` 테이블에 `(code, date)`가 사실상 PK 역할을 해야 합니다.
- 인덱스 권장:
  ```bash
  sqlite3 data/market_data.db "CREATE INDEX IF NOT EXISTS idx_daily_price_code_date ON daily_price(code, date);"
  ```

---

## 2) 유니버스 준비: “KOSPI100 + KOSDAQ150” 코드 목록

### 2.1 가장 안전한 방식(권장): “유니버스 스냅샷” 파일로 고정
refill은 시간이 오래 걸릴 수 있어, 도중에 구성 종목이 바뀌면 혼선이 생깁니다.

- `data/universe_kospi100.csv`
- `data/universe_kosdaq150.csv`

컬럼 예시:
```csv
code,name,market
005930,삼성전자,KOSPI
000660,SK하이닉스,KOSPI
...
```

> ⚠️ “KOSPI100/코스닥150 구성종목”은 특정 기준일의 지수 구성에 따라 바뀝니다.  
> **한 번 뽑아 ‘스냅샷’으로 고정한 뒤** refill을 수행하세요.

### 2.2 구현 편의 옵션(대안)
- (대안 A) “지수 구성종목”을 외부 데이터(거래소/벤더/스크래핑)로 받아 CSV 생성
- (대안 B) “시가총액 상위 N개”로 근사해 유니버스 생성  
  - 지수 구성과 동일하진 않지만 “유동성 상위 바스켓”으로는 충분한 경우가 많습니다.

---

## 3) “상장일 ~ 현재”를 얻는 방법(2가지 전략)

### 전략 1) 상장일 메타데이터를 먼저 확보(권장)
- `stock_info`(또는 별도 테이블)에 `listed_date`를 채운 뒤,
- 그 날짜부터 오늘까지 chunk로 수집합니다.

장점: 호출량이 줄고 “빈 구간 탐색”이 줄어듭니다.  
단점: 상장일 메타데이터 소스가 필요합니다(예: KRX/FDR/벤더 등).

### 전략 2) “역방향 스캔(Backward Scan)”으로 최조 데이터 시점을 찾기(외부 메타 없이 가능)
- 오늘에서 시작해 과거로 chunk를 계속 내려가며 호출합니다.
- 더 이상 데이터가 안 내려오기 시작하는 지점(연속 N번 empty 등)을 만나면 “상장 전”으로 판단하고 종료합니다.

장점: 외부 메타 없이 실행 가능  
단점: 호출량이 증가할 수 있음(특히 오래된 종목)

> 운영 추천: **전략2로 1회 실행 → 최초 데이터 날짜를 DB에 저장 → 이후부터는 전략1처럼 최적화**.

---

## 4) refill 로더 동작 명세(운영 문서용)

### 4.1 입력
- Universe: `data/universe_kospi100.csv`, `data/universe_kosdaq150.csv`
- 옵션:
  - `--chunk-days` (권장 60~120 거래일 수준으로 시작)
  - `--sleep` (기본 0.5)
  - `--start-mode listing|backward`
    - `listing`: listed_date 기반(있을 때)
    - `backward`: 오늘→과거 역방향 스캔
  - `--resume`: 중단 지점부터 재개

### 4.2 출력/저장
- `daily_price` UPSERT
- `refill_progress`(권장) 테이블로 진행상황 체크포인트 저장:
  - `code`
  - `last_fetched_end_date` (가장 최근에 “끝점”으로 시도한 날짜)
  - `min_date_in_db` (현재까지 확보된 최소 날짜)
  - `updated_at`
  - `status` (RUNNING/DONE/ERROR)

### 4.3 알고리즘(권장: “끝점 기준 역방향 chunk”)
1. `end = today`
2. `start = end - chunk_days`
3. KIS 기간별시세 API 호출(일봉)
4. 응답 레코드를 `(code, date)`로 UPSERT
5. 응답 중 최소 날짜가 `start`보다 최근이면:
   - “더 과거 데이터가 없을 가능성” → empty가 연속되면 종료 판단
6. `end = start - 1day`로 이동해 반복

> 팁: 휴장일/주말 때문에 날짜 계산은 “달력일”로 해도 되지만, 데이터가 비는 것은 자연스러우니  
> **empty 1번으로 종료하지 말고, 연속 N번 empty(예: 3~5회)**를 종료 조건으로 두면 안정적입니다.

---

## 5) 실행 예시(README 스타일)

> 아래 커맨드는 예시입니다. 실제 프로젝트에 맞게 파일명/경로를 조정하세요.

### 5.1 유니버스 스냅샷 준비
```bash
# (예시) 외부에서 받은 목록을 data/ 폴더에 저장
ls data/universe_kospi100.csv
ls data/universe_kosdaq150.csv
```

### 5.2 1종목 드라이런(권장)
```bash
python src/collectors/refill_loader.py \
  --universe data/universe_kospi100.csv \
  --code 005930 \
  --start-mode backward \
  --chunk-days 90 \
  --sleep 0.5
```

### 5.3 전체 refill (KOSPI100 + KOSDAQ150)
```bash
python -m src.collectors.refill_loader \
  --universe data/universe_kospi100.csv \
  --universe data/universe_kosdaq150.csv \
  --chunk-days 90 \
  --sleep 0.5 \
  --resume
```

> 실행 시간이 길어질 수 있으므로, 가능하면 **장 종료 후(16:30 이후) 또는 주말**에 돌리는 것을 권장합니다.

---

## 6) 검증(반드시) — “상장일 ~ 현재”가 정말 채워졌나?

### 6.1 빠른 정합성 체크
- 코드별 최소/최대 날짜
- 최근 30거래일 결측 유무
- amount 단위 sanity:
  - `amount ≈ close × volume` 스케일(천원/원 단위 차이 확인)

SQL 예시:
```sql
SELECT code, MIN(date) AS min_date, MAX(date) AS max_date, COUNT(*) AS n
FROM daily_price
GROUP BY code
ORDER BY min_date ASC;
```

### 6.2 “상장일 미만 데이터”가 안 내려오는 케이스
- API가 제공하는 히스토리 범위 제한
- 종목코드 변경/합병/분할 등 이벤트로 과거 구간이 단절
- 거래소/시장 이관(코스닥→코스피 등)으로 코드/시장 구분 혼선

> 이런 경우엔 “상장일부터 현재까지”를 100% 보장하기 어렵습니다.  
> 필요한 경우 **외부 데이터(KRX 원시데이터/상용 벤더)**로 부족 구간을 보충하는 설계를 고려하세요.

---

## 7) 운영 팁(속도/안정성)

- **병렬 처리 금지(초기)**: 먼저 단일 프로세스로 안정화 후, 필요 시 제한적으로 확장
- **429 대응**: 즉시 sleep 증가 + 백오프, 실패 지속 시 중단하고 `--resume`
- **중복 제거**: DB에 `(code, date)` UNIQUE/PK를 두고 UPSERT로 “재실행 안전”
- **로그**:
  - 종목별 진행률(최소 날짜 업데이트)
  - error code/응답 원문(민감정보 제외)

---

## 8) FAQ

### Q1. “진짜 상장일부터” 가능한가?
- 기술적으로는 “가능한 한 과거까지” 끌어오는 건 가능하지만, **데이터 제공 범위/이벤트(코드변경 등)** 때문에
  일부 종목은 상장일까지 100% 채우기 어렵습니다. 이 경우 “최초 가용 일자(min_date)”를 확보하는 것이 현실적입니다.

### Q2. chunk-days는 얼마가 좋은가?
- 처음엔 **90일** 정도로 시작 추천(실패 시 원인 파악이 쉬움).
- 안정화되면 120~180일로 늘려 호출 횟수를 줄일 수 있습니다(단, 응답 건수 제한/타임아웃 주의).

### Q3. TradingView가 필요할까?
- refill 작업은 “데이터 적재”라서 **TradingView가 필요하지 않습니다.**  
  TradingView는 모니터링/알림(관제) 용도에 가깝고, 히스토리 대량 적재는 KIS+DB로 통일하는 편이 운영이 단순합니다.

---

## 9) 다음 단계(추천)
1) refill 완료 후: `daily_loader`로 “증분”만 유지
2) 데이터 품질 검증 후: Next-Open 백테스트 → 실전(close/open/sync/cancel) 투입
3) 장기 운영 시: 유니버스 정기 갱신(분기/월) + 결측/이벤트 대응 로직 강화
