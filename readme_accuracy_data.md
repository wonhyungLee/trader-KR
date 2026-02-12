# 정확도모드 보강데이터 수집: 500 에러(끊김) 방어 가이드

이 문서는 **대량 수집(예: 2,773 종목 연속 호출)** 중 KIS OpenAPI에서 흔히 발생하는
`500 Internal Server Error` / `502/503/504` / 간헐적 네트워크 끊김에 대해,
**자동 재시도 + 백오프 + 세션 리셋 + 쿨다운**으로 끝까지 수집할 수 있도록
프로젝트를 운용하는 방법을 정리한다.

> ✅ 이 저장소에는 이미 방어 로직이 적용되어 있다.
> - `src/brokers/kis_broker.py` : 재시도/백오프/세션리셋/토큰재발급(401/403) 포함
> - `src/utils/http_retry.py`    : 재시도 유틸
> - `src/utils/kis_probe.py`     : “언제부터 다시 정상 호출되는지” 확인용 프로브

---

## 1) “언제부터 다시 연결 가능한가?”를 확인하는 방법

`500`은 **서버/게이트웨이 내부 오류**라 “정확히 몇 분 뒤 복구”를 문서만으로 확정할 수 없다.
가장 안전한 방법은 **가벼운 API를 일정 간격으로 호출해 정상화 여부를 감지**하는 것이다.

### 1.1 프로브 실행(정상화 감지)

아래 명령은 삼성전자(005930) 현재가를 주기적으로 호출한다.
**연속 3회 성공**하면 종료(ExitCode=0)한다.

```bash
source venv/bin/activate
python -m src.utils.kis_probe --interval 20 --successes 3
```

권장 운영:
- 수집 중 500이 연속될 때 → 위 프로브를 먼저 돌려 **정상화 확인 후 수집 재개**
- 프로브도 계속 실패하면 → KIS 장애/점검/네트워크 문제 가능성이 높으니 **대기 시간**을 늘린다.

---

## 2) 500 방어 로직(프로젝트 적용 내용)

### 2.1 자동 재시도(429/5xx/네트워크 예외)

`KISBroker.request()`는 아래 상황에서 자동으로 재시도한다.

- HTTP: `429, 500, 502, 503, 504`
- 예외: `requests.RequestException`(타임아웃, 커넥션 리셋 등)

재시도는 **지수 백오프(2^n)** + **랜덤 지터**를 사용한다.

### 2.2 keep-alive 꼬임 방지: 세션 리셋

장시간 대량 호출 시 커넥션이 “반쯤 죽는” 경우가 있어서,
일정 attempt에서 `requests.Session()`을 자동으로 갈아끼운다.

### 2.3 연속 오류 쿨다운

retryable 에러가 연속으로 누적되면(기본 10회),
한 번 **긴 대기(기본 180초)** 를 적용해 게이트웨이 과부하를 피한다.

### 2.4 401/403 방어: 토큰 강제 재발급

토큰 만료/폐기로 401/403이 뜨면, 토큰을 강제로 갱신 후 1회 추가 시도한다.

---

## 3) 설정 튜닝(권장 값)

`config/settings.yaml`의 `kis:` 아래에 아래 옵션을 추가/조정할 수 있다.

```yaml
kis:
  # 기존
  rate_limit_sleep_sec: 0.06   # (실전 REST 20건/초 기준 여유값)

  # 추가(방어용)
  timeout_connect_sec: 5
  timeout_read_sec: 20
  max_retries: 8
  backoff_base_sec: 2
  backoff_cap_sec: 60
  backoff_jitter_sec: 0.5
  consecutive_error_cooldown_after: 10
  consecutive_error_cooldown_sec: 180
```

팁:
- `rate_limit_sleep_sec`를 너무 낮추면(예: 0.05) 500이 더 자주 발생할 수 있다.
- 수집이 느리더라도 **안 끊기고 끝까지 가는 것**이 우선이면 0.35~0.5가 안전하다.

---

## 4) 수집 코드에서 해야 할 추가 방어(추천)

### 4.1 실패 종목 SKIP + 실패 목록 재수집

특정 종목에서만 계속 500이 반복될 수 있다. 이런 경우:

1) 종목별 최대 재시도 이후에도 실패하면 해당 종목은 SKIP
2) `data/csv/failed_codes_accuracy.csv` 등에 기록
3) 마지막에 실패 목록만 따로 재수집

### 4.2 Resume(재개) 체크포인트

긴 리스트(2,773개)를 도는 작업은 중간에 끊겨도 이어서 할 수 있어야 한다.

권장 방식:
- `data/accuracy_progress.json`에 마지막 성공 `index/code/date`를 저장
- 다음 실행 시 해당 지점부터 재개

---

## 5) CSV 자동 저장

수집 결과를 DB에 저장한 뒤 CSV로 자동 저장하려면:

`config/settings.yaml`

```yaml
export_csv:
  enabled: true
  out_dir: "data/csv"
  mode: "overwrite"     # 운영 추천: overwrite
  tables: []             # 비우면 전체 테이블 export
```

---

## 6) 참고: 빈 파라미터("" ) 제거

일부 TR은 빈 파라미터를 허용하지만, 대량 호출에서 간헐적으로 오류를 유발할 수 있다.

가능하면 요청 파라미터는 아래처럼 **None만 넣고, ""은 전달하지 않도록** 구성하는 것이 안전하다.

```python
params = {k: v for k, v in params.items() if v is not None and v != ""}
```

단, 주문/체결조회 등 일부 TR은 공란 필드가 필요할 수 있으니(예: PDNO="")
그 경우에는 기존대로 유지한다.
