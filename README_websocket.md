# WebSocket 감시/신호 모듈 가이드 (README_websocket) — KIS 동적 구독 + 배치 스캔 하이브리드

이 문서는 **Codex에게 그대로 지시**하여,
코스피100 + 코스닥150(≈250종목)을 **실시간(또는 준실시간) 감시**하면서
**API 호출 제약을 피하고**, 조건 충족 시 **텔레그램/디스코드로 신호를 보내는 기능**을 구현하기 위한 “운영 설계 + 구현 스펙”입니다.

> 핵심 아이디어(하이브리드)
>
> 1) **REST 배치 스캔**(멀티종목 시세조회)으로 250개 전체를 **30~60초 간격**으로 훑고  
> 2) “지금 중요해진 소수 종목(Top-N)”만 **WebSocket으로 동적 구독**하여 초단위로 정밀 감시  
> 3) 신호 발생 시 알림 전송 + 과도한 호출/구독 변경은 제한(쿨다운/레이트리밋)

---

## 0) 목표 / 비목표

### 목표
- 250종목을 상시 감시하면서도 **REST 호출량 폭발을 방지**
- “조건 근접 종목”만 실시간 정밀 감시(동적 구독)
- 신호 발생 시 알림(텔레그램/디스코드/로그)
- 재연결/재구독/중복 알림 방지 등 **운영 안정성**

### 비목표
- “모든 250종목을 tick 단위로 상시 실시간 스트리밍” (트래픽/제한/운영 난이도 과다)

---

## 1) 사용 API (KIS Open API, 국내주식)

### 인증(필수)
- REST 접근토큰: `auth_token`
- WebSocket 접속키(approval key): `auth_ws_token`

### 전체 스캔(REST, 호출량 절감용)
- 관심종목(멀티종목) 시세조회: `intstock_multprice`  
  → 한 번 호출로 여러 티커를 묶어서 받는 용도(배치 스캔)

### 실시간 감시(WebSocket, 동적 구독용)
- 실시간 체결(예: KRX/통합): `ccnl_*` 계열 (프로젝트에서 “가격 스트림”이 필요)
- 실시간 호가(선택): `asking_price_total` 등
- 장운영정보(선택): `market_status_total` 등

> 주의: 실제 WS “TR_ID/구독 메시지 포맷/최대 구독 수”는 문서/계정에 따라 달라질 수 있으니,
> KIS 공식 샘플/가이드에 맞춰 구현하고, **구독 상한/트래픽 상한**은 운영 테스트로 확정한다.

---

## 2) 아키텍처 개요

### 2.1 구성요소
- `src/monitor/`
  - `scanner.py` : REST 배치 스캔(250종목)
  - `ws_client.py` : WebSocket 연결/재연결/구독/해제
  - `signal_engine.py` : 신호 조건 평가 + 쿨다운(중복 알림 방지)
  - `subscription_manager.py` : 동적 구독 Top-N 선정 + 리밸런싱
  - `notifier.py` : 텔레그램/디스코드 전송(기존 notifier 재사용 가능)
  - `state_store.py` : 최소 상태 저장(구독 목록, 쿨다운, 마지막 가격 등)

### 2.2 데이터 흐름(요약)
1) `scanner`가 30~60초마다 `intstock_multprice`로 250종목 스냅샷 수집
2) `subscription_manager`가 스냅샷에서 “조건 근접 Top-N”을 계산
3) `ws_client`가 현재 구독 목록과 비교하여 `subscribe/unsubscribe` 수행(동적 구독)
4) `ws_client`가 실시간 가격 스트림을 받으면 `signal_engine`이 조건 충족 여부 판단
5) 조건 충족 시 `notifier`로 알림 전송(텔레그램/디스코드)
6) 모든 동작은 rate limit/쿨다운/백오프 규칙을 준수

---

## 3) 동적 구독(Dynamic Subscription) 규칙 (중요)

### 3.1 용어
- **스캔(Scan)**: REST 배치로 전체 250종목을 저빈도로 훑는 것(예: 60초)
- **구독(Subscribe)**: WS로 특정 종목의 실시간 데이터를 받도록 등록
- **동적 구독**: 스캔 결과에 따라 “구독 목록”을 주기적으로 바꾸는 것

### 3.2 리밸런싱 주기
- 기본: **60초마다** 구독 목록 리밸런싱
- 너무 자주 바꾸면(예: 1초마다) 오히려 불안정/차단 리스크 증가

### 3.3 구독 상한(Top-N)
- `MAX_WS_SUBS = 20` (초기 권장)
- 안정화 후 30~50까지 확장 가능(단, 트래픽/제한 테스트 필요)

### 3.4 핑퐁 방지(구독 쿨다운)
- `SUBSCRIBE_COOLDOWN_SEC = 180` (예: 한번 해제하면 3분 내 재구독 금지)
- “임계치 근처”에서 왔다갔다 하면서 구독/해제가 반복되는 것을 방지

### 3.5 우선순위 선정(Top-N 점수)
스캔 결과로 “실시간으로 더 자세히 봐야 할 종목”을 점수화:

예시 점수(간단):
- `distance_to_threshold`가 가까울수록 점수↑
- 당일 변동성/거래대금 급증 등 가중치↑(선택)

예:  
`score = w1 * (1 / (abs(disparity - thr) + eps)) + w2 * intraday_volatility + w3 * log(amount)`

> BNF-K가 일봉 기반이면, 장중 감시는 “보조 알림” 성격으로 두는 것이 운영이 단순하다.

---

## 4) 신호 조건 설계(예시) + 스팸 방지

### 4.1 신호 조건 예시(프로젝트에 맞게 교체 가능)
- “괴리율(disparity)이 임계치 이하로 하락” (Mean Reversion 진입 후보)
- “n분 수익률이 -x% 이하 급락”
- “거래대금 급증 + 하락” 등

### 4.2 알림 쿨다운(중복 알림 방지)
- 종목별/신호유형별로:
  - `ALERT_COOLDOWN_SEC = 600` (10분)
- 동일 조건이 계속 유지되어도 10분 내 중복 알림 금지

### 4.3 알림 내용(필수 포함)
- 종목코드/종목명
- 트리거 조건(임계치, 현재값)
- 관측 시각(KST)
- (선택) 링크/메모(HTS/TradingView 링크 등)

---

## 5) 레이트리밋 / 백오프 / 장애 대응(필수)

### 5.1 REST 스캔 호출 예산
- 스캔 주기: 30~60초 권장
- 멀티종목 호출은 “배치 크기(batch_size)”에 따라 호출 횟수 결정:
  - `calls_per_scan = ceil(250 / batch_size)`
- 초기 권장:
  - `batch_size = 50`
  - `scan_interval = 60s`
  - `rate_limit_sleep_sec = 0.5`

### 5.2 REST 실패 대응
- HTTP 429/일시 오류:
  - exponential backoff: 0.5 → 1 → 2 → 4
  - 최대 재시도 횟수: 5
  - 연속 실패 시:
    - 스캔 주기 자동 증가(예: 60s → 120s)
    - 알림 전송 + 안전 모드

### 5.3 WebSocket 장애 대응
- 끊김 감지 시:
  - 재연결(backoff) 1s → 2s → 4s → 8s
- 재연결 후:
  - “현재 목표 구독 목록(target_subs)”을 기준으로 **전량 재구독**
- 메시지 처리 지연(backpressure) 방지:
  - WS 수신 큐 제한(최대 큐 길이)
  - 처리 지연 시 “구독 수를 자동 축소” 옵션(advanced)

---

## 6) 구현 요구사항(“Codex 구현 체크리스트”)

### 6.1 필수 파일/모듈
- `src/monitor/ws_client.py`
  - `connect()`, `disconnect()`, `subscribe(code)`, `unsubscribe(code)`
  - `run_forever()` : 수신 루프(파서 포함)
- `src/monitor/scanner.py`
  - `scan_once(universe_codes) -> dict[code] = snapshot`
  - `scan_loop()` : 주기 실행
- `src/monitor/subscription_manager.py`
  - `compute_targets(snapshot) -> set[codes]`  # Top-N 산출
  - `rebalance(current_subs, target_subs)`      # subscribe/unsubscribe diff
- `src/monitor/signal_engine.py`
  - `on_tick(code, price, ts)` -> list[alerts]
  - 쿨다운/중복 방지 포함
- `src/monitor/monitor_main.py`
  - 실행 엔트리포인트: `python -m src.monitor.monitor_main`
- `src/utils/notifier.py` (기존 재사용 가능)
  - 텔레그램/디스코드 전송 지원(둘 중 하나만 있어도 OK)

### 6.2 설정(config/settings.yaml) 추가 키
```yaml
monitor:
  enabled: true
  scan_interval_sec: 60
  rest_batch_size: 50
  max_ws_subs: 20
  subscribe_cooldown_sec: 180
  alert_cooldown_sec: 600

  # 신호 조건(예시)
  signal:
    type: "disparity_threshold"
    disparity_threshold: -0.08
    use_intraday: true   # 장중감시 사용 여부

kis:
  rate_limit_sleep_sec: 0.5
```

### 6.3 실행 커맨드(예시)
```bash
python -m src.monitor.monitor_main
```

---

## 7) 운영 모드 추천(가장 안전한 시작)

### Mode A (가장 안전)
- REST 배치 스캔: 60초
- WS 구독: Top 10~20
- 신호: “근접 경보”만(진입 후보 알림)

### Mode B (좀 더 공격적)
- REST 배치 스캔: 30초
- WS 구독: Top 30~40
- 신호: “돌파” + “급락” 복합

> 250종목 “상시 tick 스트리밍”은 비추천. 먼저 A로 안정화 후 B로 확장.

---

## 8) 수용 기준(Acceptance Criteria)

Codex 구현 완료 후 아래가 만족되어야 함:

1) **REST 호출량이 폭발하지 않는다**
   - 60초 스캔 + 배치 호출로 안정적으로 유지
2) **WS 연결이 끊겨도 자동 복구한다**
   - 재연결 후 목표 구독 목록이 자동 복원
3) **구독 목록이 동적으로 바뀐다**
   - 스캔 결과에 따라 subscribe/unsubscribe가 수행
4) **알림이 스팸으로 폭발하지 않는다**
   - 종목/신호별 쿨다운이 동작
5) **로그/상태 저장이 된다**
   - 현재 구독 목록, 마지막 알림 시각, 최근 가격 등을 최소 저장

---

## 9) 자주 하는 실수(피해야 함)

- (금지) 250종목을 단일종목 REST API로 초단위 폴링
- (금지) WS 구독/해제를 초당 수십 번 바꾸는 설계(핑퐁)
- (금지) 429가 나와도 계속 같은 속도로 때리기
- (필수) 백오프, 재시도 제한, 안전 모드
- (필수) 알림 쿨다운

---

## 10) 다음 단계(선택)

- 신호 발생 시 “주문(order_queue PENDING 생성)”까지 자동화하고 싶다면:
  - monitor는 “알림+후보 선정”까지만
  - 실제 주문은 기존 Next-Open 루프(close/open)와 분리 유지(안전)
- TradingView는 “관제/검증” 레이어로만 추가 권장(필수 아님)

---
