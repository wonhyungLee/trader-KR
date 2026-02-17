# BNF-K 서버 배포 가이드 (Oracle Cloud, Ubuntu 가정)

## 1. 시스템 준비
- OS: Ubuntu 22.04+ (root 혹은 sudo 권한)
- 시간대: `sudo timedatectl set-timezone Asia/Seoul`
- 필수 패키지:
  ```bash
  sudo apt update && sudo apt install -y python3-pip python3.10-venv git sqlite3 ntp
  ```
- 방화벽: 필요 시 22/80/443 등 개방, 내부 only 운영이면 생략.

## 2. 코드 배치
- 이 폴더 전체를 서버로 복사: `/opt/bnf-k` 가정
  ```bash
  sudo mkdir -p /opt/bnf-k && sudo chown -R $USER:$USER /opt/bnf-k
  # 로컬에서 scp 등으로 전송
  ```
- 작업 디렉토리: `/opt/bnf-k`

## 3. 가상환경 & 의존성
```bash
cd /opt/bnf-k
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 4. 환경변수 설정 (중요: **개인정보 노출 금지**)
- 이 코드베이스는 실행 시 `.env`와 `개인정보` 파일을 자동 로딩합니다. `개인정보`에 `KIS1_*`가 있으면 기본으로 사용됩니다.
- 다른 계정을 쓰려면 `.env`에 명시적으로 덮어쓰세요.
- 최소 필요 키:
  ```bash
  export KIS_APP_KEY="..."
  export KIS_APP_SECRET="..."
  export KIS_ACCOUNT_NO="00000000-01"
  export TG_BOT_TOKEN="..."      # 텔레그램 알림 사용 시
  export TG_CHAT_ID="..."
  ```
- 템플릿: `config/settings.yaml` 에 `${VAR}` 로 치환 지원.

## 5. 디렉토리/DB 준비
```bash
mkdir -p data logs .cache
sqlite3 data/market_data.db "VACUUM;"  # 최초 생성 겸 확인
```

## 6. 초기 데이터 적재
```bash
source venv/bin/activate
python -m src.collectors.universe_loader           # 유니버스 350 고정 (KOSPI200+KOSDAQ150)
python -m src.collectors.refill_loader --resume    # KIS 일봉 백필
```
- 이후 매일: `python -m src.collectors.daily_loader`

## 7. 운영 루프 수동 실행 예시 (조회/수집/선별 전용)
```bash
python -m src.collectors.daily_loader
python -m src.collectors.accuracy_data_loader --resume
python -m src.monitor.monitor_main
python server.py
```

## 8. 크론 등록 예시 (KST 기준)
```bash
crontab -e
# 내용:
35 15 * * 1-5 cd /opt/bnf-k && /opt/bnf-k/venv/bin/python -m src.collectors.daily_loader >> logs/cron.log 2>&1
10 02 * * 1-5 cd /opt/bnf-k && /opt/bnf-k/venv/bin/python -m src.collectors.accuracy_data_loader --resume >> logs/cron.log 2>&1
30 02 * * 6 cd /opt/bnf-k && /opt/bnf-k/venv/bin/python -m src.collectors.refill_loader --resume >> logs/cron.log 2>&1
```


## 8.5 실시간 감시(Monitor) 데몬 (선택)
- `config/settings.yaml`에서 `monitor.enabled: true`로 켜고, `scan_interval_sec`/`max_ws_subs`/`ws_subscribe_interval_sec(기본 0.2초)`를 확인하세요.
- WS는 **1세션/등록 합산 41건** 제한이 있으니 `max_ws_subs`를 41 이하로 두세요.

systemd 서비스 예시(데몬): `/etc/systemd/system/bnf-k-monitor.service`
```ini
[Unit]
Description=BNF-K Monitor (REST scan + WS dynamic subscription)
After=network.target

[Service]
Type=simple
User=bnf
WorkingDirectory=/opt/bnf-k
EnvironmentFile=/opt/bnf-k/.env
ExecStart=/opt/bnf-k/venv/bin/python -m src.monitor.monitor_main
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

활성화:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now bnf-k-monitor
sudo systemctl status bnf-k-monitor
```

## 9. 서비스(선택) — systemd 단일 서비스 예시
`/etc/systemd/system/bnf-k.service`
```ini
[Unit]
Description=BNF-K daily loop (runs via cron); placeholder for future daemon
After=network.target
[Service]
Type=oneshot
User=bnf
WorkingDirectory=/opt/bnf-k
ExecStart=/bin/true
[Install]
WantedBy=multi-user.target
```
- 현재는 크론 중심이라 ExecStart는 비워둔 상태. 필요 시 scheduler/daemon으로 교체.

## 10. 로그 관리
- 기본 로그: `logs/bnf_trader.log`, `logs/cron.log`
- logrotate 권장: `/etc/logrotate.d/bnf_k`
```text
/opt/bnf-k/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
```

## 11. 동작 확인 체크리스트
- `python -m src.collectors.daily_loader --limit 3` 로 소량 증분 테스트
- `python -m src.analyzer.backtest_runner_active_universe` 실행 후
  `data/backtest_active_universe_daytrade/equity_curve.csv` 생성 확인
- 디스코드/텔레그램 알림 설정 시 메시지 수신 여부 확인

## 12. 보안 메모
- `개인정보` 파일 내용은 절대 README나 코드에 포함하지 말고, 환경변수/비밀 관리 도구로 주입
- `.cache/kis_token.json`, `data/`, `logs/`는 권한 700 권장
- 실서버 전환 전에는 수집/선별/모니터 기능만 최소 2~3일 안정화 검증

---

# 백테스트용 CSV 요청 목록
로컬에서 API 대신 CSV를 바로 적재하려면 아래 파일들을 `data/`에 제공해 주세요 (헤더 포함, UTF-8):
1) `data/stock_info.csv`
```
code,name,market,marcap
005930,삼성전자,KOSPI,350000000000000
...
```
2) `data/daily_price.csv` — 모든 종목의 일봉 합본
```
date,code,open,high,low,close,volume,amount
2023-01-02,005930,60000,61000,59500,60500,12345678,740000000000
...
```
- `amount` 단위는 원. 없으면 `close*volume`로 계산 가능.
- `ma25`, `disparity` 컬럼이 없으면 스크립트가 다시 계산합니다.

위 두 CSV를 제공해 주시면
`python -m src.analyzer.backtest_runner_active_universe`로 즉시 daytrade 백테스트를 돌릴 수 있습니다.
레거시 호환용으로 `python -m src.analyzer.backtest_runner`도 동일한 daytrade 러너를 호출합니다.
