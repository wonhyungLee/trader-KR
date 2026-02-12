#!/usr/bin/env bash
# 데이터 리필 실행 스크립트 (250 종목 유니버스 전용)
set -x

# 프로젝트 루트로 이동 (사용자 환경에 맞게 수정)
cd "$(dirname "$0")" || exit 1

# 가상환경 활성화
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "myenv" ]; then
    source myenv/bin/activate
fi

export PYTHONUNBUFFERED=1

# --source: kis (한국투자증권 REST API)
# --chunk-days: KIS 응답 건수(최대 100건)에 맞춰 안전하게 120~180일 권장.
#               어차피 응답의 min_date로 경계가 이동하므로 크게 줘도 되지만 
#               API 응답 안정성을 위해 150일 권장.
# --cooldown: kis.rate_limit_sleep_sec가 0.1이면 초당 10건. 
#             refill_loader 내부에서 KISPriceClient를 통해 broker의 설정을 따름.
# --resume: refill_progress 테이블을 참조해 중단 지점부터 재개.

python -u -m src.collectors.refill_loader \
  --source kis \
  --chunk-days 150 \
  --resume