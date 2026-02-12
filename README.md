# BNF Viewer (KR)

KOSPI200 + KOSDAQ150 종목을 대상으로 하는 **조회 전용 대시보드**입니다.

- ✅ 종목/섹터 목록 조회
- ✅ 일봉 차트(종가, MA25, 거래량) 조회
- ✅ 전략 조건 기반 '매수 후보(Selection)' 조회
- ❌ 자동매매/주문 기능 없음
- ❌ 잔고/포트폴리오 기능 없음

---

## 1) 설치

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cd frontend
npm install
npm run build
cd ..
```

## 2) 유니버스 CSV 준비 (필수)

`data/universe_kospi200.csv` + `data/universe_kosdaq150.csv`가 필요합니다.

- KOSDAQ150은 기본 파일이 포함되어 있습니다.
- KOSPI200은 환경에 따라 자동 생성 스크립트를 제공합니다:

```bash
python scripts/generate_universe_kr.py
```

## 3) DB에 유니버스 적재

```bash
python -m src.collectors.universe_loader
```

## 4) 가격 데이터 적재

초기 적재(최근 500일) 예시:

```bash
python -m src.collectors.bulk_loader --days 500
```

> `bulk_loader`는 finance-datareader를 사용합니다. 실행 환경에 인터넷 연결이 필요합니다.

## 5) 서버 실행

```bash
python server.py
# 기본 포트: http://localhost:5001
```

---

## 폴더 구조

- `server.py` : Flask API + 정적 프론트(dist) 서빙
- `frontend/` : React(Vite) 프론트엔드
- `src/collectors/` : universe_loader, bulk_loader 등 데이터 적재 스크립트
- `data/market_data.db` : SQLite DB

