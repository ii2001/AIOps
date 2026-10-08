# Renewable Energy AIOps

## MVP
최근 168시간의 태양광 전력거래량(MWh)을 입력받아 다음 24시간의 시간대별 전력거래량을 동시에 예측한다. 입력 종료시각이 하루의 끝이면 다음날 24시간에 해당한다. 시간 정보 없이 순서대로 168개 값을 전달하며 응답 hour 1~24는 예측 lead hour이다.

## Dataset
제공된 실제 KPX 지역별 시간별 태양광/풍력 데이터:
`data/raw/한국전력거래소_지역별 시간별 태양광 및 풍력 발전량_20251231.csv` (CP949).

원본 CSV는 수정하지 않는다. [데이터 프로파일](docs/data_profile.md)에 전체 통계·결측·중복·시간 누락과 선택 근거를 기록한다. 거래일 D의 거래시간 1~24를 내부적으로 D 00~23시 구간 시작(Asia/Seoul)으로 변환한다. 공식 시간구간 종료시각이라고 주장하지 않는 내부 표기 규약이다.

## Target
Region: 전라남도
Fuel: 태양광

전체 8760시간이 있고 태양광 지역 중 연간 거래량이 가장 크다. 지역/연료/기간 및 모델 계약은 `src/config.py` 한 곳에서 관리한다.

## Model
TensorFlow/Keras: Input(168,1) → LSTM(32) → Dense(32,ReLU) → Dense(24).
전력거래량 1개 feature로 시작한다. 공용 sequence builder는 (hours,features)를 받으며 feature 0이 목표값이다. 미래 확장을 위한 별도 미사용 layer/interface나 외부 날씨 수집은 추가하지 않았다.

Input: 168 hours / Output: 24 hourly forecasts. Seed 42, 최대30 epochs, Adam, batch64, shuffle=False, Validation loss EarlyStopping(patience4, best weights restore).

## Split / Scaling
원시 시간을 먼저 Train70% / Validation15% / Test15%로 분리한다. 각 파티션 안에서만 input168 + target24 시퀀스를 만들기 때문에 원시 입력과 정답 모두 경계를 넘지 않는다. Validation/Test 파티션 처음168시간은 context용이며 별도 target 평가에 포함되지 않는다.
Scaler는 Train 원시값의 min/max에만 fit하고 Validation/Test/서빙은 transform만 한다. 같은 `src/features.py` 변환과 복원을 재사용한다. 정확한 split 시각과 shape는 실제 학습 후 `artifacts/training_result.json`에 기록된다.

## Metrics
MAE(MWh), RMSE(MWh), WAPE(%) = sum(abs(actual-predicted))/sum(abs(actual))*100.
태양광의 0값 때문에 MAPE는 사용하지 않는다. WAPE를 핵심 운영 지표로 기록한다. 실제값 합이0인 평가셋의 WAPE는 정의되지 않으므로 null로 표시한다.
Metrics는 매시간 rolling origin의 모든24 horizons를 동일 가중치로 평가한다(겹치는 forecast target의 중복 포함). 실제 MWh로 복원 후 음수 예측을0으로 floor한 값으로 평가하며 서빙에도 같은 정책을 적용한다. 상한 clipping은 없다.

## Input / Output Policy
입력 음수, NaN, inf는 HTTP422로 거절한다. 관측0은 유효하며 임의 보간·야간0 강제 변환을 하지 않는다. 모델/scaler 파일이 없거나 모델 계약이 맞지 않으면 /predict는503을 반환한다. 모델/scaler는 첫 예측에서 로드하고 캐시해 요청마다 읽지 않는다.

## Run Training
프로젝트 루트에서:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/profile_data.py
python scripts/train_baseline.py
```

실제 모델: `serving_app/models/baseline.keras`
Scaler: `serving_app/models/scaler.pkl`
실제 평가: `artifacts/training_result.json`
실제 Test 하루24시간 예시: `artifacts/baseline_prediction_sample.csv`, `artifacts/baseline_prediction_chart.png`

## Run API
```bash
.venv/bin/uvicorn serving_app.main:app --host 127.0.0.1 --port 8080
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/model/info
curl -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  --data-binary @artifacts/predict_request.json
```
`predict_request.json`은 아래 실제 API 검사 실행 시 원본 Test 구간에서 자동 생성한다.

## Swagger
http://127.0.0.1:8080/docs

## Checks
```bash
.venv/bin/python tests/check_baseline.py
```
결정적 단위 검증과 실제 subprocess API 검증을 수행하고 `artifacts/api_verification.json`에 결과를 저장한다. 정상168개/24출력, 167·169개422, 음수/NaN422, 모델 로딩, finite 출력, 역변환 MWh와 offline 평가 예측의 일치, 시간경계/Train-only scaler fit을 검사한다. 자체 실행한 테스트 서버는 정상 종료한다.

## Scope
현재 단계에는 MLflow, Registry, Deployment Gate, Drift Detection, fine-tuning, 자동 재학습, Docker, LLM Sidecar가 없다. 데이터/features/model/metrics와 model_loader/route만 분리했다. 기존 개인 HAIC 프로젝트는 변경하지 않는다.

## Actual Baseline Run

| Partition | Raw time range (KST) | X shape | y shape |
|---|---|---|---|
| train | 2025-01-01T00:00:00+09:00 → 2025-09-13T11:00:00+09:00 | [5941, 168, 1] | [5941, 24] |
| validation | 2025-09-13T12:00:00+09:00 → 2025-11-07T05:00:00+09:00 | [1123, 168, 1] | [1123, 24] |
| test | 2025-11-07T06:00:00+09:00 → 2025-12-31T23:00:00+09:00 | [1123, 168, 1] | [1123, 24] |

실제 Test: MAE **191.985006 MWh**, RMSE **299.919240 MWh**, WAPE **52.288744%**. Seed42, 30 epochs 완료. WAPE가 높아 초기 기능 검증용 baseline이며 예측 품질이 우수하다고 주장하지 않는다.

[실제 Test 예측 차트](artifacts/baseline_prediction_chart.png) · [API 검사](artifacts/api_verification.json) · [전체 검증](artifacts/verification.json).

## Docker Dashboard (추가 구현)

팀 대시보드를 Docker에서도 실행할 수 있다. 초기 seed 준비 후 빌드/실행:

```bash
.venv/bin/python scripts/prepare_docker.py
docker compose up -d --build
```

접속: http://localhost:8099 · Swagger: http://localhost:8099/docs

이미 빌드했다면 `docker compose up -d`로 다시 시작한다. [Docker 실행/보존/검증 설명](docs/docker.md).

## GitHub 저장소 안내

실제 결과 요약과 차트: [reports/](reports/README.md).

가상환경·원본 CSV·모델/scaler·MLflow DB와 운영 상태는 Git에서 제외한다. Clone 후 데이터 준비와 실행 요건은 [저장소 환경 안내](docs/repository_setup.md)를 참고한다. 기존 로컬 실험 파일은 유지된다.
