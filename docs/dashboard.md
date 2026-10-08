# Renewable Energy Ops 대시보드

교수님 내부 데모 8088/8099를 실제 브라우저로 확인하고 밝은 카드, 탭, KPI, 파이프라인, 모델 요약과 이벤트 로그 정보 구조를 참고했다. HTML/JS 소스 복사가 아니라 우리 프로젝트에 맞춘 구현이다. 참고 화면과 결과 캡처는 artifacts/ui/에 저장했다.

## 접속 및 재시작

```bash
.venv/bin/python scripts/start_dashboard.py
```

- 이 PC: http://localhost:8088
- 같은 네트워크: 이 PC의 IP주소:8088 (IP는 네트워크 변경 시 바뀔 수 있음)
- Swagger: http://localhost:8088/docs

start_dashboard는 기존 8088 서비스가 우리 대시보드이면 재사용하고 다른 서비스이면 종료하지 않는다. 서버 PID 및 로그는 artifacts/ui/server_process.json, server.log에서 확인할 수 있다. 사용자가 직접 확인하도록 서버는 실행 상태로 남긴다.

## 화면 구성

- Overview: 실제 Service/Registry/rolling WAPE/threshold/3-day streak/Gate, 파이프라인, 고정 실험 시계열, benchmark, 후보 비교와 이벤트.
- Prediction: timestamp와 실제168개 MWh 샘플, 반복 가능한 실제 모델 preview, 별도의 운영 예측 저장,24시간 actual 등록.
- Drift Lab: 첫 +15% 실패, 기본-15% 미탐지, 모든 감소 강도와 최소 시험 검출-20%, 자동학습 및 독립 Gate FAIL/기존모델 유지.
- Datasets: 실제 전라남도 태양광 데이터 통계, 정확한 split 시각,7features/168→24 계약.
- Logs: 실제 INFO/WARN/OK/FAIL 필터와 로그 다운로드.

실시간 전력시장 연동 UI라고 주장하지 않는다. 2025년 실제 데이터와 명시적 구조적 simulation 결과를 표시한다. Service Healthy는 API 연결 상태이며 모델 정확도가 우수하거나 Drift가 해소됐다는 의미가 아니다.

## 보존 및 API

대시보드의 Registry/state는 artifacts/ui/operations로 SQLite backup 및 상태 복사를 수행했다. 기존 모델 artifact는 읽기만 하며, 새 학습 artifact 경로는 운영 복사본으로 설정한다. 원본 CSV·모델·이전 Registry·실험 결과는 수정하지 않는다.

기존 /health, /model/info, /predict, /actuals, /aiops/status, /batch-test, /logs를 재사용한다. 추가 /dashboard/summary와 /dashboard/example은 읽기 전용 화면 데이터다. /dashboard/predict-preview는 기존 scaler/시간features/MLflow loader를 재사용해 실제 inference를 수행하되 forecast/actual 상태를 저장하지 않는다. 따라서 같은 입력의 테스트를 반복할 수 있고, 운영 예측의 immutable 정책은 그대로 유지된다.

운영 예측 저장은 /predict를 사용한다. actual은 해당 forecast가 저장된 날짜의 실제24개 MWh가 확보된 후 /actuals로 등록한다. 같은 actual 재전송은 하루 평가/3일 연속 상태를 늘리지 않는다. 재생 버튼은 이미 실제 실행한 시나리오 결과를 다시 보여주며 새 재학습 성공을 가장하지 않는다.

## 검증

기존 설치된 Playwright를 재사용했다:
```bash
PYTHONDONTWRITEBYTECODE=1 ../project/.venv/bin/python scripts/check_dashboard_ui.py
```

desktop/mobile, 모든 탭의390px 가로 넘침, keyboard navigation, 실제preview HTTP200/24개 finite값, 반복preview의 상태 무변경,167개422, 실제데모 기록, FAIL filter, actual duplicate HTTP200 확인. 원본 source/evidence 보존은 artifacts/ui/source_validation.json에 기록한다.
