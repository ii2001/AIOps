# Baseline v2 serving contract 설계 및 검증 범위

기존 /predict는168개 MWh만 받는다. v2의 hour/weekday/day-of-year는 그 값만으로 복원할 수 없으므로 v1 endpoint/schema/model/scaler는 변경하지 않았다. v2 HTTP endpoint는 이번 단계에 구현하지 않았다.

후속 권장 endpoint: `POST /predict/v2`.
```json
{"start_datetime":"2025-11-07T06:00:00+09:00", "sequence":["실제 MWh 값 168개"]}
```
위 sequence placeholder는 계약 설명용이며 실제 요청 payload가 아니다. start_datetime은 **가장 오래된 첫 입력 관측의 구간 시작**이다. 마지막 입력 시각=start+167시간, 첫 forecast 시각=start+168시간. Asia/Seoul 또는 UTC+09:00의 정각만 허용하며 음수/NaN/inf·167/169개 입력은 거절한다. 서버 현재 시간을 임의로 넣지 않는다. 시간대없는 timestamp나 누락시간 데이터도 거절해야 한다.

sequence 각 timestamp를 start+i시간으로 생성하고 `src/time_features.make_time_features`로 학습과 동일한7features를 만든다. v2 별도 cached loader는 baseline_v2.keras/scaler_v2.pkl를 사용해야 한다. 출력은 기존24개 hour/mwh 및 region/fuel을 유지하고 model_version=baseline-v2로 구분한다. Input hour=lead hour라 timestamp 응답을 추가한다면 별도 명시한다.

2025 day-of-year는1~365를365로 나눠 sin/cos를 생성한다. 공용 함수는 윤년366을 사용한다. MWh는 Train-fitted scaler_v2로만 transform, sin/cos 재fit 없음. 미래 관측량이나 기상정보를 구성하지 않는다.

이번 실제 검증: 저장된 v2 모델을 로드하고 Test window의 첫 timestamp+168개 MWh로 timestamp를 재구성했다. 공용 feature가 offline X_v2와 동일하고24개 MWh 출력이 저장된 offline 예측과 일치하는지 tests/check_model_quality.py로 검증한다. 이는 **공용 전처리/저장 모델 smoke test**이며 v2 HTTP200을 검증했다고 주장하지 않는다. 기존 v1 API 코드와 기존HTTP검증 결과는 보존한다.
