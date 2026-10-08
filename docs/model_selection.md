# 실제 모델 후보 선택

| Model | MAE MWh | RMSE MWh | WAPE % | v1 대비 개선율 % |
|---|---:|---:|---:|---:|
| Baseline v1 | 191.985006 | 299.919240 | 52.288744 | 0.000000 |
| Previous Day Same Hour | 140.258263 | 277.651699 | 38.200527 | 26.943116 |
| Previous Week Same Hour | 138.029838 | 286.094112 | 37.593597 | 28.103845 |
| 7-Day Same-Hour Mean | 124.440547 | 228.439342 | 33.892438 | 35.182153 |
| Baseline v2 LSTM | 157.366530 | 247.058669 | 42.860109 | 18.031864 |

Best model: 7-Day Same-Hour Mean; best naive: 7-Day Same-Hour Mean. 주요지표 WAPE에 따라 선정했으며 LSTM이라는 이유로 우선하지 않았다.

Production candidate 추천: **현재는 운영용 승격 추천하지 않음**. Best WAPE 33.892438%와 계획 Gate8%를 비교해야 하며, 독립 미래기간/walk-forward 검증도 필요하다. 현재 7-Day Same-Hour Mean를 차기 개선의 성능 기준으로 사용한다. 상대적으로 가장 좋다는 의미와 운영 기준 통과는 다르다.

평가 범위: v1과 동일한1123 rolling origins ×24 horizons. 첫 target 2025-11-14T06:00:00+09:00, 마지막 horizon 2025-12-31T23:00:00+09:00. ground truth는 v1 test_actual.npy를 그대로 사용해 float32 normalization/inverse 오차까지 일치시켰다. 이전일은 입력 마지막24시간, 이전주는 입력 첫24시간, 평균은 입력7×24시간의 같은 위치 평균으로 계산했으며 모든 참조는 forecast origin 이전이다. 원시70/15/15 split과 파티션 내부 sequence, Train-only scaler 및 출력0 floor 정책을 유지했다.

모델 개선 효과는 있었으나 단순 seasonal baseline을 아직 이기지 못함
