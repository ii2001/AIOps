# 모델 성능 트러블슈팅 - Baseline v1

## 1. 문제 발견
v1은168시간 입력→24출력, FastAPI200 및validation이 동작했지만 Test MAE=191.985006 MWh, RMSE=299.919240 MWh, WAPE=52.288744%였다. 기능과 예측 품질은 별개다.

## 2. 왜 문제인가
기획 Gate WAPE≤8%, 최근7일 drift WAPE≥10% 대비 초기 baseline 오차가 높다. 코드는 정상인데 제안된 품질 기준에서는 운영 불가능한 상태다. 전체 Test WAPE만으로 개별 최근7일 경보를 단정하지 않고 실제 daily-origin7일 집계도 별도로 계산했다. 운영 threshold를 코드에 적용하거나 변경하지 않았다.

## 3. 기준 모델 비교
| Model | MAE MWh | RMSE MWh | WAPE % | v1 대비 개선율 % |
|---|---:|---:|---:|---:|
| Baseline v1 | 191.985006 | 299.919240 | 52.288744 | 0.000000 |
| Previous Day Same Hour | 140.258263 | 277.651699 | 38.200527 | 26.943116 |
| Previous Week Same Hour | 138.029838 | 286.094112 | 37.593597 | 28.103845 |
| 7-Day Same-Hour Mean | 124.440547 | 228.439342 | 33.892438 | 35.182153 |
| Baseline v2 LSTM | 157.366530 | 247.058669 | 42.860109 | 18.031864 |

평가 범위: v1과 동일한1123 rolling origins ×24 horizons. 첫 target 2025-11-14T06:00:00+09:00, 마지막 horizon 2025-12-31T23:00:00+09:00. ground truth는 v1 test_actual.npy를 그대로 사용해 float32 normalization/inverse 오차까지 일치시켰다. 이전일은 입력 마지막24시간, 이전주는 입력 첫24시간, 평균은 입력7×24시간의 같은 위치 평균으로 계산했으며 모든 참조는 forecast origin 이전이다. 원시70/15/15 split과 파티션 내부 sequence, Train-only scaler 및 출력0 floor 정책을 유지했다.

복잡한 LSTM v1이 세 단순 seasonal baseline보다 성능이 낮았다.

## 4. 원인 가설
확인된 코드 사실: v1 입력은 MWh 하나, 명시적 hour/weekday/season/weather/capacity feature가 없다. 24개 직접출력 Dense 구조와 LSTM32를30epochs 학습했다. 실제 전체 시계열의 시간대 평균 min=0.006238, max=1349.887705 MWh로 시간대 패턴이 존재한다. 월별 평균 min=351.352697, max=620.887123 MWh도 변하지만 이것만으로 weather/설비/출력제어의 인과관계를 확인할 수는 없다. [실제 진단값](../artifacts/model_comparison/v1_diagnostics.json).

가능한 원인: 시간대·요일·연중 계절 정보를 명시하지 않아 주기를 학습하기 어려웠을 수 있다. weather/설비규모/출력제어 등 외생변수 부재가 일별 변동 오차를 키웠을 수 있다. direct24출력의 학습량이나 capacity 부족도 가능하지만 확정 원인은 아니다. Train이1월~9월, Test가11~12월이므로 계절 일반화 문제도 가설이다. 야간 관측값을 임의0으로 바꾸지 않았다.

## 5. 개선 조치
MWh + hour sin/cos + weekday sin/cos + day-of-year sin/cos의7features, LSTM64→Dense64(ReLU)→Dense24를 별도 v2로 학습했다. Seed42, shuffle=False, 최대60epochs, val_loss EarlyStopping patience10/best weights restore. 실제 종료 18epoch, best epoch 8. sin/cos는 재fit하지 않으며 MWh scaler는 Train-only이다. 미래 날씨/target 시각 feature는 사용하지 않았다.

시간 feature, capacity, 학습 종료조건이 함께 바뀌었으므로 결과를 **시간 feature만의 순수 효과**로 인과 해석할 수 없다. 한 번의 고정-seed 실험이며 하이퍼파라미터 탐색은 수행하지 않았다.

## 6. 개선 전후 결과
v1 WAPE=52.288744%, v2=42.860109%, 상대 개선율=18.031864%. Best naive=7-Day Same-Hour Mean / 33.892438%.

## 7. 결론
모델 개선 효과는 있었으나 단순 seasonal baseline을 아직 이기지 못함. 전체 최저 WAPE 모델은 7-Day Same-Hour Mean / 33.892438%이다. 제안 Gate8%까지 25.892438 percentage points, drift10%까지 23.892438 percentage points 차이가 있다.
Best model의 daily-origin7일 평가 41 windows 중 41개가10% 이상이고, 범위는16.783993~61.349715%였다. 현재 정의의 경보가 초기 상태부터 빈번할 수 있으므로 정상 조건/최근7일 집계 규약과 운영 정책 검증이 필요하다. [실제7일 집계](../artifacts/model_comparison/rolling_7day_diagnostics.json). 운영 준비 판정: NOT YET.

## 8. 다음 개선 방향
우선 weather(일사량/운량)·설비용량·출력제어 관련 데이터의 접근성과 timestamp 품질을 확인하고, 현재 strongest naive를 기준으로 개선을 검증한다. 동일capacity/동일학습설정으로 시간feature ablation을 하면 기여도를 더 정확히 구분할 수 있다. XGBoost/LightGBM 등 lag+calendar 기반 비교도 후속 후보이며 Transformer/TCN 확장은 단순모델과 외생변수 검증 이후에 판단한다. 이번에는 수집/구현하지 않았다. Test를 후보 비교에 사용했으므로 Production 선택 전 새 기간의 독립 holdout이나 walk-forward 검증이 필요하다.
