# 모델 개선 v3 실제 실험과 Gate 권고

## 결과

전라남도 태양광 2025년 시간별 데이터, 기존과 동일한 Test 1,123개 origin × 24시간 전체를 평가했다. WAPE는 정확도가 아니라 오차율이며 작을수록 좋다. 중복되는 rolling 예측을 합산한 지표이므로 독립 일별 평가와 동일하지 않다.

| 모델 | Test MAE (MWh) | Test RMSE (MWh) | Test WAPE |
| --- | ---: | ---: | ---: |
| 기존 LSTM v2 | 157.37 | 247.06 | 42.8601% |
| 최근 7일 동일 시간 평균 | 124.44 | 228.44 | 33.8924% |
| 개선 residual LSTM v3 | 124.17 | 230.51 | 33.8185% |
| 고정 설정 histogram gradient boosting | 129.18 | 235.19 | 35.1845% |

v3는 v2 대비 WAPE를 9.0416%p 낮췄다. 그러나 7일 평균 대비 차이는 0.0739%p에 불과하다. 기존 LSTM의 약점을 줄인 실험이며, 상용 품질 확보나 단순 모델 대비 우월성 입증으로 해석하지 않는다. 트리 모델은 이번 한 설정에서는 평균보다 나빴다. Test 결과를 보고 설정을 반복 조정하지 않았다.

## 변경 사항과 누수 방지

v3는 기존 168시간 관측과 시간 특징을 사용하되 다음 24시간의 이미 알 수 있는 달력 특징, 과거 동일 시간의 전일/전주/7일 평균을 함께 입력한다. 7일 평균을 시작점으로 LSTM이 잔차를 학습한다. 미래 발전량이나 외부 기상 실측값은 입력하지 않는다. MAE loss를 사용한다. 고정된 평가 대상에서 WAPE 분모는 일정하므로 절대오차를 줄이는 방향과 일치한다.

Train은 기존 구간 그대로이며 scaler는 Train만 사용한다. 기존 Validation 중 앞 구간으로 EarlyStopping하고 마지막 14일을 Gate로 분리했다. 모든 24시간 target은 자신의 구간 안에 있고, context는 예측 시점 이전 관측만 사용한다. v3는 최대 50 epoch, seed42, patience8; 실제 9 epoch 후 종료하고 최선 1 epoch 가중치를 복원했다. 기존 v2는 전체 Validation으로 선택했던 모델이므로 v2와의 Gate 비교가 완전히 독립적인 모델 선택 비교는 아니다. 기존 Test도 이전 실험에서 이미 확인한 기간이다. 새로운 미래 기간을 확보해 최종 성능을 확인해야 한다.

## 이번 실험의 Gate와 운영 권고의 구분

이번 실험은 학습과 Test 확인 전에 앞 Validation의 7일 평균 WAPE 49.7324%를 상한으로 동결했다. 뒤 14일 Gate에서 v3 14.8769%, 기존 v2 38.1499%, 7일 평균 15.1524%였으며, 상한 이하 + 기존 모델보다 개선 + 평균보다 개선 조건으로 **실험 Gate PASS**다. 이 49.7324%는 실험 기준이며 사업 품질 SLA가 아니다. 앞 Validation 평균 49.73%, 뒤 Gate 평균 15.15%, Test 평균 33.89%의 차이는 단일 기간만으로 절대 상한을 정하기 어렵다는 증거다.

권장하는 팀 프로젝트의 다음 목표는 다음과 같다. 아래 수치는 확정된 산업 표준이나 통계적으로 검증된 운영 임계값이 아닌, 앞으로 사전 선언하고 검증할 개발 목표다.

- 모델 품질 목표: 새 독립 평가 기간에서 **WAPE 30% 이하**를 우선 목표로 한다. 그 이후 요구 품질을 더 낮춘다.
- 잠정 절대 Gate 상한 제안: **40%**. 현재 83.6961%보다 엄격한 개발용 안전장치이나, 사업적으로 허용 가능한 오차라는 의미는 아니다. Validation의 어려운 기간에는 정상 모델도 40%를 넘을 수 있으므로 곧바로 운영 설정에 적용하지 않는다.
- 동시 상대 Gate: 동일 시간·동일 실제값·동일 전처리로 candidate WAPE ≤ 0.95 × min(current Production WAPE, 7일 평균 WAPE). **상한 40%와 상대 조건을 모두 충족**해야 승격한다. 5% 상대 개선은 실무 표준이 아닌 팀의 사전 선언 목표다.
- 기존처럼 실패 후보는 승격하지 않는다. Drift 탐지 임계값과 모델 승격 Gate 상한은 별도 목적이므로 한 숫자로 일괄 변경하지 않는다.
- 7일 rolling drift 임계값은 새 모델의 정상 Validation 운영 재현에서 재보정한다. 모델 전체 Test WAPE와 일별/rolling WAPE는 평가 단위가 달라 숫자를 그대로 교환할 수 없다.

실제 배포를 판단하는 Gate 구간에서 v3의 평균 대비 상대 개선은 약 1.82%이며 권장 5% 조건을 충족하지 않는다. Test에서의 개선은 약 0.22%에 불과하다. 따라서 이번 실험 PASS만으로 Production을 교체하지 않았다. 향후 서로 겹치지 않는 여러 시간 구간에서 사전 선언 조건을 평가하고, 기상 **예보**·설비 용량·계절 변화 정보를 확보해 단순 평균을 명확하게 이기는지 확인해야 한다. 예측 시점에 알 수 없는 미래 기상 실측값을 넣으면 누수가 생긴다.

## 실행과 증거

```bash
.venv/bin/python scripts/train_baseline_v3.py
.venv/bin/python scripts/compare_tree_candidate.py
.venv/bin/python tests/check_model_v3.py
```

기존 결과 덮어쓰기를 방지하므로 첫 두 스크립트는 결과가 이미 있으면 종료한다. 새로운 실험은 별도 이름/출력 경로를 정해 진행한다.

로컬 실제 결과: `artifacts/model_v3/metrics.json`, `tree_metrics.json`, `gate_plan.json`, `comparison.csv`, `validation.json`, `wape_comparison.png`, `forecast_comparison.png`. 모델과 scaler는 `serving_app/models/baseline_v3.keras`, `scaler_v3.pkl`이다. 바이너리는 기존 ignore 정책으로 Git에서 제외한다. 검토용 작은 결과 사본은 `reports/model_v3/`에 둔다.

검증은 동일 Test target, 미래 target 변경 시 입력 불변, 전일/전주/평균 정렬, 저장 모델 재로딩 예측 일치, 기존 파일 hash 보존을 검사한다. v3는 다중 입력이므로 현재 단일 입력 serving/retraining과 바로 호환되지 않는다. 운영 API, MLflow Production, Docker 모델, 기존 Gate 설정은 변경하지 않았다.

사용 API 참고: [Keras Functional API](https://keras.io/guides/functional_api/), [scikit-learn HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html). 기존 설치된 TensorFlow/scikit-learn만 사용했으며 새 의존성은 없다.
