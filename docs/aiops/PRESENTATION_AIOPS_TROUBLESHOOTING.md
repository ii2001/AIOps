# Troubleshooting 1: 임의 임계값은 실제 운영에서 작동하지 않았다

| 단계 | 실제 관측과 조치 |
|---|---|
| Before | 계획 Gate 8%, Drift 10% |
| Observed | LSTM v2 Test WAPE 42.86%. 정상 Test rolling 평균 47.57%; 41회 모두 10% 초과 |
| Problem | 정상 운영도 지속적으로 Drift로 분류될 수 있음 |
| Action | 주입 전에 Validation 정상 rolling 분포 P90/P95로 정책 확정 |
| After | Warning 74.78%, Drift 83.70%. 정상 Test 3일 연속 오탐 0회 |

**배운 점:** 운영 정책은 정상 성능 분포에서 출발해야 한다. 그러나 높은 임계값의 무오탐이 좋은 검출 민감도를 뜻하지는 않는다.

# Troubleshooting 2: 구조 변화가 항상 WAPE 증가를 뜻하지 않았다

- **시나리오:** 2025-12-08부터 유의미한 발전량에 신규 설비 편입 +15% 적용. 기존 예측은 고정.
- **실제 결과:** 동일 이후 기간 WAPE 51.13% → 49.82%. MAE/RMSE는 증가했지만 WAPE는 감소.
- **Detection:** 3일 연속 조건 미충족. **자동 재학습·자동 배포 없음.**
- **추가 확인:** +20%/+25%는 별도 민감도 분석에서도 미탐지. 기본 주입률과 임계값은 바꾸지 않음.
- **별도 수동 진단:** 실제 warm-start 및 독립 14일 Gate 검증. Production 47.36% vs Candidate 46.70%, Gate PASS, Registry v1 → v2.

**검증한 것:** 일일 평가, 멱등 처리, MLflow 기록, 독립 Gate 및 안전 승격 코드.
**아직 성공하지 못한 것:** 기본 +15% 업무 시나리오의 자동 검출.

**배운 점:** 모델이 응답하고 재학습이 가능해도 업무상 Drift를 검출한다고 가정하면 안 된다. WAPE의 분모 및 오차 방향을 함께 분석해야 한다.

[정상 분포 차트](../../artifacts/aiops/threshold_distribution.png) · [주입 전후 차트](../../artifacts/aiops/normal_vs_drift_wape.png) · [수동 Gate 비교](../../artifacts/aiops/production_before_after.png)


## [SCENARIO_2_AUTOMATIC] 시행착오 2 — Drift를 넣었는데 오차율이 줄었다

- **Before:** 신규설비 +15%, WAPE51.13%→49.82%,미탐지.
- **Bias:** mean signed error +55.86 MWh, 과대예측60.42%.
- **Action:** 업무적으로 가능한 출력제어/설비성능저하. 기본-15%도미탐지, -20/-25/-30%를전부별도평가.
- **Result:** 시험한최소검출-20%, 2025-12-26 3일확정(delay18일).
- **AUTO:** Detector가warm-start를호출. 최근6주fit/2주Gate, fit안에감소실제값69시간.
- **Gate:** Production82.67% vs Candidate83.25%, **FAIL**. 모델승격없이기존Production유지.
- **Lesson:** 데이터를바꿨다는사실이아니라업무환경변화와모델오차방향을함께검토해야한다. 자동재학습성공도자동배포의근거가아니다.

별도scenario2 Registry를사용했고 이전수동v1→v2 승격은그대로보존했다. 이번새자동실험은isolated v1유지이며복구배포성공으로포장하지않는다.
