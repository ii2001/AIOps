# AIOps End-to-End — 실제 자동 탐지와 안전한 배포 거절

Normal → Daily rolling7-day WAPE → Persistent curtailment → 3-day exceed → Auto Drift Detection → Warm-start → Candidate → Independent Gate → FAIL → Production retained

| 실제 항목 | 결과 |
|---|---|
| First +15% scenario | FAIL_NOT_DETECTED, 보존 |
| 기본 -15% | 미탐지 |
| 최소 검출 강도(시험 grid) | -20% |
| 같은 이후 원본 WAPE | 51.128592% |
| 감소 시나리오 WAPE | 62.116568% |
| threshold | 83.696140% (변경 없음) |
| first/second/third exceed | 2025-12-24 / 2025-12-25 / 2025-12-26 |
| delay | 18일 |
| auto retrain | YES, 실제 MLflow Run b45409dc0a3344f7939d253c8ff9e755 |
| Production Gate WAPE | 82.671954% |
| Candidate Gate WAPE | 83.251072% |
| Gate | FAIL, 상대 개선 실패 |
| isolated Registry version | v1→v1 RETAINED |
| original Registry | 이전 수동 Production v2 보존 |

모델성능 복구/자동승격은 성공하지 않았다. Detector→재학습→평가→안전한 거절 루프는 실제 자동 실행으로 확인했다. 후보성능 개선,새기간 holdout,검출민감도 일반화는 다음 과제다.

## 재현
`.venv/bin/python scripts/run_aiops_demo.py`
최초실행은분리DB/새Run/API자동루프를수행한다. 이미demo_result가있으면기록된결과를출력해기존실험을덮어쓰거나재학습하지않는다. 실행log: artifacts/aiops/scenario_2/execution.log. baseline-v1앱과원본파일은보존한다.

검증: `.venv/bin/python tests/check_scenario2.py` (새training없이실제Run/API/로그/시점기록을검증).

[민감도 차트](../../artifacts/aiops/drift_sensitivity.png) · [감지 차트](../../artifacts/aiops/scenario_2_drift_detection.png) · [Gate 비교](../../artifacts/aiops/retraining_before_after.png) · [전체 타임라인](../../artifacts/aiops/aiops_full_timeline.png)
