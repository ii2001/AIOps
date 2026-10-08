# 실제 실행 결과 스냅샷

이 폴더는 포트폴리오 검토용 작은 결과만 보존한다. 모델 binary, 원본 CSV, DB, runtime state 및 전체 로그는 Git에 포함하지 않는다. 원본 artifacts는 로컬에서 유지된다. JSON의 개인 머신 경로는 `${PROJECT_ROOT}`로 치환했으며 metric/판정은 변경하지 않았다.

- [모델 비교](model_comparison/model_comparison.csv): v1 WAPE52.288744%, v2 42.860109%, best naive33.892438%.
- [차트](model_comparison/wape_comparison.png): 동일 Test windows/horizons.
- [첫 +15% 실험](aiops/scenario_1_capacity_increase.json): WAPE51.128592%→49.819659%, FAIL_NOT_DETECTED.
- [모든 감소 실험](aiops/drift_sensitivity.csv): 기본-15% 미탐지, 시험한 최소 검출-20%,2025-12-26 탐지.
- [자동 Candidate](aiops/candidate_metrics.json): Production82.671954% vs Candidate83.251072%, Gate FAIL/RETAINED. 자동승격 성공으로 해석하지 않는다.
- [검증](aiops/scenario_2_validation.json): Detector→warm-start→독립Gate→안전거절의 실제 기록.
- [대시보드](dashboard/dashboard.png): 팀 Docker UI 실제 캡처.

Scenario2의 분리 Registry와 이전 수동 승격 Registry는 서로 다르다. 상세한 해석은 docs/aiops/에 보존했다. reports는 고정 스냅샷이며 live dashboard는 로컬 artifacts의 상태를 조회한다.
