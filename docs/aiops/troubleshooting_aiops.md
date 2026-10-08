# AIOps 운영 정책 트러블슈팅

## 사례 1. 임의8%/10%와실제성능분포불일치
문제/증상: v1=52.288744%, v2=42.860109%,best naive=33.892438%. 정상rolling Test mean=47.570426%, 기존10%는41회전부초과.
원인: 정상분포확인전정한운영가정.
영향: 정상모델도지속경보.
해결: injection전에Validation P90/P95로freeze.
전후: Warning=74.780965%,Drift=83.696140%, 정상Test3일episode오탐0회.
배운점: 임계값은정상운영분포에서출발해야한다. 높은threshold는미탐지를늘릴수있으므로정확도SLA와구분한다.

## 사례 2. LSTM이naive보다나쁜성능
문제/증상: v2 WAPE42.86%>naive33.89%.
원인: [기존진단](../troubleshooting_model_quality.md)의가설은아직인과검증되지않음.
영향: 높은정상오차로작은구조변화감지어려움.
해결: 이번은사용자요청대로LSTM v2운영loop,naive는benchmark유지.
전후: 정확도최고모델선택이라고포장하지않음.
배운점: 교육용운영대상선택과정확도기준선택은별개.

## 사례 3. +15%가WAPE상승/탐지로이어지지않음
문제/증상: same-after WAPE 51.128592%→49.819659%,확정일=None.
원인: 상대오차분모도커지고각시간의과소/과대예측방향에따라residual변화가다름. 실제numerator/denominator는injection_error_diagnostics.json에기록.
영향: 고정P95정책은15%설비편입을미탐지.
해결: primary를FAIL_NOT_DETECTED로보존. threshold/15%를바꾸지않고20/25%별도민감도분석.
전후: 자동재학습없음;수동진단과분리.
배운점: 설비규모변화를WAPE하나로항상감지할수없다. Capacity-normalized/bias/level변화지표는후속후보이며이번에새검출기로구현하지않음.

## 사례 4. 8주전체fit+Gate중복누수방지
문제: 같은데이터로candidate를학습/검증하면낙관평가.
원인: 단일최근window를역할구분없이재사용.
영향: 안전배포판정왜곡.
해결: first42tuning(35fit+7ES),last14Gate;as-of까지만cut.
전후: Gate label은fit/ES와분리,actual weights equality 및split검사PASS.
배운점: Gate를EarlyStopping에도쓰면독립검증이아님.

## 사례 5. Gate FAIL의안전한기존Production유지
문제: 자동재학습성공을무조건배포로혼동할수있음.
원인: fitting과운영승격의역할혼동.
해결: absoluteP95 ceiling AND same-window relative improvement.
실제결과: 수동진단Production 47.362817%→Candidate 46.701117%,PASS, v1→v2.
FAIL branch는pureGate/alias불변unit check만실행했고실제FAIL training이일어났다고주장하지않는다.
배운점: Run/artifact보존과Production승격은별도.

## 사례 6. MLflow version형식검사오류
실제증상: registered_model.aliases['Production']와str version비교AssertionError(checks_failed.log).
원인: 설치된MLflow가aliasversion을int로반환.
해결: 비교전에str로정규화. 재실행API/Registry체크PASS.
영향: 첫검사가실패했으나Run/모델결과는조작하지않음.

## 설계상추가기록
- Git저장소가없어original_hashes 및new-file diff로기존모델/결과/문서보존검증.
- single-process RLock+atomic JSON, one-worker만지원. 멀티프로세스운영은transactional storage가후속요구사항.
- forecast/actual날짜불일치, duplicateforecast, conflictingactuals, out-of-order actuals를거절. actual재전송은daily count를늘리지않음.
- live자동재학습은56일의이미관측된history가없으면defer하며미래CSV로몰래채우지않는다.
- API contract check는현재v2(version2)로과거날짜를replay한기능검사다. **후행weights로과거운영성능을검증했다고주장하지않음**. Threshold/Test및Gate의시간누수검증은별도의frozen실험과독립Gate기록으로평가했다.
- 기존Baseline v1 /predict앱은보존;AIOps앱은start_datetime필수및하루midnight forecast조건으로새entrypoint에분리.
- high threshold의무오탐은15%검출민감도성공을뜻하지않는다. Docker/dashboard용기능은준비됐으나primary자동loop검출데모는완료하지못했으므로최종ready=NO.

## 최종 API 검사의 시점 및 상태 분리
최종 API 검사 날짜는 2025-12-18부터 2025-12-25까지이며 Candidate fine-tuning/ES 종료 2025-12-17T23:00:00+09:00보다 뒤이다. 별도 AIOPS_STATE_PATH=api_validation_state_chronological.json으로 기존runtime_state를 보존했다. 예측/실제값/중복멱등/모델정보/상태/로그/167·169검사 모두 PASS. 이는 API replay 기능 검사이며 새 미래기간 운영성과를 주장하지 않는다. 기존 1차기능검사결과는 api_initial_contract_verification.json에 보존했다.


# [SCENARIO_2_AUTOMATIC] 사례 6. +15% Drift에서 상대오차가 감소했다

**문제/실제:** 동일 이후 기간 WAPE 51.128592% → 49.819659%. 첫 실험은 FAIL_NOT_DETECTED, 자동 재학습 NOT_RUN으로 그대로 보존했다.
**실제 bias:** 평균 signed error(predicted-actual)=55.861175 MWh, median=40.564601 MWh, overprediction=60.416667%, underprediction=34.027778%, MAE=171.506081 MWh.
**원인 해석:** 평균 양의 bias가 확인됐지만 모든 오차가 가까워졌다는 의미는 아니다. 이전 실험의 MAE/RMSE는 증가했고, actual 분모 증가와 오차 방향이 함께 작용해 WAPE가 감소했다.
**해결:** 업무적으로 가능한 반대 변화인 출력제어/설비 성능 저하를 별도 시나리오로 설계했다. first failure 기록/CSV/기존모델을 수정하지 않았다.
**배운 점:** 평균 bias·절대오차·상대오차 분모를 함께 봐야 한다.

# 사례 7. 기본 -15%와 모든 민감도 실험

시작일 2025-12-08, 유의미한 발전량 조건=23.94099994 MWh(기존 Train peak1%, 첫 실험과 동일), 기존 threshold Warning=74.780965%,Drift=83.696140%를 유지했다.

| 감소율 | 이후 pooled WAPE | 최대 rolling WAPE | 첫 초과 | 3일 탐지 |
|---|---:|---:|---|---|
| -15% | 58.002828% | 97.715227% | 2025-12-25 | None |
| -20% | 62.116568% | 106.316570% | 2025-12-24 | 2025-12-26 |
| -25% | 67.550508% | 116.831756% | 2025-12-16 | 2025-12-26 |
| -30% | 74.602085% | 129.121359% | 2025-12-16 | 2025-12-26 |

**기본 -15%는 미탐지**였다. 시험한 grid에서 최소 검출은 -20%; 연속 변화율 전체의 수학적 최소값이라고 주장하지 않는다. Test를 이용해 데모 강도를 선정한 민감도 실험이며 새로운 운영기간에 일반화된 검출성능이라고 주장하지 않는다.

# 사례 8. Detector가 실제로 재학습을 호출했다

첫 초과=2025-12-24,2일=2025-12-25,3일=2025-12-26, delay=18일. /batch-test → replay_demo → check_and_trigger → daily_policy event → retrain 경로이며 CLI/사람이 retrain을 직접 호출하지 않았다. actual24시간을 하루 한 번 평가하며 동일 날짜로3회를 만들지 않았다.

**시간/Registry 분리:** 기존 수동 Production v2는 12월17일까지 fine-tuning/ES를 경험했으므로 12월8일 시나리오에 그대로 넣으면 시점 해석이 혼동될 수 있다. 원본 당시 Production LSTM-v2의 실제 MLflow version1 weights를 읽어 별도 `scenario_2/mlflow.db`에 초기 Production으로 복제했다. weights가 baseline_v2.keras와 같은지도 확인했다. 기존 root Registry 수동 v2와 기존 Run/artifact는 변경하지 않았다.

**데이터 반영:** 최근56일 2025-11-01T00:00:00+09:00~2025-12-26T23:00:00+09:00; first42 fit=2025-12-12T23:00:00+09:00,last14 Gate=2025-12-13T00:00:00+09:00~2025-12-26T23:00:00+09:00. 10epochs,LR1e-4,seed42,shuffle=False. 이번은 EarlyStopping 없이42일 전체를fit했고 Gate는 어떤 fitting/모델선택에도 사용하지 않았다. 이는 검출 초기에 최근 감소값이 내부ES에만 들어가는 상황을 피하기 위한 명시적 변경이다.

Optimizer fit에 실제 감소 관측 69시간이 포함됐다: 2025-12-08T08:00:00+09:00~2025-12-12T21:00:00+09:00. as-of 이후 미래값은 사용하지 않았다. Gate 입력 context는 과거fit구간과 겹칠 수 있지만 label은 겹치지 않는다. detector실험은 원본 prediction을 고정한actual-only 시뮬레이션이며 Gate에서는 두 모델에 동일한 관측된감소input을 제공했다.

# 사례 9. 자동 Candidate가 상대개선 조건을 통과하지 못했다

Production: MAE=182.419072,RMSE=286.935233,WAPE=82.671954%. Candidate: MAE=183.696922,RMSE=297.610817,WAPE=83.251072%.

실제 Gate=FAIL. Candidate는 절대상한 이하였지만 동일Gate의 Production보다 WAPE가 높아 거절됐다. 임계값/epochs를 변경해 PASS를 유도하지 않았다. 실제 [FAIL] candidate rejected → existing production retained 로그, Candidate Run/artifact 보존, isolated Production v1→v1 유지.

**API/cache:** 승격이 없으므로 기존 모델을 유지하는 것이 정상이다. API실제version,Registryversion,명시적모델예측이 일치했다. cache mismatch는 발생하지 않았고 이슈를 만들어 기록하지 않는다. Promotion/cache invalidation branch는 이번FAIL 실행에서는 실행하지 않았음을 명확히 한다.

**결론:** 자동 탐지→warm-start→독립Gate→안전한유지까지 end-to-end 제어루프는 검증됐다. 모델의 성능 복구/자동승격 성공은 아니다. Docker/dashboard/report의 다음 구현 단계는 진행 가능하지만 Candidate 품질 개선과 새기간검증은 남아 있다.
