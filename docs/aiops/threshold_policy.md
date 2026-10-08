# 실제 정상 분포 기반 운영 임계값

Production 대상은 정확도 best naive가 아니라 AIOps 교육 목표에 따라 LSTM v2이다. v2 원본 Test WAPE42.860109%, best naive33.892438%의 한계를 숨기지 않는다.

매일 midnight-issued24-hour forecast의 실제24시간이 모두 확보된 뒤 한번만 최근7일168개 **비중복** target을 평가한다. Validation/Test 각각47개 완전한 daily forecast,41개 rolling 평가다. Validation 정상 분포로 Warning=P90=74.780965%, Drift=P95=83.696140%를 확정한 뒤 Test/injection을 평가했다. std는 population standard deviation이다. 원래 Validation은 v2 EarlyStopping에도 쓰인 구간이므로 완전히 새 calibration holdout은 아니며 추가 미래기간 검증이 필요하다.

| 분포 | mean | median | std | P75 | P90 | P95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 52.841456 | 51.824532 | 16.232381 | 66.895527 | 74.780965 | 83.696140 | 32.100403 | 87.615592 |
| Test | 47.570426 | 41.812928 | 12.629129 | 58.294722 | 63.721328 | 70.852041 | 34.075208 | 80.643836 |

정상Test41회 중 기존10%초과=41, 기존8%초과=41. 새 Warning 발생2일, Drift 초과0일,3일연속 episode 오탐0회이다. Rolling windows가 겹치므로41개를 독립 표본으로 해석하지 않는다.

WAPE≥Drift를 달력상 연속된3개 **날짜**에서 확인해야 drift가 된다. 같은날actual 재전송은 멱등이며 상태를 증가시키지 않는다. 누락된날은 연속조건/rolling window를 끊고, 한 episode당3일째 한번 event를 발생시킨다.

절대Gate=P95=83.696140% AND Candidate WAPE<Production WAPE를 동일한 independent Gate에 적용한다. P95는 오차가 큰 v2의 **데모용 운영 ceiling**이며 business accuracy SLA가 아니다. Calibration7일daily와 Gate14일rolling metric은 동일 WAPE 단위지만 가중치와길이가 다르므로 이 ceiling의 일반적 통계 보장은 주장하지 않는다. 모델별/계절별 calibration과 운영정책 추가검토가 필요하다. threshold를 injection을 보고 낮추지 않았다.
