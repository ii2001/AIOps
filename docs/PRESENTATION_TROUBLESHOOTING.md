# Troubleshooting: 모델은 동작했지만 예측 품질이 낮았다

### 문제
- API168→24,200/422 정상. v1 WAPE **52.29%**.
- 세 seasonal naive보다 낮은 성능.

### 원인 가설
MWh 단일feature, 명시적 시간/요일/계절 정보 부재, weather/설비용량 부재. 인과 원인은 아직 검증하지 않음.

### 개선
Cyclical time features 6개 + LSTM64/Dense64 + EarlyStopping. v1 원본과 동일Test1123×24 유지.

### 결과
- v1 WAPE: **52.29%**
- Best Naive (7-Day Same-Hour Mean): **33.89%**
- v2 WAPE: **42.86%**
- v1 대비 상대 개선율: **18.03%**
- 모델 개선 효과는 있었으나 단순 seasonal baseline을 아직 이기지 못함

| 단계 | 내용 |
|---|---|
| Before | LSTM v1 WAPE 52.29% |
| 발견 | Best naive WAPE 33.89%로 v1보다 우수 |
| 원인 가설 | 명시적 시간정보와 외생변수 부재 |
| Action | Cyclical features + LSTM v2 + EarlyStopping |
| After | v2 WAPE 42.86%, relative improvement 18.03% |
| Next | Weather/Capacity 데이터 품질 검토 + naive 대비 재검증 |

### 배운 점
“모델이 정상 응답한다고 운영 가능한 것은 아니다. AIOps의 출발점은 모델 품질 기준을 먼저 검증하는 것이다.”

Gate8%/drift10%는 이번에 적용하지 않았다. best model WAPE 33.89%로 현재 운영 준비 판정은 NOT YET이다. feature와capacity/학습설정이 함께 바뀐 단일실험이므로 시간정보만의 효과로 단정하지 않는다.

[WAPE 비교](../artifacts/model_comparison/wape_comparison.png) · [실제 하루 forecast](../artifacts/model_comparison/forecast_comparison.png).
