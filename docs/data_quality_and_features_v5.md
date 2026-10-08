# 실제 데이터 정제·특징 개선 결과

요청 범위는 실제 데이터의 품질 점검과 입력 특징 개선이다. 원본 KPX CSV, target 정답, 기존 Train/Validation/Test 구간을 보존하고 별도 실험으로 실행했다.

## 데이터 점검 및 원본의 의미

전라남도 태양광 2025년 8,760시간에 중복키 0, 빈 필드 0, 비유한 값 0, 음수 0, 시간 누락 0을 확인했다. 관측 0값은 915건이며 최대 거래량은 2,394.099994 MWh다. 이미 유효한 값들을 누락이나 오류로 보고 임의로 삭제·보간할 근거는 없었다. 고출력 또는 저출력이라는 이유만으로 이상값으로 제거하지 않았다.

[KPX 공식 데이터 설명, PDF 5페이지](https://new.kpx.or.kr/boardDownload.es?bid=0045&list_no=71003&seq=2)에 따르면 이 자료는 전력시장 참여 발전기의 송전단 거래량이며 심야 태양광 값에 ESS 방전이 포함될 수 있다. 따라서 야간 값을 강제로 0으로 바꾸지 않았다. 이 자료의 발전량과 거래량 의미를 그대로 유지한다.

원본 SHA-256과 검사 결과를 `artifacts/data_features_v5/data_audit.json` 및 `original_hashes.json`에 저장했다. 기존 source/model/report 파일도 hash로 보존을 확인했다.

## 입력 변환

1. 예측 origin 직전 168시간 값의 95th percentile로 입력과 학습 target을 정규화한다. 0인 이력에도 동작하도록 분모를 최소 1 MWh로 한다. 평가에서는 예측을 원래 MWh 단위로 복원하고 기존 Test 실제값과 비교한다. 평가 정답 자체는 정규화하거나 평활화하지 않는다.
2. 최근 24시간의 정규화된 관측 프로필과 과거 규모 특징을 추가한다. 최근 7일 분위수는 과거 출력 규모의 proxy이며 실제 설치 용량을 측정한 값이 아니다.
3. 별도 후보에는 [NOAA의 태양 위치 계산](https://gml.noaa.gov/grad/solcalc/solareqns.PDF)을 이용한 시각·계절 특징을 추가한다. 대표점 34.8N/126.8E를 가정한 지역 proxy로, 개별 발전소 위치나 기상예보가 아니다. 내부 거래구간의 중간 시각을 사용하며 야간 target/예측을 강제로 0으로 만들지 않는다.

트리는 기존 설치된 scikit-learn을 재사용했다. 절대오차 loss에 각 origin의 복원 scale을 sample weight로 적용해 원래 MWh 절대오차와 학습 방향을 맞췄다. random 내부 Validation과 미래 실제값을 feature에 사용하지 않는다.

## 선택과 결과

7일 평균, 정규화 후보, 정규화+태양 위치 후보를 고정했다. 앞 Validation에서만 후보를 선택했고, 뒤 14일 Gate와 Test 확인 전에 `selection.json`으로 선택 결과를 동결했다.

| 선택용 Validation 후보 | WAPE |
| --- | ---: |
| 7일 평균 | 49.7324% |
| 과거 규모 정규화 | **44.3684%** |
| 정규화 + 태양 위치 proxy | 44.6449% |

선택된 후보는 **과거 규모 정규화**다. 태양 위치 특징은 선택 모델에 사용하지 않았다.

| 평가 구간 | 7일 평균 | 이전 v3 | 선택된 데이터 특징 v5 |
| --- | ---: | ---: | ---: |
| Gate | 15.1524% | 14.8769% | 16.6393% |
| Test | 33.8924% | **33.8185%** | 34.0472% |

선택 구간의 개선이 Test로 이어지지 않았다. 모델 승격 조건은 상한 40%와 동일 Gate 구간의 Production/7일 평균 중 좋은 쪽보다 5% 이상 WAPE 개선이다. 실제 Gate는 **FAIL**이며 운영 API, MLflow Production, Docker, 기존 임계값은 그대로 유지했다. 상한은 개발용 제안이며 사업 SLA나 산업 표준이 아니다.

Test는 이전 실험에서도 확인한 기간이므로 새 미래 기간에서 독립 평가가 필요하다. 이 결과를 확인한 뒤 Test 수치가 좋아질 때까지 변환 설정을 반복 검색하지 않았다. 발표에서는 이번 실험을 성능 개선 성공으로 표시하지 않는다. 앞 Validation에서만 오차가 줄었다는 제한을 함께 기록한다.

## 다음 작업에 필요한 정보

현재 입력은 발전량/거래량 이력과 달력뿐이다. 다음에는 예측 시점에 발행되어 있던 기상예보, ESS 운영 또는 별도 거래량 분해 정보, 지역 설비 용량 이력과 여러 해의 데이터를 검토한다. 확보되지 않은 정보를 만들어 넣지 않았다. 실제 계량 오류가 확인되면 출처·수정 사유·대상 행을 기록하고 원본과 별도의 정정본으로 다룬다.

## 재현 및 검증

```bash
.venv/bin/python scripts/train_data_features_v5.py
.venv/bin/python tests/check_data_features_v5.py
```

학습은 기존 결과가 있으면 덮어쓰지 않는다. 파일:

- `src/normalized_features.py`: 과거만 사용하는 입력 변환 및 단일 예측 함수
- `scripts/train_data_features_v5.py`: audit, 사전 계획, 후보 선택, 실제 평가
- `tests/check_data_features_v5.py`: 원본/기존 파일 보존, 동일 Test 정답, 미래 target 변경 시 feature 불변, 0 이력 처리, 태양 위치 계절 sanity, 모델 재로딩 예측 일치, Gate 판정 검사
- `artifacts/data_features_v5/`: 실제 결과 JSON/CSV 및 예측 배열
- `reports/data_features_v5/`: Git 검토용 작은 결과 사본과 비교 차트
- `serving_app/models/data_features_v5.pkl`: 로컬 후보 모델, 기존 ignore 정책으로 Git 제외

기존 백엔드에 연결하지 않은 실험 후보이며, 원본 CSV를 정정한 파일은 만들지 않았다. 정정할 실제 데이터 오류가 발견되지 않았기 때문이다.
