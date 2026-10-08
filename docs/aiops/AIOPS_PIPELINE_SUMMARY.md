# AIOps 운영파이프라인및실제실행요약

KPXCSV →168h/7features →ProductionLSTM v2 →24hforecast저장 →/actuals의24h도착 →daily rolling7-day WAPE →P90Warning/P95Drift →3 consecutive calendar days →warm-start →Candidate →independentGate →MLflowRegistry/Productionalias →version-keyedServingcache.

**실제primary15%는NOT_DETECTED**. 자동training/배포아님. 수동diagnostic에서Gate PASS 및v1→v2만별도검증했다. naivemodel은benchmark로만유지.

## 실행
```bash
.venv/bin/python -m pip install -r requirements-aiops.txt
.venv/bin/python scripts/analyze_aiops.py
.venv/bin/python scripts/run_aiops.py
.venv/bin/uvicorn serving_app.aiops_main:app --host 127.0.0.1 --port 8080 --workers 1
```
run_aiops는미탐지시명시적manualdiagnostic으로Candidate/Gate를검증하는실험runner이다. liveAPI자동분기는register_actuals의3일event및56일observedhistory만사용한다. 이미실행한동일holdout을계속반복튜닝하면독립성을약화시키므로새운영기간으로검증해야한다.

## API
- /health: 모델cache로드여부.
- /predict: start_datetime=첫과거관측의KST정각,sequence168개; forecast=start+168h가midnight. 저장된forecast는immutable. 반환24hour/mwh와정확한Registryversion.
- /actuals: date,actuals24개; forecast없는날/충돌/역순거절;동일actual재전송멱등.
- /aiops/status: 실제rollingWAPE,threshold,streak,version. 최소7개연속완전days이전은insufficient_data.
- /model/info: MLflowProductionalias의actualversion,metrics,params,featureconfig.
- /logs: 실제aiops.log.
- /batch-test: scenario=normal|drift; threshold/injection실험의초기v2reference예측으로고정된실제결과를replay한다. currentProductionversion도별도표시하며새모델의성능데모라고혼동하지않는다.

기존baseline v1앱은serving_app.main에서그대로사용가능하다. AIOps는새entrypoint를사용해기존168-only 계약을깨지않았다. 모델file/scaler/이전artifacts/문서는hash보존검증.

## 상태및남은작업
primary15%검출=미탐지,수동Gate=PASS,MLflow/API검사PASS. Docker/dashboard ready=NO(주요기본시나리오민감도미검증). 다음은Capacity/level/bias 지표의업무적적합성검토,새기간calibration/holdout,live동시성/데이터수집정책을정의하는것이다. Docker나dashboard는이번에구현하지않았다.
