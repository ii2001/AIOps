"""AIOps app; original serving_app.main (baseline v1) stays unchanged."""
from datetime import date,datetime,timedelta
from typing import Annotated,Literal
import json
from fastapi import FastAPI,HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel,Field
from serving_app.main import validation_error
from src.aiops_registry import OUT,NAME,current,client,assets
from src.aiops_runtime import LOCK,read_state,save_state,forecast,register_actuals,status,log
from src.aiops_retraining import retrain
from src.time_features import FEATURE_NAMES

MWh=Annotated[float,Field(ge=0,allow_inf_nan=False)]
app=FastAPI(title='Renewable Energy AIOps — LSTM v2 Operations')
app.add_exception_handler(RequestValidationError,validation_error)


class Prediction(BaseModel):
    start_datetime:datetime
    sequence:list[MWh]=Field(min_length=168,max_length=168)


class Actuals(BaseModel):
    date:date
    actuals:list[MWh]=Field(min_length=24,max_length=24)


class Batch(BaseModel):
    scenario:Literal['normal','drift','output_curtailment']


@app.get('/health')
def health():return {'status':'ok','model_loaded':bool(assets.cache_info().currsize)}


@app.post('/predict')
def predict(req:Prediction):
    try:
        with LOCK:
            return forecast(req.sequence,req.start_datetime)
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc


@app.post('/actuals')
def actuals(req:Actuals):
    try:
        with LOCK:
            from src.aiops_trigger import check_and_trigger
            return check_and_trigger(req.date,req.actuals)
    except ValueError as exc:raise HTTPException(409,str(exc)) from exc


def req_timezone():
    from zoneinfo import ZoneInfo
    return ZoneInfo('Asia/Seoul')


@app.get('/aiops/status')
def aiops_status():
    with LOCK:return status()


@app.get('/model/info')
def model_info():
    version=current();run=client().get_run(version.run_id)
    return {'model_name':NAME,'production_version':str(version.version),'stage':version.current_stage,'model_version':str(version.version),'feature_configuration':FEATURE_NAMES,'input_hours':168,'output_hours':24,'metrics':run.data.metrics,'params':run.data.params,'cache_version_keys':'version-specific; cleared on promotion'}


@app.get('/logs')
def logs():return {'content':(OUT/'aiops.log').read_text() if (OUT/'aiops.log').exists() else ''}


@app.post('/batch-test')
def batch_test(req:Batch):
    if req.scenario=='output_curtailment':
        import os
        if os.getenv('AIOPS_SCENARIO2_DEMO')!='1':raise HTTPException(403,'Use the isolated scenario2 experiment entrypoint')
        from src.aiops_scenario2 import replay_demo
        return replay_demo()
    # Reference experiment: fixed original LSTM v2 predictions, same as threshold/injection evidence.
    result=json.loads((OUT/'drift_result.json').read_text())
    records=result['normal_records'] if req.scenario=='normal' else result['drift_records']
    log(f'[INFO] batch-test scenario={req.scenario}, reference=initial-baseline-v2, immutable precomputed forecasts')
    return {'scenario':req.scenario,'reference_model':'initial-baseline-v2','current_production_version':str(current().version),'factor':0 if req.scenario=='normal' else .15,'actual_only_injection':True,'evaluations':records,'drift_detected':any(r['event'] for r in records)}


@app.get('/dashboard/summary')
def dashboard_summary():
    from src.dashboard_view import summary
    with LOCK:return summary()


@app.get('/dashboard/example')
def dashboard_example():
    from src.dashboard_view import example
    return example()


@app.post('/dashboard/predict-preview')
def prediction_preview(req:Prediction):
    from src.dashboard_view import preview
    try:
        with LOCK:return preview(req.sequence,req.start_datetime)
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc


# API routes precede the static dashboard so /predict and /docs retain their contracts.
from fastapi.staticfiles import StaticFiles
from src.config import ROOT
app.mount('/',StaticFiles(directory=ROOT/'serving_app/static',html=True),name='dashboard')
