"""Single-process durable local state for daily forecast/actual matching."""
import json,os,threading
from datetime import datetime,timedelta,date
from pathlib import Path
import numpy as np
from src.config import ROOT,TARGET_REGION,TARGET_FUEL
from src.time_features import make_time_features
from src.features import restore_predictions
from src.aiops_registry import current,assets,OUT,NAME
from src.aiops_policy import daily_policy

# ponytail: one process-local lock; run one worker. Use transactional storage before multi-worker deployment.
LOCK=threading.RLock()
STATE=Path(os.getenv('AIOPS_STATE_PATH', str(OUT/'runtime_state.json')))


def log(message):
    with (OUT/'aiops.log').open('a') as f:f.write(datetime.now().astimezone().isoformat()+' '+message+'\n')


def read_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {'forecasts':{},'days':[]}


def save_state(state):
    temporary=STATE.with_suffix('.tmp');temporary.write_text(json.dumps(state));os.replace(temporary,STATE)


def status(state=None):
    state=read_state() if state is None else state
    p=json.loads((OUT/'threshold_analysis.json').read_text())['recommended']
    records=daily_policy(state['days'],p['warning'],p['drift'])
    last=records[-1] if records else {'status':'insufficient_data','rolling_wape':None,'consecutive_drift_days':0}
    return {'status':last['status'],'rolling_7d_wape':last['rolling_wape'],'warning_threshold':p['warning'],'drift_threshold':p['drift'],'consecutive_drift_days':last['consecutive_drift_days'],'production_version':str(current().version),'evaluation_date':last.get('date'),'actual_days':len(state['days'])}


def forecast(sequence,start):
    with LOCK:
        times=[start+timedelta(hours=i) for i in range(168)];target=start+timedelta(hours=168)
        if target.hour!=0:raise ValueError('Daily AIOps forecast must start at midnight KST')
        state=read_state();key=target.date().isoformat()
        if key in state['forecasts']:raise ValueError('Forecast date already stored; immutable prediction prevents actual leakage')
        observed=state.setdefault('observed',{})
        for timestamp,value in zip(times,sequence):
            key_time=timestamp.isoformat()
            if key_time in observed and observed[key_time]!=value:raise ValueError('Input disagrees with previously registered observation')
            observed[key_time]=value
        version=str(current().version);model,scaler=assets(version)
        values=restore_predictions(model.predict(make_time_features(times,sequence,scaler)[None,:,:],verbose=0),scaler)[0].tolist()
        state['forecasts'][key]={'date':key,'timestamps':[(target+timedelta(hours=i)).isoformat() for i in range(24)],'predicted':values,'model_version':version}
        save_state(state);log(f'[INFO] prediction generated date={key}, production=v{version}')
        return {'predictions':[{'hour':i+1,'mwh':v} for i,v in enumerate(values)],'forecast_date':key,'model_version':version,'region':TARGET_REGION,'fuel':TARGET_FUEL}


def register_actuals(day,actual):
    with LOCK:
        state=read_state();key=day.isoformat();record=state['forecasts'].get(key)
        if record is None:raise ValueError('No stored forecast for this actual date')
        duplicate=next((r for r in state['days'] if r['date']==key),None)
        if duplicate:
            if duplicate['actual']!=actual:raise ValueError('Conflicting actuals; prior evaluation cannot be rewritten')
            return status(state),False
        if state['days'] and key<=state['days'][-1]['date']:raise ValueError('Actuals must arrive in chronological order')
        state['days'].append({**record,'actual':actual})
        p=json.loads((OUT/'threshold_analysis.json').read_text())['recommended'];r=daily_policy(state['days'],p['warning'],p['drift'])[-1]
        save_state(state);log(f'[INFO] daily WAPE calculated date={key}, rolling7d={r["rolling_wape"]}, streak={r["consecutive_drift_days"]}')
        if r['rolling_wape'] is not None and r['rolling_wape']>=p['warning']:log('[WARN] warning threshold exceeded')
        if r['consecutive_drift_days']:log('[WARN] drift threshold exceeded')
        if r['event']:log(f'[WARN] drift detected date={key}')
        return status(state),r['event']
