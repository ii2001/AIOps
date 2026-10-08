"""Shared actual-arrival -> detector -> warm-start/Gate; no manual detection override."""
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
from src.aiops_runtime import LOCK,register_actuals,read_state,save_state,status,log
from src.aiops_retraining import retrain


def check_and_trigger(day,actual):
    with LOCK:
        _,event=register_actuals(day,actual)
        state=read_state();observed=state.setdefault('observed',{})
        start=datetime.combine(day,datetime.min.time(),tzinfo=ZoneInfo('Asia/Seoul'))
        for i,value in enumerate(actual):observed[(start+timedelta(hours=i)).isoformat()]=value
        if event:state['pending_retrain']=True
        save_state(state);result=None
        if state.get('pending_retrain'):
            end=start.replace(hour=23)
            times=[end-timedelta(hours=56*24-1-i) for i in range(56*24)]
            if all(t.isoformat() in observed for t in times):
                # All42 fit days are used; fixed10 epochs avoids holding all new drift days solely in ES.
                result=retrain(times,[observed[t.isoformat()] for t in times],day,'automatic-live-actuals-drift',log,validation_days=0)
                state['pending_retrain']=False;state['last_retraining']=result;save_state(state)
            else:log('[INFO] retrain deferred: insufficient56days of observed history; no future CSV substitution')
        return {'aiops':status(),'retraining':result,'drift_event_emitted':event}
