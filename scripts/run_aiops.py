import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv,json,pickle
from datetime import date
import numpy as np
from src.config import ROOT,INPUT_LEN
from src.data_loader import load_target
from src.features import chronological_splits
from src.time_features import make_time_features
from src.metrics import evaluate
from src.aiops_policy import daily_policy
from src.aiops_registry import initialize,current,assets,OUT
from src.aiops_retraining import retrain
from src.aiops_runtime import log


def simulate(factor=.15):
    normal=json.loads((OUT/'normal_daily.json').read_text())['test'];policy=json.loads((OUT/'threshold_analysis.json').read_text())['recommended']
    start=normal[len(normal)//2]['date'];times,values=load_target();splits=chronological_splits(values)
    minimum=float(values[:splits[0][1]].max())*.01
    shifted=np.array([v*(1+factor) if t.date().isoformat()>=start and v>minimum else v for t,v in zip(times,values)])
    drift=[]
    for day in normal:
        t=date.fromisoformat(day['date']);actual=[shifted[i] for i,ts in enumerate(times) if ts.date()==t]
        drift.append({**day,'actual':actual})
    records=daily_policy(drift,policy['warning'],policy['drift']);base=daily_policy(normal,policy['warning'],policy['drift'])
    after=[r for r in records if r['date']>=start]
    detected=next((r for r in after if r['event']),None)
    result={'factor':factor,'start_date':start,'minimum_mwh':minimum,'minimum_rule':'1% of original Train peak, frozen before injection; significant output, not an astronomical daylight classifier',
            'normal_wape_before':evaluate([d['actual'] for d in normal if d['date']<start],[d['predicted'] for d in normal if d['date']<start])['wape'],
            'normal_wape_same_after_period':evaluate([d['actual'] for d in normal if d['date']>=start],[d['predicted'] for d in normal if d['date']>=start])['wape'],
            'drift_wape_after':evaluate([d['actual'] for d in drift if d['date']>=start],[d['predicted'] for d in drift if d['date']>=start])['wape'],
            'first_exceed_date':next((r['date'] for r in after if r['consecutive_drift_days']>=1),None),'two_consecutive_date':next((r['date'] for r in after if r['consecutive_drift_days']>=2),None),
            'detection_date':detected['date'] if detected else None,'detection_delay_days':(date.fromisoformat(detected['date'])-date.fromisoformat(start)).days if detected else None,'normal_records':base,'drift_records':records}
    if factor==.15:
        with (OUT/'drift_simulation.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['timestamp','actual_mwh','drift_actual_mwh','injected'])
            w.writerows((t.isoformat(),float(v),float(s),bool(v!=s)) for t,v,s in zip(times,values,shifted))
        (OUT/'drift_result.json').write_text(json.dumps(result,indent=2))
        (OUT/'drift_daily.json').write_text(json.dumps(drift))
    return result,times,shifted


def main():
    times,values=load_target()
    with (ROOT/'serving_app/models/scaler_v2.pkl').open('rb') as f:scaler=pickle.load(f)
    version=initialize(make_time_features(times[:168],values[:168],scaler)[None,:,:])
    log(f'[INFO] initial Production LSTM-v2 v{version}; naive remains benchmark only')
    result,times,shifted=simulate()
    print('Primary +15%:',json.dumps({k:v for k,v in result.items() if 'records' not in k}),flush=True)
    if result['detection_date']:
        asof=date.fromisoformat(result['detection_date']);reason='automatic-drift-15percent'
        log(f'[WARN] drift detected primary15percent date={asof}')
    else:
        log('[INFO] primary15percent NOT_DETECTED; no automatic retraining or threshold change')
        # Explicit separate diagnostic validates warm-start/Gate, never claimed as an automatic detection.
        asof=times[-1].date();reason='manual-diagnostic-primary-drift-not-detected'
        log('[INFO] manual diagnostic requested by experiment plan; automatic detector did not fire')
    candidate=retrain(times,shifted,asof,reason,log)
    with (OUT/'production_comparison.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['role','production_version','mae','rmse','wape','evaluation','reason'])
        for role,scores in [('Production',candidate['production']),('Candidate',candidate['candidate'])]:w.writerow([role,candidate['production_before'] if role=='Production' else candidate['production_after'],scores['mae'],scores['rmse'],scores['wape'],'same independent last14day Gate',reason])
    (OUT/'execution_result.json').write_text(json.dumps({'primary':{k:v for k,v in result.items() if 'records' not in k},'retraining':candidate,'initial_production':version,'final_production':str(current().version)},indent=2))

if __name__=='__main__':main()
