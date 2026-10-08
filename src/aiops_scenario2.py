"""Frozen real Production forecasts + actual-only curtailment, followed by shared automatic trigger."""
import csv,json
from datetime import date
import numpy as np
from src.config import ROOT
from src.data_loader import load_target
from src.aiops_registry import OUT,current
from src.aiops_runtime import LOCK,read_state,save_state,status,log
from src.aiops_policy import daily_policy
from src.metrics import evaluate
from src.aiops_trigger import check_and_trigger

PARENT=ROOT/'artifacts/aiops'


def analyze_scenarios():
    first=json.loads((PARENT/'drift_result.json').read_text())
    days=json.loads((PARENT/'normal_daily.json').read_text())['test']
    policy=json.loads((PARENT/'threshold_analysis.json').read_text())['recommended']
    original_after=[d for d in days if d['date']>=first['start_date']]
    sensitivity=[]
    for percent in [15,20,25,30]:
        changed=[{**d,'actual':[v*(1-percent/100) if d['date']>=first['start_date'] and v>first['minimum_mwh'] else v for v in d['actual']]} for d in days]
        records=daily_policy(changed,policy['warning'],policy['drift']);after=[r for r in records if r['date']>=first['start_date']]
        detected=next((r for r in after if r['event']),None)
        result={'scenario':'output_curtailment','change_percent':-percent,'start_date':first['start_date'],'minimum_mwh':first['minimum_mwh'],
            'original_wape_after':evaluate([d['actual'] for d in original_after],[d['predicted'] for d in original_after])['wape'],
            'mean_wape_after':evaluate([d['actual'] for d in changed if d['date']>=first['start_date']],[d['predicted'] for d in original_after])['wape'],
            'mean_rolling_wape_after':float(np.mean([r['rolling_wape'] for r in after if r['rolling_wape'] is not None])),
            'max_rolling_wape':max(r['rolling_wape'] or 0 for r in after),'first_warning_date':next((r['date'] for r in after if r['rolling_wape'] is not None and r['rolling_wape']>=policy['warning']),None),
            'first_exceed_date':next((r['date'] for r in after if r['consecutive_drift_days']>=1),None),'two_consecutive_date':next((r['date'] for r in after if r['consecutive_drift_days']>=2),None),
            'detected':detected is not None,'detection_date':detected['date'] if detected else None,'detection_delay':(date.fromisoformat(detected['date'])-date.fromisoformat(first['start_date'])).days if detected else None,
            'records':records,'days':changed,'prediction_reference':'original MLflow Production LSTM-v2/version1, unchanged actual-only injection'}
        (OUT/f'output_curtailment_{percent}.json').write_text(json.dumps(result,indent=2));sensitivity.append(result)
    (PARENT/'scenario_2_output_curtailment_15.json').write_text(json.dumps({k:v for k,v in sensitivity[0].items() if k not in ['records','days']},indent=2))
    fields=['scenario','change_percent','mean_wape_after','mean_rolling_wape_after','max_rolling_wape','first_exceed_date','detected','detection_date','detection_delay']
    with (PARENT/'drift_sensitivity.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in sensitivity)
    chosen=next((r for r in sensitivity if r['detected']),None)
    (OUT/'selection.json').write_text(json.dumps({k:v for k,v in chosen.items() if k not in ['records','days']} if chosen else {'detected':False,'change_percent':None},indent=2))
    return chosen


def replay_demo():
    with LOCK:
        if (OUT/'demo_result.json').exists():return json.loads((OUT/'demo_result.json').read_text())
        chosen=analyze_scenarios()
        if chosen is None:
            log('[INFO] Scenario2 all strengths NOT_DETECTED; no automatic retraining')
            return {'detected':False,'auto_retrain':False}
        version=str(current().version)
        if version!='1':raise ValueError('Use a fresh isolated registry seeded with original Production for reproducible replay')
        times,values=load_target();first=date.fromisoformat(chosen['days'][0]['date'])
        state=read_state()
        if state['days']:raise ValueError('Fresh isolated actual state required')
        # Pre-existing historical observations strictly before the first evaluated day.
        state['observed']={t.isoformat():float(v) for t,v in zip(times,values) if t.date()<first}
        save_state(state)
        log('[INFO] normal operation; frozen real reference forecasts; isolated Registry v1; old manual Registry v2 preserved')
        log('[INFO] scenario_1 capacity increase +15%; prior FAIL_NOT_DETECTED preserved')
        log(f'[INFO] scenario_2 output curtailment started at {chosen["start_date"]}; selected {chosen["change_percent"]}% after all sensitivity results stored')
        events=[];candidate=None
        for day in chosen['days']:
            current_state=read_state()
            if day['date'] not in current_state['forecasts']:
                current_state['forecasts'][day['date']]={'date':day['date'],'predicted':day['predicted'],'model_version':version,'source':'real immutable original LSTM-v2 forecast'}
                save_state(current_state);log(f'[INFO] prediction generated from frozen genuine model output date={day["date"]}, version={version}')
            response=check_and_trigger(date.fromisoformat(day['date']),day['actual'])
            events.append({'date':day['date'],**response})
            if response['retraining'] is not None:candidate=response['retraining'];break
        if candidate:
            fit_end=candidate['split']['fit_end'];start=candidate['split']['known_start']
            minvalue=chosen['minimum_mwh'];changed=[t for t,v in zip(times,values) if start<=t.isoformat()<=fit_end and t.date().isoformat()>=chosen['start_date'] and v>minvalue]
            assert changed,'No drift observations in optimizer fit dataset'
            candidate['retrain_data_contains_drift']={'hours':len(changed),'first':changed[0].isoformat(),'last':changed[-1].isoformat(),'change_percent':chosen['change_percent']}
            (PARENT/'scenario_2_candidate_metrics.json').write_text(json.dumps(candidate,indent=2))
        result={'scenario':'output_curtailment','base_change':-15,'base_detected':json.loads((OUT/'output_curtailment_15.json').read_text())['detected'],'selected':{k:v for k,v in chosen.items() if k not in ['records','days']},
            'detected':any(e['drift_event_emitted'] for e in events),'auto_retrain':candidate is not None,'candidate':candidate,'version_before':version,'version_after':str(current().version),'events':events,
            'registry_scope':'artifacts/aiops/scenario_2/mlflow.db; original Registry/manual v2 not reset','trigger_path':'check_and_trigger -> daily_policy event -> retrain, no manual retrain call'}
        (OUT/'demo_result.json').write_text(json.dumps(result,indent=2));return result
