"""Validate actual automatic replay artifacts and immutable first experiment; never runs retraining again."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['AIOPS_RUN_DIR']=str(ROOT/'artifacts/aiops/scenario_2')
os.environ['AIOPS_STATE_PATH']=str(ROOT/'artifacts/aiops/scenario_2/runtime_state.json')
import json,hashlib
from datetime import datetime
import numpy as np
from src.aiops_registry import current,client
from src.aiops_policy import daily_policy
from src.aiops_retraining import gate_passes


def main():
    out=ROOT/'artifacts/aiops/scenario_2';parent=out.parent
    result=json.loads((out/'demo_result.json').read_text());candidate=result['candidate'];selected=result['selected'];api=json.loads((out/'api_verification.json').read_text())
    events=result['events'];first=next(e for e in events if e['drift_event_emitted'])
    assert first['date']==selected['detection_date'] and first['retraining'] is not None
    assert first['retraining']['reason']=='automatic-live-actuals-drift'
    before_events=[e for e in events if e['date'] in [selected['first_exceed_date'],selected['two_consecutive_date'],selected['detection_date']]]
    assert [e['aiops']['consecutive_drift_days'] for e in before_events]==[1,2,3]
    assert selected['change_percent']==-20 and not result['base_detected']
    assert selected['mean_wape_after']>selected['original_wape_after']
    assert candidate['warm_start_weights_verified'] and candidate['gate_independent_of_fit_and_early_stopping']
    assert candidate['split']['fine_tuning_days']==42 and candidate['split']['gate_days']==14
    assert datetime.fromisoformat(candidate['split']['fit_end'])<datetime.fromisoformat(candidate['split']['gate_start'])
    assert candidate['retrain_data_contains_drift']['hours']>0
    assert candidate['gate']=='PASS' if gate_passes(candidate['candidate']['wape'],candidate['production']['wape'],candidate['gate_threshold']) else candidate['gate']=='FAIL'
    version=str(current().version)
    if candidate['gate']=='FAIL':assert version==result['version_before']==result['version_after']
    else:assert version==result['version_after']!=result['version_before']
    run=client().get_run(candidate['run_id']);assert run.info.status=='FINISHED' and run.data.tags['reason']=='automatic-live-actuals-drift'
    assert np.isclose(run.data.metrics['wape'],candidate['candidate']['wape'])
    assert api['status']=='PASS' and api['registry_version']==api['after_model_version'] and not api['cache_mismatch']
    logs=(out/'aiops.log').read_text();start=logs.index('[WARN] drift detected');chain=['[WARN] drift detected','[INFO] retrain triggered','[INFO] candidate evaluated']
    chain+=['[OK] gate passed','[OK] production promoted'] if candidate['gate']=='PASS' else ['[FAIL] candidate rejected','[INFO] existing production retained']
    indices=[logs.index(text,start) for text in chain];assert indices==sorted(indices)
    scenario1=json.loads((parent/'scenario_1_capacity_increase.json').read_text());assert scenario1['result']=='FAIL_NOT_DETECTED' and scenario1['wape_after']<scenario1['wape_before'] and not scenario1['auto_retrain']
    old=json.loads((out/'original_hashes.json').read_text())
    allowed={'src/aiops_registry.py','src/aiops_retraining.py','serving_app/aiops_main.py','docs/aiops/troubleshooting_aiops.md','docs/aiops/PRESENTATION_AIOPS_TROUBLESHOOTING.md','artifacts/aiops/validation.json','artifacts/aiops/aiops.log'}
    changed=[p for p,h in old.items() if not (ROOT/p).exists() or hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    assert set(changed)<=allowed,changed
    checks={k:'PASS' for k in ['scenario_1_preserved','scenario_1_fail_not_detected','scenario_1_wape_decreased','scenario_2_created','scenario_2_wape_increased','scenario_2_threshold_exceeded','scenario_2_three_day_detection','auto_retrain_triggered','warm_start_verified','retrain_data_contains_drift','gate_independent_validation','gate_checked','production_promoted_or_retained','mlflow_verified','api_verified','aiops_log_order_verified','cache_verified']}
    checks.update({'actual_gate':candidate['gate'],'actual_promotion':'AUTO' if candidate['gate']=='PASS' else 'RETAINED','automatic_promotion_branch_executed':candidate['gate']=='PASS','base_minus15_detection':'NOT_DETECTED','minimum_tested_detectable_change':selected['change_percent'],'registry_scope':str(out/'mlflow.db'),'changed_previous_files':changed})
    existing=json.loads((out/'validation.json').read_text()) if (out/'validation.json').exists() else {}
    existing.update(checks)
    (out/'validation.json').write_text(json.dumps(existing,indent=2));print(json.dumps(existing,indent=2))

if __name__=='__main__':main()
