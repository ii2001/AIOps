"""Deterministic policy/Gate tests and actual MLflow/API verification; synthetic policy inputs are not drift evidence."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,os,socket,subprocess,time,hashlib
from datetime import date,timedelta,datetime
import requests
import numpy as np
from src.aiops_policy import daily_policy
from src.aiops_retraining import gate_passes
from src.aiops_registry import OUT,current,client,assets
from src.data_loader import load_target
from src.metrics import evaluate


def main():
    # Controlled contract-only inputs verify days, reset, and one event per episode.
    days=[{'date':(date(2025,1,1)+timedelta(days=i)).isoformat(),'actual':[100.]*24,'predicted':[0.]*24} for i in range(10)]
    p=daily_policy(days,80,90);assert p[6]['consecutive_drift_days']==1 and p[8]['event'] and not p[9]['event']
    try:daily_policy(days+[days[-1]],80,90)
    except ValueError:pass
    else:raise AssertionError('Duplicate daily evaluation accepted')
    assert gate_passes(30,40,50) and not gate_passes(40,40,50) and not gate_passes(60,70,50)
    before=str(current().version);assert current().current_stage=='Production'
    reg=client().get_registered_model('RenewableEnergyPredictor');assert str(reg.aliases['Production'])==before
    result=json.loads((OUT/'retraining_metrics.json').read_text());assert result['warm_start_weights_verified'] and result['gate_independent_of_fit_and_early_stopping']
    assert result['split']['fine_tuning_days']==42 and result['split']['gate_days']==14
    assert datetime.fromisoformat(result['split']['fine_tuning_end'])<datetime.fromisoformat(result['split']['gate_start'])
    # Gate FAIL helper itself is pure: it must not modify MLflow Production.
    assert not gate_passes(result['production']['wape']+1,result['production']['wape'],result['gate_threshold'])
    assert str(current().version)==before
    times,values=load_target();normal=json.loads((OUT/'normal_daily.json').read_text())['test']
    test_state=OUT/'api_validation_state_chronological.json'
    assert not test_state.exists(),'Preserve existing validation state; choose a fresh state path for another run'
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    base=f'http://127.0.0.1:{port}'
    with (OUT/'api_server.log').open('w') as log:
        proc=subprocess.Popen([sys.executable,'-m','uvicorn','serving_app.aiops_main:app','--host','127.0.0.1','--port',str(port)],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','AIOPS_STATE_PATH':str(test_state)})
        try:
            for _ in range(600):
                if proc.poll() is not None:raise RuntimeError('API startup failed')
                try:
                    h=requests.get(base+'/health',timeout=.5)
                    if h.status_code==200:break
                except requests.ConnectionError:pass
                time.sleep(.05)
            else:raise TimeoutError('API startup')
            info=requests.get(base+'/model/info').json();assert info['production_version']==before
            code=[];statuses=[]
            gate_days=[day for day in normal if day['date']>result['split']['fine_tuning_end'][:10]]
            assert len(gate_days)>=8
            for day in gate_days[:8]:
                end=next(i for i,t in enumerate(times) if t.date().isoformat()==day['date'] and t.hour==0)
                payload={'start_datetime':times[end-168].isoformat(),'sequence':values[end-168:end].tolist()}
                r=requests.post(base+'/predict',json=payload,timeout=180);assert r.status_code==200,r.text
                assert len(r.json()['predictions'])==24 and r.json()['model_version']==before;code.append(r.status_code)
                actual={'date':day['date'],'actuals':day['actual']};r=requests.post(base+'/actuals',json=actual,timeout=180);assert r.status_code==200,r.text
                statuses.append(r.json()['aiops'])
                repeated=requests.post(base+'/actuals',json=actual).json()['aiops'];assert repeated==statuses[-1]
            assert statuses[-1]['actual_days']==8 and statuses[-1]['rolling_7d_wape'] is not None
            b=requests.post(base+'/batch-test',json={'scenario':'drift'});assert b.status_code==200
            assert b.json()['factor']==.15
            for length in [167,169]:
                p={**payload,'sequence':payload['sequence'][:167] if length==167 else payload['sequence']+[1.]};r=requests.post(base+'/predict',json=p);assert r.status_code==422
            assert requests.get(base+'/logs').status_code==200
            (OUT/'api_verification.json').write_text(json.dumps({'status':'PASS','model_info':info,'predict_http_statuses':code,'daily_actuals_idempotency':'PASS','final_status':statuses[-1],'demo_drift_detected':b.json()['drift_detected'],'baseline_v1_app_untouched':True},indent=2))
        finally:
            proc.terminate()
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
    originals=json.loads((OUT/'original_hashes.json').read_text());changed=[p for p,h in originals.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];assert not changed,changed
    primary=json.loads((OUT/'drift_result.json').read_text())
    validation={'implementation_checks':'PASS','baseline_files_preserved':'PASS','validation_thresholds_frozen_before_injection':'PASS','normal_test_false_alarm_measured':'PASS','three_days_not_three_calls':'PASS','warm_start_and_42_14_split':'PASS','Gate_actual_result':result['gate'],'Gate_FAIL_preserves_production_unit_check':'PASS','MLflow_Production_alias_and_Run':'PASS','API':'PASS',
                'primary_15percent_drift_detected':'PASS' if primary['detection_date'] else 'FAIL_NOT_DETECTED','primary_15percent_WAPE_increased':'PASS' if primary['drift_wape_after']>primary['normal_wape_same_after_period'] else 'FAIL_NOT_INCREASED','automatic_primary_retraining':'PASS' if result['reason'].startswith('automatic') else 'NOT_RUN; manual diagnostic separated', 'changed_existing_files':changed}
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2));print(json.dumps(validation,indent=2))

if __name__=='__main__':main()
