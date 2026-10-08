"""Fresh isolated automatic E2E experiment; original failed experiment and live Registry stay untouched."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'artifacts/aiops/scenario_2'
os.environ['AIOPS_RUN_DIR']=str(OUT)
os.environ['AIOPS_STATE_PATH']=str(OUT/'runtime_state.json')
os.environ['AIOPS_SCENARIO2_DEMO']='1'
import json,shutil,socket,subprocess,time,pickle
from datetime import datetime,timedelta
import numpy as np
import requests,mlflow,mlflow.tensorflow
from tensorflow import keras
from src.aiops_registry import current,client,log_model,promote,assets,NAME
from src.data_loader import load_target
from src.time_features import make_time_features
from src.features import restore_predictions


def main():
    parent=ROOT/'artifacts/aiops'
    first_failure=json.loads((parent/'drift_result.json').read_text())
    print('Scenario1 preserved FAIL_NOT_DETECTED:',json.dumps({k:first_failure[k] for k in ['start_date','normal_wape_same_after_period','drift_wape_after','detection_date']}),flush=True)
    print('Actual prediction bias:',(parent/'scenario_1_bias_analysis.json').read_text(),flush=True)
    if (OUT/'demo_result.json').exists():
        print((OUT/'demo_result.json').read_text());return
    shutil.copy2(ROOT/'artifacts/aiops/threshold_analysis.json',OUT/'threshold_analysis.json')
    # Load the original real Production LSTM-v2 from existing MLflow, not a reconstructed/fake prediction model.
    mlflow.set_tracking_uri(f'sqlite:///{ROOT/"artifacts/aiops/mlflow.db"}')
    original=mlflow.tensorflow.load_model(f'models:/{NAME}/1')
    local=keras.models.load_model(ROOT/'serving_app/models/baseline_v2.keras')
    assert all(np.array_equal(a,b) for a,b in zip(original.get_weights(),local.get_weights()))
    times,values=load_target()
    with (ROOT/'serving_app/models/scaler_v2.pkl').open('rb') as f:scaler=pickle.load(f)
    client()
    run=log_model(original,{'epochs':18,'learning_rate':.001,'retraining_window':'original v2 Train/Validation, before Test','threshold':json.loads((OUT/'threshold_analysis.json').read_text())['recommended']['drift'],'source_original_production_version':'1'},json.loads((ROOT/'artifacts/model_comparison/v2_metrics.json').read_text())['metrics']['test'],'initial-production',make_time_features(times[:168],values[:168],scaler)[None,:,:])
    promote(run)
    first=json.loads((ROOT/'artifacts/aiops/normal_daily.json').read_text())['test'][0]
    index=next(i for i,t in enumerate(times) if t.date().isoformat()==first['date'] and t.hour==0)
    payload={'start_datetime':times[index-168].isoformat(),'sequence':values[index-168:index].tolist()}
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    base=f'http://127.0.0.1:{port}'
    with (OUT/'api_server.log').open('w') as log:
        proc=subprocess.Popen([sys.executable,'-m','uvicorn','serving_app.aiops_main:app','--host','127.0.0.1','--port',str(port)],cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdout=log,stderr=subprocess.STDOUT)
        try:
            for _ in range(600):
                if proc.poll() is not None:raise RuntimeError('AIOps server exited')
                try:
                    if requests.get(base+'/health',timeout=.5).status_code==200:break
                except requests.ConnectionError:pass
                time.sleep(.05)
            else:raise TimeoutError('AIOps server readiness')
            before=requests.post(base+'/predict',json=payload,timeout=180);assert before.status_code==200,before.text
            np.testing.assert_allclose([p['mwh'] for p in before.json()['predictions']],first['predicted'],rtol=1e-4,atol=.001)
            r=requests.post(base+'/batch-test',json={'scenario':'output_curtailment'},timeout=600);assert r.status_code==200,r.text
            result=r.json();candidate=result.get('candidate')
            if candidate:
                end=datetime.fromisoformat(candidate['split']['gate_end']);state=json.loads((OUT/'runtime_state.json').read_text())
                observed=state['observed'];start=end-timedelta(hours=167);sequence=[observed[(start+timedelta(hours=i)).isoformat()] for i in range(168)]
                after=requests.post(base+'/predict',json={'start_datetime':start.isoformat(),'sequence':sequence},timeout=180);assert after.status_code==200,after.text
                info=requests.get(base+'/model/info').json();assert after.json()['model_version']==info['production_version']==result['version_after']
                model,scaler2=assets(result['version_after']);expected=restore_predictions(model.predict(make_time_features([start+timedelta(hours=i) for i in range(168)],sequence,scaler2)[None,:,:],verbose=0),scaler2)[0]
                np.testing.assert_allclose([p['mwh'] for p in after.json()['predictions']],expected,rtol=1e-4,atol=.001)
                checks={'status':'PASS','before_model_version':before.json()['model_version'],'after_model_version':after.json()['model_version'],'registry_version':info['production_version'],'served_predictions_match_explicit_registry_model':True,'cache_mismatch':False,'health':requests.get(base+'/health').json(),'model_info':info}
                (OUT/'api_verification.json').write_text(json.dumps(checks,indent=2))
                print(json.dumps({k:v for k,v in result.items() if k!='events'},indent=2),flush=True)
        finally:
            proc.terminate()
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()

if __name__=='__main__':main()
