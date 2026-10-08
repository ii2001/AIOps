"""Executable checks with deterministic unit inputs and real trained-model HTTP requests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json,math,os,pickle,socket,subprocess,time
import numpy as np
import requests
from src.config import ROOT, MODEL_PATH, SCALER_PATH, INPUT_LEN, OUTPUT_LEN
from src.data_loader import load_target
from src.features import fit_scaler,transform,inverse,build_sequences,chronological_splits
from src.metrics import evaluate


def main():
    unit_scaler=fit_scaler([0,10]);assert np.allclose(inverse(transform([0,5,10],unit_scaler),unit_scaler),[0,5,10])
    # Known index values prove the builder uses the next 24 hours and no future feature rows.
    values=np.arange(200,dtype='float32');X,y=build_sequences(values)
    assert X.shape==(9,168,1) and y.shape==(9,24)
    assert X[0,-1,0]==167 and np.array_equal(y[0],np.arange(168,192))
    assert X[8,-1,0]==175 and y[8,-1]==199
    scores=evaluate([1,3],[2,1]);assert scores['mae']==1.5 and math.isclose(scores['rmse'],math.sqrt(2.5)) and scores['wape']==75
    assert evaluate([0,0],[0,0])['wape'] is None
    times,actual=load_target();splits=chronological_splits(actual)
    with SCALER_PATH.open('rb') as file:scaler=pickle.load(file)
    assert scaler==fit_scaler(actual[:splits[0][1]])
    for (a,b),(c,d) in zip(splits,splits[1:]):assert times[b-1]<times[c]
    result=json.loads((ROOT/'artifacts/training_result.json').read_text())
    for name,(a,b) in zip(['train','validation','test'],splits):
        assert result['partitions'][name]['X_shape'][0]==b-a-INPUT_LEN-OUTPUT_LEN+1
        assert result['partitions'][name]['end']==times[b-1].isoformat()
    assert MODEL_PATH.is_file() and MODEL_PATH.stat().st_size>0
    payload={'sequence':actual[splits[2][0]:splits[2][0]+INPUT_LEN].tolist()}
    (ROOT/'artifacts/predict_request.json').write_text(json.dumps(payload,indent=2))
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    base=f'http://127.0.0.1:{port}'
    with (ROOT/'artifacts/api_server.log').open('w') as log:
        proc=subprocess.Popen([sys.executable,'-m','uvicorn','serving_app.main:app','--host','127.0.0.1','--port',str(port)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
        try:
            for _ in range(600):
                if proc.poll() is not None:raise RuntimeError('API exited')
                try:
                    health=requests.get(base+'/health',timeout=.5)
                    if health.status_code==200:break
                except requests.ConnectionError:pass
                time.sleep(.05)
            else:raise TimeoutError('API startup')
            before=health.json();assert before=={'status':'ok','model_loaded':False}
            info=requests.get(base+'/model/info').json();assert info['input_hours']==168 and info['output_hours']==24
            response=requests.post(base+'/predict',json=payload,timeout=180);assert response.status_code==200,response.text
            prediction=response.json();p=prediction['predictions'];assert len(p)==24 and [x['hour'] for x in p]==list(range(1,25))
            predicted=np.array([x['mwh'] for x in p]);assert np.isfinite(predicted).all() and (predicted>=0).all()
            offline=np.load(ROOT/'artifacts/test_predicted.npy')[0]
            assert np.allclose(predicted,offline,atol=1e-3,rtol=1e-4),'API/offline MWh predictions differ'
            validation={}
            for length in [167,169]:
                sequence=payload['sequence'][:167] if length==167 else payload['sequence']+[payload['sequence'][-1]]
                r=requests.post(base+'/predict',json={'sequence':sequence});assert r.status_code==422;validation[str(length)]={'http_status':r.status_code,'body':r.json()}
            negative=payload['sequence'].copy();negative[0]=-1;r=requests.post(base+'/predict',json={'sequence':negative});assert r.status_code==422
            for value in [float('nan'),float('inf')]:
                nonfinite=json.dumps({'sequence':[value]+payload['sequence'][1:]})
                r=requests.post(base+'/predict',data=nonfinite,headers={'Content-Type':'application/json'})
                assert r.status_code==422
            after=requests.get(base+'/health').json();assert after['model_loaded'] is True
            (ROOT/'artifacts/api_verification.json').write_text(json.dumps({'status':'PASS','health_before':before,'health_after':after,'model_info':info,'predict_http_status':response.status_code,'predict_response':prediction,'length_validation':validation,'negative_input':'422','nonfinite_input':'422','model_loaded':'PASS','training_serving_mwh_parity':'PASS','finite_output':'PASS','split_leakage_check':'PASS','train_only_scaler_fit':'PASS'},ensure_ascii=False,indent=2))
            print('PASS: deterministic sequence/scaler/metric checks; real model load; /health, /model/info, 168->24; 167/169/negative/NaN/inf 422; offline MWh parity; no split leakage')
        finally:
            proc.terminate()
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()

if __name__=='__main__':main()
