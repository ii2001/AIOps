"""Real container verification; inference preview does not write operational forecasts."""
import json,math,time
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
URL='http://127.0.0.1:8099'


def main():
    for _ in range(180):
        try:
            h=requests.get(URL+'/health',timeout=2)
            if h.status_code==200:break
        except requests.ConnectionError:pass
        time.sleep(.5)
    else:raise TimeoutError('Docker service readiness failed')
    summary=requests.get(URL+'/dashboard/summary',timeout=30);assert summary.status_code==200,summary.text
    example=requests.get(URL+'/dashboard/example',timeout=30);assert example.status_code==200,example.text
    payload=example.json();payload={k:payload[k] for k in ['start_datetime','sequence']}
    before=requests.get(URL+'/aiops/status').json()
    r=requests.post(URL+'/dashboard/predict-preview',json=payload,timeout=180);assert r.status_code==200,r.text
    result=r.json();assert len(result['predictions'])==24 and all(math.isfinite(p['mwh']) and p['mwh']>=0 for p in result['predictions'])
    assert result['model_version']==summary.json()['model']['version']
    invalid=requests.post(URL+'/dashboard/predict-preview',json={**payload,'sequence':payload['sequence'][:-1]});assert invalid.status_code==422
    assert requests.get(URL+'/aiops/status').json()==before
    assert requests.get(URL+'/').status_code==200 and requests.get(URL+'/docs').status_code==200
    report={'status':'PASS','health':h.json(),'model_info':requests.get(URL+'/model/info').json(),'preview_http_status':r.status_code,'predictions_count':24,'finite_nonnegative_MWh':True,'length167_status':invalid.status_code,'forecast_state_unchanged':True,'dashboard_HTTP200':True,'Swagger_HTTP200':True,'forecast_response':result}
    (ROOT/'artifacts/docker/validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('PASS: actual Docker Registry/model/scaler load,24h inference,167->422,UI,Swagger; state preserved')

if __name__=='__main__':main()
