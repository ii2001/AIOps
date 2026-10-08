"""Check source/target preservation, input-only normalization and saved inference."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,pickle,hashlib
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
import numpy as np
from src.config import ROOT
from src.data_loader import load_target
from src.features import inverse
from src.residual_features import windows,forecast_inputs
from src.normalized_features import normalized_inputs,predict_normalized,solar_proxy
from src.metrics import evaluate


def main():
    out=ROOT/'artifacts/data_features_v5'
    result=json.loads((out/'metrics.json').read_text())
    selection=json.loads((out/'selection.json').read_text())
    assert selection['selected']==min(selection['metrics'],key=lambda n:selection['metrics'][n]['wape'])
    with (ROOT/'serving_app/models/data_features_v5.pkl').open('rb') as f:saved=pickle.load(f)
    times,values=load_target();start=int(len(values)*.7)+int(len(values)*.15)+168
    X,y,origins=windows(times,values,saved['scaler'],start,len(values))
    assert np.array_equal(inverse(y,saved['scaler']),np.load(ROOT/'artifacts/test_actual.npy'))
    changed_values=values.copy();changed_values[start:]=0
    changed,_,changed_origins=windows(times,changed_values,saved['scaler'],start,start+24)
    for solar in [False,True]:
        features,scale=normalized_inputs([a[:1] for a in X],saved['scaler'],times,[start],solar)
        altered,other_scale=normalized_inputs(changed,saved['scaler'],times,changed_origins,solar)
        assert np.array_equal(features,altered) and np.array_equal(scale,other_scale)
        assert np.isclose(scale[0],max(np.quantile(values[start-168:start],.95),1),rtol=1e-5)
    if saved['name']=='mean7':prediction=values[start-168:start].reshape(7,24).mean(axis=0)
    else:prediction=predict_normalized(saved['model'],saved['name']=='normalized_solar',times[start-168:start],values[start-168:start],saved['scaler'])
    cached=np.load(out/'test_predicted.npy')
    assert np.allclose(prediction,cached[0],rtol=1e-5,atol=.001)
    assert np.isfinite(cached).all() and (cached>=0).all()
    scores=evaluate(inverse(y,saved['scaler']),cached)
    assert all(np.isclose(scores[k],result['metrics']['test']['v5'][k]) for k in scores)
    zero=forecast_inputs(times[:168],np.zeros(168),saved['scaler'])
    features,scale=normalized_inputs([a[None] for a in zero],saved['scaler'],times,[168],True)
    assert scale[0]==1 and np.isfinite(features).all()
    summer=[datetime(2025,6,21,tzinfo=ZoneInfo('Asia/Seoul'))+timedelta(hours=h) for h in range(24)]
    winter=[datetime(2025,12,21,tzinfo=ZoneInfo('Asia/Seoul'))+timedelta(hours=h) for h in range(24)]
    sun=solar_proxy(summer);win=solar_proxy(winter)
    assert sun.shape==(24,4) and sun[0,1]==0 and sun[12,1]>.7 and sun[12,2]>win[12,2]
    gate=result['metrics']['gate'];passed=gate['v5']['wape']<=min(40,.95*min(gate['v2']['wape'],gate['mean7']['wape']))
    assert result['gate']==('PASS' if passed else 'FAIL')
    hashes=json.loads((out/'original_hashes.json').read_text())
    changed_files=[n for n,d in hashes.items() if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=d]
    assert not changed_files,changed_files
    validation={'status':'PASS','raw_csv_unchanged':True,'same_test_actuals':True,'past_only_features':True,
                'zero_history_finite':True,'solar_season_sanity':True,'saved_inference_parity':True,
                'existing_files_preserved':True,'production_switched':False,'gate':result['gate']}
    (out/'validation.json').write_text(json.dumps(validation,indent=2));print(json.dumps(validation))


if __name__=='__main__':main()
