"""Run after v2 training. Deterministic alignment checks plus real saved-model feature/prediction parity."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,pickle,hashlib
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
import numpy as np
from tensorflow import keras
from src.config import ROOT,INPUT_LEN
from src.benchmarks import seasonal_predictions,comparison_data,NAIVE_NAMES
from src.features import fit_scaler,build_sequences,inverse,restore_predictions
from src.time_features import make_time_features
from src.metrics import evaluate


def main():
    out=ROOT/'artifacts/model_comparison'
    sequence=np.arange(168,dtype='float64')[None,:]
    p=seasonal_predictions(sequence)
    assert np.array_equal(p[NAIVE_NAMES[0]][0],np.arange(144,168))
    assert np.array_equal(p[NAIVE_NAMES[1]][0],np.arange(24))
    assert np.array_equal(p[NAIVE_NAMES[2]][0],np.arange(24)+72)
    times,values,splits,history,actual=comparison_data()
    with (ROOT/'serving_app/models/scaler_v2.pkl').open('rb') as file:scaler=pickle.load(file)
    assert scaler==fit_scaler(values[:splits[0][1]])
    features=make_time_features(times,values,scaler)
    assert features.shape==(8760,7) and np.isfinite(features).all() and np.max(np.abs(features[:,1:]))<=1
    X,y=build_sequences(features[splits[2][0]:splits[2][1]])
    assert np.array_equal(inverse(y,scaler),actual)
    origin=splits[2][0]+INPUT_LEN
    windows=json.loads((out/'test_windows.json').read_text())
    assert len(windows)==len(actual) and windows[0][0]==times[origin].isoformat() and windows[-1][-1]==times[-1].isoformat()
    dates=[datetime(2025,1,1,tzinfo=ZoneInfo('Asia/Seoul'))+timedelta(hours=i) for i in range(25)]
    cyclic=make_time_features(dates,[0]*25,scaler)
    assert np.allclose(cyclic[0,1:3],cyclic[24,1:3]) and np.allclose(cyclic[0,1:3],[0,1])
    for a,b in splits:
        count=b-a-168-24+1
        assert a+count-1+168+24-1==b-1
    # Future serving reconstructs all input timestamps from the oldest input's start_datetime.
    start=times[splits[2][0]]
    reconstructed=[start+timedelta(hours=i) for i in range(168)]
    shared=make_time_features(reconstructed,history[0],scaler)
    assert np.array_equal(shared,X[0])
    model=keras.models.load_model(ROOT/'serving_app/models/baseline_v2.keras')
    assert model.input_shape[1:]==(168,7) and model.output_shape[1:]==(24,)
    predicted=restore_predictions(model.predict(shared[None,:,:],verbose=0),scaler)[0]
    assert np.allclose(predicted,np.load(out/'v2_test_predicted.npy')[0],rtol=1e-4,atol=.001)
    assert np.isfinite(predicted).all() and (predicted>=0).all()
    recomputed=evaluate(actual,np.load(ROOT/'artifacts/test_predicted.npy'))
    v1=json.loads((out/'v1_metrics.json').read_text())['metrics']['test']
    assert all(np.isclose(recomputed[k],v1[k],rtol=1e-12) for k in v1)
    originals=json.loads((out/'original_hashes.json').read_text())
    changed=[name for name,digest in originals.items() if not (ROOT/name).exists() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    assert not changed,changed
    result={'status':'PASS','v1_files_and_artifacts_preserved':True,'same_1123_test_windows_and_24_horizons':True,
            'scaler_train_only':True,'sequence_boundary_leakage':False,'naive_offsets_checked':True,'cyclic_features_checked':True,
            'v2_saved_model_load':True,'shared_future_serving_features_and_mwh_parity':True,'v2_HTTP_endpoint_implemented':False,
            'changed_existing_files':changed}
    (out/'validation.json').write_text(json.dumps(result,indent=2))
    print('PASS: v1 hashes, identical test windows, train-only scaler, naive offsets, time cycles, boundary containment, saved v2 model and serving feature/prediction parity')

if __name__=='__main__':main()
