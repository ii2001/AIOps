import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,pickle,hashlib
import numpy as np
from tensorflow import keras
from src.config import ROOT,SEED,INPUT_LEN,TARGET_REGION,TARGET_FUEL
from src.features import fit_scaler,build_sequences,inverse,restore_predictions
from src.time_features import make_time_features,FEATURE_NAMES
from src.model_v2 import build_model_v2
from src.metrics import evaluate
from src.benchmarks import comparison_data,benchmark_metrics


def main():
    out=ROOT/'artifacts/model_comparison';out.mkdir(exist_ok=True)
    keras.utils.set_random_seed(SEED)
    times,values,splits,history,actual=comparison_data()
    naive=benchmark_metrics(history,actual)
    (out/'naive_metrics.json').write_text(json.dumps(naive,indent=2))
    print('Seasonal benchmarks:',json.dumps(naive),flush=True)
    scaler=fit_scaler(values[:splits[0][1]])
    with (ROOT/'serving_app/models/scaler.pkl').open('rb') as f:assert scaler==pickle.load(f)
    features=make_time_features(times,values,scaler)
    datasets=[build_sequences(features[a:b]) for a,b in splits]
    assert np.array_equal(inverse(datasets[2][1],scaler),actual)
    model=build_model_v2();model.summary()
    training=model.fit(*datasets[0],validation_data=datasets[1],epochs=60,batch_size=64,shuffle=False,
                       callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)],verbose=2)
    predictions=[restore_predictions(model.predict(X,verbose=0),scaler) for X,_ in datasets[1:]]
    metrics={name:evaluate(inverse(y,scaler),pred) for name,(_,y),pred in zip(['validation','test'],datasets[1:],predictions)}
    model.save(ROOT/'serving_app/models/baseline_v2.keras')
    with (ROOT/'serving_app/models/scaler_v2.pkl').open('wb') as f:pickle.dump(scaler,f)
    target_start=splits[2][0]+INPUT_LEN
    windows=[[times[target_start+i].isoformat(),times[target_start+i+23].isoformat()] for i in range(len(actual))]
    (out/'test_windows.json').write_text(json.dumps(windows,indent=2))
    result={'model_version':'baseline-v2','region':TARGET_REGION,'fuel':TARGET_FUEL,'seed':SEED,
            'feature_names':FEATURE_NAMES,'architecture':'Input(168,7) -> LSTM(64) -> Dense(64,relu) -> Dense(24)',
            'epochs_completed':len(training.history['loss']),'best_epoch':int(np.argmin(training.history['val_loss']))+1,
            'max_epochs':60,'early_stopping_patience':10,'shuffle':False,
            'scaler':scaler,'metrics':metrics,'shapes':{name:{'X':list(X.shape),'y':list(y.shape)} for name,(X,y) in zip(['train','validation','test'],datasets)},
            'history':training.history,'test_windows_sha256':hashlib.sha256((out/'test_windows.json').read_bytes()).hexdigest(),
            'feature_convention':'KST hour0..23, weekday0..6, day-of-year1..365 for2025 (366 leap years); input timestamps only',
            'evaluation':'same v1 origins and all24 horizons; MWh inverse and zero floor; train-only scaler'}
    (out/'v2_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    (ROOT/'serving_app/models/baseline_v2_metadata.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    np.save(out/'v2_test_predicted.npy',predictions[1])
    print('V2 actual metrics:',json.dumps(metrics),flush=True)
    print('Epochs completed:',result['epochs_completed'],'best epoch:',result['best_epoch'],flush=True)

if __name__=='__main__':main()
