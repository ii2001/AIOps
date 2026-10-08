import json
import numpy as np
from src.config import ROOT, INPUT_LEN, OUTPUT_LEN
from src.data_loader import load_target
from src.features import chronological_splits
from src.metrics import evaluate


NAIVE_NAMES = ['Previous Day Same Hour', 'Previous Week Same Hour', '7-Day Same-Hour Mean']


def seasonal_predictions(history):
    history = np.asarray(history,dtype='float64')
    if history.ndim != 2 or history.shape[1] != INPUT_LEN or not np.isfinite(history).all():
        raise ValueError('Expected finite (samples,168) input histories')
    return dict(zip(NAIVE_NAMES,[history[:,-24:],history[:,:24],history.reshape(-1,7,24).mean(axis=1)]))


def comparison_data():
    times,values = load_target()
    splits = chronological_splits(values)
    original = json.loads((ROOT/'artifacts/training_result.json').read_text())
    for name,(a,b) in zip(['train','validation','test'],splits):
        assert original['partitions'][name]['start']==times[a].isoformat()
        assert original['partitions'][name]['end']==times[b-1].isoformat()
    start,end = splits[2]
    samples = end-start-INPUT_LEN-OUTPUT_LEN+1
    history = np.stack([values[start+i:start+i+INPUT_LEN] for i in range(samples)])
    targets = np.stack([values[start+i+INPUT_LEN:start+i+INPUT_LEN+OUTPUT_LEN] for i in range(samples)])
    actual = np.load(ROOT/'artifacts/test_actual.npy')
    assert actual.shape==targets.shape==(samples,OUTPUT_LEN)
    assert np.allclose(actual,targets,atol=.001,rtol=0)
    # Reuse v1's float32-scaled/inversed ground truth so metrics are numerically identical.
    return times,values,splits,history,actual


def benchmark_metrics(history,actual):
    return {name:evaluate(actual,prediction) for name,prediction in seasonal_predictions(history).items()}
