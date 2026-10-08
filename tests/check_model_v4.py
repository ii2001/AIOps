"""Validate fixed selection, no future-target input, saved inference and Gate rejection."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hashlib, json, pickle
import numpy as np
from src.config import ROOT
from src.data_loader import load_target
from src.features import inverse, restore_predictions
from src.residual_features import windows
from src.model_v4 import seasonal_candidates, tree_features, predict_v4
from src.metrics import evaluate


def main():
    out = ROOT/'artifacts/model_v4'
    result = json.loads((out/'metrics.json').read_text())
    selection = json.loads((out/'selection.json').read_text())
    assert selection['selected'] == min(selection['metrics'], key=lambda n: selection['metrics'][n]['wape'])
    with (ROOT/'serving_app/models/baseline_v4.pkl').open('rb') as f:
        saved = pickle.load(f)
    times, values = load_target()
    start = int(len(values)*.7) + int(len(values)*.15) + 168
    X, y, _ = windows(times, values, saved['scaler'], start, len(values))
    assert np.array_equal(inverse(y, saved['scaler']), np.load(ROOT/'artifacts/test_actual.npy'))
    altered = values.copy(); altered[start:] = 0
    changed, _, _ = windows(times, altered, saved['scaler'], start, start+24)
    assert np.array_equal(tree_features([a[:1] for a in X]), tree_features(changed))
    predicted = predict_v4(saved['model'], saved['name'], times[start-168:start], values[start-168:start], saved['scaler'])
    cached = np.load(out/'test_predicted.npy')
    assert np.allclose(predicted, cached[0], atol=.001, rtol=1e-5)
    assert np.isfinite(cached).all() and (cached >= 0).all()
    score = evaluate(inverse(y, saved['scaler']), cached)
    assert all(np.isclose(score[k], result['metrics']['test']['v4'][k]) for k in score)
    simple = np.arange(168).reshape(1, 7, 24)
    candidates = seasonal_candidates(simple)
    assert np.array_equal(candidates['median7'][0], np.arange(24)+72)
    assert np.array_equal(candidates['median3'][0], np.arange(24)+120)
    gate = result['metrics']['gate']
    passed = gate['v4']['wape'] <= min(40, .95*min(gate['v2']['wape'],gate['mean7']['wape']))
    assert result['gate'] == ('PASS' if passed else 'FAIL')
    original = json.loads((out/'original_hashes.json').read_text())
    changed_files = [name for name, digest in original.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    assert not changed_files, changed_files
    validation = {'status': 'PASS', 'validation_only_selection': True, 'same_test_targets': True,
                  'past_only_features': True, 'saved_model_prediction_parity': True,
                  'original_files_preserved': True, 'gate': result['gate'], 'production_switched': False}
    (out/'validation.json').write_text(json.dumps(validation, indent=2))
    print(json.dumps(validation))


if __name__ == '__main__':
    main()
