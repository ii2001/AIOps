"""Run after train_baseline_v3.py: past-only inputs, boundaries and saved predictions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hashlib, json
import numpy as np
from tensorflow import keras
from src.config import ROOT
from src.data_loader import load_target
from src.features import chronological_splits, fit_scaler, inverse, restore_predictions
from src.residual_features import forecast_inputs, windows
from src.metrics import evaluate
from scripts.compare_tree_candidate import tree_features


def main():
    out = ROOT / 'artifacts/model_v3'
    times, values = load_target()
    splits = chronological_splits(values)
    scaler = fit_scaler(values[:splits[0][1]])
    start, end = splits[2][0] + 168, splits[2][1]
    X, y, origins = windows(times, values, scaler, start, end)
    assert len(origins) == 1123 and origins[-1] + 24 == end
    assert np.array_equal(inverse(y, scaler), np.load(ROOT / 'artifacts/test_actual.npy'))
    altered = values.copy()
    altered[start:] = 0
    changed, _, _ = windows(times, altered, scaler, start, start + 24)
    assert all(np.array_equal(a[:1], b) for a, b in zip(X, changed)), 'Future target leaked into inputs'
    assert tree_features(X).shape == (1123*24, 12)
    assert np.array_equal(tree_features([a[:1] for a in X]), tree_features(changed))
    raw = values[start-168:start].reshape(7, 24)
    assert np.allclose(inverse(X[3][0, :, 0], scaler), raw.mean(axis=0), atol=.001)
    assert np.allclose(inverse(X[2][0, :, 0], scaler), raw[-1], atol=.001)
    assert np.allclose(inverse(X[2][0, :, 1], scaler), raw[0], atol=.001)
    rebuilt = forecast_inputs(times[start-168:start], values[start-168:start], scaler)
    assert all(np.array_equal(a[0], b) for a, b in zip(X, rebuilt))
    try:
        windows(times, values, scaler, 167, 200)
    except ValueError:
        pass
    else:
        raise AssertionError('Missing context accepted')
    model = keras.models.load_model(ROOT / 'serving_app/models/baseline_v3.keras')
    predicted = restore_predictions(model.predict([a[:2] for a in X], verbose=0), scaler)
    assert np.allclose(predicted, np.load(out / 'test_predicted.npy')[:2], rtol=1e-4, atol=.001)
    scores = evaluate(inverse(y, scaler), np.load(out / 'test_predicted.npy'))
    recorded = json.loads((out / 'metrics.json').read_text())
    assert all(np.isclose(scores[k], recorded['metrics']['test']['v3'][k]) for k in scores)
    original = json.loads((out / 'original_hashes.json').read_text())
    changed_files = [name for name, digest in original.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    assert not changed_files, changed_files
    result = {'status': 'PASS', 'same_test_targets': True, 'past_only_inputs': True,
              'saved_model_prediction_parity': True, 'original_files_preserved': True,
              'production_switched': False}
    (out / 'validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
