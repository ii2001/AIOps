"""Fixed candidate list selected on early Validation; Gate and Test remain held out."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hashlib, json, pickle, csv
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from src.config import ROOT, SEED
from src.data_loader import load_target
from src.features import chronological_splits, fit_scaler, inverse, restore_predictions
from src.residual_features import windows
from src.model_v4 import seasonal_candidates, tree_features
from src.metrics import evaluate


def main():
    out = ROOT / 'artifacts/model_v4'
    if out.exists():
        raise RuntimeError('Existing v4 experiment preserved')
    out.mkdir()
    originals = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for folder in ['src', 'scripts', 'serving_app', 'docs', 'reports', 'tests']
                 for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    (out/'original_hashes.json').write_text(json.dumps(originals, indent=2))
    plan = {'candidates': ['mean7', 'median7', 'median3', 'quantile35', 'quantile65', 'trimmed_mean7', 'enhanced_tree'],
            'selection': 'Minimum WAPE on early Validation only; no Gate/Test candidate selection',
            'tree_params': {'loss': 'absolute_error', 'max_iter': 200, 'max_leaf_nodes': 31,
                            'min_samples_leaf': 50, 'learning_rate': .05, 'l2_regularization': 1,
                            'early_stopping': False, 'random_state': SEED},
            'gate': {'absolute_wape_cap': 40, 'max_relative_to_best_production_or_naive': .95},
            'production_switched': False}
    (out/'plan.json').write_text(json.dumps(plan, indent=2))
    times, values = load_target()
    (_, train_end), (val_start, val_end), (test_start, test_end) = chronological_splits(values)
    scaler = fit_scaler(values[:train_end])
    bounds = {'train': (168, train_end), 'selection': (val_start+168, val_end-336),
              'gate': (val_end-336, val_end), 'test': (test_start+168, test_end)}
    train_X, train_y, _ = windows(times, values, scaler, *bounds['train'])
    model = HistGradientBoostingRegressor(**plan['tree_params'])
    model.fit(tree_features(train_X), train_y.reshape(-1))
    X, y, origins = windows(times, values, scaler, *bounds['selection'])
    raw = np.stack([values[i-168:i].reshape(7, 24) for i in origins])
    candidates = seasonal_candidates(raw)
    candidates['enhanced_tree'] = restore_predictions(model.predict(tree_features(X)).reshape(y.shape), scaler)
    actual = inverse(y, scaler)
    scores = {name: evaluate(actual, pred) for name, pred in candidates.items()}
    selected = min(scores, key=lambda n: scores[n]['wape'])
    # Freeze winner before looking at Gate or Test. Keep all selection scores for audit.
    (out/'selection.json').write_text(json.dumps({'selected': selected, 'metrics': scores}, indent=2))
    print('Validation selection:', selected, scores, flush=True)
    with (ROOT/'serving_app/models/baseline_v4.pkl').open('wb') as f:
        pickle.dump({'name': selected, 'model': model if selected == 'enhanced_tree' else None, 'scaler': scaler}, f)
    metrics, daily = {}, {}
    for split in ['gate', 'test']:
        X, y, origins = windows(times, values, scaler, *bounds[split])
        actual = inverse(y, scaler)
        raw = np.stack([values[i-168:i].reshape(7, 24) for i in origins])
        mean = raw.mean(axis=1)
        predicted = (restore_predictions(model.predict(tree_features(X)).reshape(y.shape), scaler)
                     if selected == 'enhanced_tree' else seasonal_candidates(raw)[selected])
        np.save(out/f'{split}_predicted.npy', predicted)
        metrics[split] = {'v4': evaluate(actual, predicted), 'mean7': evaluate(actual, mean)}
        existing = json.loads((ROOT/'artifacts/model_v3/metrics.json').read_text())['metrics'][split]
        metrics[split]['v2'] = existing['v2']; metrics[split]['v3'] = existing['v3']
        midnight = [j for j, i in enumerate(origins) if times[i].hour == 0]
        daily[split] = {'samples': len(midnight), 'v4': evaluate(actual[midnight], predicted[midnight]),
                       'mean7': evaluate(actual[midnight], mean[midnight])}
        print(split, metrics[split], 'midnight:', daily[split], flush=True)
    gate = metrics['gate']; cap = min(gate['v2']['wape'], gate['mean7']['wape']) * .95
    passed = gate['v4']['wape'] <= min(40, cap)
    result = {'selected': selected, 'selection': scores, 'metrics': metrics, 'midnight_metrics': daily,
              'gate': 'PASS' if passed else 'FAIL', 'gate_effective_cap': cap,
              'plan': plan, 'segments': {k: {'start': times[a].isoformat(), 'end': times[b-1].isoformat()} for k,(a,b) in bounds.items()},
              'limitations': ['Existing Test already examined in past experiments; independent future evaluation still required',
                              'No weather forecasts or capacity data', 'No serving/retraining integration or Production promotion']}
    (out/'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (ROOT/'serving_app/models/baseline_v4_metadata.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    with (out/'comparison.csv').open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['split', 'model', 'mae', 'rmse', 'wape'])
        for split, models in metrics.items():
            for name, score in models.items(): writer.writerow([split, name, score['mae'], score['rmse'], score['wape']])
    print('Declared Gate:', result['gate'], flush=True)


if __name__ == '__main__':
    main()
