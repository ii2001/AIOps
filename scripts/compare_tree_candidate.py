"""One fixed, past-only tree benchmark; no Test-based parameter tuning."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from src.config import ROOT, SEED, INPUT_LEN
from src.data_loader import load_target
from src.features import chronological_splits, fit_scaler, inverse
from src.residual_features import windows
from src.metrics import evaluate


def tree_features(X):
    history, future, references, base = X
    days = history[:, :, 0].reshape(-1, 7, 24)
    stats = np.stack([days.std(axis=1), days.min(axis=1), days.max(axis=1)], axis=-1)
    return np.concatenate([future, references, stats], axis=-1).reshape(-1, 12)


def main():
    out = ROOT / 'artifacts/model_v3'
    destination = out / 'tree_metrics.json'
    if destination.exists():
        raise RuntimeError('Existing benchmark preserved')
    times, values = load_target()
    (_, train_end), (val_start, val_end), (test_start, test_end) = chronological_splits(values)
    scaler = fit_scaler(values[:train_end])
    segments = {'train': (INPUT_LEN, train_end), 'early_stopping_calibration': (val_start + INPUT_LEN, val_end - 336), 'gate': (val_end - 336, val_end), 'test': (test_start + INPUT_LEN, test_end)}
    train_X, train_y, _ = windows(times, values, scaler, *segments['train'])
    model = HistGradientBoostingRegressor(loss='absolute_error', max_iter=100, max_leaf_nodes=15,
                                         min_samples_leaf=50, learning_rate=.05,
                                         l2_regularization=1, early_stopping=False, random_state=SEED)
    model.fit(tree_features(train_X), train_y.reshape(-1))
    metrics = {}
    for name in ['early_stopping_calibration', 'gate', 'test']:
        X, y, _ = windows(times, values, scaler, *segments[name])
        predicted = np.maximum(inverse(model.predict(tree_features(X)).reshape(y.shape), scaler), 0)
        metrics[name] = evaluate(inverse(y, scaler), predicted)
        np.save(out / f'tree_{name}_predicted.npy', predicted)
        print(name, metrics[name], flush=True)
    destination.write_text(json.dumps({'params': model.get_params(), 'metrics': metrics,
        'production_switched': False, 'features': 'Known future calendar, past day/week/mean and seven-day same-hour std/min/max',
        'selection': 'One fixed benchmark; Train only, no random internal Validation, no Test-based tuning'}, indent=2))


if __name__ == '__main__':
    main()
