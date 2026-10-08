"""Past-only robust seasonal and tree features; no extra data or dependencies."""
import numpy as np
from src.residual_features import forecast_inputs


def seasonal_candidates(history):
    days = np.asarray(history, dtype='float64')
    if days.ndim != 3 or days.shape[1:] != (7, 24) or not np.isfinite(days).all() or (days < 0).any():
        raise ValueError('Expected finite nonnegative (samples,7,24) history')
    return {
        'mean7': days.mean(axis=1),
        'median7': np.median(days, axis=1),
        'median3': np.median(days[:, -3:], axis=1),
        'quantile35': np.quantile(days, .35, axis=1),
        'quantile65': np.quantile(days, .65, axis=1),
        'trimmed_mean7': np.sort(days, axis=1)[:, 1:-1].mean(axis=1),
    }


def tree_features(X):
    history, future, _, _ = X
    count = len(history)
    days = history[:, :, 0].reshape(count, 7, 24)
    lags = days.transpose(0, 2, 1)
    stats = np.stack([days.mean(axis=1), np.median(days, axis=1), days.std(axis=1),
                      days.min(axis=1), days.max(axis=1)], axis=-1)
    daily = np.repeat(days.mean(axis=2)[:, None, :], 24, axis=1)
    recent = np.repeat(history[:, -6:, 0][:, None, :], 24, axis=1)
    horizon = np.broadcast_to(np.arange(1, 25)[None, :, None] / 24, (count, 24, 1))
    return np.concatenate([future, lags, stats, daily, recent, horizon], axis=-1).reshape(count*24, 32)


def predict_v4(model, name, times, values, scaler):
    from src.features import restore_predictions
    X = forecast_inputs(times, values, scaler)
    if name == 'enhanced_tree':
        predicted = model.predict(tree_features([a[None] for a in X])).reshape(1, 24)
        return restore_predictions(predicted, scaler)[0]
    return seasonal_candidates(np.asarray(values).reshape(1, 7, 24))[name][0]
