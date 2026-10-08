import numpy as np


def evaluate(actual, predicted):
    actual, predicted = np.asarray(actual, dtype='float64'), np.asarray(predicted, dtype='float64')
    if actual.shape != predicted.shape or not actual.size:
        raise ValueError('Metrics require matching nonempty shapes')
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('Metrics require finite values')
    error = actual - predicted
    denominator = float(np.abs(actual).sum())
    return {'mae': float(np.abs(error).mean()), 'rmse': float(np.sqrt((error ** 2).mean())),
            'wape': float(np.abs(error).sum() / denominator * 100) if denominator else None}
