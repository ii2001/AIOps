import numpy as np
from src.config import INPUT_LEN, OUTPUT_LEN, TRAIN_RATIO, VAL_RATIO


def chronological_splits(values):
    train_end = int(len(values) * TRAIN_RATIO)
    val_end = train_end + int(len(values) * VAL_RATIO)
    return [(0, train_end), (train_end, val_end), (val_end, len(values))]


def fit_scaler(train):
    train = np.asarray(train)
    if not train.size or not np.isfinite(train).all():
        raise ValueError('Scaler requires finite training data')
    minimum, maximum = float(train.min()), float(train.max())
    return {'min': minimum, 'max': maximum, 'span': maximum - minimum or 1.0}


def transform(values, scaler):
    return ((np.asarray(values) - scaler['min']) / scaler['span']).astype('float32')


def inverse(values, scaler):
    return np.asarray(values, dtype='float64') * scaler['span'] + scaler['min']


def restore_predictions(values, scaler):
    predictions = inverse(values, scaler)
    if not np.isfinite(predictions).all():
        raise ValueError('Model returned NaN/inf')
    # Energy is nonnegative. The same zero floor is used for evaluation and serving; no upper clipping.
    return np.maximum(predictions, 0.0)


def build_sequences(features):
    features = np.asarray(features, dtype='float32')
    if features.ndim == 1:
        features = features[:, None]
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError('Expected finite (hours, features) data; target is feature 0')
    samples = len(features) - INPUT_LEN - OUTPUT_LEN + 1
    if samples <= 0:
        raise ValueError('Partition is shorter than input + output horizon')
    X = np.stack([features[i:i + INPUT_LEN] for i in range(samples)])
    y = np.stack([features[i + INPUT_LEN:i + INPUT_LEN + OUTPUT_LEN, 0] for i in range(samples)])
    return X, y
