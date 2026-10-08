from functools import lru_cache
import pickle
import numpy as np
from src.config import MODEL_PATH, SCALER_PATH, INPUT_LEN, OUTPUT_LEN
from src.features import transform, restore_predictions


@lru_cache(maxsize=1)
def load_assets():
    from tensorflow import keras
    model = keras.models.load_model(MODEL_PATH)
    with SCALER_PATH.open('rb') as file: scaler = pickle.load(file)
    if model.input_shape[1:] != (INPUT_LEN, 1) or model.output_shape[1:] != (OUTPUT_LEN,):
        raise ValueError('Saved model contract does not match the configured input/output horizon')
    return model, scaler


def predict(sequence):
    model, scaler = load_assets()
    x = transform(sequence, scaler).reshape(1, INPUT_LEN, 1)
    scaled = model.predict(x, verbose=0)
    values = restore_predictions(scaled, scaler)[0]
    if values.shape != (OUTPUT_LEN,) or not np.isfinite(values).all():
        raise ValueError('Invalid model output')
    return values.tolist()
