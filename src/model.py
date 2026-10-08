from tensorflow import keras
from src.config import INPUT_LEN, OUTPUT_LEN


def build_model(feature_dim=1):
    model = keras.Sequential([
        keras.layers.Input(shape=(INPUT_LEN, feature_dim)),
        keras.layers.LSTM(32),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(OUTPUT_LEN),
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
