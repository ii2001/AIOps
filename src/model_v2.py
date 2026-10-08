from tensorflow import keras
from src.config import INPUT_LEN, OUTPUT_LEN


def build_model_v2():
    model = keras.Sequential([
        keras.layers.Input(shape=(INPUT_LEN,7)),
        keras.layers.LSTM(64),
        keras.layers.Dense(64,activation='relu'),
        keras.layers.Dense(OUTPUT_LEN),
    ])
    model.compile(optimizer='adam',loss='mse')
    return model
