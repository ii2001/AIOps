from tensorflow import keras


def build_model_v3():
    history=keras.Input((168,7),name='history')
    future=keras.Input((24,6),name='known_future_calendar')
    references=keras.Input((24,3),name='past_day_week_mean')
    base=keras.Input((24,1),name='seasonal_mean')
    encoded=keras.layers.LSTM(32)(history)
    context=keras.layers.RepeatVector(24)(encoded)
    context=keras.layers.Concatenate()([context,future,references])
    hidden=keras.layers.Dense(32,activation='relu')(context)
    residual=keras.layers.Dense(1,kernel_initializer='zeros',bias_initializer='zeros',name='residual')(hidden)
    output=keras.layers.Reshape((24,))(keras.layers.Add()([base,residual]))
    model=keras.Model([history,future,references,base],output,name='seasonal_residual_lstm_v3')
    # On a fixed dataset WAPE's denominator is constant: minimizing MAE minimizes its numerator.
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=.001),loss='mae')
    return model
