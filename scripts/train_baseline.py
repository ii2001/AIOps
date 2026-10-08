import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv, json, pickle
import numpy as np
from tensorflow import keras
from src.config import ROOT, SEED, EPOCHS, INPUT_LEN, OUTPUT_LEN, MODEL_PATH, SCALER_PATH, TARGET_REGION, TARGET_FUEL
from src.data_loader import load_target
from src.features import chronological_splits, fit_scaler, transform, inverse, restore_predictions, build_sequences
from src.model import build_model
from src.metrics import evaluate


def main():
    keras.utils.set_random_seed(SEED)
    times, values = load_target()
    splits = chronological_splits(values)
    scaler = fit_scaler(values[splits[0][0]:splits[0][1]])
    datasets = [build_sequences(transform(values[a:b], scaler)) for a, b in splits]
    partitions = {}
    for name, (a, b), (X, y) in zip(['train', 'validation', 'test'], splits, datasets):
        partitions[name] = {'start': times[a].isoformat(), 'end': times[b-1].isoformat(), 'raw_rows': b-a,
                            'first_target': times[a+INPUT_LEN].isoformat(), 'last_target': times[b-1].isoformat(),
                            'X_shape': list(X.shape), 'y_shape': list(y.shape)}
        assert X.shape[1:] == (INPUT_LEN, 1) and y.shape[1] == OUTPUT_LEN
        print(f'{name}: {partitions[name]}', flush=True)
    assert times[splits[0][1]-1] < times[splits[1][0]] and times[splits[1][1]-1] < times[splits[2][0]]
    model = build_model()
    model.summary()
    history = model.fit(*datasets[0], validation_data=datasets[1], epochs=EPOCHS, batch_size=64,
                        shuffle=False, callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)], verbose=2)
    predictions = [restore_predictions(model.predict(X, verbose=0), scaler) for X, _ in datasets[1:]]
    metrics = {name: evaluate(inverse(y, scaler), predicted) for name, (_, y), predicted in zip(['validation', 'test'], datasets[1:], predictions)}
    model.save(MODEL_PATH)
    with SCALER_PATH.open('wb') as file: pickle.dump(scaler, file)
    result = {'seed': SEED, 'region': TARGET_REGION, 'fuel': TARGET_FUEL, 'feature_columns': ['전력거래량(MWh)'],
              'partitions': partitions, 'scaler': scaler, 'metrics': metrics, 'epochs_completed': len(history.history['loss']),
              'best_validation_loss': min(history.history['val_loss']), 'evaluation_policy': 'all hourly rolling origins and 24 horizons; inverse MWh and zero floor; WAPE null if actual denominator=0',
              'architecture': 'Input(168,1) -> LSTM(32) -> Dense(32,relu) -> Dense(24)', 'history': history.history}
    (ROOT / 'artifacts/training_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    for name, scores in metrics.items(): print(f'{name} MAE={scores["mae"]:.6f} RMSE={scores["rmse"]:.6f} WAPE={scores["wape"]:.6f}%', flush=True)
    # Select a real test target that begins at midnight, rather than a fabricated pattern.
    test_start = splits[2][0]
    index = next(i for i in range(len(predictions[1])) if times[test_start+INPUT_LEN+i].hour == 0)
    actual = values[test_start+INPUT_LEN+index:test_start+INPUT_LEN+index+OUTPUT_LEN]
    predicted = predictions[1][index]
    example_times = times[test_start+INPUT_LEN+index:test_start+INPUT_LEN+index+OUTPUT_LEN]
    with (ROOT / 'artifacts/baseline_prediction_sample.csv').open('w', newline='') as file:
        writer = csv.writer(file); writer.writerow(['hour', 'datetime', 'actual', 'predicted'])
        writer.writerows((i+1, t.isoformat(), float(a), float(p)) for i, (t, a, p) in enumerate(zip(example_times, actual, predicted)))
    np.save(ROOT / 'artifacts/test_actual.npy', inverse(datasets[2][1], scaler))
    np.save(ROOT / 'artifacts/test_predicted.npy', predictions[1])
    (ROOT / 'data/processed/target_hourly.csv').write_text('datetime,mwh\n'+''.join(f'{t.isoformat()},{v}\n' for t,v in zip(times,values)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10,4)); ax.plot(range(24),actual,label='Actual',marker='o');ax.plot(range(24),predicted,label='LSTM forecast',marker='.')
    ax.set(title=f'{example_times[0].date()} / Jeollanam-do solar / real test data', xlabel='KST hour (interval start)', ylabel='MWh', xticks=range(0,24,2));ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(ROOT/'artifacts/baseline_prediction_chart.png',dpi=160);plt.close(fig)
    print(f'Saved model: {MODEL_PATH}\nSaved scaler: {SCALER_PATH}',flush=True)

if __name__ == '__main__': main()
