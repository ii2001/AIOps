import json
import numpy as np
from tensorflow import keras
from src.config import ROOT,SEED,INPUT_LEN
from src.features import build_sequences,inverse,restore_predictions
from src.time_features import make_time_features
from src.metrics import evaluate
from src.aiops_registry import current,assets,log_model,promote,client,OUT


def gate_passes(candidate,production,limit):
    return candidate is not None and production is not None and np.isfinite(candidate) and candidate<=limit and candidate<production


def retrain(times,values,as_of,reason,log,validation_days=7):
    end=next(i for i,t in enumerate(times) if t.date()==as_of and t.hour==23)+1
    start=end-56*24
    if start<0:raise ValueError('56 days of known history are required')
    known_times=times[start:end];known_values=np.asarray(values[start:end])
    assert len(known_times)==1344 and known_times[-1].date()==as_of
    version=str(current().version);production,scaler=assets(version)
    features=make_time_features(known_times,known_values,scaler)
    # First42days tuning; inner35 fit / final7 ES. Last14days are untouched Gate targets.
    fit_end=(42-validation_days)*24;gate_start=42*24
    X,y=build_sequences(features[:fit_end])
    validation_data=build_sequences(features[fit_end-INPUT_LEN:gate_start]) if validation_days else None
    Xg,yg=build_sequences(features[gate_start-INPUT_LEN:])
    assert start+gate_start>=start+fit_end and len(known_times[gate_start:])==14*24
    truth=inverse(yg,scaler)
    prod_metrics=evaluate(truth,restore_predictions(production.predict(Xg,verbose=0),scaler))
    keras.utils.set_random_seed(SEED)
    candidate=keras.models.clone_model(production);candidate.set_weights(production.get_weights())
    assert all(np.array_equal(a,b) for a,b in zip(candidate.get_weights(),production.get_weights()))
    candidate.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),loss='mse')
    log(f'[INFO] retrain triggered reason={reason}, warm_start=v{version}, known_end={known_times[-1].isoformat()}')
    h=candidate.fit(X,y,validation_data=validation_data,epochs=20 if validation_days else 10,batch_size=64,shuffle=False,callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=4,restore_best_weights=True)] if validation_days else [],verbose=2)
    cand_metrics=evaluate(truth,restore_predictions(candidate.predict(Xg,verbose=0),scaler))
    policy=json.loads((OUT/'threshold_analysis.json').read_text())['recommended'];limit=policy['gate']
    passed=gate_passes(cand_metrics['wape'],prod_metrics['wape'],limit)
    params={'epochs':len(h.history['loss']),'learning_rate':1e-4,'retraining_window':f'56days: first{42-validation_days}fit+{validation_days}ES, last14Gate','threshold':policy['drift'],'gate_threshold':limit,'as_of':as_of.isoformat(),'warm_start_version':version}
    run=log_model(candidate,params,cand_metrics,reason,X[:1])
    client().set_tag(run,'gate','PASS' if passed else 'FAIL')
    client().log_metric(run,'production_gate_wape',prod_metrics['wape'])
    log(f'[INFO] candidate evaluated production_wape={prod_metrics["wape"]:.6f}, candidate_wape={cand_metrics["wape"]:.6f}, absolute_gate={limit:.6f}')
    before,after=version,version
    if passed:
        log('[OK] gate passed: absolute ceiling AND same-window improvement')
        before,after=promote(run);log(f'[OK] production promoted v{before} -> v{after}')
    else:
        log('[FAIL] candidate rejected');log(f'[INFO] existing production retained v{version}')
    result={'reason':reason,'as_of':as_of.isoformat(),'production':prod_metrics,'candidate':cand_metrics,'gate':'PASS' if passed else 'FAIL','gate_threshold':limit,'run_id':run,'production_before':before,'production_after':after,'warm_start_weights_verified':True,'epochs_completed':len(h.history['loss']),
            'split':{'known_start':known_times[0].isoformat(),'fit_end':known_times[fit_end-1].isoformat(),'early_stopping_start':known_times[fit_end].isoformat() if validation_days else None,'fine_tuning_end':known_times[gate_start-1].isoformat(),'gate_start':known_times[gate_start].isoformat(),'gate_end':known_times[-1].isoformat(),'fine_tuning_days':42,'gate_days':14},'gate_independent_of_fit_and_early_stopping':True,'metrics_convention':'identical rolling origins with targets only in last14days; same shifted known-input features for both models'}
    (OUT/'retraining_metrics.json').write_text(json.dumps(result,indent=2))
    return result
