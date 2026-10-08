"""Two fixed data transforms, no source/target edits, early-Validation selection."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv, hashlib, json, pickle
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from src.config import ROOT, CSV_PATH, SEED
from src.data_loader import load_target, read_rows
from src.features import chronological_splits, fit_scaler, inverse
from src.residual_features import windows
from src.normalized_features import normalized_inputs
from src.metrics import evaluate


def main():
    out = ROOT/'artifacts/data_features_v5'
    if out.exists():
        raise RuntimeError('Existing data-feature experiment preserved')
    out.mkdir()
    original_files = [CSV_PATH] + [p for folder in ['src','scripts','tests','serving_app','docs','reports']
                                  for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
    original = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in original_files}
    (out/'original_hashes.json').write_text(json.dumps(original, indent=2))
    rows = read_rows(); times, values = load_target(rows)
    audit = {'total_raw_rows':len(rows), 'target_rows':len(values), 'missing_or_duplicate_hours':False,
             'nonfinite_or_negative':False, 'observed_zero_rows':int((values==0).sum()),
             'raw_csv_sha256':original[str(CSV_PATH.relative_to(ROOT))], 'raw_rows_edited':0,
             'targets_smoothed_or_clipped':False, 'difficult_test_rows_removed':0,
             'night_values_preserved':True, 'ESS_source':'https://new.kpx.or.kr/boardDownload.es?bid=0045&list_no=71003&seq=2'}
    (out/'data_audit.json').write_text(json.dumps(audit, indent=2))
    (_, train_end), (val_start,val_end), (test_start,test_end) = chronological_splits(values)
    scaler = fit_scaler(values[:train_end])
    bounds = {'train':(168,train_end),'selection':(val_start+168,val_end-336),
              'gate':(val_end-336,val_end),'test':(test_start+168,test_end)}
    params = dict(loss='absolute_error',max_iter=200,max_leaf_nodes=31,min_samples_leaf=50,
                  learning_rate=.05,l2_regularization=1,early_stopping=False,random_state=SEED)
    plan = {'candidates':['mean7','normalized','normalized_solar'], 'params':params,
            'selection':'Minimum early Validation WAPE only; winner frozen before Gate/Test',
            'normalization':'Past168h 95th percentile, floor1 MWh; sample-weighted absolute loss',
            'solar':'NOAA calendar geometry; assumed regional proxy 34.8N/126.8E, hourly midpoint, KST; not weather or exact plant location',
            'gate':{'absolute_cap':40,'relative_to_best_production_or_mean7':.95},'production_switched':False}
    (out/'plan.json').write_text(json.dumps(plan, indent=2))
    train_X,train_y,train_origins = windows(times,values,scaler,*bounds['train'])
    select_X,select_y,select_origins = windows(times,values,scaler,*bounds['selection'])
    actual = inverse(select_y,scaler)
    mean = np.stack([values[i-168:i].reshape(7,24).mean(axis=0) for i in select_origins])
    scores = {'mean7':evaluate(actual,mean)}; models = {}
    for name,solar in [('normalized',False),('normalized_solar',True)]:
        train_features,scale = normalized_inputs(train_X,scaler,times,train_origins,solar)
        target = inverse(train_y,scaler)/scale[:,None]
        model = HistGradientBoostingRegressor(**params)
        # Weighting by each origin's scale matches absolute loss in original MWh units.
        model.fit(train_features,target.reshape(-1),sample_weight=np.repeat(scale,24))
        features,scale = normalized_inputs(select_X,scaler,times,select_origins,solar)
        predicted = np.maximum(model.predict(features).reshape(actual.shape)*scale[:,None],0)
        scores[name] = evaluate(actual,predicted); models[name] = model
        print('Selection',name,scores[name],flush=True)
    selected = min(scores,key=lambda n:scores[n]['wape'])
    (out/'selection.json').write_text(json.dumps({'selected':selected,'metrics':scores},indent=2))
    saved = {'name':selected,'model':models.get(selected),'scaler':scaler}
    with (ROOT/'serving_app/models/data_features_v5.pkl').open('wb') as f:pickle.dump(saved,f)
    metrics = {}; daily = {}
    for split in ['gate','test']:
        X,y,origins = windows(times,values,scaler,*bounds[split]); actual = inverse(y,scaler)
        mean = np.stack([values[i-168:i].reshape(7,24).mean(axis=0) for i in origins])
        if selected=='mean7':predicted=mean
        else:
            features,scale=normalized_inputs(X,scaler,times,origins,selected=='normalized_solar')
            predicted=np.maximum(models[selected].predict(features).reshape(actual.shape)*scale[:,None],0)
        np.save(out/f'{split}_predicted.npy',predicted)
        metrics[split]={'v5':evaluate(actual,predicted),'mean7':evaluate(actual,mean)}
        previous=json.loads((ROOT/'artifacts/model_v3/metrics.json').read_text())['metrics'][split]
        metrics[split].update(v2=previous['v2'],v3=previous['v3'])
        midnight=[j for j,i in enumerate(origins) if times[i].hour==0]
        daily[split]={'count':len(midnight),'v5':evaluate(actual[midnight],predicted[midnight]),'mean7':evaluate(actual[midnight],mean[midnight])}
        print(split,metrics[split],flush=True)
    gate=metrics['gate']; passed=gate['v5']['wape']<=min(40,.95*min(gate['v2']['wape'],gate['mean7']['wape']))
    result={'selected':selected,'selection':scores,'metrics':metrics,'midnight_metrics':daily,
            'gate':'PASS' if passed else 'FAIL','plan':plan,'audit':audit,
            'limitations':['Existing Test inspected in prior experiments; independent future evaluation required',
                           'No forecast weather or measured capacity; history quantile is not installed capacity',
                           'No serving/retraining integration or promotion']}
    (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    (ROOT/'serving_app/models/data_features_v5_metadata.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['split','model','mae','rmse','wape'])
        for split,s in metrics.items():
            for name,m in s.items():w.writerow([split,name,m['mae'],m['rmse'],m['wape']])
    print('Selected',selected,'Gate',result['gate'],flush=True)


if __name__=='__main__':main()
