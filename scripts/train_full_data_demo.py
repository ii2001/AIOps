"""Train on all observed2025 data for a labeled demo. Scores are in-sample, never Test performance."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hashlib,json,pickle
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from src.config import ROOT,CSV_PATH,SEED
from src.data_loader import load_target
from src.features import fit_scaler,inverse,chronological_splits
from src.residual_features import windows
from src.normalized_features import normalized_inputs
from src.metrics import evaluate


def main():
    out=ROOT/'artifacts/full_data_demo'
    model_file=ROOT/'serving_app/models/full_data_demo.pkl'
    if out.exists() or model_file.exists():raise RuntimeError('Preserve existing demo outputs')
    out.mkdir()
    original={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [CSV_PATH]+[p for folder in ['src','scripts','tests','serving_app','docs','reports']
                  for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']}
    (out/'original_hashes.json').write_text(json.dumps(original,indent=2))
    times,values=load_target();scaler=fit_scaler(values)
    params=dict(loss='absolute_error',max_iter=500,max_leaf_nodes=63,min_samples_leaf=20,
                learning_rate=.05,l2_regularization=1,early_stopping=False,random_state=SEED)
    plan={'purpose':'All-data training and in-sample presentation demo',
          'trained_on_prior_validation_and_test':True,'evaluation_is_independent':False,
          'params':params,'gate':'NOT_EVALUATED_NO_HOLDOUT','production_switched':False,
          'proposed_future_gate':{'cap_wape':15,'minimum_relative_improvement':.05,
                                  'comparators':['current Production','7-Day Same-Hour Mean'],
                                  'requires_new_independent_targets':True}}
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    X,y,origins=windows(times,values,scaler,168,len(values))
    features,scale=normalized_inputs(X,scaler,times,origins,False)
    actual=inverse(y,scaler)
    model=HistGradientBoostingRegressor(**params)
    print('Fitting all2025 targets:',features.shape,'including prior Validation and Test',flush=True)
    model.fit(features,(actual/scale[:,None]).reshape(-1),sample_weight=np.repeat(scale,24))
    predicted=np.maximum(model.predict(features).reshape(actual.shape)*scale[:,None],0)
    _,_,(test_start,test_end)=chronological_splits(values)
    mask=np.array(origins)>=test_start+168
    old_actual=np.load(ROOT/'artifacts/test_actual.npy')
    assert np.array_equal(actual[mask],old_actual)
    np.save(out/'all_in_sample_predicted.npy',predicted)
    np.save(out/'former_test_in_sample_predicted.npy',predicted[mask])
    result={**plan,'model':'normalized histogram gradient boosting / full-data-demo',
            'training_target_start':times[168].isoformat(),'training_target_end':times[-1].isoformat(),
            'training_origins':len(origins),'metrics':{'all_training_origins_in_sample':evaluate(actual,predicted),
                                                     'former_test_origins_in_sample':evaluate(actual[mask],predicted[mask])},
            'previous_v3_evaluation_wape':json.loads((ROOT/'artifacts/model_v3/metrics.json').read_text())['metrics']['test']['v3']['wape'],
            'metrics_label':'학습 데이터 재평가 오차 (기존 Test 포함 학습); 독립 운영 성능 아님',
            'source_csv_sha256':original[str(CSV_PATH.relative_to(ROOT))]}
    with model_file.open('wb') as f:pickle.dump({'model':model,'scaler':scaler,'with_solar':False,'metadata':result},f)
    (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
