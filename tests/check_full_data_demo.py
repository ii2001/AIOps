"""Verify all-data demo scores and labels; never treat trained-on targets as a holdout."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hashlib,json,pickle
import numpy as np
from src.config import ROOT
from src.data_loader import load_target
from src.features import chronological_splits
from src.normalized_features import predict_normalized
from src.metrics import evaluate


def main():
    out=ROOT/'artifacts/full_data_demo'
    result=json.loads((out/'metrics.json').read_text())
    assert result['trained_on_prior_validation_and_test'] and not result['evaluation_is_independent']
    assert result['gate']=='NOT_EVALUATED_NO_HOLDOUT' and not result['production_switched']
    assert '독립 운영 성능 아님' in result['metrics_label']
    times,values=load_target();_,_,(start,end)=chronological_splits(values);origin=start+168
    with (ROOT/'serving_app/models/full_data_demo.pkl').open('rb') as f:saved=pickle.load(f)
    predicted=predict_normalized(saved['model'],False,times[origin-168:origin],values[origin-168:origin],saved['scaler'])
    cached=np.load(out/'former_test_in_sample_predicted.npy')
    assert cached.shape==(1123,24) and np.isfinite(cached).all() and (cached>=0).all()
    assert np.allclose(predicted,cached[0],rtol=1e-5,atol=.001)
    metrics=evaluate(np.load(ROOT/'artifacts/test_actual.npy'),cached)
    assert all(np.isclose(metrics[k],result['metrics']['former_test_origins_in_sample'][k]) for k in metrics)
    hashes=json.loads((out/'original_hashes.json').read_text())
    changed=[n for n,h in hashes.items() if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h]
    assert set(changed)<= {'scripts/train_full_data_demo.py'},changed
    policy=json.loads((ROOT/'reports/full_data_demo/current_gate_policy.json').read_text())
    assert policy['cap_wape']==15 and policy['minimum_relative_improvement']==.05
    assert policy['requires_new_independent_targets'] and policy['status']=='NOT_EVALUATED_NO_HOLDOUT'
    validation={'status':'PASS','former_test_in_training':True,'scores_recomputed':True,
                'saved_model_inference_parity':True,'in_sample_labels_explicit':True,
                'original_data_models_preserved':True,'authorized_training_policy_code_change':changed,
                'current_proposed_gate_cap':policy['cap_wape'],'independent_gate':'NOT_EVALUATED_NO_HOLDOUT',
                'production_switched':False}
    (out/'validation.json').write_text(json.dumps(validation,indent=2));print(json.dumps(validation))


if __name__=='__main__':main()
