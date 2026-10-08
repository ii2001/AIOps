"""Dedicated local MLflow store; baseline artifacts stay untouched."""
import json,pickle,os
from functools import lru_cache
from pathlib import Path
import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
from tensorflow import keras
from src.config import ROOT,INPUT_LEN,TARGET_REGION,TARGET_FUEL
from src.time_features import make_time_features,FEATURE_NAMES
from src.features import restore_predictions

OUT=Path(os.getenv('AIOPS_RUN_DIR',str(ROOT/'artifacts/aiops'))).resolve()
NAME='RenewableEnergyPredictor'


def client():
    mlflow.set_tracking_uri(f'sqlite:///{OUT / "mlflow.db"}')
    c=MlflowClient()
    experiments=c.search_experiments(filter_string="name = 'renewable-aiops'")
    if not experiments:c.create_experiment('renewable-aiops',artifact_location=(OUT/'mlflow_artifacts').as_uri())
    mlflow.set_experiment('renewable-aiops')
    return c


def current():
    return client().get_model_version_by_alias(NAME,'Production')


@lru_cache(maxsize=2)
def assets(version):
    model=mlflow.tensorflow.load_model(f'models:/{NAME}/{version}')
    with (ROOT/'serving_app/models/scaler_v2.pkl').open('rb') as f:scaler=pickle.load(f)
    return model,scaler


def log_model(model,params,scores,reason,example):
    client()
    with mlflow.start_run(run_name=reason) as run:
        mlflow.log_params({'model_type':'LSTM-v2','input_len':168,'output_len':24,'feature_count':7,**params})
        mlflow.set_tags({'phase':'baseline' if reason=='initial-production' else 'retraining','target_region':TARGET_REGION,'fuel':TARGET_FUEL,'reason':reason})
        mlflow.log_metrics({k:v for k,v in scores.items() if v is not None})
        mlflow.tensorflow.log_model(model,name='model',input_example=example)
        return run.info.run_id


def promote(run_id):
    c=client();before=None
    try:before=str(c.get_model_version_by_alias(NAME,'Production').version)
    except mlflow.exceptions.MlflowException:pass
    version=mlflow.register_model(f'runs:/{run_id}/model',NAME)
    # Alias identifies exactly one current Production; stage retained for instructor-compatible UI.
    c.transition_model_version_stage(NAME,version.version,'Production',archive_existing_versions=True)
    c.set_registered_model_alias(NAME,'Production',version.version)
    assets.cache_clear()
    return before,str(version.version)


def initialize(example):
    c=client()
    if c.search_model_versions(f"name='{NAME}'"):
        return str(current().version)
    metadata=json.loads((ROOT/'artifacts/model_comparison/v2_metrics.json').read_text())
    model=keras.models.load_model(ROOT/'serving_app/models/baseline_v2.keras')
    params={'epochs':metadata['epochs_completed'],'learning_rate':.001,'retraining_window':'original Train / original Validation early stopping','threshold':json.loads((OUT/'threshold_analysis.json').read_text())['recommended']['drift']}
    run_id=log_model(model,params,metadata['metrics']['test'],'initial-production',example)
    _,version=promote(run_id)
    return version
