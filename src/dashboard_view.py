"""Read-only UI summaries of actual experiments and the isolated dashboard operations store."""
import csv,json
from datetime import datetime,timedelta
from src.config import ROOT,TARGET_REGION,TARGET_FUEL
from src.aiops_registry import OUT,current,client,assets
from src.aiops_runtime import status
from src.features import restore_predictions
from src.time_features import make_time_features

PARENT=ROOT/'artifacts/aiops'


def read(path):return json.loads(path.read_text())


def summary():
    version=current();run=client().get_run(version.run_id)
    profile=read(ROOT/'artifacts/data_profile.json')
    target=next(x for x in profile['groups'] if x['region']==TARGET_REGION and x['fuel']==TARGET_FUEL)
    with (ROOT/'artifacts/model_comparison/model_comparison.csv').open() as f:comparison=list(csv.DictReader(f))
    return {'model':{'name':version.name,'version':str(version.version),'stage':version.current_stage,'metrics':run.data.metrics,'params':run.data.params},
            'status':status(),'dataset':{**target,'total_rows':profile['rows'],'start':profile['start'],'end':profile['end'],'encoding':profile['encoding']},
            'threshold':read(PARENT/'threshold_analysis.json'),'baseline':read(ROOT/'artifacts/model_comparison/v2_metrics.json'),'partitions':read(ROOT/'artifacts/training_result.json')['partitions'],
            'comparison':comparison,'scenario1':read(PARENT/'scenario_1_capacity_increase.json'),'scenario2':read(PARENT/'scenario_2/demo_result.json'),
            'normal_series':read(PARENT/'drift_result.json')['normal_records'],'drift_series':read(PARENT/'scenario_2/output_curtailment_20.json')['records'],
            'sensitivity':[ {k:v for k,v in read(PARENT/f'scenario_2/output_curtailment_{x}.json').items() if k not in ['records','days']} for x in [15,20,25,30]],
            'scope':'dashboard sandbox / Scenario2 Registry copy; original model and experiment files preserved',
            'model_loaded':bool(assets.cache_info().currsize)}


def example():
    from src.data_loader import load_target
    times,values=load_target();selected=read(PARENT/'scenario_2/selection.json')
    sequence=[float(v)*(1+selected['change_percent']/100) if t.date().isoformat()>=selected['start_date'] and v>selected['minimum_mwh'] else float(v) for t,v in zip(times[-168:],values[-168:])]
    return {'start_datetime':times[-168].isoformat(),'sequence':sequence,'source':'actual2025 CSV copy with selected -20% experiment transformation; no future actual fabricated'}


def preview(sequence,start):
    target=start+timedelta(hours=168)
    if target.hour!=0:raise ValueError('Daily forecast must start at midnight KST')
    times=[start+timedelta(hours=i) for i in range(168)]
    version=str(current().version);model,scaler=assets(version)
    values=restore_predictions(model.predict(make_time_features(times,sequence,scaler)[None,:,:],verbose=0),scaler)[0].tolist()
    return {'predictions':[{'hour':i+1,'mwh':v} for i,v in enumerate(values)],'forecast_date':target.date().isoformat(),
            'model_version':version,'region':TARGET_REGION,'fuel':TARGET_FUEL,'mode':'preview; not stored as an operational forecast'}
