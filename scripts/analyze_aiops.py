import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv,json,pickle
import numpy as np
from tensorflow import keras
from src.config import ROOT,INPUT_LEN
from src.data_loader import load_target
from src.features import chronological_splits,build_sequences,restore_predictions
from src.time_features import make_time_features
from src.aiops_policy import distribution,daily_policy


def main():
 out=ROOT/'artifacts/aiops';times,values=load_target();splits=chronological_splits(values)
 with (ROOT/'serving_app/models/scaler_v2.pkl').open('rb') as f:scaler=pickle.load(f)
 model=keras.models.load_model(ROOT/'serving_app/models/baseline_v2.keras');features=make_time_features(times,values,scaler)
 normal={}
 for name,(a,b) in zip(['validation','test'],splits[1:]):
  X,_=build_sequences(features[a:b]);pred=restore_predictions(model.predict(X,verbose=0),scaler)
  normal[name]=[{'date':times[a+INPUT_LEN+i].date().isoformat(),'actual':values[a+INPUT_LEN+i:a+INPUT_LEN+i+24].tolist(),'predicted':pred[i].tolist(),'model_version':'baseline-v2'} for i in range(len(X)) if times[a+INPUT_LEN+i].hour==0]
 val=daily_policy(normal['validation'],float('inf'),float('inf'));v=[r['rolling_wape'] for r in val if r['rolling_wape'] is not None]
 stats=distribution(v);warning,drift=stats['p90'],stats['p95']
 # Freeze thresholds before computing or injecting any drift. P95 is also the absolute candidate Gate.
 test=daily_policy(normal['test'],warning,drift);t=[r['rolling_wape'] for r in test if r['rolling_wape'] is not None]
 analysis={'old_policy':{'gate':8.,'drift':10.},'normal_validation':stats,'normal_test':distribution(t),
           'recommended':{'warning':warning,'drift':drift,'gate':drift,'source':'Validation daily-origin rolling7-day P90/P95; gate=P95, plus improvement on independent gate'},
           'false_alarms':{'evaluation_days':len(t),'warning_days':sum(r['rolling_wape'] is not None and r['rolling_wape']>=warning for r in test),'drift_exceed_days':sum(r['rolling_wape'] is not None and r['rolling_wape']>=drift for r in test),'drift_events':sum(r['event'] for r in test)},
           'old_policy_test_days_at_or_above_10':sum(w>=10 for w in t),'old_gate_test_days_above_8':sum(w>8 for w in t)}
 (out/'threshold_analysis.json').write_text(json.dumps(analysis,indent=2))
 (out/'normal_daily.json').write_text(json.dumps(normal))
 with (out/'normal_rolling_wape.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['split','date','rolling_wape','production_model','status']);writer.writeheader()
  for name,days in normal.items():
   for r in daily_policy(days,warning,drift):writer.writerow({'split':name,'date':r['date'],'rolling_wape':r['rolling_wape'],'production_model':'LSTM-v2','status':r['status']})
 print(json.dumps(analysis,indent=2),flush=True)

if __name__=='__main__':main()
