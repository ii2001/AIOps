import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv,json,pickle
import numpy as np
from tensorflow import keras
from src.config import ROOT,SEED,INPUT_LEN,OUTPUT_LEN
from src.data_loader import load_target
from src.features import chronological_splits,fit_scaler,inverse,restore_predictions
from src.residual_features import windows
from src.model_v3 import build_model_v3
from src.benchmarks import seasonal_predictions
from src.metrics import evaluate


def main():
 out=ROOT/'artifacts/model_v3'
 if (out/'metrics.json').exists():raise RuntimeError('Existing v3 result preserved; choose another named experiment')
 keras.utils.set_random_seed(SEED);times,values=load_target();splits=chronological_splits(values)
 train_start,train_end=splits[0];val_start,val_end=splits[1];test_start,test_end=splits[2];gate_start=val_end-14*24
 scaler=fit_scaler(values[:train_end])
 with (ROOT/'serving_app/models/scaler_v2.pkl').open('rb') as f:assert pickle.load(f)==scaler
 segments={'train':(train_start+INPUT_LEN,train_end),'early_stopping_calibration':(val_start+INPUT_LEN,gate_start),'gate':(gate_start,val_end),'test':(test_start+INPUT_LEN,test_end)}
 sets={name:windows(times,values,scaler,a,b) for name,(a,b) in segments.items()}
 assert segments['early_stopping_calibration'][1]==segments['gate'][0] and segments['gate'][1]<=segments['test'][0]
 cal_X,cal_y,_=sets['early_stopping_calibration'];cal_actual=inverse(cal_y,scaler)
 cal_naive=restore_predictions(cal_X[3][:,:,0],scaler);cap=evaluate(cal_actual,cal_naive)['wape']
 gate_plan={'cap_wape':cap,'source':'7-day same-hour mean WAPE on early Validation calibration only; frozen before training/Test','gate_start':times[gate_start].isoformat(),'gate_end':times[val_end-1].isoformat(),'require_lower_than_production_and_same_window_naive':True,'business_accuracy_SLA':False}
 (out/'gate_plan.json').write_text(json.dumps(gate_plan,indent=2));print('Frozen Gate plan:',json.dumps(gate_plan),flush=True)
 model=build_model_v3();model.summary()
 history=model.fit(sets['train'][0],sets['train'][1],validation_data=(cal_X,cal_y),epochs=50,batch_size=64,shuffle=False,
                   callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=8,restore_best_weights=True)],verbose=2)
 model.save(ROOT/'serving_app/models/baseline_v3.keras')
 with (ROOT/'serving_app/models/scaler_v3.pkl').open('wb') as f:pickle.dump(scaler,f)
 original=keras.models.load_model(ROOT/'serving_app/models/baseline_v2.keras')
 metrics={};predictions={};references={}
 for name in ['early_stopping_calibration','gate','test']:
  X,y,origins=sets[name];actual=inverse(y,scaler)
  pred=restore_predictions(model.predict(X,verbose=0),scaler)
  old=restore_predictions(original.predict(X[0],verbose=0),scaler)
  raw=np.stack([values[i-168:i] for i in origins]);naive=seasonal_predictions(raw)
  metrics[name]={'v3':evaluate(actual,pred),'v2':evaluate(actual,old),**{k:evaluate(actual,p) for k,p in naive.items()}}
  predictions[name]=pred;references[name]=naive['7-Day Same-Hour Mean']
  np.save(out/f'{name}_predicted.npy',pred)
  print(name,json.dumps(metrics[name]),flush=True)
 test_actual=inverse(sets['test'][1],scaler);assert np.array_equal(test_actual,np.load(ROOT/'artifacts/test_actual.npy'))
 gm=metrics['gate'];passed=gm['v3']['wape']<=cap and gm['v3']['wape']<gm['v2']['wape'] and gm['v3']['wape']<gm['7-Day Same-Hour Mean']['wape']
 result={'model_version':'baseline-v3-experimental','architecture':'history LSTM32 + known future calendar + past day/week/mean -> shared Dense32 -> residual added to seasonal mean','seed':SEED,'max_epochs':50,'epochs_completed':len(history.history['loss']),'best_epoch':int(np.argmin(history.history['val_loss']))+1,'loss':'mae','segments':{n:{'target_start':times[a].isoformat(),'target_end':times[b-1].isoformat(),'samples':len(sets[n][1])} for n,(a,b) in segments.items()},'metrics':metrics,'gate_plan':gate_plan,'gate':'PASS' if passed else 'FAIL','history':history.history,'production_switched':False,'limitations':['single predetermined architecture; no Test-based hyperparameter search','Test already examined in prior experiments; new independent future period still needed','v2 comparator originally used full Validation for EarlyStopping; Gate is withheld only from v3 fitting/selection','multiple-input v3 requires serving/retraining integration before production deployment']}
 (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));(ROOT/'serving_app/models/baseline_v3_metadata.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 with (out/'comparison.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['split','model','mae','rmse','wape'])
  for split,scores in metrics.items():
   for name,score in scores.items():w.writerow([split,name,score['mae'],score['rmse'],score['wape']])
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 names=['v2','7-Day Same-Hour Mean','v3'];fig,ax=plt.subplots(figsize=(9,4));scores=[metrics['test'][n]['wape'] for n in names];bars=ax.bar(names,scores,color=['#94a3b8','#60a5fa','#087f6b'])
 for bar,v in zip(bars,scores):ax.text(bar.get_x()+bar.get_width()/2,v+.3,f'{v:.2f}%',ha='center')
 ax.set(title='Same1123 test origins / all24 horizons / actual WAPE',ylabel='WAPE (%)',ylim=(0,max(scores)*1.2));fig.tight_layout();fig.savefig(out/'wape_comparison.png',dpi=180);plt.close(fig)
 sample=next(i for i,t in enumerate(sets['test'][2]) if times[t].hour==0)
 fig,ax=plt.subplots(figsize=(10,4));ax.plot(range(24),test_actual[sample],label='Actual',marker='o');ax.plot(range(24),references['test'][sample],label='7-day mean');ax.plot(range(24),predictions['test'][sample],label='Residual LSTM v3');ax.set(title=f'{times[sets["test"][2][sample]].date()} real Test day',xlabel='KST hour',ylabel='MWh');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'forecast_comparison.png',dpi=180);plt.close(fig)
 print('Actual V3 Gate:',result['gate'],'Test WAPE:',metrics['test']['v3']['wape'],flush=True)

if __name__=='__main__':main()
