"""LSTM v3 input: observed history, known future calendar, past-only seasonal references."""
from datetime import timedelta
import calendar
import numpy as np
from src.config import INPUT_LEN,OUTPUT_LEN
from src.features import transform
from src.time_features import make_time_features


def forecast_inputs(times,values,scaler):
    if len(times)!=INPUT_LEN or len(values)!=INPUT_LEN:
        raise ValueError('Exactly168 hourly observations required')
    history=make_time_features(times,values,scaler)
    future=[times[-1]+timedelta(hours=i+1) for i in range(OUTPUT_LEN)]
    fractions=np.array([[t.hour/24,t.weekday()/7,t.timetuple().tm_yday/(366 if calendar.isleap(t.year) else 365)] for t in future])
    calendar_features=np.column_stack([np.sin(2*np.pi*fractions[:,j]) if k%2==0 else np.cos(2*np.pi*fractions[:,j]) for k,j in enumerate([0,0,1,1,2,2])]).astype('float32')
    raw=np.asarray(values,dtype='float64').reshape(7,24)
    base=raw.mean(axis=0)
    references=transform(np.column_stack([raw[-1],raw[0],base]),scaler)
    return [history,calendar_features,references,transform(base,scaler)[:,None]]


def windows(times,values,scaler,target_start,target_end):
    if len(times)!=len(values) or not INPUT_LEN<=target_start<target_end<=len(values):
        raise ValueError('Partition requires 168 past hours and targets inside the data')
    # All24 target hours remain inside the declared partition. Context may use already-known past only.
    origins=range(target_start,target_end-OUTPUT_LEN+1)
    inputs=[forecast_inputs(times[i-INPUT_LEN:i],values[i-INPUT_LEN:i],scaler) for i in origins]
    if not inputs:raise ValueError('Empty evaluation partition')
    X=[np.stack([sample[j] for sample in inputs]) for j in range(4)]
    y=np.stack([transform(values[i:i+OUTPUT_LEN],scaler) for i in origins])
    return X,y,list(origins)
