"""Daily-only rolling policy; thresholds are estimated before any drift injection."""
from datetime import date, timedelta
import numpy as np
from src.metrics import evaluate


def distribution(values):
    values=np.asarray(values,dtype=float)
    if not len(values) or not np.isfinite(values).all():raise ValueError('Finite nonempty calibration WAPE required')
    return {**{name:float(fn(values)) for name,fn in [('mean',np.mean),('median',np.median),('std',np.std),('min',np.min),('max',np.max)]},
            **{f'p{p}':float(np.percentile(values,p)) for p in [75,90,95]}}


def daily_policy(days, warning, drift):
    """Input chronological complete daily records. A gap resets consecutive-days count; event once per episode."""
    results=[];streak=0;previous=None
    for i,day in enumerate(days):
        current=date.fromisoformat(day['date'])
        if previous is not None and current<=previous:raise ValueError('Duplicate or unordered daily actuals')
        if previous is None or (current-previous).days!=1:streak=0
        previous=current
        window=days[max(0,i-6):i+1]
        complete=len(window)==7 and (current-date.fromisoformat(window[0]['date'])).days==6
        wape=evaluate([d['actual'] for d in window],[d['predicted'] for d in window])['wape'] if complete else None
        if wape is None:streak=0;status='insufficient_data'
        else:
            streak=streak+1 if wape>=drift else 0
            status='drift' if streak>=3 else 'warning' if wape>=warning else 'healthy'
        results.append({'date':day['date'],'rolling_wape':wape,'consecutive_drift_days':streak,'status':status,'event':streak==3})
    return results
