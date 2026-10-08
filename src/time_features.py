import calendar
import numpy as np
from src.features import transform

FEATURE_NAMES = ['normalized_mwh', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'day_of_year_sin', 'day_of_year_cos']


def make_time_features(times, values, scaler):
    """Shared training/future-serving transformation; timestamps must be hourly, aware KST."""
    if len(times) != len(values) or not times:
        raise ValueError('Matching nonempty timestamps and MWh are required')
    if any(t.tzinfo is None or t.utcoffset().total_seconds() != 9 * 3600 or t.minute or t.second or t.microsecond for t in times):
        raise ValueError('Use hourly Asia/Seoul timestamps (UTC+09:00)')
    if any((b-a).total_seconds() != 3600 for a,b in zip(times,times[1:])):
        raise ValueError('Timestamps must be consecutive hours')
    values = np.asarray(values)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('MWh must be finite and nonnegative')
    hour = np.array([t.hour for t in times]) / 24
    weekday = np.array([t.weekday() for t in times]) / 7
    year_day = np.array([t.timetuple().tm_yday / (366 if calendar.isleap(t.year) else 365) for t in times])
    return np.column_stack([transform(values,scaler),np.sin(2*np.pi*hour),np.cos(2*np.pi*hour),
                            np.sin(2*np.pi*weekday),np.cos(2*np.pi*weekday),
                            np.sin(2*np.pi*year_day),np.cos(2*np.pi*year_day)]).astype('float32')
