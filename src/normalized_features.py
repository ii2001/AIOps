"""Local, past-only amplitude normalization plus optional calendar solar proxy."""
import calendar
from datetime import timedelta
import numpy as np
from src.model_v4 import tree_features
from src.residual_features import forecast_inputs
from src.features import inverse


def solar_proxy(times):
    # ponytail: one regional representative point; site coordinates required for site-level forecasts.
    latitude = np.deg2rad(34.8)
    longitude = 126.8
    hour = np.array([t.hour + .5 for t in times])
    gamma = np.array([2*np.pi/(366 if calendar.isleap(t.year) else 365) *
                      (t.timetuple().tm_yday-1+(t.hour+.5-12)/24) for t in times])
    eqtime = 229.18*(.000075+.001868*np.cos(gamma)-.032077*np.sin(gamma)
                     -.014615*np.cos(2*gamma)-.040849*np.sin(2*gamma))
    decl = (.006918-.399912*np.cos(gamma)+.070257*np.sin(gamma)-.006758*np.cos(2*gamma)
            +.000907*np.sin(2*gamma)-.002697*np.cos(3*gamma)+.00148*np.sin(3*gamma))
    angle = np.deg2rad((hour*60+eqtime+4*longitude-60*9)/4-180)
    zenith_cos = np.sin(latitude)*np.sin(decl)+np.cos(latitude)*np.cos(decl)*np.cos(angle)
    daylight = 24/np.pi*np.arccos(np.clip(-np.tan(latitude)*np.tan(decl), -1, 1))
    return np.column_stack([zenith_cos, np.maximum(zenith_cos, 0), daylight/24, decl])


def normalized_inputs(X, scaler, times, origins, with_solar=False):
    history = inverse(X[0][:, :, 0], scaler)
    scale = np.maximum(np.quantile(history, .95, axis=1), 1.0)
    copied = [a.copy() for a in X]
    copied[0][:, :, 0] = history/scale[:, None]
    features = tree_features(copied).reshape(len(history), 24, -1)
    # Preserve recent cloud/ESS/profile information; no smoothing or deletion of target values.
    recent = np.repeat((history[:, -24:]/scale[:, None])[:, None, :], 24, axis=1)
    amplitude = np.repeat(np.log1p(scale)[:, None, None], 24, axis=1)
    parts = [features, recent, amplitude]
    if with_solar:
        future = [[times[i-1]+timedelta(hours=h+1) for h in range(24)] for i in origins]
        parts.append(np.stack([solar_proxy(window) for window in future]))
    combined = np.concatenate(parts, axis=-1)
    if not np.isfinite(combined).all():
        raise ValueError('Features must be finite')
    return combined.reshape(len(history)*24, -1), scale


def predict_normalized(model, with_solar, times, values, scaler):
    X = forecast_inputs(times, values, scaler)
    features, scale = normalized_inputs([a[None] for a in X], scaler, times, [168], with_solar)
    return np.maximum(model.predict(features)*scale[0], 0)
