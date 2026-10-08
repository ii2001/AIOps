import csv
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
from src.config import CSV_PATH, CSV_ENCODING, TARGET_REGION, TARGET_FUEL, VALUE_COLUMN


def read_rows(path=CSV_PATH):
    with open(path, encoding=CSV_ENCODING, newline='') as file:
        reader = csv.DictReader(file)
        required = {'거래일', '거래시간', '지역', '연료원', VALUE_COLUMN}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing CSV columns: {required - set(reader.fieldnames or [])}')
        return list(reader)


def timestamp(row):
    hour = int(row['거래시간'])
    if not 1 <= hour <= 24:
        raise ValueError(f'Invalid trading hour: {hour}')
    # Internal interval-start convention: D/hour 1 -> D 00:00, D/hour 24 -> D 23:00 KST.
    return datetime.strptime(row['거래일'], '%Y-%m-%d').replace(tzinfo=ZoneInfo('Asia/Seoul')) + timedelta(hours=hour - 1)


def load_target(rows=None):
    rows = read_rows() if rows is None else rows
    selected = sorted((timestamp(r), float(r[VALUE_COLUMN])) for r in rows
                      if r['지역'] == TARGET_REGION and r['연료원'] == TARGET_FUEL)
    if not selected:
        raise ValueError('Target series is empty')
    times, values = zip(*selected)
    values = np.asarray(values, dtype='float64')
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Target values must be finite and nonnegative')
    if any(b - a != timedelta(hours=1) for a, b in zip(times, times[1:])):
        raise ValueError('Target contains duplicate or missing hours; no implicit imputation')
    return list(times), values
