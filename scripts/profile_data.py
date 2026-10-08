import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv, hashlib, json, math
from collections import Counter, defaultdict
from datetime import timedelta
from src.config import ROOT, CSV_PATH, VALUE_COLUMN, TARGET_REGION, TARGET_FUEL
from src.data_loader import read_rows, timestamp


def main():
    rows = read_rows()
    fields = list(rows[0])
    groups = defaultdict(list)
    missing = Counter()
    keys = Counter()
    full_rows = Counter()
    for row in rows:
        for column in fields:
            if row[column] is None or not row[column].strip(): missing[column] += 1
        key = (row['지역'], row['연료원'])
        groups[key].append((timestamp(row), float(row[VALUE_COLUMN])))
        keys[(*key, row['거래일'], int(row['거래시간']))] += 1
        full_rows[tuple(row.values())] += 1
    start = min(timestamp(r) for r in rows)
    end = max(timestamp(r) for r in rows)
    expected = {start + timedelta(hours=i) for i in range(int((end - start).total_seconds() / 3600) + 1)}
    summary = []
    for (region, fuel), points in sorted(groups.items()):
        values = [v for _, v in points]
        summary.append({'region': region, 'fuel': fuel, 'rows': len(points),
                        'missing_hours': len(expected - {t for t, _ in points}),
                        'min': min(values), 'max': max(values), 'mean': math.fsum(values) / len(values),
                        'total': math.fsum(values), 'zeros': sum(v == 0 for v in values)})
    all_values = [float(r[VALUE_COLUMN]) for r in rows]
    profile = {'csv_sha256': hashlib.sha256(CSV_PATH.read_bytes()).hexdigest(), 'encoding': 'cp949',
               'columns': fields, 'rows': len(rows), 'start': start.isoformat(), 'end': end.isoformat(),
               'trading_hours': sorted({int(r['거래시간']) for r in rows}), 'missing': dict(missing),
               'duplicate_keys': sum(v - 1 for v in keys.values()), 'duplicate_rows': sum(v - 1 for v in full_rows.values()),
               'negative_values': sum(v < 0 for v in all_values), 'nonfinite_values': sum(not math.isfinite(v) for v in all_values),
               'min': min(all_values), 'max': max(all_values), 'mean': math.fsum(all_values) / len(all_values), 'groups': summary}
    (ROOT / 'artifacts/data_profile.json').write_text(json.dumps(profile, ensure_ascii=False, indent=2))
    table = '| 지역 | 연료 | 행 수 | 누락시간 | 최소 MWh | 최대 MWh | 평균 MWh | 합계 MWh | 0시간 |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for group in summary:
        table += '| ' + ' | '.join(str(group[k]) if k in ['region', 'fuel', 'rows', 'missing_hours', 'zeros'] else f'{group[k]:.6f}' for k in ['region', 'fuel', 'rows', 'missing_hours', 'min', 'max', 'mean', 'total', 'zeros']) + ' |\n'
    target = next(g for g in summary if g['region'] == TARGET_REGION and g['fuel'] == TARGET_FUEL)
    assert target['missing_hours'] == 0 and target['rows'] == 8760
    assert target['total'] == max(g['total'] for g in summary if g['fuel'] == TARGET_FUEL)
    doc = f'''# 실제 KPX CSV 데이터 프로파일

파일: {CSV_PATH.name}, CP949, 원본 bytes 변경 없음. SHA-256: `{profile['csv_sha256']}`.

- 컬럼: {fields}
- 행 수: {len(rows)}. 거래일: 2025-01-01~2025-12-31, 거래시간 1~24.
- 지역 목록: {sorted({r['지역'] for r in rows})}
- 연료원: {sorted({r['연료원'] for r in rows})}
- 빈 필드: {dict(missing)}. 전체 중복행: {profile['duplicate_rows']}, 지역/연료/날짜/시간 중복키: {profile['duplicate_keys']}.
- 음수: {profile['negative_values']}, NaN/inf: {profile['nonfinite_values']}.
- 전체 전력거래량 min={profile['min']}, max={profile['max']}, mean={profile['mean']} MWh.

## 지역/연료별 통계 및 시간 누락
{table}
## MVP 선택
{TARGET_REGION} / {TARGET_FUEL}: 8760시간 전체, 누락/중복/결측 없음. 태양광 지역 중 연간 총 거래량이 가장 큰 {target['total']:.6f} MWh이므로 선택했다. 0값 {target['zeros']}건은 관측값으로 유지하며 보간하거나 삭제하지 않는다. 밤에도 0 초과 값이 관측될 수 있으며 원본을 그대로 사용한다. 품질 검사상 오류는 없지만 연료별 지역 집계 수준이 다름(태양광 시도별, 풍력 육지/제주)과 CP949 인코딩을 확인했다.

## 시간 의미
원본 거래시간 1~24는 모델 내부에서 거래일 D의 00~23시 **구간 시작 표기**로 매핑한다: D/1→D 00:00, D/24→D 23:00 (Asia/Seoul). 이는 연속 시퀀스를 위한 내부 표기 규약이며 공식 거래구간 종료시각으로 해석한다고 주장하지 않는다. 원본은 수정하지 않는다. 마지막 시각은 2025-12-31 23:00이다.

## 누수 방지와 처리 정책
원시 timestamp를 오름차순 정렬하고 앞70%/다음15%/마지막15%를 먼저 분리한다. 각 파티션 안에서만 168→24 시퀀스를 생성해 입력과 정답 모두 분리 경계를 넘지 않는다. scaler는 Train 원시값에만 fit하고 Validation/Test 및 서빙은 동일 transform을 사용한다. 음수/NaN/inf 입력은 거절하며, 예측 복원값은 물리적 하한0으로만 clip한다. 실제 train/eval 기간과 shape는 artifacts/training_result.json에 기록된다.
'''
    (ROOT / 'docs/data_profile.md').write_text(doc)
    print(json.dumps({'rows': len(rows), 'target': target}, ensure_ascii=False))

if __name__ == '__main__': main()
