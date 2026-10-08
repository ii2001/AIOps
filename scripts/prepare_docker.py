"""Read-only host snapshot for Docker. No original Registry or experiment changes."""
import json,shutil,sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'artifacts/ui/operations'
    seed=ROOT/'artifacts/docker/seed';seed.mkdir(parents=True,exist_ok=True)
    if not (source/'mlflow.db').exists():
        source=ROOT/'artifacts/aiops/scenario_2'
    original=sqlite3.connect((source/'mlflow.db').as_uri()+'?mode=ro',uri=True)
    dest=sqlite3.connect(seed/'mlflow.db');original.backup(dest);dest.close();original.close()
    for name in ['threshold_analysis.json','demo_result.json','runtime_state.json','aiops.log']:
        shutil.copy2(source/name,seed/name)
    drift=source/'drift_result.json'
    shutil.copy2(drift if drift.exists() else ROOT/'artifacts/aiops/drift_result.json',seed/'drift_result.json')
    (ROOT/'artifacts/docker/seed_manifest.json').write_text(json.dumps({'source':str(source),'files':[p.name for p in seed.iterdir()],'policy':'SQLite backup; originals read-only; no retraining'},indent=2))
    print('Docker seed snapshot ready')

if __name__=='__main__':main()
