"""Launch a durable local preview with copies of Registry/state; prior evidence stays untouched."""
import json,os,shutil,socket,sqlite3,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/ui';OPERATIONS=OUT/'operations';SOURCE=ROOT/'artifacts/aiops/scenario_2'


def main():
    OUT.mkdir(exist_ok=True);OPERATIONS.mkdir(exist_ok=True)
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1',8088))==0:
            import requests
            r=requests.get('http://127.0.0.1:8088/dashboard/summary',timeout=10)
            if r.status_code==200:print('Dashboard already running: http://127.0.0.1:8088');return
            raise RuntimeError('Port8088 is used by another service; existing process preserved')
    if not (OPERATIONS/'mlflow.db').exists():
        source=sqlite3.connect(SOURCE/'mlflow.db');dest=sqlite3.connect(OPERATIONS/'mlflow.db');source.backup(dest);source.close()
        # Existing logged models still point to original immutable artifacts; new runs write only to sandbox.
        dest.execute('UPDATE experiments SET artifact_location=? WHERE name=?',((OPERATIONS/'mlflow_artifacts').as_uri(),'renewable-aiops'));dest.commit();dest.close()
    for name in ['threshold_analysis.json','demo_result.json','runtime_state.json','aiops.log']:
        if not (OPERATIONS/name).exists():shutil.copy2(SOURCE/name,OPERATIONS/name)
    if not (OPERATIONS/'drift_result.json').exists():shutil.copy2(ROOT/'artifacts/aiops/drift_result.json',OPERATIONS/'drift_result.json')
    env={**os.environ,'AIOPS_RUN_DIR':str(OPERATIONS),'AIOPS_STATE_PATH':str(OPERATIONS/'runtime_state.json'),
         'AIOPS_SCENARIO2_DEMO':'1','PYTHONDONTWRITEBYTECODE':'1'}
    with (OUT/'server.log').open('a') as log:
        proc=subprocess.Popen([sys.executable,'-m','uvicorn','serving_app.aiops_main:app','--host','0.0.0.0','--port','8088','--workers','1'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    import requests
    for _ in range(600):
        if proc.poll() is not None:raise RuntimeError('Server exited; inspect artifacts/ui/server.log')
        try:
            if requests.get('http://127.0.0.1:8088/health',timeout=1).status_code==200:
                (OUT/'server_process.json').write_text(json.dumps({'pid':proc.pid,'url':'http://127.0.0.1:8088','state':str(OPERATIONS)}));print('Dashboard running: http://127.0.0.1:8088');return
        except requests.ConnectionError:pass
        time.sleep(.1)
    raise TimeoutError('Server readiness timeout')

if __name__=='__main__':main()
