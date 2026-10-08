"""Initialize a durable Docker sandbox and relocate only its copied MLflow paths."""
import os,shutil,sqlite3
from pathlib import Path
from urllib.parse import unquote,urlparse
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ['AIOPS_RUN_DIR'])


def relocate(value):
    if not value or value.startswith('models:'):return value
    path=unquote(urlparse(value).path) if value.startswith('file:') else value
    if '/artifacts/' not in path:return value
    mapped=ROOT/'artifacts'/path.split('/artifacts/',1)[1]
    return mapped.as_uri() if value.startswith('file:') else str(mapped)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for source in (ROOT/'artifacts/docker/seed').iterdir():
        target=OUT/source.name
        if not target.exists():shutil.copy2(source,target)
    db=sqlite3.connect(OUT/'mlflow.db')
    for table,columns in {'experiments':['artifact_location'],'runs':['artifact_uri'],'logged_models':['artifact_location'],'model_versions':['source','storage_location']}.items():
        for column in columns:
            for rowid,value in db.execute(f'SELECT rowid,{column} FROM {table}').fetchall():
                mapped=relocate(value)
                if mapped!=value:db.execute(f'UPDATE {table} SET {column}=? WHERE rowid=?',(mapped,rowid))
    db.execute('UPDATE experiments SET artifact_location=? WHERE name=?',((OUT/'mlflow_artifacts').as_uri(),'renewable-aiops'))
    db.commit();db.close()
    print('Docker MLflow paths ready; persistent sandbox initialized without retraining',flush=True)
    os.execvp('python',['python','-m','uvicorn','serving_app.aiops_main:app','--host','0.0.0.0','--port','8088','--workers','1'])

if __name__=='__main__':main()
