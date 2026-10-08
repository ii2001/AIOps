# 팀 프로젝트 Docker 대시보드

## 실행

Docker Desktop을 실행한 뒤 팀 프로젝트 루트에서:

```bash
# 처음 빌드하거나 호스트 운영 정보를 새로 이미지에 넣을 때
.venv/bin/python scripts/prepare_docker.py
docker compose up -d --build

# 이미 빌드된 상태에서 다시 시작
docker compose up -d
```

- 대시보드: http://localhost:8099
- Swagger: http://localhost:8099/docs
- 컨테이너: `renewable-energy-aiops`
- 이미지: `renewable-energy-aiops:latest`
- 내부 포트: 8088, 호스트 포트: 8099

로컬 Python 대시보드의8088과 함께 사용할 수 있다. 컨테이너는 Docker Desktop이 실행 중일 때 동작한다.

## 확인 / 중지

```bash
docker compose ps
docker compose logs -f --tail=100
docker compose down
```

`down`은 운영 볼륨을 지우지 않는다. 실제 저장 상태를 유지하려면 `down -v`로 볼륨을 삭제하지 않는다. 재시작 정책은 `unless-stopped`이다.

## 기존 데이터 보존

이미 검증된 모델을 이미지에 넣으므로 빌드 시 학습하거나 지표를 새로 만들지 않는다. host Registry는 SQLite read-only backup으로 seed를 만들고, container의 복사본에만 artifact 경로를 /app 기준으로 교정한다. 예전 Run/model/scaler/CSV/실험은 유지한다.

운영 상태는 `renewable-energy-aiops_renewable_energy_state` named volume에 저장한다. 초기값은 파일이 없는 첫 기동에서만 복사한다. 컨테이너를 다시 생성해도 forecast/actual/Registry/new artifact가 유지된다. 새로운 MLflow artifact는 volume 안에 기록하고 원본 host 경로에 쓰지 않는다.

`.dockerignore`로 .venv, Git, cache, 실제 .env, review_bundle, ZIP 및 Docker 검증 로그를 제외한다. 실험 JSON/CSV와 모델은 대시보드 및 추론을 위해 이미지에 포함한다. timezone은 Asia/Seoul이다. 모델파일이나 전체artifact가 없는 경량review ZIP만으로는 이 이미지를 다시 빌드할 수 없다.

## 실제 검증 결과

Docker image build, healthy startup, MLflow model/scaler load, 실제24개finite/비음수MWh 예측 HTTP200,167개422,대시보드/Swagger200 및 실제브라우저테스트 모두PASS. 컨테이너 force-recreate 후 별도probe로named-volume보존을확인했다.

검증: `scripts/check_docker.py`, 기록: `artifacts/docker/validation.json`, build/container/browser PNG 및 log.

## 기존 HAIC 이미지 정리

우리 컨테이너 검증 후 `serving_app-serving-app:latest`, `haic-serving:before-ui-20261007` 두 HAIC 이미지만 삭제했다. 다른 프로젝트의 컨테이너/이미지/DB/volume과 개인 HAIC 소스 파일은 변경하지 않았다.
