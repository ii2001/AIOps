# GitHub 저장소와 로컬 실행 데이터

이 저장소는 코드·문서·작은 실제 결과(reports/)를 관리한다. `.gitignore`로 가상환경, 실제 .env, 원본/processed 데이터, 모델/scaler, MLflow DB/artifact, runtime state, 전체로그와 review ZIP을 제외했다. 기존 로컬 파일은 삭제하거나 이동하지 않았다.

## Clone 후 기본 환경

```bash
git clone https://github.com/ii2001/AIOps.git
cd AIOps
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-aiops.txt
```

원본 KPX CSV를 `data/raw/`에 준비한 뒤 기본 학습은 다음 명령으로 생성할 수 있다.

```bash
python scripts/profile_data.py
python scripts/train_baseline.py
python scripts/train_baseline_v2.py
python scripts/compare_models.py
```

## AIOps / Dashboard / Docker의 입력

Git clone만으로 기존 trained model, MLflow Registry, 실행 state가 복원되지는 않는다. 기존 실험을 그대로 실행하려면 신뢰할 수 있는 로컬 백업의 모델/scaler와 `artifacts/aiops/`, 필요한 비교 산출물을 복원해야 한다. 새 실험은 analyze_aiops.py → run_aiops.py → run_aiops_demo.py 흐름을 사용하지만, 원래 실험의 bias/snapshot 등 준비 산출물도 필요하다. 일부 검증 스크립트는 수집된 실제 결과를 검사하므로 빈 환경에서 곧바로 모두 실행된다고 주장하지 않는다.

기존 로컬 데이터가 준비된 환경:

```bash
python scripts/start_dashboard.py
python scripts/prepare_docker.py
docker compose up -d --build
```

Dashboard: localhost:8088. Docker: localhost:8099. 경량 review ZIP은 Docker build context를 대체하지 않는다.

브라우저 검증이 필요하면 개발 환경에 Playwright를 설치한다(`pip install playwright`, `python -m playwright install chromium`). Docker 실행에는 Playwright가 필요하지 않다.

README는 Baseline 단계의 설명을 포함하며, 최신 결과는 reports/와 추가된 docs/aiops/PRESENTATION_AIOPS_FINAL_FLOW.md를 함께 확인한다. 과거 문서의 artifacts 링크는 로컬 실행 결과를 가리켜 GitHub에서 열리지 않을 수 있다. GitHub에서는 reports/README.md의 링크로 실제 결과를 확인한다.
