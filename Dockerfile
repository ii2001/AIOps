FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements-aiops.txt ./
RUN pip install --no-cache-dir -r requirements-aiops.txt
COPY src/ src/
COPY scripts/ scripts/
COPY serving_app/ serving_app/
COPY data/ data/
COPY artifacts/ artifacts/
ENV TZ=Asia/Seoul \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AIOPS_RUN_DIR=/app/artifacts/docker/operations \
    AIOPS_STATE_PATH=/app/artifacts/docker/operations/runtime_state.json \
    AIOPS_SCENARIO2_DEMO=1
EXPOSE 8088
CMD ["python", "scripts/docker_entrypoint.py"]
