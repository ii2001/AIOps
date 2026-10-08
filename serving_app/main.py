from math import isfinite
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from serving_app.routers import health, predict

app = FastAPI(title='Renewable Energy AIOps — Baseline MVP')
app.include_router(health.router)
app.include_router(predict.router)


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    # Preserve rejection while safely representing NaN/inf in validation error details.
    detail = jsonable_encoder(exc.errors(), custom_encoder={float: lambda v: v if isfinite(v) else str(v)})
    return JSONResponse(status_code=422, content={'detail': detail})
