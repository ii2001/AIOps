from fastapi import APIRouter, HTTPException
from src.config import TARGET_REGION, TARGET_FUEL, MODEL_VERSION
from serving_app.schemas import PredictRequest, PredictResponse, HourlyPrediction
from serving_app.model_loader import predict as forecast

router = APIRouter()


@router.post('/predict', response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        values = forecast(request.sequence)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictResponse(predictions=[HourlyPrediction(hour=i+1, mwh=value) for i,value in enumerate(values)],
                           model_version=MODEL_VERSION, region=TARGET_REGION, fuel=TARGET_FUEL)
