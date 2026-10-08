from typing import Annotated
from pydantic import BaseModel, Field
from src.config import INPUT_LEN

NonnegativeMWh = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class PredictRequest(BaseModel):
    sequence: list[NonnegativeMWh] = Field(min_length=INPUT_LEN, max_length=INPUT_LEN)


class HourlyPrediction(BaseModel):
    hour: int
    mwh: NonnegativeMWh


class PredictResponse(BaseModel):
    predictions: list[HourlyPrediction]
    model_version: str
    region: str
    fuel: str
