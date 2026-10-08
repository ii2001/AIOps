from fastapi import APIRouter
from serving_app.model_loader import load_assets
from src.config import MODEL_VERSION, TARGET_REGION, TARGET_FUEL, INPUT_LEN, OUTPUT_LEN

router = APIRouter()


@router.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': bool(load_assets.cache_info().currsize)}


@router.get('/model/info')
def model_info():
    return {'model_version': MODEL_VERSION, 'region': TARGET_REGION, 'fuel': TARGET_FUEL,
            'input_hours': INPUT_LEN, 'output_hours': OUTPUT_LEN}
