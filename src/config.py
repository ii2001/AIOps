from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / 'data/raw/한국전력거래소_지역별 시간별 태양광 및 풍력 발전량_20251231.csv'
CSV_ENCODING = 'cp949'
TARGET_REGION = '전라남도'
TARGET_FUEL = '태양광'
VALUE_COLUMN = '전력거래량(MWh)'
INPUT_LEN = 168
OUTPUT_LEN = 24
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
SEED = 42
EPOCHS = 30
MODEL_PATH = ROOT / 'serving_app/models/baseline.keras'
SCALER_PATH = ROOT / 'serving_app/models/scaler.pkl'
MODEL_VERSION = 'baseline-v1'
