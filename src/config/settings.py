from dotenv import load_dotenv
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]  # sky-root-analysis

DATA_DIR = BASE_DIR / "data"
RESULT_DIR = BASE_DIR / "result"
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

# API_KEYS
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# PATHs
OPEN_DATA_DIR = DATA_DIR / "open"

RAW_DATA_DIR = DATA_DIR / "raw"

INTERIM_DATA_DIR = DATA_DIR / "interim"
COMPRESSED_DATA_DIR = INTERIM_DATA_DIR / "compressed"
MAPPING_DATA_DIR = INTERIM_DATA_DIR / "mapping"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULT_TRIP_DIR= RESULT_DIR / "trip"
RESULT_REGION_DIR= RESULT_DIR / "region"
RESULT_SENSITIVITY_DIR = RESULT_DIR / "sensitivity"