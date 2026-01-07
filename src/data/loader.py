import pandas as pd
from functools import lru_cache
from src.config.settings import OPEN_DATA_DIR, RAW_DATA_DIR, COMPRESSED_DATA_DIR, MAPPING_DATA_DIR, PROCESSED_DATA_DIR, RESULT_TRIP_DIR, RESULT_REGION_DIR, RESULT_SENSITIVITY_DIR


@lru_cache
def load_city_bus():
    return pd.read_csv(
        OPEN_DATA_DIR / "도시버스노선별정류소정보"
    )
      
@lru_cache
def load_air_bus():
    return pd.read_csv(
        OPEN_DATA_DIR / "인천공항버스노선별정류소정보"
    )
    
@lru_cache
def load_emd():
    return pd.read_json(
        OPEN_DATA_DIR / "emd_WGS84.json"
    )
    
@lru_cache
def load_raw_data():
    return pd.read_csv(
        RAW_DATA_DIR / "2024-08-19.csv"
    )
    
@lru_cache
def load_compressed_data():
    return pd.read_csv(
        COMPRESSED_DATA_DIR / "2024-08-19.csv"
    )
    
@lru_cache
def load_mapping_data():
    return pd.read_csv(
        MAPPING_DATA_DIR / "2024-08-19.csv"
    )
    
@lru_cache
def load_processed_data():
    return pd.read_csv(
        PROCESSED_DATA_DIR / "2024-08-19.csv"
    )