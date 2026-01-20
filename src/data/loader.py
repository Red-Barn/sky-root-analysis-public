import pandas as pd
import geopandas as gpd
from functools import lru_cache
from src.config.settings import OPEN_DATA_DIR, DATA_DIR, RESULT_REGION_DIR, RESULT_TRIP_DIR


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
def load_gpd_emd():
    return gpd.read_file(
        OPEN_DATA_DIR / "emd_WGS84.json"
    )
    
@lru_cache
def load_all_trips():
    return pd.read_csv(
        DATA_DIR / "processed_all_trips.csv"
    )
    
@lru_cache
def load_analysis_trips():
    return pd.read_csv(
        RESULT_TRIP_DIR / "routes_analysis_all_trips.csv"
    )
    
@lru_cache
def load_analysis_region():
    return pd.read_csv(
        RESULT_REGION_DIR / "region_analysis_all_trips.csv"
    )