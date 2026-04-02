import pandas as pd
import traceback
from datetime import datetime, timedelta
from tqdm import tqdm

from src.analysis.extraction.extractor import extract_actual_trip_coords
from src.analysis.extraction.generation import get_bus_candidate_routes


def get_departure_time_for_api(df_trip: pd.DataFrame) -> int:
    """과거 데이터는 api를 통해 경로를 구할 때 에러가 날 수 있기에 현재 기준 내일 동일시간으로 경로 추출"""
    t = pd.to_datetime(df_trip.iloc[0]["DPR_MT1_UNIT_TM"])
    base = datetime.now() + timedelta(days=1)
    dt = base.replace(hour=t.hour, minute=t.minute, second=0)
    return int(dt.timestamp())
    
    
def build_total_api_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    실제 경로의 출발 좌표, 도착 좌표, 출발 시간으로 Google Map에서 대중교통 최적 경로를 추출

    Args:
        df (pd.DataFrame): 실제 이동 경로

    Returns:
        pd.DataFrame: TRIP_NO 별 Google Map으로 추출한 최적 경로들의 데이터프레임
    """
    results = []
    
    grouped = df.groupby("TRIP_NO")
    pbar = tqdm(grouped, total=grouped.ngroups, desc="Building total api info")

    for trip_no, df_trip in pbar:
        try:
            pbar.set_postfix_str(f"ID: {trip_no}")
            
            actual_coords = extract_actual_trip_coords(df_trip)
            if len(actual_coords) < 10:
                continue
            
            origin_lat, origin_lon = actual_coords[0]
            dest_lat, dest_lon = actual_coords[-1]
            departure_time = get_departure_time_for_api(df_trip)
            
            candidate_total_info = get_bus_candidate_routes(trip_no, origin_lat, origin_lon, dest_lat, dest_lon, departure_time)

            results.extend(candidate_total_info)
            
        except Exception as e:
            tqdm.write(f"[Error] {trip_no}: {e}")
            tqdm.write(traceback.format_exc())
            
    return pd.DataFrame(results)