import pandas as pd
import traceback
from datetime import datetime, timedelta
from tqdm import tqdm

from src.analysis.route.generation import get_bus_candidate_routes

# 출발 시간 처리
def get_departure_time_for_api(df_trip):
    """
    csv의 시간(HH:MM:SS)만 사용하고 날짜는 내일로 설정
    """
    t = pd.to_datetime(df_trip.iloc[0]["DPR_MT1_UNIT_TM"])
    base = datetime.now() + timedelta(days=1)
    dt = base.replace(hour=t.hour, minute=t.minute, second=0)
    return int(dt.timestamp())


# 실제 Trip 좌표 추출
def extract_actual_trip_coords(df_trip):
    df = df_trip.copy()

    # 정지 / 체류 제거
    df = df[df["DYNA_MVMT_SPED"] > 3]

    # 시간 정렬
    df = df.sort_values("DPR_MT1_UNIT_TM")

    return list(zip(df["DPR_CELL_YCRD"], df["DPR_CELL_XCRD"]))  # (lat, lon)
    
    
def build_total_api_info(df):
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