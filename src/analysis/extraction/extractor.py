import pandas as pd
import traceback
from tqdm import tqdm
from typing import Any

from src.analysis.extraction.generation import get_candidate_routes_info
from src.analysis.extraction.similarity import select_best_route_gpu
from src.data.loader import load_all_api_info

# 실제 Trip 좌표 추출
def extract_actual_trip_coords(df_trip: pd.DataFrame) -> list[tuple[float, float]]:
    df = df_trip.copy()
    df = df.sort_values("DPR_MT1_UNIT_TM")

    return list(zip(df["DPR_CELL_YCRD"], df["DPR_CELL_XCRD"]))  # (lat, lon)


# Trip 단위 최적 경로 추출
def extract_candidate_trip(trip_no: str, df_trip: pd.DataFrame, df_api_info: pd.DataFrame) -> dict[str, Any]:
    """
    캐쉬된 후보 경로들을 추출하고, 추출된 후보 경로 중 최적 경로를 판단하고 선택하는 파이프라인
    
    Args:
        trip_no (str): 실제 이동 경로 번호
        df_trip (pd.DataFrame): 실제 이동 경로
        df_api_info (pd.DataFrame): api 추출된 후보 경로

    Returns:
        candidate_routes: [{TRIP_NO, ROUTE_NO, POINTS}]
        dtw : 최종 dtw 길이
        alignment: 실제 이동경로의 각 좌표를 기준으로 가장 거리가 짧은 최적 경로 좌표의 튜플 정보들
        distances: alignment의 길이 정보
    """
    emd_code = df_trip.iloc[0]["EMD_CODE"]
    actual_coords = extract_actual_trip_coords(df_trip)

    if len(actual_coords) < 10:
        tqdm.write(f"{trip_no}: 좌표 부족({len(actual_coords)}개) -> 스킵")
        return None
    
    # 후보 경로 생성
    candidate_routes = get_candidate_routes_info(trip_no, df_api_info)

    if not candidate_routes:
        tqdm.write("버스 후보 경로 없음 -> 스킵")
        return None

    # 최적 경로 선택
    best_idx, metrics = select_best_route_gpu(actual_coords, candidate_routes)

    return {
        "TRIP_NO": trip_no,
        "EMD_CODE": emd_code,
        "best_route_idx": best_idx,
        "dtw": metrics["dtw"],
        "alignment": metrics["alignment"],
        "distances": metrics["distances"].tolist(),
    }
    
def extract_candidate_trips(df):
    results = []
    df_api_info = load_all_api_info()
    
    grouped = df.groupby("TRIP_NO")
    pbar = tqdm(grouped, total=grouped.ngroups, desc="Extracting Candidate Trips", position=1, leave=False)

    for trip_no, df_trip in pbar:
        try:
            pbar.set_postfix_str(f"ID: {trip_no}")
            
            res = extract_candidate_trip(trip_no, df_trip, df_api_info)
            
            if res:
                results.append(res)
                
        except Exception as e:
            tqdm.write(f"[Error] {trip_no}: {e}")
            tqdm.write(traceback.format_exc())

    return pd.DataFrame(results)