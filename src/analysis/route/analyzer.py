import pandas as pd
import traceback
from datetime import datetime, timedelta
from tqdm import tqdm

from src.analysis.route.generation import get_candidate_routes_info
from src.analysis.route.similarity import select_best_route_gpu
from src.analysis.route.improvement import is_improvement_required
from src.data.loader import load_all_api_info

# 실제 Trip 좌표 추출
def extract_actual_trip_coords(df_trip):
    df = df_trip.copy()

    # 정지 / 체류 제거
    df = df[df["DYNA_MVMT_SPED"] > 3]

    # 시간 정렬
    df = df.sort_values("DPR_MT1_UNIT_TM")

    return list(zip(df["DPR_CELL_YCRD"], df["DPR_CELL_XCRD"]))  # (lat, lon)


# Trip 단위 분석
def analyze_trip(trip_no, df_trip, df_api_info, ctx):
    emd_code = df_trip.iloc[0]["EMD_CODE"]
    actual_coords = extract_actual_trip_coords(df_trip)

    if len(actual_coords) < 10:
        tqdm.write(f"{trip_no}: 좌표 부족({len(actual_coords)}개) -> 스킵")
        return None

    # Step 1: 후보 경로 생성
    candidate_routes = get_candidate_routes_info(trip_no, df_api_info)

    if not candidate_routes:
        tqdm.write("버스 후보 경로 없음")
        return {
            "TRIP_NO": trip_no,
            "has_candidate": False
        }

    # Step 2: 최적 경로 선택
    best_idx, metrics = select_best_route_gpu(actual_coords, candidate_routes, policy=ctx.similarity, device=ctx.device)
    
    # Step 3: 경로 개선 필요 판별
    improvement = is_improvement_required(metrics["distances"], policy=ctx.improvement)

    return {
        "TRIP_NO": trip_no,
        "EMD_CODE": emd_code,
        "has_candidate": True,
        "best_route_idx": best_idx,
        "dtw": metrics["dtw"],
        "aligment": metrics["aligment"],
        "distances": metrics["distances"].tolist(),
        "improve_required": improvement["need_improvement"],
        "deviation_ratio": improvement["deviation_ratio"],
        "mean_confidence": improvement["mean_confidence"],
        "longest_deviation": improvement["longest_deviation"],
        "separation": improvement["separation"]
    }
    
def analyze_trips(df, ctx):
    results = []
    df_api_info = load_all_api_info()
    
    grouped = df.groupby("TRIP_NO")
    pbar = tqdm(grouped, total=grouped.ngroups, desc="Analyzing Trips", position=1, leave=False)

    for trip_no, df_trip in pbar:
        try:
            pbar.set_postfix_str(f"ID: {trip_no}")
            
            res = analyze_trip(trip_no, df_trip, df_api_info, ctx)
            
            if res:
                results.append(res)
                
        except Exception as e:
            tqdm.write(f"[Error] {trip_no}: {e}")
            tqdm.write(traceback.format_exc())

    return pd.DataFrame(results)