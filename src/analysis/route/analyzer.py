import ast
import numpy as np
import pandas as pd
from src.analysis.route.improvement import is_improvement_required

    
def analyze_trips(df: pd.DataFrame, ctx) -> pd.DataFrame:
    """
    경로의 dtw distance를 바탕으로 gmm을 계산하여 해당 경로가 개선이 필요한지 판단하는 파이프라인

    Args:
        df (pd.DataFrame): 실제 이동 경로

    Returns:
        improve_required: 개선 필요 여부
        deviation_ratio: 전체 경로의 이탈 비율
        mean_conf: 이탈 분류 점들의 실제 이탈 확률
        longest_deviation: 최대 이탈 길이
        longest_deviation_ratio: 전체 경로의 최대 이탈 거리 비율
        separation: 정상/이탈 군집의 분리도
        is_deviated: 각 거리값이 정싱/이탈인지 bool로 표시(정상: false, 이탈: true)
    """
    results = []

    for _, trip in df.iterrows():
        
        trip_no = trip["TRIP_NO"]
        emd_code = trip["EMD_CODE"]
        best_idx = trip["best_route_idx"]
        dtw = trip["dtw"]
        alignment = ast.literal_eval(trip["alignment"])
        distances = np.array(ast.literal_eval(trip["distances"]), dtype=np.float32)

        # 경로 개선 필요 판별
        improvement = is_improvement_required(distances, policy=ctx.improvement)

        res = {
            "TRIP_NO": trip_no,
            "EMD_CODE": emd_code,
            "best_route_idx": best_idx,
            "dtw": dtw,
            "alignment": alignment,
            "distances": distances.tolist(),
            "improve_required": improvement["need_improvement"],
            "deviation_ratio": improvement["deviation_ratio"],
            "mean_confidence": improvement["mean_confidence"],
            "longest_deviation": improvement["longest_deviation"],
            "longest_deviation_ratio": improvement["longest_deviation_ratio"],
            "separation": improvement["separation"],
            "is_deviated": improvement["is_deviated"],
        }
        
        if res:
            results.append(res)
                
    return pd.DataFrame(results)