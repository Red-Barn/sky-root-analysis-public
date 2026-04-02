import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

from src.config.settings import PROCESSED_DATA_DIR, DATA_DIR

AIRPORT_EMD = 28110147  # 인천공항

def sum_all_dataframes() -> pd.DataFrame:
    all_tripDF = pd.DataFrame()
    results = []
    
    for file_path in tqdm(list(PROCESSED_DATA_DIR.glob("*.csv")), desc="Summing DataFrames", position=0):
        tripDF = pd.read_csv(file_path)
        results.append(tripDF)
        
    all_tripDF = pd.concat(results, ignore_index=True)
    return all_tripDF


def get_exit_time(trip_group: pd.DataFrame) -> pd.Timestamp:
    start_emd = trip_group.iloc[0]["EMD_CODE"]
    exited = trip_group[trip_group["EMD_CODE"] != start_emd]
    
    if exited.empty:
        return None

    return pd.to_datetime(exited.iloc[0]["DPR_MT1_UNIT_TM"])


def get_airport_entry_time(trip_group: pd.DataFrame) -> pd.Timestamp:
    airport = trip_group[trip_group["EMD_CODE"] == AIRPORT_EMD]
    
    if airport.empty:
        return None

    return pd.to_datetime(airport.iloc[0]["DPR_MT1_UNIT_TM"])


def get_airport_exit_time(trip_group: pd.DataFrame) -> pd.Timestamp:
    if trip_group.iloc[-1]["EMD_CODE"] != AIRPORT_EMD:
        return None
    
    return pd.to_datetime(trip_group.iloc[-1]["DPR_MT1_UNIT_TM"])


def get_access_time(trip_group: pd.DataFrame) -> tuple[float, float]:
    exit_time = get_exit_time(trip_group)
    entry_time = get_airport_entry_time(trip_group)
    final_time = get_airport_exit_time(trip_group)

    if exit_time is None or entry_time is None or final_time is None:
        return None, None
    
    tail_time = (final_time - entry_time).total_seconds() / 60  # 인천공항에서 머문 시간 (분 단위)
    body_time = (entry_time - exit_time).total_seconds() / 60  # 인천공항까지 걸린 시간 (분 단위)

    return tail_time, body_time


def get_boxplot(trip_df: pd.DataFrame) -> tuple[list, dict]:
    """boxplot에 필요한 시간 정보 추출"""
    body_access_times = defaultdict(list)
    tail_access_times = list()
    
    for trip_no, group in trip_df.groupby("TRIP_NO"):
        start_emd = group.iloc[0]["EMD_CODE"]
        if start_emd == AIRPORT_EMD:  # 인천공항 출발인 경우 제외
            continue
        tail_time, body_time = get_access_time(group)
        if body_time is not None:
            body_access_times[start_emd].append(body_time)
        if tail_time is not None:
            tail_access_times.append(tail_time)
                
    return tail_access_times, body_access_times
        
                
def get_body_outlier(access_times: dict) -> dict:
    """
    경유 노선을 동일 지역에서 출발하는 모든 경로에 대해 upper를 넘는 시간의 노선으로 판단
    boxplot 데이터를 csv로 저장
    """
    bounds = {}
    results = []
    
    for emd, times in access_times.items():
        Q1 = np.percentile(times, 25)
        Q2 = np.percentile(times, 50)
        Q3 = np.percentile(times, 75)
        min_val = min(times)
        max_val = max(times)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        results.append((emd, min_val, lower, Q1, Q2, Q3, upper, max_val))
        bounds[emd] = upper
        
    df = pd.DataFrame(results, columns=["EMD_CODE", "min", "lower", "Q1", "Q2", "Q3", "upper", "max"])
    df.to_csv(DATA_DIR / "access_time_boxplot.csv", index=False)
        
    return bounds


def get_tail_outlier(access_times: list) -> float:
    """
    공항 지연 데이터를 마지막 EMD 도착후 지난 모든 시간에 대해 upper를 넘는 시간의 노선으로 판단
    boxplot 데이터를 시각화하여 저장
    """
    Q1 = np.percentile(access_times, 25)
    Q2 = np.percentile(access_times, 50)
    Q3 = np.percentile(access_times, 75)
    IQR = Q3 - Q1
    upper = Q3 + 1.5 * IQR
    
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    
    fig, ax = plt.subplots(figsize=(16, 9))
    box = ax.boxplot(access_times, patch_artist=True, vert=False)
    for patch in box["boxes"]:
        patch.set_alpha(0.6)
        
    left_whisker = min(box["whiskers"][0].get_xdata())
    right_whisker = max(box["whiskers"][1].get_xdata())
    
    y = 1
    
    ax.text(Q1, y + 0.12, f"Q1={Q1:.1f}", ha="center", fontsize=9)
    ax.text(Q2, y + 0.20, f"Q2={Q2:.1f}", ha="center", fontsize=9)
    ax.text(Q3, y + 0.12, f"Q3={Q3:.1f}", ha="center", fontsize=9)
    
    ax.text(left_whisker, y - 0.08, f"Min={left_whisker:.1f}", ha="center", fontsize=9)
    ax.text(right_whisker, y - 0.08, f"Max={right_whisker:.1f}", ha="center", fontsize=9)

    for flier in box["fliers"]:
        xs = flier.get_xdata()
        ys = flier.get_ydata()
        for xi, yi in zip(xs, ys):
            ax.text(xi, yi + 0.05, f"{xi:.1f}", ha="center", fontsize=8)

    ax.set_title("마지막 지역 도착 후 인천공항 도달까지 걸리는 시간 분포")
    ax.set_xlabel("시간(분)")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(DATA_DIR / "tail_access_time_boxplot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    return upper


def outlier_filter(trip_df: pd.DataFrame, bounds: dict, upper: float) -> pd.DataFrame:
    """
    body_time : 동일 EMD들의 출발 EMD 탈출 후 마지막 EMD 도착까지 걸리는 시간 리스트의 딕셔너리
    tail_time : 모든 경로의 마지막 EMD 도착 후 경로가 끝날 때까지 걸리는 시간의 리스트
    각 time의 upper를 넘는 경로를 각각 경유 노선, 이상치로 판단하여 제거

    Args:
        trip_df (pd.DataFrame): 실제 이동 데이터프레임
        bounds (dict): body_time 딕셔너리
        upper (float): tail_time 리스트의 upper 값

    Returns:
        pd.DataFrame: 경유 노선 및 이상치가 제거된 데이터프레임
    """
    filtered_results = []
    grouped = trip_df.groupby("TRIP_NO")
    
    for trip_no, group in tqdm(grouped, total=grouped.ngroups, desc="Filtering Outliers", position=0):
        start_emd = group.iloc[0]["EMD_CODE"]
        if start_emd == AIRPORT_EMD:  # 인천공항 출발인 경우 제외
            continue
        tail_time, body_time = get_access_time(group)
        if (body_time is not None and
            tail_time is not None and
            body_time <= bounds[start_emd] and
            tail_time <= upper):
            filtered_results.append(group)
            
    filtered_trip_df = pd.concat(filtered_results, ignore_index=True)
    return filtered_trip_df


def delete_outlier():
    trip_df = sum_all_dataframes()
    tail_access_times, body_access_times = get_boxplot(trip_df)
    bounds = get_body_outlier(body_access_times)
    upper = get_tail_outlier(tail_access_times)
    filtered_trip_df = outlier_filter(trip_df, bounds, upper)
    filtered_trip_df.to_csv(DATA_DIR / "filtered_all_trips.csv", index=False)