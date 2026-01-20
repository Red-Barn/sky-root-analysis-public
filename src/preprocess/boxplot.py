import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from src.config.settings import PROCESSED_DATA_DIR, DATA_DIR
from src.trajectory.builder import group_py_NO

AIRPORT_EMD = 28110147  # 인천공항

def sum_all_dataframes():
    all_tripDF = pd.DataFrame()
    
    results = []
    
    for file_path in tqdm(list(PROCESSED_DATA_DIR.glob("*.csv")), desc="Summing DataFrames", position=0):
        tripDF = pd.read_csv(file_path)
        results.append(tripDF)
        
    all_tripDF = pd.concat(results, ignore_index=True)
    return all_tripDF


def get_exit_time(trip_group):
    start_emd = trip_group.iloc[0]["EMD_CODE"]
    
    exited = trip_group[trip_group["EMD_CODE"] != start_emd]
    
    if exited.empty:
        return None

    return pd.to_datetime(exited.iloc[0]["DPR_MT1_UNIT_TM"])


def get_airport_entry_time(trip_group):
    airport = trip_group[trip_group["EMD_CODE"] == AIRPORT_EMD]
    
    if airport.empty:
        return None

    return pd.to_datetime(airport.iloc[0]["DPR_MT1_UNIT_TM"])


def get_access_time(trip_group):
    exit_time = get_exit_time(trip_group)
    entry_time = get_airport_entry_time(trip_group)

    if exit_time is None or entry_time is None:
        return None

    return (entry_time - exit_time).total_seconds() / 60  # 분 단위 반환


def get_boxplot(trip_df):
    access_times = defaultdict(list)
    
    for trip_no, group in trip_df.groupby("TRIP_NO"):
        start_emd = group.iloc[0]["EMD_CODE"]
        if start_emd == AIRPORT_EMD:  # 인천공항 출발인 경우 제외
            continue
        t = get_access_time(group)
        if t is not None:
            access_times[start_emd].append(t)
                
    return access_times
        
                
def get_outlier(access_times):
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


def outlier_filter(trip_df, bounds):
    filtered_results = []
    grouped = trip_df.groupby("TRIP_NO")
    
    for trip_no, group in tqdm(grouped, total=grouped.ngroups, desc="Filtering Outliers", position=0):
        start_emd = group.iloc[0]["EMD_CODE"]
        if start_emd == AIRPORT_EMD:  # 인천공항 출발인 경우 제외
            continue
        t = get_access_time(group)
        if t is None or t <= bounds[start_emd]:
            filtered_results.append(group)
            
    filtered_trip_df = pd.concat(filtered_results, ignore_index=True)
    
    return filtered_trip_df


def delete_outlier():
    trip_df = sum_all_dataframes()
    access_times = get_boxplot(trip_df)
    bounds = get_outlier(access_times)
    filtered_trip_df = outlier_filter(trip_df, bounds)
    filtered_trip_df.to_csv(DATA_DIR / "processed_all_trips.csv", index=False)