import pandas as pd
from tqdm import tqdm

from src.config.settings import PROCESSED_DATA_DIR

def get_exit_time(trip_group):
    start_emd = trip_group.iloc[0]["EMD_CODE"]
    
    exited = trip_group[trip_group["EMD_CODE"] != start_emd]
    
    if exited.empty:
        return None

    return exited.iloc[0]["DPR_MT1_UNIT_TM"]

AIRPORT_EMD = "28110147"  # 인천공항

def get_airport_entry_time(trip_group):
    airport = trip_group[trip_group["EMD_CODE"] == AIRPORT_EMD]
    
    if airport.empty:
        return None

    return airport.iloc[0]["DPR_MT1_UNIT_TM"]

def get_access_time(trip_group):
    exit_time = get_exit_time(trip_group)
    entry_time = get_airport_entry_time(trip_group)

    if exit_time is None or entry_time is None:
        return None

    return (entry_time - exit_time).total_seconds() / 60  # minutes

def get_boxplot():
    access_times = {}
    
    for file_path in tqdm(list(PROCESSED_DATA_DIR.glob("*.csv")), desc="Creating Box Plot", position=0):
        tripDF = pd.read_csv(file_path)
        
        for trip_no, group in tripDF.groupby("TRIP_NO"):
            start_emd = group.iloc[0]["EMD_CODE"]
            t = get_access_time(group)
            if t is not None:
                access_times[start_emd].append(t)
                
    return access_times
                
def get_outlier(access_times: dict) -> dict:
    bounds = {}
    
    for emd, times in access_times.items():
        s = pd.Series(times)
        Q1 = s.quantile(0.25)
        Q3 = s.quantile(0.75)
        IQR = Q3 - Q1
        
        upper = Q3 + 1.5 * IQR
        bounds[emd] = upper
        
    return bounds

def delete_outlier():
    bounds = get_outlier(get_boxplot())
    
    for file_path in tqdm(list(PROCESSED_DATA_DIR.glob("*.csv")), desc="Delete Routes with Box Plot", position=0):
        tripDF = pd.read_csv(file_path)
        
        for trip_no, group in tripDF.groupby("TRIP_NO"):
            start_emd = group.iloc[0]["EMD_CODE"]
            t = get_access_time(group)
            
        if t is None:
            continue

        if start_emd not in bounds:
            continue

        if t <= bounds[start_emd]:
            valid_trip_nos.add(trip_no)