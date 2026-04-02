import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from src.trajectory.haversine import harversine_torch

def check_paths_city_bus_stops_GPU(paths: dict, bus_stops: pd.DataFrame, bus_threshold_m: int, device: torch.device) -> dict:
    """
    실측된 이동 경로 중 공항 버스를 이용하기 전 경로에 대해 도시버스 정류장을 매핑
    paths의 각 경로에 대해 bus_stops와의 거리를 계산하여 bus_threshold_m 이내에 있는 버스 정류장을 매핑
    매핑된 정류장을 지나는 버스 노선 데이터 추출
    passed_bus_info: [time, bus_id, station_name, bus_type]

    Args:
        paths (dict): 공항버스 이용하기 전의 실측된 이동 경로
        bus_stops (pd.DataFrame): 도시 버스 노선 및 정류장 데이터
        bus_threshold_m (int): 버스 정류장 매핑 판단 기준
        device (torch.device): 기본 GPU, GPU 없으면 CPU

    Returns:
        dict: {TRIP_NO: passed_bus_info}
    """
    bus_result = {}

    # 가상정류장 제거
    bus_stops = bus_stops[~bus_stops['정류소명'].str.contains(r'\(가상\)|\(미정차\)', regex=True)]
      
    bus_stop_coords = torch.tensor(bus_stops[['X좌표', 'Y좌표']].values).to(device)
    
    for person_code, path in tqdm(paths.items(), total=len(paths), desc='Checking paths', position=1, leave=False):
        path = np.array(path)
        passed_bus_info = []
        
        times = path[:, 1]  # 시간
        coords = path[:, 2:-1].astype(float)
        
        # 기준치 이내에 있는 버스 정류장 매핑
        path_points = torch.tensor(coords).to(device)
        bus_distances = harversine_torch(path_points, bus_stop_coords)
        near_bus_stops = bus_distances <= bus_threshold_m
        torch.cuda.empty_cache()

        # 매핑된 정류장을 바탕으로 해당 정류장을 지나는 버스 노선 정보 추출
        for i in range(len(times)):
            time = times[i]
            busroot = bus_stops['노선명'].values[near_bus_stops[i].cpu()]
            busstop_name = bus_stops['정류소명'].values[near_bus_stops[i].cpu()]
            
            if busroot.size > 0:
                passed_bus_info.append([time, list(set(busroot)), list(set(busstop_name)), '일반버스'])
                      
        if not passed_bus_info or len(passed_bus_info) == 1:
            continue

        bus_result[person_code] = passed_bus_info   
                
    return bus_result