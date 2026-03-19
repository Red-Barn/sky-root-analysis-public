import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from src.trajectory.haversine import harversine_torch

def check_paths_city_bus_stops_GPU(paths, bus_stops, bus_threshold_m=50, device = torch.device('cpu')):
    bus_result = {}

    # 가상정류장 제거
    bus_stops = bus_stops[~bus_stops['정류소명'].str.contains(r'\(가상\)|\(미정차\)', regex=True)]
      
    bus_stop_coords = torch.tensor(bus_stops[['X좌표', 'Y좌표']].values).to(device)
    
    for person_code, path in tqdm(paths.items(), total=len(paths), desc='Checking paths', position=1, leave=False):
        path = np.array(path)
        passed_bus_info = []
        
        times = path[:, 1]  # 시간
        coords = path[:, 2:-1].astype(float)  # 좌표
        
        path_points = torch.tensor(coords).to(device)
        bus_distances = harversine_torch(path_points, bus_stop_coords)  # harversine 거리 계산
        near_bus_stops = bus_distances <= bus_threshold_m
        torch.cuda.empty_cache()

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