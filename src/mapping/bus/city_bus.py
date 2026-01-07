import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from src.trajectory.haversine import cdist

# 경로가 어떤 버스 정류장을 지나는지 확인하는 함수
def check_paths_city_bus_stops_GPU(paths, bus_stops, bus_threshold_m=50, device = torch.device('cpu')):
    bus_result = {}
    bus_stops = bus_stops[~bus_stops['정류소명'].str.contains(r'\(가상\)|\(미정차\)', regex=True)]
      
    # 모든 좌표를 텐서로 변환하여 한 번에 처리
    bus_stop_coords = torch.tensor(bus_stops[['X좌표', 'Y좌표']].values).to(device)
    
    for person_code, path in tqdm(paths.items(), total=len(paths), desc='Checking paths'):
        path = np.array(path)
        passed_bus_info = []
        
        # 각 경로의 좌표와 시간 데이터 분리
        times = path[:, 1]  # 시간
        coords = path[:, 2:-1].astype(float)  # 좌표
            
        # 경로 좌표를 텐서로 변환
        path_points = torch.tensor(coords).to(device)
        
        # 거리 계산을 벡터화하여 한 번에 수행
        bus_distances = cdist(path_points, bus_stop_coords)  # 각 경로와 버스 정류장 간의 거리 계산
        torch.cuda.empty_cache()
        
        # 설정한 거리 기준, 가까운 정류장 찾기
        near_bus_stops = bus_distances <= bus_threshold_m

        for i in range(len(times)):
            time = times[i]
            busroot = bus_stops['노선명'].values[near_bus_stops[i].cpu()]
            busstop_name = bus_stops['정류소명'].values[near_bus_stops[i].cpu()]
            
            # 버스 정류장을 지날 경우 bus_passed_entries에 추가
            if busroot.size > 0:
                passed_bus_info.append([time, list(set(busroot)), list(set(busstop_name)), '일반버스'])
                      
        # passed_bus_info가 비어있거나, 1개만 남으면 데이터 삭제
        if not passed_bus_info or len(passed_bus_info) == 1:
            continue

        bus_result[person_code] = passed_bus_info   
                
    return bus_result