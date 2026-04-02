import pandas as pd
import torch
from tqdm import tqdm
from src.trajectory.haversine import harversine_torch

def check_paths_air_bus_stops_GPU(paths: dict, bus_stops: pd.DataFrame, bus_threshold_m: int, device: torch.device) -> dict:
    """
    paths의 각 경로에 대해 bus_stops와의 거리를 계산하여 bus_threshold_m 이내에 있는 버스 정류장을 매핑
    매핑된 정류장을 지나는 버스 노선 데이터 추출
    도착 후 동일 장소에 오래 머물러 매핑된 중복된 정류장 제거
    최대 빈도의 버스 노선 중 마지막이 공항 정류장이고, 중간에 다른 정류장이 있는 노선을 실제 이동 경로의 버스 노선으로 판단
    passed_bus_info: [time, bus_id, ars_id, station_name, bus_type]

    Args:
        paths (dict): 실측된 이동 경로들
        bus_stops (pd.DataFrame): 공항 버스 노선 및 정류장 데이터
        bus_threshold_m (int): 버스 정류장 매핑 판단 기준
        device (torch.device): 기본 GPU, GPU 없으면 CPU

    Returns:
        dict: {TRIP_NO: passed_bus_info}
    """
    bus_result = {}
    
    # 가상정류장, 미정차 정류장 제거
    bus_stops = bus_stops[~bus_stops['정류소명'].str.contains(r'\(가상\)|\(미정차\)', regex=True)]
    arrived_bus_stops = bus_stops[bus_stops['정류소명'].str.contains('인천공항') &
                                     ~bus_stops['정류소명'].str.contains(r'\(가상\)|\(미정차\)|KT|전망대|검역소|충전소|주차장', regex=True)]['ARS_ID'].unique()
    
    bus_stop_coords = torch.tensor(bus_stops[['X좌표', 'Y좌표']].values).to(device)
    
    for person_code, path in tqdm(paths.items(), total=len(paths), desc='Checking paths', position=1, leave=False):
        passed_bus_info = []
        visited_bus_stops = set()

        times = path[:, 1]
        coords = path[:, 2:].astype(float)
        
        # 기준치 이내에 있는 버스 정류장 매핑
        path_points = torch.tensor(coords).to(device)
        bus_distances = harversine_torch(path_points, bus_stop_coords)
        near_bus_stops = bus_distances <= bus_threshold_m
        torch.cuda.empty_cache()

        # 매핑된 정류장을 바탕으로 해당 정류장을 지나는 버스 노선 정보 추출
        for i in range(len(times)):
            time = times[i]
            busroot = bus_stops['노선명'].values[near_bus_stops[i].cpu()]
            busstop = bus_stops['ARS_ID'].values[near_bus_stops[i].cpu()]
            busstop_name = bus_stops['정류소명'].values[near_bus_stops[i].cpu()]
            visited_bus_stops.update(set(busstop))
    
            if busroot.size > 0:
                passed_bus_info.append([time, list(set(busroot)), list(set(busstop)), list(set(busstop_name)), '공항버스'])

        # 도착 정류장 이후 중복 정류장 제거
        for i in range(len(passed_bus_info) -1, 0, -1):
            currentstop = passed_bus_info[i][3]
            nextstop = passed_bus_info[i-1][3]
            if currentstop == nextstop:
                del passed_bus_info[i]
            else:
                break 
        
        if not passed_bus_info or len(passed_bus_info) == 1:
            continue
        
        # 최대 빈도 노선 필터링
        all_bus_routes = [route for entry in passed_bus_info for route in entry[1]]
        filtered_routes = [route for route in all_bus_routes]
        route_counts = pd.Series(filtered_routes).value_counts()
        most_common_route = route_counts.idxmax() if not route_counts.empty else None
    
        if most_common_route:
            route_bus_stops = bus_stops[bus_stops['노선명'] == most_common_route]
            ars_to_name = dict(zip(route_bus_stops['ARS_ID'], route_bus_stops['정류소명']))
            passed_bus_info = [entry for entry in passed_bus_info if most_common_route in entry[1]]

            for entry in passed_bus_info:
                entry[2] = [stop for stop in entry[2] if stop in ars_to_name]
                entry[3] = [ars_to_name[stop] for stop in entry[2]]
                entry[1] = [most_common_route]
                
        if not any(stop in arrived_bus_stops for stop in passed_bus_info[-1][2]):
            continue

        if not any(stop not in arrived_bus_stops for entry in passed_bus_info for stop in entry[2]):
            continue
                  
        # 버스 시간 차이가 20분 이상일 때 유효 버스 경로로 간주
        if passed_bus_info[-1][0] - passed_bus_info[0][0] > pd.Timedelta(minutes=20):
            bus_result[person_code] = passed_bus_info   
                
    return bus_result