import pandas as pd
import numpy as np
import create_trajectory
import haversine
import torch
from tqdm import tqdm

import os

# GPU로 이동
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_name(0))

# 처리할 폴더 경로
folder_path = r'C:\mygit\SkyRoot\compressed_data'

# 결과 저장 폴더
output_folder = r'C:\mygit\SkyRoot\mapping_data'
os.makedirs(output_folder, exist_ok=True)

# 거리 설정
bus_threshold_m = 100

airbusrootDF = pd.read_csv(r"C:\mygit\SkyRoot\open_data\인천공항버스노선별정류소정보")
citybusrootDF = pd.read_csv(r"C:\mygit\SkyRoot\open_data\도시버스노선별정류소정보")

# 경로가 어떤 버스 정류장을 지나는지 확인하는 함수
# GPU 사용
def check_paths_air_bus_stops_GPU(paths, bus_stops, bus_threshold_m=50):
    bus_result = {}
    
    bus_stops = bus_stops[~bus_stops['정류소명'].str.contains(r'\(가상\)|\(미정차\)', regex=True)]
    # 조건에 맞는 ARS_ID 추출
    arrived_bus_stops = bus_stops[bus_stops['정류소명'].str.contains('인천공항') &
                                     ~bus_stops['정류소명'].str.contains(r'\(가상\)|\(미정차\)|KT|전망대|검역소|충전소|주차장', regex=True)]['ARS_ID'].unique()

    
    # 모든 좌표를 텐서로 변환하여 한 번에 처리
    bus_stop_coords = torch.tensor(bus_stops[['X좌표', 'Y좌표']].values).to(device)
    
    for person_code, path in tqdm(paths.items(), total=len(paths), desc='Checking paths'):
        passed_bus_info = []
        visited_bus_stops = set()
        
        # 각 경로의 좌표와 시간 데이터 분리
        times = path[:, 1]  # 시간
        coords = path[:, 2:].astype(float)  # 좌표
        
        # 경로 좌표를 텐서로 변환
        path_points = torch.tensor(coords).to(device)
        
        # 거리 계산을 벡터화하여 한 번에 수행
        bus_distances = haversine.cdist(path_points, bus_stop_coords)  # 각 경로와 버스 정류장 간의 거리 계산
        torch.cuda.empty_cache()
        
        # 설정한 거리 기준, 가까운 정류장 찾기
        near_bus_stops = bus_distances <= bus_threshold_m

        for i in range(len(times)):
            time = times[i]
            busroot = bus_stops['노선명'].values[near_bus_stops[i].cpu()]
            busstop = bus_stops['ARS_ID'].values[near_bus_stops[i].cpu()]
            busstop_name = bus_stops['정류소명'].values[near_bus_stops[i].cpu()]
            
            # 버스 정류장 통과 기록
            visited_bus_stops.update(set(busstop))
            
            # 버스 정류장을 지날 경우 bus_passed_entries에 추가
            if busroot.size > 0:
                passed_bus_info.append([time, list(set(busroot)), list(set(busstop)), list(set(busstop_name)), '공항버스'])
              
        # 뒤에서부터 순회로 도착 중복 정류장 제거
        for i in range(len(passed_bus_info) -1, 0, -1):
            currentstop = passed_bus_info[i][3]
            nextstop = passed_bus_info[i-1][3]
            if currentstop == nextstop:
                del passed_bus_info[i]
            else:
                break 
        
        # passed_bus_info가 비어있거나, 1개만 남으면 데이터 삭제
        if not passed_bus_info or len(passed_bus_info) == 1:
            continue
        
        # # 마지막 ARS_ID가 arrived_bus_stops 에 없을 시 데이터 삭제
        # if not any(stop in arrived_bus_stops for stop in passed_bus_info[-1][2]):
        #     continue
        
        # 모든 노선명을 수집
        all_bus_routes = [route for entry in passed_bus_info for route in entry[1]]

        # # 마지막 항목의 노선명만 수집
        # last_bus_routes = set(passed_bus_info[-1][1])
        
        # 마지막 노선들에 해당하는 모든 노선들의 빈도 계산
        # filtered_routes = [route for route in all_bus_routes if route in last_bus_routes]
        filtered_routes = [route for route in all_bus_routes]
        route_counts = pd.Series(filtered_routes).value_counts()

        # 가장 많이 나타난 노선명 추출
        most_common_route = route_counts.idxmax() if not route_counts.empty else None
    
        # passed_bus_info에서 most_common_route가 없는 항목 삭제
        if most_common_route:
            # most_common_route에 해당하는 정류장 정보 필터링
            route_bus_stops = bus_stops[bus_stops['노선명'] == most_common_route]
            
            # ARS_ID와 정류소명을 딕셔너리로 매핑
            ars_to_name = dict(zip(route_bus_stops['ARS_ID'], route_bus_stops['정류소명']))
            
            # passed_bus_info 필터링 및 업데이트
            passed_bus_info = [entry for entry in passed_bus_info if most_common_route in entry[1]]
            
            # passed_bus_info의 모든 list(set(busroot))를 most_common_route로 변경
            for entry in passed_bus_info:
                # 기존의 busstop(ARS_ID)와 정류소명 업데이트
                entry[2] = [stop for stop in entry[2] if stop in ars_to_name]  # 유효한 ARS_ID만 유지
                entry[3] = [ars_to_name[stop] for stop in entry[2]]  # 해당 정류소명 매핑
        
                # 모든 entry[1]을 most_common_route로 변경
                entry[1] = [most_common_route]
                
        # 마지막 ARS_ID가 arrived_bus_stops 에 없을 시 데이터 삭제
        if not any(stop in arrived_bus_stops for stop in passed_bus_info[-1][2]):
            continue

        # arrived_bus_stops에 있는 정류장을 제외한 정류장이 없을 시 데이터 삭제
        if not any(stop not in arrived_bus_stops for entry in passed_bus_info for stop in entry[2]):
            continue
                  
        # 버스 시간 차이가 20분 이상일 때
        if passed_bus_info[-1][0] - passed_bus_info[0][0] > pd.Timedelta(minutes=20):
            bus_result[person_code] = passed_bus_info   
                
    return bus_result

# 경로가 어떤 버스 정류장을 지나는지 확인하는 함수
# GPU 사용
def check_paths_city_bus_stops_GPU(paths, bus_stops, bus_threshold_m=50):
    bus_result = {}
    bus_stops = bus_stops[~citybusrootDF['정류소명'].str.contains(r'\(가상\)|\(미정차\)', regex=True)]
      
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
        bus_distances = haversine.cdist(path_points, bus_stop_coords)  # 각 경로와 버스 정류장 간의 거리 계산
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

# 한 정거장에 여러 버스가 정거하는 도시 버스 특성상 그 연속성을 교집합으로 계산
def find_routes_passing_stops(bus):
    check_normal_bus = {}
    
    for key, values in bus.items():
        times = [value[0] for value in values]
        roots = [value[1] for value in values]
        stops = [value[2] for value in values]
        types = [value[3] for value in values]
        
        results = []     # 결과들을 저장할 리스트
        current_intersection = roots[0]  # 첫 번째 집합으로 시작
        previous_intersection = current_intersection  # 초기 상태 설정
        temp_stops = [stops[0]]
        temp_times = [times[0]]                 # 첫 번째 시간으로 시작
        intersection_count = 1  # 연속된 교집합의 개수를 세는 변수
        
        for root, time, stop in zip(roots[1::], times[1::], stops[1::]):
            # 이전 경로의 버스 정류장 리스트와 동일한지 확인
            if stop == temp_stops[-1]:
                continue
            # 현재 교집합 계산
            current_intersection = set(current_intersection).intersection(root)
            
            # 공집합이 발생하면 직전 교집합 상태를 저장하고, 새로운 집합으로 교집합을 시작
            if not current_intersection:
                # 공집합이 되기 전 상태를 저장
                if intersection_count > 2:
                    for (t, s) in zip(temp_times, temp_stops):
                        results.append([t, list(previous_intersection), s, '일반버스'])
                        
                # 공집합 발생 시 해당 집합을 새로운 교집합 시작점으로 초기화
                current_intersection = root
                previous_intersection = current_intersection
                temp_stops = [stop] # 새로운 교집합의 정류장 설정
                temp_times = [time] # 새로운 교집합의 시간 설정
                intersection_count = 1 # 교집합의 개수 초기화
            else:
                # 공집합이 아닐 때만 `previous_intersection` 업데이트
                previous_intersection = current_intersection
                temp_stops.append(stop)
                temp_times.append(time)
                intersection_count += 1
        
        # 마지막 남은 공집합이 아닌 교집합 추가
        if current_intersection and intersection_count > 2:
            for (t, s) in zip(temp_times, temp_stops):
                results.append([t, list(previous_intersection), s, '일반버스'])
        
        if results:        
            check_normal_bus[key] = results
        
    return check_normal_bus   


for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)

        # 공항버스
        peopleDF = pd.read_csv(file_path)
        peopletraj = create_trajectory.normal_paths(peopleDF)
        bus = check_paths_air_bus_stops_GPU(peopletraj, airbusrootDF, bus_threshold_m)
        
        # TRIP_NO를 기준으로 그룹화하여 반복 처리
        output = pd.DataFrame()
        output = peopleDF[peopleDF['TRIP_NO'].isin(list(bus.keys()))].copy()

        output['BUS_ID'] = None
        output['STATION'] = None
        output['TRANSPORT_TYPE'] = None

        # TRIP_NO별 데이터 업데이트
        for trip_no, values in bus.items():
            # busroot = list(set(values[-1][1]).intersection(values[0][1]))
            for value in values:
                mask = (output['TRIP_NO'] == trip_no) & (pd.to_datetime(output['DPR_MT1_UNIT_TM']) == value[0])
                if value[4] == '공항버스':
                    output.loc[mask, 'BUS_ID'] = ', '.join(map(str, value[1]))
                    output.loc[mask, 'STATION'] = ', '.join(map(str, value[3]))
                    output.loc[mask, 'TRANSPORT_TYPE'] = value[4]
                    
        # filtered_groups = []

        # # 그룹화 및 필터링
        # for trip_no, group in output.groupby('TRIP_NO'):
        #     if trip_no in bus:
        #         # 조건에 맞는 행만 필터링
        #         filtered_group = group[group['DPR_MT1_UNIT_TM'] <= bus[trip_no][-1][0]]
        #         filtered_groups.append(filtered_group)

        # # 그룹화 해제 (모든 그룹을 다시 하나의 데이터프레임으로 결합)
        # output = pd.concat(filtered_groups, ignore_index=True)
        
        # 도시버스
        peopleDF = output
        peopletraj = create_trajectory.transport_path(peopleDF)
        bus = check_paths_city_bus_stops_GPU(peopletraj, citybusrootDF, bus_threshold_m)
        check_normal_bus = find_routes_passing_stops(bus)

        # TRIP_NO를 기준으로 그룹화하여 반복 처리
        output = pd.DataFrame()
        output = peopleDF

        # TRIP_NO별 데이터 업데이트
        for trip_no, values in check_normal_bus.items():
            # busroot = list(set(values[-1][1]).intersection(values[0][1]))
            for value in values:
                mask = (output['TRIP_NO'] == trip_no) & (pd.to_datetime(output['DPR_MT1_UNIT_TM']) == value[0])
                if value[3] == '일반버스':
                    output.loc[mask, 'BUS_ID'] = ', '.join(map(str, value[1]))
                    output.loc[mask, 'STATION'] = ', '.join(map(str, value[2]))
                    output.loc[mask, 'TRANSPORT_TYPE'] = value[3]

        # 결과를 저장
        output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.csv")
        output.to_csv(output_file, index=False)