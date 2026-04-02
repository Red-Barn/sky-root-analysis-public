from tqdm import tqdm

def find_routes_passing_stops(bus: dict) -> dict:
    """
    도시버스의 특성상 겹치는 버스정류장이 많기에 출발지부터 공항버스 탑승 전까지 도시 버스정류장의 교집합을 통해 버스정류장 매핑
    도시버스의 정류장 이동이 최소 3정류장 이상일때만 매핑

    Args:
        bus (dict): 도시버스의 매핑 데이터

    Returns:
        dict: 교집합을 통해 정제된 매핑 데이터
    """
    check_normal_bus = {}
    
    for key, values in tqdm(bus.items(), total=len(bus), desc='Intersecting paths', position=1, leave=False):
        times = [value[0] for value in values]
        roots = [value[1] for value in values]
        stops = [value[2] for value in values]
        types = [value[3] for value in values]
        
        results = []
        current_intersection = roots[0]
        previous_intersection = current_intersection
        temp_stops = [stops[0]]
        temp_times = [times[0]]
        intersection_count = 1      # 교집합이 유지된 횟수 카운트
        
        for root, time, stop in zip(roots[1::], times[1::], stops[1::]):
            if stop == temp_stops[-1]:  # 동일 정류장 연속 시
                continue

            current_intersection = set(current_intersection).intersection(root)
       
            # 새로운 정류장 교집합 계산시 공집합
            if not current_intersection:
                # 전 교집합의 개수가 2개 초과면 저장
                if intersection_count > 2:
                    for (t, s) in zip(temp_times, temp_stops):
                        results.append([t, list(previous_intersection), s, '일반버스'])
                        
                # 교집합 초기화
                current_intersection = root
                previous_intersection = current_intersection
                temp_stops = [stop]
                temp_times = [time]
                intersection_count = 1
            # 새로운 정류장 교집합 계산시 교집합
            else:
                previous_intersection = current_intersection
                temp_stops.append(stop)
                temp_times.append(time)
                intersection_count += 1
        
        if current_intersection and intersection_count > 2:
            for (t, s) in zip(temp_times, temp_stops):
                results.append([t, list(previous_intersection), s, '일반버스'])
        
        if results:        
            check_normal_bus[key] = results
        
    return check_normal_bus   