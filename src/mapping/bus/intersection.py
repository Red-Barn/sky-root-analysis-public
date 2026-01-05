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