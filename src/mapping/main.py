from pathlib import Path
import pandas as pd
import torch

from config.settings import COMPRESSED_DATA_DIR, MAPPING_DATA_DIR
from data.loader import load_air_bus, load_city_bus
from trajectory.builder import normal_paths, transport_path

from bus.airport_bus import check_paths_air_bus_stops_GPU
from bus.city_bus import check_paths_city_bus_stops_GPU
from bus.intersection import find_routes_passing_stops
from bus.updater import update_air_bus_output, update_city_bus_output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_name(0))

# 거리 설정
bus_threshold_m = 100

def run_mapping():
    airbusrootDF = load_air_bus()
    citybusrootDF = load_city_bus()
    
    for file_path in COMPRESSED_DATA_DIR.glob("*.csv"):
        peopleDF = pd.read_csv(file_path)
        
        # 공항버스
        peopletraj = normal_paths(peopleDF)
        airport_bus = check_paths_air_bus_stops_GPU(peopletraj, airbusrootDF, bus_threshold_m, device)
        
        output = peopleDF[peopleDF['TRIP_NO'].isin(airport_bus.keys())].copy()
        output[['BUS_ID', 'STATION', 'TRANSPORT_TYPE']] = None
        
        update_air_bus_output(output, airport_bus)
                    
        # 도시버스
        peopletraj = transport_path(output)
        city_bus = check_paths_city_bus_stops_GPU(peopletraj, citybusrootDF, bus_threshold_m, device)
        city_bus = find_routes_passing_stops(city_bus)

        update_city_bus_output(output, city_bus)
        
        # 결과를 저장
        output_path = MAPPING_DATA_DIR / file_path.name
        output.to_csv(output_path, index=False)
        
        
if __name__ == "__main__":
    run_mapping()