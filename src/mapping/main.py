import pandas as pd
from tqdm import tqdm

from src.config.settings import COMPRESSED_DATA_DIR, MAPPING_DATA_DIR
from src.config.runtime import create_runtime_context
from src.data.loader import load_air_bus, load_city_bus
from src.trajectory.builder import normal_paths, transport_path

from src.mapping.bus.airport_bus import check_paths_air_bus_stops_GPU
from src.mapping.bus.city_bus import check_paths_city_bus_stops_GPU
from src.mapping.bus.intersection import find_routes_passing_stops
from src.mapping.bus.updater import update_air_bus_output, update_city_bus_output

def run_mapping():
    ctx = create_runtime_context(verbose=True)
    
    airbusrootDF = load_air_bus()
    citybusrootDF = load_city_bus()
    
    for file_path in tqdm(list(COMPRESSED_DATA_DIR.glob("*.csv")), desc="Mapping files", position=0):
        peopleDF = pd.read_csv(file_path)
        
        # 공항버스
        peopletraj = normal_paths(peopleDF)
        airport_bus = check_paths_air_bus_stops_GPU(peopletraj, airbusrootDF, ctx.distance.bus_threshold_m, ctx.device)
        
        output = peopleDF[peopleDF['TRIP_NO'].isin(airport_bus.keys())].copy()
        output[['BUS_ID', 'STATION', 'TRANSPORT_TYPE']] = None
        
        update_air_bus_output(output, airport_bus)
                    
        # 도시버스
        peopletraj = transport_path(output)
        city_bus = check_paths_city_bus_stops_GPU(peopletraj, citybusrootDF, ctx.distance.bus_threshold_m, ctx.device)
        city_bus = find_routes_passing_stops(city_bus)

        update_city_bus_output(output, city_bus)
        
        # 결과를 저장
        output_path = MAPPING_DATA_DIR / file_path.name
        output.to_csv(output_path, index=False)
        
        
if __name__ == "__main__":
    run_mapping()