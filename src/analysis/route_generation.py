import googlemaps
import polyline
from datetime import datetime
import time

# =========================
# 1. Google Maps Client
# =========================
from config.settings import API_KEY
gmaps = googlemaps.Client(key=API_KEY)


# =========================
# 2. 버스 경로 추출
# =========================
def extract_bus_route_coords(route):
    """
    하나의 route에서 버스 경로에 대한 데이터만 수집
    """
    coords = []
    
    for leg in route["legs"]:
        for step in leg["steps"]:   
            if step["travel_mode"] != "TRANSIT":
                continue
            
            transit = step.get("transit_details")
            if not transit:
                continue
            
            if transit["line"]["vehicle"]["type"] != "BUS":
                continue
            
            step_coords = polyline.decode(step["polyline"]["points"])
            coords.extend(step_coords)
            
    return coords if len(coords) > 0 else None


# =========================
# 3. Trip 하나에 대한 후보 버스 경로 생성
# =========================
def get_bus_candidate_routes(
    origin_lat,
    origin_lon,
    dest_lat,
    dest_lon,
    departure_time
):
    
    res = gmaps.directions(
        origin = (origin_lat, origin_lon),
        destination = (dest_lat, dest_lon),
        mode = "transit",
        departure_time = departure_time,
        language = "ko",
        alternatives = True
    )
    
    bus_routes = []
    
    for route in res:
        coords = extract_bus_route_coords(route)
        if coords:
            bus_routes.append(coords)
    """
    bus_routes: [[(lat, lon), ...], [(lat, lon), ...], ...]
    """
    return bus_routes