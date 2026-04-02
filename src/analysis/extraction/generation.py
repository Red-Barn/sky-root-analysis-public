from typing import Any
import pandas as pd
import requests
import polyline

from src.config.settings import API_KEY
url = "https://maps.googleapis.com/maps/api/directions/json"

def get_candidate_total_info(trip_no: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for route_no, route in enumerate(data["routes"]):
        for step in route["legs"][0]["steps"]:
            row = {
                "TRIP_NO": trip_no,
                "ROUTE_NO": route_no,
                "DISTANCE": step["distance"]["value"],     # 미터
                "DURATION": step["duration"]["value"],     # 초
                "START_LNG": step["start_location"]["lng"],
                "START_LAT": step["start_location"]["lat"],
                "END_LNG": step["end_location"]["lng"],
                "END_LAT": step["end_location"]["lat"],
                "POLYLINE": step["polyline"]["points"],
                "TRAVEL_MODE": step["travel_mode"],
                "BUS_NAME": None,
                "BUS_TYPE": None
            }
            
            if step["travel_mode"] == "TRANSIT":
                transit = step.get("transit_details", {})
                line = transit.get("line", {})
                row["BUS_NAME"] = line.get("short_name")
                row["BUS_TYPE"] = line.get("name")

            rows.append(row)
    return rows

def get_candidate_routes_info(trip_no: str, df_api_info: pd.DataFrame) -> list[dict[str, Any]]:
    """
    캐시로 저장된 후보 경로 데이터프레임을 list로 추출하여 반환

    Args:
        trip_no (str): TRIP_NO
        df_api_info (pd.DataFrame): 후보 경로 데이터프레임

    Returns:
        list[dict[str, Any]]: [{TRIP_NO, ROUTE_NO, POINTS}]
    """
    if df_api_info.empty:
        return []

    trip_df = df_api_info[df_api_info["TRIP_NO"] == trip_no]
    if trip_df.empty:
        return []

    rows = []
    for route_no, route_df in trip_df.groupby("ROUTE_NO", sort=True):
        coords = []
        for encoded in route_df["POLYLINE"].dropna().tolist():
            coords.extend(polyline.decode(encoded))

        if not coords:
            continue

        rows.append({
            "TRIP_NO": trip_no,
            "ROUTE_NO": int(route_no),
            "POINTS": coords,   # [(lat, lon), ...]
        })

    return rows

def get_bus_candidate_routes(trip_no: str, origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float, departure_time: int) -> list[dict[str, Any]]:
    
    params = {
        "origin": f"{origin_lat},{origin_lon}",
        "destination": f"{dest_lat},{dest_lon}",
        "departure_time": f"{departure_time}",
        "mode": "transit",  # 대중교통
        "transit_mode": "bus",  # 버스
        "transit_routing_preference": "fewer_transfers",    #최소 환승
        "alternatives": "true", # 대안 노선 허용
        "language": "ko",
        "key": API_KEY
    }
    
    res = requests.get(url, params=params)
    data = res.json()
    
    candidate_total_info = get_candidate_total_info(trip_no, data)
    
    return candidate_total_info
