import requests
from datetime import datetime, timezone, timedelta

def get_bus_routes(origin, destination, departure_time, api_key):
    url = "https://maps.googleapis.com/maps/api/directions/json"

    params = {
        "origin": origin,              # "lat,lng"
        "destination": destination,    # "lat,lng"
        "mode": "transit",
        "allowedTravelModes": "BUS",    # 선호 교통수단 : 버스
        "routingPerference": "FEWER_TRANSFERS", # 선호 경로 환경설정 : 적은 환승
        "departure_time": departure_time,
        "alternatives": "true",
        "language": "ko",
        "key": api_key
    }

    res = requests.get(url, params=params)
    data = res.json()

    if data.get("status") != "OK":
        raise Exception(f"API Error: {data.get('status')}")

    bus_routes = []

    for route in data["routes"]:
        steps = route["legs"][0]["steps"]
        is_bus_only = True
        bus_steps = []

        for step in steps:
            mode = step.get("travel_mode")

            # 🚶 도보는 허용
            if mode == "WALKING":
                continue

            if mode == "TRANSIT":
                transit = step.get("transit_details", {})
                vehicle_type = (
                    transit.get("line", {})
                           .get("vehicle", {})
                           .get("type")
                )

                # ❌ 버스가 아니면 이 route 탈락
                if vehicle_type != "BUS":
                    is_bus_only = False
                    break

                # 버스 정보 저장
                bus_steps.append({
                    "bus_number": transit["line"].get("short_name"),
                    "bus_name": transit["line"].get("name"),
                    "departure_stop": transit["departure_stop"]["name"],
                    "arrival_stop": transit["arrival_stop"]["name"],
                    "num_stops": transit.get("num_stops"),
                    "duration": step["duration"]["text"]
                })

        if is_bus_only and bus_steps:
            bus_routes.append({
                "total_duration": route["legs"][0]["duration"]["text"],
                "total_distance": route["legs"][0]["distance"]["text"],
                "bus_steps": bus_steps
            })

    return bus_routes

def decode_polyline(polyline_str):
    coords = []
    index = lat = lng = 0

    while index < len(polyline_str):
        result = shift = 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if result & 1 else (result >> 1)
        lat += dlat

        result = shift = 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if result & 1 else (result >> 1)
        lng += dlng

        coords.append((lat / 1e5, lng / 1e5))

    return coords

def get_departure_time(time_str: str) -> int: 
    kst = timezone(timedelta(hours=9))
    now = datetime.now(kst)
    
    hour, minute = map(int, time_str.split(":"))
    
    tomorrow = now + timedelta(days=1)
    target_time = tomorrow.replace(
        hour=hour,
        minute=minute,
        second=0,
        microsecond=0
    )
    
    return int(target_time.timestamp())
    
# 🔎 사용 예시
if __name__ == "__main__":
    API_KEY = "AIzaSyC4KSVDuXq_Nm4yxvMLY_jYVbFACaGdKrU"
    origin = "37.49242416,126.9555956"
    destination = "37.46724129,126.4344681"
    departure_time = get_departure_time("8:15")

    routes = get_bus_routes(origin, destination, departure_time, API_KEY)

    if not routes:
        print("❌ 버스만 이용하는 경로가 없습니다.")
    else:
        for i, r in enumerate(routes, 1):
            print(f"\n[버스 경로 {i}]")
            print(f"총 소요 시간: {r['total_duration']}")
            print(f"총 거리: {r['total_distance']}")

            # for b in r["bus_steps"]:
            #     print(
            #         f"  🚌 {b['bus_number']} ({b['bus_name']}) | "
            #         f"{b['departure_stop']} → {b['arrival_stop']} "
            #         f"({b['num_stops']}정거장, {b['duration']})"
            #     )
