import pandas as pd

def load_actual_points(csv_path, trip_no):
    df = pd.read_csv(csv_path)

    trip = df[df["TRIP_NO"] == trip_no].copy()

    # DPR 좌표만 사용
    points = trip[["DPR_CELL_YCRD", "DPR_CELL_XCRD"]] \
                .dropna() \
                .values.tolist()

    return points  # [(lat, lng), ...]

from datetime import datetime, timedelta

def get_departure_time_from_csv(csv_path, trip_no):
    df = pd.read_csv(csv_path)

    row = df[df["TRIP_NO"] == trip_no].iloc[0]

    # 문자열 → datetime
    time_str = row["DPR_MT1_UNIT_TM"]
    t = pd.to_datetime(time_str)

    return t.hour, t.minute

def get_origin_destination_from_csv(csv_path, trip_no):
    df = pd.read_csv(csv_path)

    trip = df[df["TRIP_NO"] == trip_no].copy()

    trip["DPR_MT1_UNIT_TM"] = pd.to_datetime(trip["DPR_MT1_UNIT_TM"])
    trip = trip.sort_values("DPR_MT1_UNIT_TM")

    origin = (
        trip.iloc[0]["DPR_CELL_YCRD"],  # lat
        trip.iloc[0]["DPR_CELL_XCRD"],  # lng
    )

    destination = (
        trip.iloc[-1]["ARV_CELL_YCRD"],  # lat
        trip.iloc[-1]["ARV_CELL_XCRD"],  # lng
    )

    return origin, destination

def make_departure_timestamp(hour, minute):
    now = datetime.now()

    tomorrow = now.date() + timedelta(days=1)

    dep_dt = datetime(
        year=tomorrow.year,
        month=tomorrow.month,
        day=tomorrow.day,
        hour=hour,
        minute=minute
    )

    return int(dep_dt.timestamp())

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

def get_transit_routes_response(origin, destination, departure_time, api_key):
    import requests

    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{origin[0]},{origin[1]}",
        "destination": f"{destination[0]},{destination[1]}",
        "mode": "transit",
        "departure_time": departure_time,
        "allowedTravelModes": "BUS",    # 선호 교통수단 : 버스
        "routingPerference": "FEWER_TRANSFERS", # 선호 경로 환경설정 : 적은 환승
        "alternatives": "true",         # 여러 경로 허용
        "language": "ko",
        "key": api_key
    }

    res = requests.get(url, params=params).json()
    
    if res.get("status") != "OK":
        return None

    return res

from pyproj import Transformer

transformer = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)

def project(points):
    return [transformer.transform(lng, lat) for lat, lng in points]


import numpy as np
from shapely.geometry import Point, LineString

def evaluate_routes(res_json, actual_xy, distance_threshold=100):
    """
    res_json: Directions API 전체 응답
    actual_xy: 실제 이동 좌표 (투영된 좌표)
    """

    results = []

    for idx, route in enumerate(res_json["routes"]):
        try:
            polyline = route["overview_polyline"]["points"]
            route_coords = decode_polyline(polyline)
        except KeyError:
            continue

        # route 좌표 투영
        route_xy = project(route_coords)

        line = LineString(route_xy)

        distances = [
            Point(p).distance(line)
            for p in actual_xy
            if np.isfinite(Point(p).distance(line))
        ]

        if not distances:
            continue

        distances = np.array(distances)

        result = {
            "route_index": idx,
            "mean_distance": float(distances.mean()),
            "median_distance": float(np.median(distances)),
            "inlier_ratio": float(np.mean(distances <= distance_threshold)),
            "max_distance": float(distances.max()),
            "route_xy": route_xy
        }

        results.append(result)

    return results

def select_best_route(results):
    """
    1) 근접 비율 최대
    2) 평균 거리 최소
    """

    if not results:
        return None

    results_sorted = sorted(
        results,
        key=lambda r: (-r["inlier_ratio"], r["mean_distance"])
    )

    return results_sorted[0]


def process_single_trip(csv_path, trip_no, api_key, distance_threshold=100):
    try:
        actual = load_actual_points(csv_path, trip_no)
        actual_xy = project(actual)

        if len(actual_xy) < 5:
            return None

        origin, destination = get_origin_destination_from_csv(csv_path, trip_no)

        hour, minute = get_departure_time_from_csv(csv_path, trip_no)
        departure_time = make_departure_timestamp(hour, minute)

        res = get_transit_routes_response(origin, destination, departure_time, api_key)

        if res is None:
            return None

        results = evaluate_routes(res, actual_xy, distance_threshold)

        if not results:
            return None

        best = select_best_route(results)
        
        # # Debugging
        # for r in results:
        #     print(
        #         f"[{r['route_index']}] "
        #         f"inlier={r['inlier_ratio']:.2%}, "
        #         f"mean={r['mean_distance']:.0f}m, "
        #         f"median={r['median_distance']:.0f}m"
        # )

        return {
            "TRIP_NO": trip_no,
            "route_index": best["route_index"],
            "mean_distance": best["mean_distance"],
            "median_distance": best["median_distance"],
            "max_distance": best["max_distance"],
            "inlier_ratio": best["inlier_ratio"],
            "num_routes": len(results)
        }

    except Exception as e:
        print(f"⚠️ TRIP {trip_no} 실패: {e}")
        return None


def get_all_trip_nos(csv_path):
    df = pd.read_csv(csv_path)
    return df["TRIP_NO"].unique()


def process_all_trips(csv_path, api_key, limit=None):
    trip_nos = get_all_trip_nos(csv_path)

    results = []

    for i, trip_no in enumerate(trip_nos):
        if limit and i >= limit:
            break

        print(f"[{i+1}/{len(trip_nos)}] Processing {trip_no}")

        r = process_single_trip(csv_path, trip_no, api_key)

        if r:
            results.append(r)

    return pd.DataFrame(results)

API_KEY = r"AIzaSyC4KSVDuXq_Nm4yxvMLY_jYVbFACaGdKrU"
csv_path = r"C:\mygit\SkyRoot\result\2024-08-19.csv"

df_result = process_all_trips(csv_path, API_KEY)

df_result.to_csv("trip_route_similarity_result.csv", index=False)

print("✅ 완료:", len(df_result), "trips")
