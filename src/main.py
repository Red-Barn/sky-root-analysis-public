import pandas as pd
from datetime import datetime, timedelta
import torch

from route_generation import get_bus_candidate_routes
from route_similarity import select_best_route_gpu
from route_deviation import detect_deviation_clusters
from route_improvement import is_improvement_required


# =========================
# 환경 설정
# =========================
CSV_PATH = r"C:\mygit\SkyRoot\result\2024-08-19.csv"
OUTPUT_PATH = r"C:\mygit\SkyRoot\trip_analysis_result\2024-08-19.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 출발 시간 처리
# =========================
def get_departure_time_for_api(df_trip):
    """
    csv의 시간(HH:MM:SS)만 사용하고 날짜는 내일로 설정
    """
    t = pd.to_datetime(df_trip.iloc[0]["DPR_MT1_UNIT_TM"])
    base = datetime.now() + timedelta(days=1)
    dt = base.replace(hour=t.hour, minute=t.minute, second=0)
    return int(dt.timestamp())


# =========================
# 실제 Trip 좌표 추출
# =========================
def extract_actual_trip_coords(df_trip):
    df = df_trip.copy()

    # 정지 / 체류 제거
    df = df[df["DYNA_MVMT_SPED"] > 3]

    # 시간 정렬
    df = df.sort_values("DPR_MT1_UNIT_TM")

    return list(zip(df["DPR_CELL_YCRD"], df["DPR_CELL_XCRD"]))  # (lat, lon)


# =========================
# Trip 단위 분석
# =========================
def analyze_trip(trip_no, df_trip):
    print(f"\n[Trip 분석 시작] {trip_no}")

    emd_code = df_trip.iloc[0]["EMD_CODE"]
    actual_coords = extract_actual_trip_coords(df_trip)

    if len(actual_coords) < 10:
        print("좌표 부족 → 스킵")
        return None

    origin_lat, origin_lon = actual_coords[0]
    dest_lat, dest_lon = actual_coords[-1]

    departure_time = get_departure_time_for_api(df_trip)

    # Step 1: 후보 경로 생성
    candidate_routes = get_bus_candidate_routes(
        origin_lat, origin_lon,
        dest_lat, dest_lon,
        departure_time
    )

    if not candidate_routes:
        print("버스 후보 경로 없음")
        return {
            "TRIP_NO": trip_no,
            "has_candidate": False
        }

    # Step 2: 최적 경로 선택
    best_idx, metrics = select_best_route_gpu(
        actual_coords,
        candidate_routes
    )

    # Step 3: 이탈 지점 탐지
    clusters, max_cluster_size, has_deviation = detect_deviation_clusters(
        actual_coords,
        metrics["distances"]
    )
    
    # Step 4: 경로 개선 필요 판별
    improve_required = is_improvement_required(
        metrics, 
        clusters
    )

    return {
        "TRIP_NO": trip_no,
        "EMD_CODE": emd_code,
        "has_candidate": True,
        "best_route_idx": best_idx,
        "mean_dist": metrics["mean"],
        "median_dist": metrics["median"],
        "near_ratio": metrics["near_ratio"],
        "max_dist": metrics["max"],
        "has_deviation": has_deviation,
        "num_deviation_clusters": len(clusters),
        "max_cluster_size": max_cluster_size,
        "improve_required": improve_required
    }


# =========================
# 메인 실행
# =========================
def main():
    df = pd.read_csv(CSV_PATH)

    results = []

    for trip_no, df_trip in df.groupby("TRIP_NO"):
        try:
            res = analyze_trip(trip_no, df_trip)
            if res:
                results.append(res)
        except Exception as e:
            print(f"[에러] {trip_no}: {e}")

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_PATH, index=False)
    print("\n분석 완료 → 결과 저장:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
