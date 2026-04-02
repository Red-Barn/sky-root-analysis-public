from typing import Any
from src.trajectory.dtw import dtw_cost_haversine, dtw_path_haversine

def select_best_route_gpu(actual_coords: list[tuple[float, float]], candidate_routes: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    """
    실제 경로와 후보 경로들의 dtw 계산으로 최적 경로를 판단 및 추출하고 추출된 최적 경로의 dtw 계산 정보를 반환
    계산량을 줄이기 위해 dtw 값만을 먼저 계산해 최적 경로를 추출하고 해당 경로의 dtw 정보를 다시 계산

    Args:
        actual_coords (list[tuple[float, float]]): 실제 이동 경로
        candidate_routes (list[dict[str, Any]]): 후보 이동 경로들

    Returns:
        tuple[int, dict[str, Any]]: 최적 경로 번호, 최적 경로 dtw 계산 정보
    """
    if not actual_coords or not candidate_routes:
        return None, None

    best_idx = None
    best_score = float("inf")
    best_route_coords = None

    for route in candidate_routes:
        route_no = route["ROUTE_NO"]
        route_coords = route["POINTS"]
        if not route_coords:
            continue
    
        # dtw 비용 정보만 계산
        score = dtw_cost_haversine(actual_coords, route_coords, cutoff=best_score)
        if score < best_score:
            best_score = score
            best_idx = route_no
            best_route_coords = route_coords

    if best_idx is None:
        return None, None

    # dtw 비용, 정렬 번호, 정렬 거리 계산
    dtw, alignment, distances = dtw_path_haversine(actual_coords, best_route_coords)
    return best_idx, {
        "dtw": dtw,
        "alignment": alignment,
        "distances": distances,
    }