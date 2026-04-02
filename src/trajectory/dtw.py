import numpy as np
from numpy.typing import NDArray
from src.trajectory.haversine import haversine_radians, haversine_radian

EARTH_R = 6371000.0

def coords_to_rad(coords):
    arr = np.asarray(coords, dtype=np.float32)
    lat = np.deg2rad(arr[:, 0]).astype(np.float32, copy=False)
    lon = np.deg2rad(arr[:, 1]).astype(np.float32, copy=False)
    return lat, lon


def backtrack_collapsed_to_actual(
    steps: NDArray[np.int8], 
    latA: NDArray[np.float32], 
    lonA: NDArray[np.float32], 
    cos_latA: NDArray[np.float32], 
    latR: NDArray[np.float32], 
    lonR: NDArray[np.float32], 
    cos_latR: NDArray[np.float32]
    ) -> tuple[list[tuple[int, int]], NDArray[np.float32]]:
    """
    actual의 각 점마다 dtw path상 가장 가까운 route 점의 인덱스와 거리를 반환

    Args:
        steps (NDArray[np.int8]): backtrack을 위한 backpointer 행렬 (0: diag, 1: up, 2: left)
        latA (NDArray[np.float32]): actual 위도
        lonA (NDArray[np.float32]): actual 경도
        cos_latA (NDArray[np.float32]): actual 위도 cos 값
        latR (NDArray[np.float32]): route 위도
        lonR (NDArray[np.float32]): route 경도
        cos_latR (NDArray[np.float32]): route 위도 cos 값

    Raises:
        RuntimeError: actual 각 점에서 route와 한 점도 매칭되지 않는 점이 있을 시 에러

    Returns:
        tuple[list[tuple[int, int]], NDArray[np.float32]]: alignment, distances
    """
    n_rows, n_cols = steps.shape
    i = n_rows - 1
    j = n_cols - 1

    # 가장 짧은 route index j, 거리 d
    best_j = np.full(n_rows, -1, dtype=np.int64)
    best_d = np.full(n_rows, np.inf, dtype=np.float32)

    while True:
        d = haversine_radian(
            latA[i], lonA[i], cos_latA[i],
            latR[j], lonR[j], cos_latR[j],
        )

        # 동일 i에 대해 j가 여러 개 나올 경우 최소 j만 저장
        if d < best_d[i]:
            best_d[i] = d
            best_j[i] = j

        # dtw path 시작점 도달 시 종료
        if i == 0 and j == 0:
            break

        step = steps[i, j]
        if step == 0:   # diag
            i -= 1
            j -= 1
        elif step == 1: # up
            i -= 1
        else:           # left
            j -= 1

    if np.any(best_j < 0):
        raise RuntimeError("Failed to build collapsed DTW alignment for some actual points.")

    alignment = [(idx, int(best_j[idx])) for idx in range(n_rows)]
    distances = best_d.astype(np.float32, copy=False)
    return alignment, distances


def dtw_cost_haversine(actual_coords: list[tuple[float, float]], route_coords: list[tuple[float, float]], cutoff=np.inf) -> float:
    """
    dtw cost만 계산
    cutoff로 계산량 절감

    Args:
        actual_coords (list[tuple[float, float]]): actual 좌표들
        route_coords (list[tuple[float, float]]): route 좌표들
        cutoff: 이미 계산된 dtw cost중 최소값

    Returns:
        float: dtw cost
    """
    latA, lonA = coords_to_rad(actual_coords)
    latR, lonR = coords_to_rad(route_coords)

    N = latA.shape[0]
    M = latR.shape[0]
    if N == 0 or M == 0:
        return float("inf")

    cos_latA = np.cos(latA)
    cos_latR = np.cos(latR)

    prev = np.full(M + 1, np.inf, dtype=np.float32)
    curr = np.full(M + 1, np.inf, dtype=np.float32)
    prev[0] = 0.0

    for i in range(1, N + 1):
        curr[0] = np.inf

        row = haversine_radians(
            latA[i - 1], lonA[i - 1], cos_latA[i - 1],
            latR, lonR, cos_latR
        )

        row_min = np.inf
        for j in range(1, M + 1):
            diag = prev[j - 1]
            up = prev[j]
            left = curr[j - 1]

            # 우선순위: diag -> up -> left
            if diag <= up and diag <= left:
                m = diag
            elif up <= left:
                m = up
            else:
                m = left

            curr[j] = row[j - 1] + m
            if curr[j] < row_min:
                row_min = curr[j]

        # 누적된 cost가 cutoff보다 크면 종료
        if row_min > cutoff:
            return float("inf")

        # 다음 actual 점으로 넘어가니 지금 curr 값을 prev 값으로 재사용
        prev, curr = curr, prev

    return float(prev[M])


def dtw_path_haversine(actual_coords: list[tuple[float, float]], route_coords: list[tuple[float, float]]) -> tuple[float, list[tuple[int, int]], NDArray[np.float32]]:
    """
    cost, alignment, distances 계산
    backtracking을 위한 step 저장

    Args:
        actual_coords (list[tuple[float, float]]): actual 좌표들
        route_coords (list[tuple[float, float]]): dtw 비용이 최소인 route 좌표들

    Returns:
        tuple[float, list[tuple[int, int]], NDArray[np.float32]]: dtw cost, alignment, distances
    """
    latA, lonA = coords_to_rad(actual_coords)
    latR, lonR = coords_to_rad(route_coords)

    N = latA.shape[0]
    M = latR.shape[0]
    if N == 0 or M == 0:
        return float("inf"), [], np.empty((0,), dtype=np.float32)

    cos_latA = np.cos(latA)
    cos_latR = np.cos(latR)

    prev = np.full(M + 1, np.inf, dtype=np.float32) # dp 이전 행 값
    curr = np.full(M + 1, np.inf, dtype=np.float32) # dp 현재 행 값

    # backpointer (0: diag, 1: up, 2: left)
    steps = np.empty((N, M), dtype=np.int8)

    prev[0] = 0.0

    for i in range(1, N + 1):
        curr[0] = np.inf

        # 현재 actual 점 1개에 대한 route 전체 점들 사이의 거리 계산
        row = haversine_radians(
            latA[i - 1], lonA[i - 1], cos_latA[i - 1],
            latR, lonR, cos_latR
        )

        for j in range(1, M + 1):
            diag = prev[j - 1]
            up = prev[j]
            left = curr[j - 1]

            # 우선순위: diag -> up -> left
            if diag <= up and diag <= left:
                m = diag
                step = 0
            elif up <= left:
                m = up
                step = 1
            else:
                m = left
                step = 2

            # 누적된 cost
            curr[j] = row[j - 1] + m
            steps[i - 1, j - 1] = step

        # 다음 actual 점으로 넘어가니 지금 curr 값을 prev 값으로 재사용
        prev, curr = curr, prev

    cost = float(prev[M])
    alignment, distances = backtrack_collapsed_to_actual(steps, latA, lonA, cos_latA, latR, lonR, cos_latR)

    return cost, alignment, distances