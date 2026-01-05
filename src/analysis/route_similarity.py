import numpy as np
import torch
from trajectory.haversine import cdist

def to_tensor(coords, device):
    """
    coords: [(lat, lon), ...]
    return: torch tensor [[lon, lat], ...]
    """
    return torch.tensor(
        [[lon, lat] for lat, lon in coords],
        dtype=torch.float32,
        device=device
    )
    
def evaluate_route_gpu(
    actual_coords,
    route_coords,
    device = "cuda" if torch.cuda.is_available() else "cpu",
    near_threshold = 100    # 오차범위 = 100m
):
    A = to_tensor(actual_coords, device)
    R = to_tensor(route_coords, device)
    
    # 거리 행렬(N x M)
    dist_matrix = cdist(A, R)
    
    # 거리 행렬의 최소값 -> actual_coords의 한 점에서 가장 가까운 route_coords 거리
    min_distances = torch.min(dist_matrix, dim=1).values
    
    distances = min_distances.cpu().numpy()
    
    return {
        "mean": distances.mean(),
        "median": float(torch.median(min_distances).cpu()),
        "max": distances.max(),
        "near_ratio": (distances <= near_threshold).sum() / len(distances),
        "distances" : distances
    }
    
def select_best_route_gpu(actual_coords, candidate_routes):
    best_idx = None
    best_score = float("inf")
    best_result = None
    
    for i, route in enumerate(candidate_routes):
        if len(actual_coords) == 0 or len(route) == 0:
            return None
        
        res = evaluate_route_gpu(actual_coords, route)
        
        """
        이 부분의 핵심기준의 정의가 좀더 필요함
        """
        score = res["median"] # 핵심 기준
        
        if score < best_score:
            best_score = score
            best_idx = i
            best_result = res
            
    return best_idx, best_result