import torch

# GPU harversine 거리 계산
def cdist(coords1, coords2):
    """
    여러 쌍의 위도, 경도 좌표 간의 Haversine 거리를 GPU에서 한 번에 계산합니다.
    
    coords1과 coords2는 [경도, 위도] 형식의 좌표입니다.
    반환값은 각 좌표 쌍 사이의 거리를 미터로 반환합니다.
    """
    # 지구 반지름 (미터)
    R = 6371000.0
    
    # 경도와 위도를 라디안으로 변환
    lon1, lat1 = torch.deg2rad(coords1[:, 0]), torch.deg2rad(coords1[:, 1])
    lon2, lat2 = torch.deg2rad(coords2[:, 0]), torch.deg2rad(coords2[:, 1])
    
    # 각 좌표 쌍 간 차이 계산 (broadcasting)
    dlon = lon2.unsqueeze(0) - lon1.unsqueeze(1)
    dlat = lat2.unsqueeze(0) - lat1.unsqueeze(1)
    
    # Haversine 공식 적용
    a = torch.sin(dlat / 2)**2 + torch.cos(lat1.unsqueeze(1)) * torch.cos(lat2.unsqueeze(0)) * torch.sin(dlon / 2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    
    # 거리를 미터로 계산
    distance = R * c
    
    return distance
