import torch
from torch import Tensor
import numpy as np
from numpy.typing import NDArray

EARTH_R = 6371000.0

def harversine_torch(coords1: Tensor, coords2: Tensor) -> Tensor:
    lon1, lat1 = torch.deg2rad(coords1[:, 0]), torch.deg2rad(coords1[:, 1])
    lon2, lat2 = torch.deg2rad(coords2[:, 0]), torch.deg2rad(coords2[:, 1])
    dlon = lon2.unsqueeze(0) - lon1.unsqueeze(1)
    dlat = lat2.unsqueeze(0) - lat1.unsqueeze(1)
    a = torch.sin(dlat / 2)**2 + torch.cos(lat1.unsqueeze(1)) * torch.cos(lat2.unsqueeze(0)) * torch.sin(dlon / 2)**2
    c = 2.0 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    
    distance = EARTH_R * c
    return distance


def harversine_degree(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    sin_dlat = np.sin(dlat * 0.5)
    sin_dlon = np.sin(dlon * 0.5)
    a = sin_dlat * sin_dlat + np.cos(lat1) * np.cos(lat2) * (sin_dlon * sin_dlon)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    
    distance = EARTH_R * c
    return distance


def haversine_radians(lat1: float, lon1: float, cos_lat1: float, lat2: NDArray[np.float32], lon2: NDArray[np.float32], cos_lat2: NDArray[np.float32]) -> NDArray[np.float32]:
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    sin_dlat = np.sin(dlat * 0.5)
    sin_dlon = np.sin(dlon * 0.5)
    a = sin_dlat * sin_dlat + cos_lat1 * cos_lat2 * (sin_dlon * sin_dlon)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    
    distance = (EARTH_R * c).astype(np.float32, copy=False)
    return distance


def haversine_radian(lat1: float, lon1: float, cos1: float, lat2: float, lon2: float, cos2: float) -> float:
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    sin_dlat = np.sin(dlat * 0.5)
    sin_dlon = np.sin(dlon * 0.5)
    a = sin_dlat * sin_dlat + cos1 * cos2 * (sin_dlon * sin_dlon)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    
    distance = (EARTH_R * c).astype(np.float32, copy=False)
    return distance