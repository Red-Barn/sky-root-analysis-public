from dataclasses import dataclass


@dataclass(frozen=True)
class RouteSimilarityPolicy:
    near_threshold: float = 100 # 기본값 100


@dataclass(frozen=True)
class DeviationPolicy:
    dist_threshold: float = 150 # 이탈 범위 = 150m
    eps: float = 50             # 약 50m (위도 기준)
    min_samples: int = 5


@dataclass(frozen=True)
class ImprovementPolicy:
    max_cluster_size_threshold: int = 5
    median_dist_threshold: float = 150.0
    near_ratio_threshold: float = 0.5
    max_dist_threshold: float = 1000.0
    
    
@dataclass(frozen=True)
class BusDistancePolicy:
    bus_threshold_m = 100   # 기본값 50
    
    
@dataclass(frozen=True)
class SeverityScorePolicy:
    improve_ratio_threshold = 0.5
    median_norm_threshold = 0.3
    cluster_norm_threshold = 0.2
    min_total_trips = 5