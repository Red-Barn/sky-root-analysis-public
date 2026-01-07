import pandas as pd
from itertools import product

def is_improvement_required(metrics, deviation_clusters, max_cluster_size, policy):
    """
    metrics: Step 2 결과 dict
    deviation_clusters: Step 3 결과
    max_cluster_size: Step3 결과
    """

    if not deviation_clusters:
        return False

    A = max_cluster_size >= policy.max_cluster_size_threshold
    B = metrics["median"] >= policy.median_dist_threshold and metrics["near_ratio"] <= policy.near_ratio_threshold
    C = metrics["max"] >= policy.max_dist_threshold

    return A and (B or C)

def improvement_required_custom(
    max_cluster_size,
    median,
    near_raio,
    max,
    cluster_size_th,
    median_th,
    near_ratio_th
):
    
    A = max_cluster_size >= cluster_size_th
    B = median >= median_th and near_raio <= near_ratio_th
    C = max >= 1000

    return A and (B or C)