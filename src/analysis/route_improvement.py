import pandas as pd
from itertools import product

def is_improvement_required(
    metrics,
    deviation_clusters,
    max_cluster_size
    ):
    """
    metrics: Step 2 결과 dict
    deviation_clusters: Step 3 결과
    max_cluster_size: Step3 결과
    """

    if not deviation_clusters:
        return False

    A = max_cluster_size >= 5
    B = metrics["median"] >= 150 and metrics["near_ratio"] <= 0.5
    C = metrics["max"] >= 1000

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