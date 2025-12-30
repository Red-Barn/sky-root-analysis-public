import pandas as pd
from itertools import product

def is_improvement_required(metrics, deviation_clusters):
    """
    metrics: Step 2 결과 dict
    deviation_clusters: Step 3 결과
    """

    if not deviation_clusters:
        return False

    cluster_sizes = [len(v) for v in deviation_clusters.values()]
    max_cluster_size = max(cluster_sizes)

    A = max_cluster_size >= 5
    B = metrics["median"] >= 150 and metrics["near_ratio"] <= 0.5
    C = metrics["max"] >= 1000

    return A and (B or C)

def improvement_required_custom(
    metrics,
    deviation_clusters,
    cluster_size_th,
    median_th,
    near_ratio_th
):
    if not deviation_clusters:
        return False

    max_cluster_size = max(len(v) for v in deviation_clusters.values())

    A = max_cluster_size >= cluster_size_th
    B = metrics["median"] >= median_th and metrics["near_ratio"] <= near_ratio_th
    C = metrics["max"] >= 1000

    return A and (B or C)