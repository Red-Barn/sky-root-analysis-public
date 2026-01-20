import numpy as np
from sklearn.cluster import DBSCAN

def build_deviation_features(actual_coords, distances):
    """
    actual_coords: [(lat, lon)]
    distances: 실제점 → 경로 최소 거리 (m)
    """
    X = []

    for (lat, lon), d in zip(actual_coords, distances):
        X.append([lat, lon, d])

    return np.array(X)

def detect_deviation_clusters(actual_coords, distances, policy):
    """
    return:
      - deviation_clusters: {cluster_id: [indices]}
      - max_cluster_size: int
      - has_deviation: bool
    """
    dist_threshold=policy.dist_threshold 
    eps=policy.eps         
    min_samples=policy.min_samples

    # 1. 이탈 후보만 추림
    idx = np.where(distances > dist_threshold)[0]

    if len(idx) < min_samples:
        return {}, 0, False

    filtered_coords = [actual_coords[i] for i in idx]

    X = np.array(filtered_coords)

    # 2. DBSCAN (위도/경도만 사용)
    db = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="euclidean"
    ).fit(X)

    labels = db.labels_

    clusters = {}
    for i, label in enumerate(labels):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(idx[i])
        
    max_cluster_size = max([len(v) for v in clusters.values()]) if clusters else 0

    return clusters, max_cluster_size, len(clusters) > 0
