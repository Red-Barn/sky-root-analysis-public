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

def detect_deviation_clusters(
    actual_coords,
    distances,
    dist_threshold=150,   # 이 이상이면 '이탈 후보'
    eps=0.0005,           # 약 50m (위도 기준)
    min_samples=5
):
    """
    return:
      - deviation_clusters: {cluster_id: [indices]}
      - has_deviation: bool
    """

    # 1. 이탈 후보만 추림
    idx = np.where(distances > dist_threshold)[0]

    if len(idx) < min_samples:
        return {}, False

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

    return clusters, len(clusters) > 0
