import numpy as np
from typing import Any
from numpy.typing import NDArray
from sklearn.mixture import GaussianMixture


def gmm_deviation_clusters(distances: NDArray[np.float32]) -> tuple[NDArray[np.int_], int, int, GaussianMixture]:
    """
    정상 좌표와 이탈 좌표를 GMM을 통해서 계산, 정상/이탈 2개로 구분하기에 군집 개수를 2개로 고정

    Args:
        distances (NDArray[np.float32]): actual의 dtw path 거리

    Returns:
        labels (NDArray[np.int_]): 각 거리값의 군집 번호
        normal_label (np.intp): 정상 군집 번호
        dev_label (np.intp): 이탈 군집 번호
        gmm (GaussianMixture): 학습된 모델
        
    """
    X = distances.reshape(-1, 1)  # (N, 1)
    
    gmm = GaussianMixture(
        n_components=2,     # normal / deviation
        covariance_type='full',
        random_state=0)
    gmm.fit(X)
    
    labels = gmm.predict(X)  # (N,)
    means = gmm.means_.flatten()  # (2,)
    
    normal_label = np.argmin(means)
    dev_label = np.argmax(means)
    
    return labels, normal_label, dev_label, gmm


def longest_run(mask: NDArray[np.bool_]) -> int:
    """연속된 이탈 좌표의 최대 길이 반환"""
    max_run, cur = 0, 0
    for v in mask:
        if v:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run


def is_improvement_required(distances: NDArray[np.float32], policy) -> dict[str, Any]:
    """
    GMM을 통해서 구한 정상/이탈 지점을 기반으로 개선 필요 여부 판단
    deviation_score, longest_deviation, longest_deviation_ratio, separation를 사용해 개선 필요성 여부 판단

    Args:
        distances (NDArray[np.float32]): actual의 dtw path 거리
        policy: 정책

    Returns:
        need_improvement: 개선 필요 여부
        is_deviated: 각 거리값이 정싱/이탈인지 bool로 표시(정상: false, 이탈: true)
        probs: 각 거리값이 얼마나 정상/이탈인지 확률로 표시
        mean_conf: 이탈 분류 점들의 실제 이탈 확률
        deviation_ratio: 전체 경로의 이탈 비율
        deviation_score: 전체 경로의 이탈 점수(deviation_ratio * mean_conf)
        longest_deviation: 최대 이탈 길이
        longest_deviation_ratio: 전체 경로의 최대 이탈 거리 비율
        separation: 정상/이탈 군집의 분리도
    """
    labels, normal_label, dev_label, gmm = gmm_deviation_clusters(distances)
    is_deviated = (labels == dev_label)
    
    probs = gmm.predict_proba(distances.reshape(-1, 1))
    mean_conf = probs[:, dev_label][is_deviated].mean() if is_deviated.any() else 0.0
    
    deviation_ratio = is_deviated.mean()
    deviation_score = deviation_ratio * mean_conf
    
    point_count = len(distances)
    longest_deviation = longest_run(is_deviated)
    longest_deviation_ratio = longest_deviation / point_count
    
    mu = gmm.means_.flatten()
    sigma = np.sqrt(gmm.covariances_.flatten())
    separation = abs(mu[0] - mu[1]) / (sigma[0] + sigma[1] + 1e-12)
    
    need_improvement = (
        deviation_score >= policy.deviation_score_threshold and
        longest_deviation >= policy.longest_deviation_threshold and
        longest_deviation_ratio >= policy.longest_deviation_ratio_threshold and
        separation >= policy.separation_threshold
    )
    
    return {
        "need_improvement": need_improvement,
        "deviation_ratio": deviation_ratio,
        "mean_confidence": mean_conf,
        "longest_deviation": longest_deviation,
        "longest_deviation_ratio": longest_deviation_ratio,
        "separation": separation,
        "is_deviated": is_deviated.tolist(),
    }