import pandas as pd
import numpy as np
from numpy.typing import NDArray


def normalize(series: pd.Series) -> pd.Series:
    return (series - series.min()) / (series.max() - series.min() + 1e-6)


def wilson_lower_bound(successes: NDArray[np.int_], total: NDArray[np.int_], z=1.96) -> NDArray[np.float64]:
    """
    표본 수를 고려하여 보수적인 하한 값 계산
    표본 수가 적을수록 같은 확률이어도 수치가 낮음

    Args:
        successes (NDArray[np.int_]): 개선 필요 경로 개수
        total (NDArray[np.int_]): 전체 경로 수

    Returns:
        NDArray[np.float64]: 표본 수를 고려한 전체 경로 대비 개선 필요 경로 비율
    """
    successes = np.asarray(successes, dtype=float)
    total = np.asarray(total, dtype=float)

    lower_bound = np.zeros_like(total, dtype=float)
    valid = total > 0
    if not valid.any():
        return lower_bound

    phat = successes[valid] / total[valid]
    z2 = z ** 2
    denominator = 1.0 + z2 / total[valid]
    centre = phat + z2 / (2.0 * total[valid])
    adjusted_std = z * np.sqrt((phat * (1.0 - phat) + z2 / (4.0 * total[valid])) / total[valid])
    lower_bound[valid] = (centre - adjusted_std) / denominator
    return lower_bound


def region_level_analysis(df: pd.DataFrame, policy) -> pd.DataFrame:
    """
    route 분석 결과를 EMD_CODE 단위로 묶어서 region 단위 분석
    improve_ratio_lower_bound, avg_deviation_ratio, avg_longest_deviation_ratio를 사용해 need_attention, severity_score 계산

    Args:
        df (pd.DataFrame): route 단위 분석 결과 데이터프레임

    Returns:
        improve_ratio: 지역별 개선 필요 비율
        improve_ratio_lower_bound: 표본 수를 고려한 지역별 개선 필요 비율
        need_attention: 지역별 개선 필요성 여부 판단
        severity_score: 지역별 개선 필요성 심각도
    """
    df = df[df["EMD_CODE"].notna()].copy()

    df["improve_required"] = df["improve_required"].astype(bool)
    for col in ["deviation_ratio", "mean_confidence", "longest_deviation", "longest_deviation_ratio", "separation"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = df.groupby("EMD_CODE").agg(
        total_trips=("TRIP_NO", "count"),
        improve_trips=("improve_required", "sum"),
        avg_deviation_ratio=("deviation_ratio", "mean"),
        avg_mean_confidence=("mean_confidence", "mean"),
        avg_longest_deviation=("longest_deviation", "mean"),
        avg_longest_deviation_ratio=("longest_deviation_ratio", "mean"),
        avg_separation=("separation", "mean"),
    ).reset_index()

    grouped = grouped[grouped["total_trips"] >= policy.min_total_trips]
    grouped["improve_ratio"] = grouped["improve_trips"] / grouped["total_trips"]
    grouped["improve_ratio_lower_bound"] = wilson_lower_bound(
        grouped["improve_trips"].to_numpy(),
        grouped["total_trips"].to_numpy(),
        z=policy.wilson_z,
    )

    grouped["deviation_norm"] = normalize(grouped["avg_deviation_ratio"])
    grouped["longest_dev_norm"] = normalize(grouped["avg_longest_deviation"])
    grouped["longest_dev_ratio_norm"] = normalize(grouped["avg_longest_deviation_ratio"])
    
    grouped["needs_attention"] = (
        (grouped["improve_ratio_lower_bound"] >= policy.min_improve_ratio_lower_bound) &
        (grouped["avg_deviation_ratio"] >= policy.min_avg_deviation_ratio) &
        (grouped["avg_longest_deviation_ratio"] >= policy.min_avg_longest_deviation_ratio)
    )

    grouped["severity_score"] = (
        policy.improve_ratio_weight * grouped["improve_ratio_lower_bound"]
        + policy.deviation_ratio_weight * grouped["deviation_norm"]
        + policy.longest_deviation_weight * grouped["longest_dev_ratio_norm"]
    )

    grouped["improve_ratio_pct"] = grouped["improve_ratio"] * 100
    grouped["improve_ratio_lower_bound_pct"] = grouped["improve_ratio_lower_bound"] * 100
    grouped = grouped.sort_values(["needs_attention", "severity_score"], ascending=[False, False]).reset_index(drop=True)
    grouped["priority_rank"] = grouped.index + 1

    return grouped
