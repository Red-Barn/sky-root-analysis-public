import pandas as pd

def normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-6)

def region_level_analysis(df):
    
    df = df[df["has_candidate"] == True].copy()
    
    grouped = df.groupby("EMD_CODE").agg(
        total_trips=("TRIP_NO", "count"),
        improve_trips=("improve_required", "sum"),
        avg_median_dist=("median_dist", "mean"),
        avg_max_cluster=("max_cluster_size", "mean")
    ).reset_index()

    grouped["improve_ratio"] = grouped["improve_trips"] / grouped["total_trips"]

    # 정규화
    grouped["median_norm"] = normalize(grouped["avg_median_dist"])
    grouped["cluster_norm"] = normalize(grouped["avg_max_cluster"])

    # Severity Score (가중치는 보고서에서 설명)
    grouped["severity_score"] = (
        0.5 * grouped["improve_ratio"]
        + 0.3 * grouped["median_norm"]
        + 0.2 * grouped["cluster_norm"]
    )

    return grouped.sort_values("severity_score", ascending=False)
