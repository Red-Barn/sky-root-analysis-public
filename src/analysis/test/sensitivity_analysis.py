import pandas as pd
from itertools import product
from src.analysis.route.improvement import improvement_required_custom
from src.config.settings import RESULT_TRIP_DIR, RESULT_SENSITIVITY_DIR

def run_sensitivity_analysis(df):
    cluster_sizes = [3, 5, 7]
    medians = [100, 150, 200]
    near_ratios = [0.4, 0.5, 0.6]

    records = []
    
    df = df[df["has_candidate"] == True].copy()

    for cs, md, nr in product(cluster_sizes, medians, near_ratios):
        df["improvement_required"] = df.apply(
            lambda r: improvement_required_custom(
                r["max_cluster_size"],
                r["median_dist"],
                r["near_ratio"],
                r["max_dist"],
                cs,
                md,
                nr
            ),
            axis=1
        )

        records.append({
            "cluster_size_th": cs,
            "median_th": md,
            "near_ratio_th": nr,
            "improvement_ratio": df["improvement_required"].mean()
        })

    return pd.DataFrame(records)


def main():
    df = pd.read_csv(RESULT_TRIP_DIR)
    result = run_sensitivity_analysis(df)
    result.to_csv(RESULT_SENSITIVITY_DIR, index=False)
    print("민감도 분석 완료:",RESULT_SENSITIVITY_DIR)


if __name__ == "__main__":
    main()
