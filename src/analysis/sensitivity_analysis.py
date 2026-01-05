import pandas as pd
from itertools import product
from route_improvement import improvement_required_custom
from config.settings import ROUTE_OUTPUT_PATH, SENSITIVITY_OUTPUT_PATH

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
    df = pd.read_csv(ROUTE_OUTPUT_PATH)
    result = run_sensitivity_analysis(df)
    result.to_csv(SENSITIVITY_OUTPUT_PATH, index=False)
    print("민감도 분석 완료:",SENSITIVITY_OUTPUT_PATH)


if __name__ == "__main__":
    main()
