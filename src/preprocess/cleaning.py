from typing import List

import pandas as pd
from tqdm import tqdm
from pathlib import Path

from src.config.runtime import create_runtime_context
from src.trajectory.haversine import harversine_degree


def speed_kmh(distance_m: float, start_tm: pd.Timestamp, end_tm: pd.Timestamp) -> float:
    seconds = (end_tm - start_tm).total_seconds()
    return float(distance_m / seconds * 3.6)


def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    for col in ["DPR_MT1_UNIT_TM", "ARV_MT1_UNIT_TM"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        
    for col in ["DPR_CELL_XCRD", "DPR_CELL_YCRD", "ARV_CELL_XCRD", "ARV_CELL_YCRD", "DYNA_MVMT_SPED"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    return df


def recalc_row_speed(row: pd.Series) -> float:
    dist_m = harversine_degree(
        row["DPR_CELL_XCRD"],
        row["DPR_CELL_YCRD"],
        row["ARV_CELL_XCRD"],
        row["ARV_CELL_YCRD"],
    )
    return speed_kmh(dist_m, row["DPR_MT1_UNIT_TM"], row["ARV_MT1_UNIT_TM"])


def is_spike_pair(prev_row: pd.Series, cur_row: pd.Series, next_row: pd.Series, policy) -> bool:
    """현재 row가 spike인지 판단"""
    d_in = harversine_degree(
        prev_row["DPR_CELL_XCRD"],
        prev_row["DPR_CELL_YCRD"],
        cur_row["DPR_CELL_XCRD"],
        cur_row["DPR_CELL_YCRD"],
    )
    d_out = harversine_degree(
        cur_row["DPR_CELL_XCRD"],
        cur_row["DPR_CELL_YCRD"],
        next_row["DPR_CELL_XCRD"],
        next_row["DPR_CELL_YCRD"],
    )
    d_skip = harversine_degree(
        prev_row["DPR_CELL_XCRD"],
        prev_row["DPR_CELL_YCRD"],
        next_row["DPR_CELL_XCRD"],
        next_row["DPR_CELL_YCRD"],
    )

    v_in = speed_kmh(d_in, prev_row["DPR_MT1_UNIT_TM"], cur_row["DPR_MT1_UNIT_TM"])
    v_out = speed_kmh(d_out, cur_row["DPR_MT1_UNIT_TM"], next_row["DPR_MT1_UNIT_TM"])
    v_skip = speed_kmh(d_skip, prev_row["DPR_MT1_UNIT_TM"], next_row["DPR_MT1_UNIT_TM"])

    bad_in = (d_in >= policy.max_spike_distance_m) or (v_in >= policy.max_speed_kmh)
    bad_out = (d_out >= policy.max_spike_distance_m) or (v_out >= policy.max_speed_kmh)
    recover_skip = (d_skip < policy.max_spike_distance_m) or (v_skip < policy.max_speed_kmh)

    return bool(bad_in and bad_out and recover_skip)


def merge_two_rows(prev_row: pd.Series, next_row: pd.Series) -> pd.Series:
    """spike 제거시 이전, 이후 row 데이터의 연속성 유지"""
    merged = prev_row.copy()
    merged["ARV_MT1_UNIT_TM"] = next_row["DPR_MT1_UNIT_TM"]
    merged["ARV_CELL_ID"] = next_row["DPR_CELL_ID"]
    merged["ARV_CELL_XCRD"] = next_row["DPR_CELL_XCRD"]
    merged["ARV_CELL_YCRD"] = next_row["DPR_CELL_YCRD"]
    merged["DYNA_MVMT_SPED"] = recalc_row_speed(merged)

    return merged


def remove_spike_points(trip_df: pd.DataFrame, policy) -> tuple[pd.DataFrame, int]:
    """spike 제거 및 데이터 연속성 유지"""
    rows: List[pd.Series] = [row.copy() for _, row in trip_df.iterrows()]
    removed_count = 0
    i = 1

    while i < len(rows) - 1:
        prev_row = rows[i - 1]
        cur_row = rows[i]
        next_row = rows[i + 1]

        if is_spike_pair(prev_row, cur_row, next_row, policy):
            rows[i - 1] = merge_two_rows(prev_row, next_row)
            del rows[i]
            removed_count += 1
            if i > 1:
                i -= 1  # 변경된 이전 row의 spike 가능성 여부 검사
                
            continue    # del 후 row 인덱스가 -1 되어서 continue로 다음 루프 진입하여 i 증가 방지
        i += 1

    cleaned = pd.DataFrame(rows)
    cleaned = cleaned.reset_index(drop=True)
    return cleaned, removed_count


def clean_trip_points(df: pd.DataFrame, policy) -> tuple[pd.DataFrame, list]:
    df = prepare_input(df)
    original_columns = df.columns.tolist()

    cleaned_groups = []
    summary_rows = []
    grouped = df.groupby("TRIP_NO", sort=False)

    for trip_no, trip_df in tqdm(grouped, total=grouped.ngroups, desc="Cleaning trip points", position=1, leave=False):
        trip_df = trip_df.sort_values(["DPR_MT1_UNIT_TM", "ARV_MT1_UNIT_TM"]).reset_index(drop=True)
        original_len = len(trip_df)
        
        trip_df, spike_removed = remove_spike_points(trip_df, policy)

        # spike 제거 후 데이터가 너무 적은 경우 전체 trip 제거
        if len(trip_df) < policy.min_trip_points:
            continue

        cleaned_groups.append(trip_df[original_columns])
        summary_rows.append(
            {
                "TRIP_NO": trip_no,
                "original_rows": original_len,
                "spike_rows_removed": spike_removed,
                "final_rows": len(trip_df),
            }
        )

    cleaned_df = pd.concat(cleaned_groups, ignore_index=True)
    cleaned_df = cleaned_df[original_columns]
    
    return cleaned_df, summary_rows

    
def cleaning_folder(input_dir: Path, output_dir: Path):
    ctx = create_runtime_context(verbose=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    
    for file_path in tqdm(list(input_dir.glob("*.csv")), desc="Cleaning files", position=0):
        df = pd.read_csv(file_path)
        cleaned_df, summary_rows = clean_trip_points(df, policy=ctx.preprocess)
        summary.extend(summary_rows)
        output_path = output_dir / file_path.name
        cleaned_df.to_csv(output_path, index=False)
        
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / "cleaning_summary.csv", index=False)
    