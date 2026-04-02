import pandas as pd
from tqdm import tqdm
from pathlib import Path

def compress_folder(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path in tqdm(list(input_dir.glob("*.csv")), desc="Compressing files", position=0):
        df = pd.read_csv(file_path)
        compressed_df = compress_dataframe(df)
        output_path = output_dir / file_path.name
        compressed_df.to_csv(output_path, index=False)
        
def compress_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    시간 순으로 정렬된 데이터프레임에서 DYNA_MVMT_SPED가 0 이고
    DPR_CELL_ID가 동일한 연속된 구간을 하나의 행으로 압축하는 함수

    Args:
        df (pd.DataFrame): 원본 데이터프레임

    Returns:
        pd.DataFrame: 압축된 데이터프레임
    """
    df = df.copy()
    df['DPR_MT1_UNIT_TM'] = pd.to_datetime(df['DPR_MT1_UNIT_TM'])
    df = df.sort_values(['TRIP_NO', 'DPR_MT1_UNIT_TM']).reset_index(drop=True)

    results = []
    grouped = df.groupby('TRIP_NO')
    
    for _, group in tqdm(grouped, total=grouped.ngroups, desc="Compressing trips", position=1, leave=False):
        zero_speed_df = group[group['DYNA_MVMT_SPED'] == 0].copy()
        zero_speed_df['group'] = (zero_speed_df['DPR_CELL_ID'] != zero_speed_df['DPR_CELL_ID'].shift()).cumsum()

        compressed_group = zero_speed_df.groupby('group').agg({
            'DPR_MT1_UNIT_TM': 'min',   # 최소 출발 시간
            'ARV_MT1_UNIT_TM': 'max',   # 최대 도착 시간
            'DPR_CELL_ID': 'first',
            **{col: 'first' for col in zero_speed_df.columns if col not in ['DPR_MT1_UNIT_TM', 'ARV_MT1_UNIT_TM', 'DPR_CELL_ID', 'group']}  # 기타 열들
        }).reset_index(drop=True)

        compressed_group = compressed_group[group.columns]
        non_zero_speed_df = group[group['DYNA_MVMT_SPED'] != 0]
        final_group = pd.concat([compressed_group, non_zero_speed_df], ignore_index=True)
        results.append(final_group)

    compressed_results = pd.concat(results, ignore_index=True)
    compressed_results = compressed_results.sort_values(['TRIP_NO', 'DPR_MT1_UNIT_TM']).reset_index(drop=True)
    return compressed_results