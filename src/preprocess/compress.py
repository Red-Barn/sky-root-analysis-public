import pandas as pd
from tqdm import tqdm
from pathlib import Path

def compress_folder(input_dir: Path, output_dir: Path):
    """
    input_dir 내 CSV 파일들을 압축처리하여 output_dir에 저장
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path in tqdm(list(input_dir.glob("*.csv")), desc="Compressing files"):
        df = pd.read_csv(file_path)
        
        compressed_df = compress_dataframe(df)
        
        output_path = output_dir / file_path.name
        compressed_df.to_csv(output_path, index=False)
        
def compress_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    단일 DataFrame 압축 로직
    """
    df = df.copy()
    
        # 'DPR_MT1_UNIT_TM'을 datetime 형식으로 변환
    df['DPR_MT1_UNIT_TM'] = pd.to_datetime(df['DPR_MT1_UNIT_TM'])

    # 시간순으로 정렬
    df = df.sort_values(['TRIP_NO', 'DPR_MT1_UNIT_TM']).reset_index(drop=True)

    results = []
    
    # 각 TRIP_NO 그룹에 대해 처리
    for _, group in df.groupby('TRIP_NO'):
        # DYNA_MVMT_SPED가 0인 행만 필터링
        zero_speed_df = group[group['DYNA_MVMT_SPED'] == 0].copy()
        
        # 동일 위치(DPR_CELL_ID)에서 정지하고 있는 연속된 행을 그룹화
        zero_speed_df['group'] = (zero_speed_df['DPR_CELL_ID'] != zero_speed_df['DPR_CELL_ID'].shift()).cumsum()
        
        # 그룹별 시작 시간과 도착 시간, 다른 컬럼의 첫 번째 값 유지
        compressed_group = zero_speed_df.groupby('group').agg({
            'DPR_MT1_UNIT_TM': 'min',       # 시작 시간
            'ARV_MT1_UNIT_TM': 'max',       # 도착 시간
            'DPR_CELL_ID': 'first',         # 위치 ID
            **{col: 'first' for col in zero_speed_df.columns if col not in ['DPR_MT1_UNIT_TM', 'ARV_MT1_UNIT_TM', 'DPR_CELL_ID', 'group']}
        }).reset_index(drop=True)

        # 원본 컬럼 순서 유지
        compressed_group = compressed_group[group.columns]

        # DYNA_MVMT_SPED가 0이 아닌 행을 원본 그대로 유지
        non_zero_speed_df = group[group['DYNA_MVMT_SPED'] != 0]

        # 압축된 데이터와 비압축 데이터를 합침
        final_group = pd.concat([compressed_group, non_zero_speed_df], ignore_index=True)
        
        results.append(final_group)

    # 최종 결과를 시간순으로 정렬
    compressed_results = pd.concat(results, ignore_index=True)
    compressed_results = compressed_results.sort_values(['TRIP_NO', 'DPR_MT1_UNIT_TM']).reset_index(drop=True)
    
    return compressed_results
