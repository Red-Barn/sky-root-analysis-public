import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon
from tqdm import tqdm
from pathlib import Path

def makeEMD_folder(input_dir: Path, output_dir: Path, emdDF):
    """
    input_dir 내 CSV 파일들에 행정동 데이터를 추가하여 output_dir에 저장
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path in tqdm(list(input_dir.glob("*.csv")), desc="Making EMD files"):
        df = pd.read_csv(file_path)
        
        makeEMD_df = makeEMD_dataframe(df, emdDF)
        
        output_path = output_dir / file_path.name
        makeEMD_df.to_csv(output_path, index=False)

def makeEMD_dataframe(df: pd.DataFrame, emdDF) -> pd.DataFrame:
    """
    단일 DataFrame 행정동 데이터 추가 로직
    """
    result = []
    df = df.copy()
    
    for data in emdDF:
        name = data['properties']['EMD_KOR_NM']
        code = data['properties']['EMD_CD']
        if data['geometry']:
            if data['geometry']['type'] == "Polygon":
                polygon = Polygon(np.round(np.float64(data['geometry']['coordinates'][0]), decimals = 9))
            else:
                polygons = [Polygon(np.round(np.float64(coords[0]), decimals = 9)) for coords in data['geometry']['coordinates']]
                polygon = MultiPolygon(polygons)
                
        result.append([name, code, polygon])
        
        df['EMD_CODE'] = None
        
        # peopleDF 데이터의 각 행에 대해 처리
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Checking EMD"):
            # Point 객체 생성
            point = Point(np.round(np.float64(row['DPR_CELL_XCRD']), decimals = 9), np.round(np.float64(row['DPR_CELL_YCRD']), decimals = 9))
            
            # result 리스트에서 폴리곤을 확인
            for name, code, polygon in result:
                if polygon.contains(point):  # 포인트가 폴리곤 내부에 있는지 확인
                    # 값을 변경
                    df.at[index, 'DPR_ADNG_NM'] = name
                    df.at[index, 'EMD_CODE'] = code
                    break  # 매칭된 경우 더 이상 확인하지 않음
                
        df = df.dropna(subset=['EMD_CODE']).reset_index(drop=True)
        
        return df