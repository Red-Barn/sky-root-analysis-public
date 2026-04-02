import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon
from tqdm import tqdm
from pathlib import Path

from src.data.loader import load_emd

def makeEMD_folder(input_dir: Path, output_dir: Path):
    emdDF = load_emd()["features"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path in tqdm(list(input_dir.glob("*.csv")), desc="Making EMD files", position=0):
        df = pd.read_csv(file_path)
        makeEMD_df = makeEMD_dataframe(df, emdDF)
        output_path = output_dir / file_path.name
        makeEMD_df.to_csv(output_path, index=False)

def makeEMD_dataframe(df: pd.DataFrame, emdDF: pd.DataFrame) -> pd.DataFrame:
    """
    emdDF의 EMD_CODE 및 폴리곤 좌표 추출
    df의 DPR_CELL_XCRD, DPR_CELL_YCRD가 해당된 emd 폴리곤 좌표 안에 있으면 그 행에 EMD_CODE 데이터 추가

    Args:
        df (pd.DataFrame): 실제 이동 데이터
        emdDF (pd.DataFrame): EMD 정보 데이터

    Returns:
        pd.DataFrame: EMD 정보가 추가된 실제 이동 데이터
    """
    result = []
    df = df.copy()
    
    # EMD 정보 추출
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
    
    # 실제 이동 경로에 EMD 정보 추가
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Checking EMD", position=1, leave=False):
        point = Point(np.round(np.float64(row['DPR_CELL_XCRD']), decimals = 9), np.round(np.float64(row['DPR_CELL_YCRD']), decimals = 9))

        for name, code, polygon in result:
            if polygon.contains(point):
                df.at[index, 'DPR_ADNG_NM'] = name
                df.at[index, 'EMD_CODE'] = code
                break
            
    df = df.dropna(subset=['EMD_CODE']).reset_index(drop=True)
    return df