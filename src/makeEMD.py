import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon
from tqdm import tqdm

import os

emdDF = pd.read_json(r"C:\mygit\SkyRoot\open_data\emd_WGS84.json")["features"]

def makeEMD(emdDF, peopleDF):
    result = []

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
    
    peopleDF['EMD_CODE'] = None

    # peopleDF 데이터의 각 행에 대해 처리
    for index, row in tqdm(peopleDF.iterrows(), total=len(peopleDF), desc="Checking EMD"):
        # Point 객체 생성
        point = Point(np.round(np.float64(row['DPR_CELL_XCRD']), decimals = 9), np.round(np.float64(row['DPR_CELL_YCRD']), decimals = 9))
        
        # result 리스트에서 폴리곤을 확인
        for name, code, polygon in result:
            if polygon.contains(point):  # 포인트가 폴리곤 내부에 있는지 확인
                # 값을 변경
                peopleDF.at[index, 'DPR_ADNG_NM'] = name
                peopleDF.at[index, 'EMD_CODE'] = code
                break  # 매칭된 경우 더 이상 확인하지 않음
            
    peopleDF = peopleDF.dropna(subset=['EMD_CODE']).reset_index(drop=True)
    
    return peopleDF

# 처리할 CSV 파일이 있는 폴더 경로 지정
folder_path = r"C:\mygit\SkyRoot\mapping_data"

# 결과를 저장할 폴더 경로 지정
output_folder = r"C:\mygit\SkyRoot\result"  # 결과 저장 폴더
os.makedirs(output_folder, exist_ok=True)

# 폴더 내 모든 CSV 파일 처리
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        
        # CSV 파일 불러오기
        df = pd.read_csv(file_path)

        output = makeEMD(emdDF, df)
                
        # 결과 저장
        output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.csv")
        output.to_csv(output_file, index=False)