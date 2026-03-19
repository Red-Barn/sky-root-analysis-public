from src.config.settings import RAW_DATA_DIR, COMPRESSED_DATA_DIR, MAPPING_DATA_DIR, PROCESSED_DATA_DIR, CLEANING_DATA_DIR

from src.preprocess.compress import compress_folder
from src.preprocess.makeEMD import makeEMD_folder
from src.preprocess.boxplot import delete_outlier
from src.preprocess.cleaning import cleaning_folder

from src.mapping.main import run_mapping
from src.analysis.main import run_analysis_routes, run_analysis_regions

def main():
    # 1. 원본데이터 -> 압축데이터
    compress_folder(RAW_DATA_DIR, COMPRESSED_DATA_DIR)
    
    # 2. 점핑 데이터 제거
    cleaning_folder(COMPRESSED_DATA_DIR, CLEANING_DATA_DIR)
    
    # 3. 압축데이터에 버스정류장 매핑
    run_mapping()
    
    # 4. 매핑데이터에 EMD코드 부착
    makeEMD_folder(MAPPING_DATA_DIR, PROCESSED_DATA_DIR)
    
    # 5. 경유지 및 경로 꼬리 이상치 제거
    delete_outlier()
    
    # 6. 전처리 데이터로 경로 및 지역 분석
    run_analysis_routes()
    run_analysis_regions()
    

if __name__ == "__main__":
    main()