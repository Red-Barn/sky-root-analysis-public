from src.config.settings import RAW_DATA_DIR, COMPRESSED_DATA_DIR, MAPPING_DATA_DIR, PROCESSED_DATA_DIR

from src.preprocess.compress import compress_folder
from src.preprocess.makeEMD import makeEMD_folder
from src.preprocess.boxplot import delete_outlier

from src.mapping.main import run_mapping
from src.analysis.main import run_anaysis

def main():
    # # 1. 원본데이터 -> 압축데이터
    # compress_folder(RAW_DATA_DIR, COMPRESSED_DATA_DIR)
    
    # # 2. 압축데이터에 버스정류장 매핑
    # run_mapping()
    
    # # 3. 매핑데이터에 EMD코드 부착
    # makeEMD_folder(MAPPING_DATA_DIR, PROCESSED_DATA_DIR)
    
    # 4. 이상치 제거
    delete_outlier()
    
    # # 5. 전처리 데이터로 분석
    # run_anaysis()
    

if __name__ == "__main__":
    main()