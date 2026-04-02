from src.config.settings import RAW_DATA_DIR, COMPRESSED_DATA_DIR, MAPPING_DATA_DIR, PROCESSED_DATA_DIR, CLEANING_DATA_DIR

from src.preprocess.compress import compress_folder
from src.preprocess.makeEMD import makeEMD_folder
from src.preprocess.boxplot import delete_outlier
from src.preprocess.cleaning import cleaning_folder

from src.mapping.main import run_mapping
from src.analysis.main import run_analysis_routes, run_analysis_regions, run_extract_best_routes, run_build_candidate_total_info_cache
from src.visualization.visualize import run_visualization

def main():
    # 1. 원본 데이터 압축
    compress_folder(RAW_DATA_DIR, COMPRESSED_DATA_DIR)
    
    # 2. 데이터 정제
    cleaning_folder(COMPRESSED_DATA_DIR, CLEANING_DATA_DIR)
    
    # 3. 버스 정류장 및 버스 노선 매핑
    run_mapping()
    
    # 4. EMD 데이터 생성
    makeEMD_folder(MAPPING_DATA_DIR, PROCESSED_DATA_DIR)
    
    # 5. 경유 노선 및 이상치 제거
    delete_outlier()
    
    # 6. 후보 경로 정보 캐시 구축
    run_build_candidate_total_info_cache()
    
    # 7. 최적 경로 추출
    run_extract_best_routes()
    
    # 8. 경로 및 지역 분석
    run_analysis_routes()
    run_analysis_regions()
    
    # 9. 시각화
    run_visualization()
    

if __name__ == "__main__":
    main()