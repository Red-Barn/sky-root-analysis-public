import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from src.analysis.route.analyzer import analyze_trips
from src.analysis.extraction.extractor import extract_candidate_trips
from src.analysis.extraction.api_info_bulider import build_total_api_info
from src.analysis.region.region_analysis import region_level_analysis
from src.config.settings import RESULT_EXTRACTION_DIR, RESULT_TRIP_DIR, RESULT_REGION_DIR, DATA_DIR
from src.config.runtime import create_runtime_context
from src.data.loader import load_all_trips, load_analysis_trips, load_extracted_trips


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


# 대중교통 후보 경로 정보 구축
def run_build_candidate_total_info_cache():
    peopleDF = load_all_trips()
    cache_df = build_total_api_info(peopleDF)
    cache_df.to_csv(DATA_DIR / "total_api_info.csv", index=False)
        
        
# 최적 경로 추출
def run_extract_best_routes():
    peopleDF = load_all_trips()
    
    extract_df = extract_candidate_trips(peopleDF)
    extract_output_path = RESULT_EXTRACTION_DIR / "extracted_best_routes.csv"
    ensure_dir(RESULT_EXTRACTION_DIR)
    extract_df.to_csv(extract_output_path, index=False)
    

# 경로 분석
def run_analysis_routes():
    ctx = create_runtime_context(verbose=True)
    
    peopleDF = load_extracted_trips()
    
    route_result_df = analyze_trips(peopleDF, ctx)
    route_output_path = RESULT_TRIP_DIR / "routes_analysis_all_trips.csv"
    ensure_dir(RESULT_TRIP_DIR)
    route_result_df.to_csv(route_output_path, index=False)
    

# 지역 분석
def run_analysis_regions():
    ctx = create_runtime_context(verbose=True)
    
    peopleDF = load_analysis_trips()
    
    region_result_df = region_level_analysis(peopleDF, ctx.severity)
    region_output_path = RESULT_REGION_DIR / "region_analysis_all_trips.csv"
    ensure_dir(RESULT_REGION_DIR)
    region_result_df.to_csv(region_output_path, index=False)


if __name__ == "__main__":
    run_extract_best_routes()
    run_analysis_routes()
    run_analysis_regions()
