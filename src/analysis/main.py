import pandas as pd

from src.analysis.route.analyzer import analyze_trips
from src.analysis.region.region_analysis import region_level_analysis
from src.config.settings import PROCESSED_DATA_DIR, RESULT_TRIP_DIR, RESULT_REGION_DIR
from src.config.runtime import create_runtime_context

def run_anaysis():
    ctx = create_runtime_context(verbose=True)
    
    for file_path in PROCESSED_DATA_DIR.glob("*.csv"):
        peopleDF = pd.read_csv(file_path)
        
        # 경로 분석 결과
        route_result_df = analyze_trips(peopleDF, ctx)
        route_result_df.to_csv(RESULT_TRIP_DIR, index=False)
        
        # 지역 분석 결과
        region_result_df = region_level_analysis(route_result_df, ctx.severity)
        region_result_df.to_csv(RESULT_REGION_DIR, index=False)
        

if __name__ == "__main__":
    run_anaysis()
