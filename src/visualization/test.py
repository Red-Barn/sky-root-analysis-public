import ast
from pathlib import Path
from typing import List, Sequence, Tuple

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.extraction.extractor import extract_actual_trip_coords
from src.analysis.extraction.generation import get_candidate_routes_info
from src.data.loader import load_all_api_info, load_all_trips, load_analysis_region, load_analysis_trips, load_gpd_emd


plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def add_region_names(region_df, gpd_emd):
    meta = gpd_emd[["EMD_CD", "EMD_KOR_NM"]].copy()
    out = region_df.copy()

    out = out.merge(meta, left_on="EMD_CODE", right_on="EMD_CD", how="left")
    out = out.rename(columns={"EMD_KOR_NM": "EMD_NAME"})
    out = out.drop(columns=["EMD_CD"])
    return out


def build_actual_coords_for_trip(processed_df, trip_no):
    trip_df = processed_df[processed_df["TRIP_NO"] == trip_no].copy()
    return extract_actual_trip_coords(trip_df)


def build_candidate_coords_for_trip(api_df, trip_no, best_route_idx):
    candidate_routes = get_candidate_routes_info(trip_no, api_df)
    for route in candidate_routes:
        if int(route["ROUTE_NO"]) == int(best_route_idx):
            return route["POINTS"]
    return []


def parse_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    return ast.literal_eval(str(value))


def parse_bool_list(value: object) -> List[bool]:
    parsed = parse_list(value)
    result: List[bool] = []
    for item in parsed:
        if isinstance(item, bool):
            result.append(item)
        else:
            result.append(str(item).lower() == "true")
    return result


def select_single_trip_case(region_df: pd.DataFrame, route_df: pd.DataFrame, trip_no: str) -> pd.Series:
    trip_no = str(trip_no)

    df = route_df.copy()
    df["TRIP_NO"] = df["TRIP_NO"].astype(str)
    df["EMD_CODE"] = df["EMD_CODE"].astype(str)

    df = df[df["TRIP_NO"] == trip_no].copy()
    if df.empty:
        raise ValueError(f"TRIP_NO={trip_no} 가 load_analysis_trips 결과에 없습니다.")

    df["deviation_score"] = df["deviation_ratio"] * df["mean_confidence"]

    merge_cols = ["EMD_CODE"]
    for col in ["EMD_NAME", "priority_rank", "severity_score"]:
        if col in region_df.columns:
            merge_cols.append(col)

    df = df.merge(region_df[merge_cols], on="EMD_CODE", how="left")

    # 같은 TRIP_NO가 여러 행이면 가장 대표성이 높은 1개를 선택
    df = df.sort_values(
        by=["improve_required", "deviation_score", "longest_deviation_ratio", "separation", "dtw"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)

    return df.iloc[0]


def plot_case_map(
    actual_coords: Sequence[Tuple[float, float]],
    candidate_coords: Sequence[Tuple[float, float]],
    case_row: pd.Series,
    gpd_emd: gpd.GeoDataFrame,
    out_path: Path,
) -> None:
    all_coords = list(actual_coords) + list(candidate_coords)
    if not all_coords:
        return

    center_lat = float(np.mean([lat for lat, _ in all_coords]))
    center_lon = float(np.mean([lon for _, lon in all_coords]))
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="OpenStreetMap", control_scale=True)

    emd_code = str(case_row["EMD_CODE"])
    region_gdf = gpd_emd[gpd_emd["EMD_CD"].astype(str) == emd_code]
    if not region_gdf.empty:
        folium.GeoJson(
            region_gdf.__geo_interface__,
            name="분석 지역",
            style_function=lambda _: {
                "color": "#444444",
                "weight": 2,
                "fillColor": "#ffd54f",
                "fillOpacity": 0.15,
            },
            tooltip=folium.GeoJsonTooltip(fields=["EMD_KOR_NM"], aliases=["지역명"]),
        ).add_to(fmap)

    if candidate_coords:
        folium.PolyLine(
            locations=[(lat, lon) for lat, lon in candidate_coords],
            tooltip="API 대중교통 후보 경로",
            popup="API 대중교통 후보 경로",
            color="#d62728",
            weight=4,
            opacity=0.8,
            dash_array="10, 10",
        ).add_to(fmap)

    if actual_coords:
        folium.PolyLine(
            locations=[(lat, lon) for lat, lon in actual_coords],
            tooltip="실제 이동 경로",
            popup="실제 이동 경로",
            color="#1f77b4",
            weight=5,
            opacity=0.9,
        ).add_to(fmap)
        folium.Marker(actual_coords[0], tooltip="출발", icon=folium.Icon(color="green")).add_to(fmap)
        folium.Marker(actual_coords[-1], tooltip="도착", icon=folium.Icon(color="black")).add_to(fmap)
        

    min_lat = min(lat for lat, _ in all_coords)
    max_lat = max(lat for lat, _ in all_coords)
    min_lon = min(lon for _, lon in all_coords)
    max_lon = max(lon for _, lon in all_coords)
    fmap.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.save(out_path)


def plot_case_distance_profile(case_row: pd.Series, out_path: Path) -> None:
    distances = [float(value) for value in parse_list(case_row["distances"])]
    is_deviated = parse_bool_list(case_row["is_deviated"])

    if not distances or len(distances) != len(is_deviated):
        return

    x = np.arange(len(distances))
    y = np.asarray(distances)
    mask = np.asarray(is_deviated)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, linewidth=1.8, alpha=0.85, label="DTW 정렬 거리")
    ax.scatter(x[~mask], y[~mask], s=18, alpha=0.7, label="정상 구간")
    ax.scatter(x[mask], y[mask], s=22, alpha=0.9, label="이탈 구간")

    ax.set_title(
        f"거리 프로파일 | TRIP_NO={case_row['TRIP_NO']} | "
        f"deviation_ratio={case_row['deviation_ratio']:.3f}, "
        f"longest_deviation_ratio={case_row['longest_deviation_ratio']:.1f}"
    )
    ax.set_xlabel("정렬 index")
    ax.set_ylabel("거리")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)



def prepare_inputs():
    region_df = load_analysis_region().copy()
    route_df = load_analysis_trips().copy()
    processed_df = load_all_trips().copy()
    api_df = load_all_api_info().copy()
    gpd_emd = load_gpd_emd().copy()

    region_df["EMD_CODE"] = pd.to_numeric(region_df["EMD_CODE"], errors="coerce").astype(int).astype(str)
    route_df["EMD_CODE"] = pd.to_numeric(route_df["EMD_CODE"], errors="coerce").astype(int).astype(str)
    processed_df["EMD_CODE"] = pd.to_numeric(processed_df["EMD_CODE"], errors="coerce").astype(int).astype(str)
    gpd_emd["EMD_CD"] = pd.to_numeric(gpd_emd["EMD_CD"], errors="coerce").astype(int).astype(str)
    
    region_df = add_region_names(region_df, gpd_emd)

    return region_df, route_df, processed_df, api_df, gpd_emd



def generate_single_trip_visuals(region_df, route_df, processed_df, api_df, gpd_emd, out_dir: Path, trip_no: str):
    ensure_dir(out_dir)

    case_row = select_single_trip_case(region_df, route_df, trip_no)

    trip_no = str(case_row["TRIP_NO"])
    best_route_idx = int(case_row["best_route_idx"])
    emd_code = str(case_row["EMD_CODE"])

    prefix = f"{emd_code}_trip{trip_no}_route{best_route_idx}"

    actual_coords = build_actual_coords_for_trip(processed_df, trip_no)
    candidate_coords = build_candidate_coords_for_trip(api_df, trip_no, best_route_idx)

    if not actual_coords and not candidate_coords:
        raise ValueError(f"TRIP_NO={trip_no} 에 대해 actual/candidate 좌표를 모두 찾지 못했습니다.")

    plot_case_map(
        actual_coords=actual_coords,
        candidate_coords=candidate_coords,
        case_row=case_row,
        gpd_emd=gpd_emd,
        out_path=out_dir / f"{prefix}_map.html",
    )
    plot_case_distance_profile(
        case_row=case_row,
        out_path=out_dir / f"{prefix}_distance_profile.png",
    )
        

def main():
    region_df, route_df, processed_df, api_df, gpd_emd = prepare_inputs()
    generate_single_trip_visuals(
        region_df=region_df,
        route_df=route_df,
        processed_df=processed_df,
        api_df=api_df,
        gpd_emd=gpd_emd,
        out_dir=Path("result/report_figures/test_maps"),
        trip_no="TRIP_439053",
    )


if __name__ == "__main__":
    main()
