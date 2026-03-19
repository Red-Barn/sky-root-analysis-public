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


def select_case_trips(top_regions, route_df):
    top_region_codes = set(top_regions["EMD_CODE"].astype(str))

    df = route_df.copy()
    df["EMD_CODE"] = df["EMD_CODE"].astype(str)
    df = df[df["EMD_CODE"].isin(top_region_codes)]
    df["deviation_score"] = df["deviation_ratio"] * df["mean_confidence"]

    df = df.merge(
        top_regions[["EMD_CODE", "EMD_NAME", "priority_rank", "severity_score"]],
        on="EMD_CODE",
        how="left",
    )

    selected = (
        df.sort_values(
            by=["EMD_CODE", "improve_required", "deviation_score", "longest_deviation_ratio", "separation", "dtw"],
            ascending=[True, False, False, False, False, True],
        )
        .sort_values(["priority_rank", "deviation_score"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return selected



def plot_top_regions_bar(top_regions: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))

    labels = top_regions.apply(
        lambda row: f"{int(row['priority_rank'])}. {row['EMD_NAME'] if pd.notna(row['EMD_NAME']) else row['EMD_CODE']}",
        axis=1,
    )

    ax.barh(labels, top_regions["severity_score"])
    ax.invert_yaxis()
    ax.set_title("우선 개선 지역 Top N")
    ax.set_xlabel("severity_score")
    ax.set_ylabel("지역")

    for i, (_, row) in enumerate(top_regions.iterrows()):
        text = (
            f"개선비율 {row['improve_ratio_lower_bound_pct']:.1f}% | "
            f"평균 이탈비율 {row['avg_deviation_ratio']:.3f} | "
            f"trip {int(row['total_trips'])}건"
        )
        ax.text(row["severity_score"], i, f"  {text}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)



def plot_region_priority_scatter(region_df: pd.DataFrame, top_regions: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))

    ax.scatter(
        region_df["improve_ratio_lower_bound_pct"],
        region_df["avg_deviation_ratio"],
        s=region_df["total_trips"] * 10,
        alpha=0.4,
        label="전체 지역",
    )

    highlight = region_df[region_df["EMD_CODE"].isin(top_regions["EMD_CODE"])]
    ax.scatter(
        highlight["improve_ratio_lower_bound_pct"],
        highlight["avg_deviation_ratio"],
        s=highlight["total_trips"] * 12,
        alpha=0.9,
        label="Top 지역",
    )

    for _, row in highlight.iterrows():
        label = row["EMD_NAME"] if pd.notna(row["EMD_NAME"]) else row["EMD_CODE"]
        ax.annotate(
            label,
            (row["improve_ratio_lower_bound_pct"], row["avg_deviation_ratio"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_title("지역 우선순위 산점도")
    ax.set_xlabel("improve_ratio_lower_bound_pct (%)")
    ax.set_ylabel("avg_deviation_ratio")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)



def plot_region_choropleth(region_df: pd.DataFrame, gpd_emd: gpd.GeoDataFrame, out_path: Path) -> None:
    geo = gpd_emd.copy()
    geo["EMD_CD"] = geo["EMD_CD"].astype(str)

    data = region_df.copy()
    data["EMD_CODE"] = data["EMD_CODE"].astype(str)

    merged = geo.merge(data, left_on="EMD_CD", right_on="EMD_CODE", how="left")

    fig, ax = plt.subplots(figsize=(16, 9))
    geo.plot(ax=ax, color="#f2f2f2", linewidth=0.2, edgecolor="white")
    merged.dropna(subset=["severity_score"]).plot(
        ax=ax,
        column="severity_score",
        legend=True,
        linewidth=0.2,
        edgecolor="white",
        cmap="OrRd",
    )
    ax.set_title("지역별 severity_score 지도")
    ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)



def plot_top_region_boxplot(route_top_df: pd.DataFrame, top_regions: pd.DataFrame, out_path: Path) -> None:
    data = []
    labels = []

    for _, region_row in top_regions.iterrows():
        code = region_row["EMD_CODE"]
        values = route_top_df.loc[route_top_df["EMD_CODE"] == code, "deviation_ratio"].dropna().tolist()
        if values:
            data.append(values)
            labels.append(region_row["EMD_NAME"] if pd.notna(region_row["EMD_NAME"]) else code)

    if not data:
        return

    fig, ax = plt.subplots(figsize=(16, 9))
    box = ax.boxplot(data, patch_artist=True, labels=labels)
    for patch in box["boxes"]:
        patch.set_alpha(0.6)

    ax.set_title("Top 지역의 trip-level deviation_ratio 분포")
    ax.set_xlabel("지역")
    ax.set_ylabel("deviation_ratio")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)



def plot_policy_scatter(route_df: pd.DataFrame, out_path: Path) -> None:
    df = route_df.copy()
    df["deviation_score"] = df["deviation_ratio"] * df["mean_confidence"]

    fig, ax = plt.subplots(figsize=(16, 9))

    for label, part in df.groupby("improve_required"):
        ax.scatter(
            part["deviation_score"],
            part["separation"],
            s=part["longest_deviation_ratio"] * 120,
            alpha=0.7,
            label=f"improve_required={label}",
        )

    ax.axvline(0.2, linestyle="--", alpha=0.8)
    ax.axhline(1.1, linestyle="--", alpha=0.8)
    
    ax.set_title("전체 trip의 개선 판정 기준 시각화")
    ax.set_xlabel("deviation_score")
    ax.set_ylabel("separation")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)



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



def generate_visuals(region_df, route_df, processed_df, api_df, gpd_emd, out_dir, top_n):
    ensure_dir(out_dir)
    case_dir = out_dir / "case_maps"
    ensure_dir(case_dir)

    top_regions = region_df.sort_values("severity_score", ascending=False).head(top_n).reset_index(drop=True).copy()

    route_top_df = route_df[route_df["EMD_CODE"].isin(top_regions["EMD_CODE"])].copy()
    route_top_df = route_top_df.merge(top_regions[["EMD_CODE", "EMD_NAME", "priority_rank"]],on="EMD_CODE",how="left")
    route_top_df["deviation_score"] = route_top_df["deviation_ratio"] * route_top_df["mean_confidence"]

    selected_cases = select_case_trips(top_regions, route_df)

    top_regions.to_csv(out_dir / "top_regions_summary.csv", index=False, encoding="utf-8-sig")
    selected_cases.to_csv(out_dir / "selected_case_trips.csv", index=False, encoding="utf-8-sig")

    plot_top_regions_bar(top_regions, out_dir / "01_top_regions_bar.png")
    plot_region_priority_scatter(region_df, top_regions, out_dir / "02_region_priority_scatter.png")
    plot_region_choropleth(region_df, gpd_emd, out_dir / "03_region_severity_map.png")
    plot_top_region_boxplot(route_top_df, top_regions, out_dir / "04_top_region_trip_boxplot.png")
    plot_policy_scatter(route_df, out_dir / "05_policy_scatter.png")

    for _, case_row in selected_cases.iterrows():
        trip_no = case_row["TRIP_NO"]
        best_route_idx = case_row["best_route_idx"]
        emd_name = case_row["EMD_NAME"]
        rank = int(case_row["priority_rank"])
        prefix = f"{trip_no}_{best_route_idx}"
        
        emd_dir = case_dir / f"{rank:02d}_{emd_name}"
        ensure_dir(emd_dir)
        actual_coords = build_actual_coords_for_trip(processed_df, trip_no)
        candidate_coords = build_candidate_coords_for_trip(api_df, trip_no, best_route_idx)

        plot_case_map(
            actual_coords=actual_coords,
            candidate_coords=candidate_coords,
            case_row=case_row,
            gpd_emd=gpd_emd,
            out_path=emd_dir / f"{prefix}_map.html",
        )
        plot_case_distance_profile(
            case_row=case_row,
            out_path=emd_dir / f"{prefix}_distance_profile.png",
        )
        

def main():
    region_df, route_df, processed_df, api_df, gpd_emd = prepare_inputs()
    generate_visuals(
        region_df=region_df,
        route_df=route_df,
        processed_df=processed_df,
        api_df=api_df,
        gpd_emd=gpd_emd,
        out_dir=Path("result/report_figures"),
        top_n=10,
    )


if __name__ == "__main__":
    main()
