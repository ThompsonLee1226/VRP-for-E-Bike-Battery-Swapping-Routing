import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from Utility_Function import utility


DATA_PATH = "battery_swapping_routing_data_valid_time30.csv"
OUTPUT_DIR_DEFAULT = "utility_viz_random_hour"

# Utility parameters
U_TIME = 0.5
T_HOURS = 1.0
Y_MODE = "low"

RANDOM_SEED = 42
TOP_N = 20
HIST_BINS = 20
FIXED_HOUR = "2025-10-26 18:00:00"


def pick_random_hour(df: pd.DataFrame) -> pd.Timestamp:
    dt_series = pd.to_datetime(df["datetime"], errors="coerce")
    valid_dt = dt_series.dropna()
    if valid_dt.empty:
        raise ValueError("No valid datetime values found in dataset.")

    hours = valid_dt.dt.floor("h").drop_duplicates().sort_values()
    rng = np.random.default_rng(RANDOM_SEED)
    return pd.Timestamp(rng.choice(hours.to_numpy()))


def pick_fixed_hour(df: pd.DataFrame, fixed_hour: str) -> pd.Timestamp:
    dt_series = pd.to_datetime(df["datetime"], errors="coerce")
    valid_dt = dt_series.dropna()
    if valid_dt.empty:
        raise ValueError("No valid datetime values found in dataset.")

    target = pd.Timestamp(pd.to_datetime(fixed_hour)).floor("h")
    hours = valid_dt.dt.floor("h").drop_duplicates().sort_values()
    if target not in hours.values:
        raise ValueError(f"Fixed hour {target} not found in dataset.")
    return target


def build_hour_slice(df: pd.DataFrame, hour_ts: pd.Timestamp) -> pd.DataFrame:
    dt_series = pd.to_datetime(df["datetime"], errors="coerce")
    hour_series = dt_series.dt.floor("h")
    return df.loc[hour_series == hour_ts].copy()


def compute_utility(df: pd.DataFrame) -> pd.Series:
    if Y_MODE == "low+soon":
        y = df["low_power_bike_count"].to_numpy() + df["soon_low_power_bike_count"].to_numpy()
    else:
        y = df["low_power_bike_count"].to_numpy()

    return utility(
        rent_rate=df["rent"].to_numpy(),
        return_rate=df["return"].to_numpy(),
        u=np.full(len(df), U_TIME, dtype=float),
        y=y,
        n_low=df["low_power_bike_count"].to_numpy(),
        n_soon=df["soon_low_power_bike_count"].to_numpy(),
        n_normal=df["normal_power_bike_count"].to_numpy(),
        T=T_HOURS,
    )


def print_top_utilities(df: pd.DataFrame, top_n: int) -> None:
    cols = ["h3", "rent", "return", "utility"]
    top_df = df.sort_values("utility", ascending=False).head(top_n)
    print("\nTop utilities:")
    print(top_df[cols].to_string(index=False))


def plot_distribution(df: pd.DataFrame, output_path: str, title_suffix: str) -> None:
    values = df["utility"].dropna().to_numpy()
    if len(values) == 0:
        raise ValueError("No valid utility values to plot.")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(values, bins=HIST_BINS, color="#2a788e", edgecolor="#1b4d5c", alpha=0.85)
    ax.set_title(f"Utility Distribution ({title_suffix})")
    ax.set_xlabel("Utility")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close(fig)

    counts, bin_edges = np.histogram(values, bins=HIST_BINS)
    dist_df = pd.DataFrame({
        "bin_left": bin_edges[:-1],
        "bin_right": bin_edges[1:],
        "count": counts,
    })
    print("\nUtility distribution (bins):")
    print(dist_df.to_string(index=False))


def main(output_dir: str = OUTPUT_DIR_DEFAULT, fixed_hour: str | None = FIXED_HOUR) -> None:
    df = pd.read_csv(DATA_PATH)
    if fixed_hour:
        hour_ts = pick_fixed_hour(df, fixed_hour)
        title_suffix = "Fixed Hour"
    else:
        hour_ts = pick_random_hour(df)
        title_suffix = "Random Hour"
    hour_df = build_hour_slice(df, hour_ts)
    if hour_df.empty:
        raise ValueError("Selected hour slice is empty. Choose another hour or seed.")

    hour_df["utility"] = compute_utility(hour_df)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp(hour_ts).strftime("%Y%m%d_%H%M")

    print(f"Selected hour: {hour_ts}")
    print(f"Rows: {len(hour_df)}")
    print_top_utilities(hour_df, TOP_N)
    plot_distribution(hour_df, str(output_path / f"utility_distribution_{stamp}.png"), title_suffix)
    print(f"Saved distribution plot: {output_path}")


if __name__ == "__main__":
    main()
