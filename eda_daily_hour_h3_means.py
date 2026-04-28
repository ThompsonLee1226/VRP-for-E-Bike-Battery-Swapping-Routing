import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = "battery_swapping_routing_data_valid_time30.csv"
DEFAULT_OUTPUT = "EDA_Results/hour_h3_means.csv"
DEFAULT_OUTPUT_HOUR = "EDA_Results/hour_means.csv"


def build_hour_h3_means(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    df["hour"] = df["datetime"].dt.hour

    metrics = [
        "rent",
        "return",
        "low_power_bike_count",
        "soon_low_power_bike_count",
        "normal_power_bike_count",
    ]

    missing = [col for col in ["h3", "datetime", *metrics] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    grouped = df.groupby(["hour", "h3"], as_index=False)[metrics].mean()

    return grouped


def build_hour_means(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    df["hour"] = df["datetime"].dt.hour

    metrics = [
        "rent",
        "return",
        "low_power_bike_count",
        "soon_low_power_bike_count",
        "normal_power_bike_count",
    ]

    missing = [col for col in ["datetime", *metrics] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    grouped = df.groupby(["hour"], as_index=False)[metrics].mean()

    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute hourly mean rent/return and bike counts per H3 cell, and overall per hour."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument(
        "--output-hour",
        default=DEFAULT_OUTPUT_HOUR,
        help="Output CSV path for per-hour averages",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    result = build_hour_h3_means(df)
    result_hour = build_hour_means(df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    output_hour_path = Path(args.output_hour)
    output_hour_path.parent.mkdir(parents=True, exist_ok=True)
    result_hour.to_csv(output_hour_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Saved: {output_hour_path}")
    print(f"Rows: {len(result)}")
    print(f"Rows: {len(result_hour)}")


if __name__ == "__main__":
    main()
