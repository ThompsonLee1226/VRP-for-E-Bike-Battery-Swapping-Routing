import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = "battery_swapping_routing_data_valid_time30.csv"
DEFAULT_OUTPUT_MEAN = "EDA_Results/hour_h3_means.csv"
DEFAULT_OUTPUT_HOUR_MEAN = "EDA_Results/hour_means.csv"
DEFAULT_OUTPUT_MEDIAN = "EDA_Results/hour_h3_medians.csv"
DEFAULT_OUTPUT_HOUR_MEDIAN = "EDA_Results/hour_medians.csv"
DEFAULT_OUTPUT_MAX = "EDA_Results/hour_h3_max.csv"
DEFAULT_OUTPUT_HOUR_MAX = "EDA_Results/hour_max.csv"


def build_hour_h3_stats(df: pd.DataFrame, stat: str) -> pd.DataFrame:
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

    if stat == "median":
        grouped = df.groupby(["hour", "h3"], as_index=False)[metrics].median()
    elif stat == "max":
        grouped = df.groupby(["hour", "h3"], as_index=False)[metrics].max()
    else:
        grouped = df.groupby(["hour", "h3"], as_index=False)[metrics].mean()

    return grouped


def build_hour_stats(df: pd.DataFrame, stat: str) -> pd.DataFrame:
    metrics = [
        "rent",
        "return",
        "low_power_bike_count",
        "soon_low_power_bike_count",
        "normal_power_bike_count",
    ]

    if stat == "max":
        hour_h3 = build_hour_h3_stats(df, stat)
        hour_h3["rent_return_sum"] = hour_h3["rent"] + hour_h3["return"]
        idx = hour_h3.groupby("hour")["rent_return_sum"].idxmax()
        return (
            hour_h3.loc[idx, ["hour", *metrics]]
            .sort_values("hour")
            .reset_index(drop=True)
        )

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    df["hour"] = df["datetime"].dt.hour

    missing = [col for col in ["datetime", *metrics] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if stat == "median":
        grouped = df.groupby(["hour"], as_index=False)[metrics].median()
    else:
        grouped = df.groupby(["hour"], as_index=False)[metrics].mean()

    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute hourly mean/median/max rent/return and bike counts per H3 cell, and overall per hour."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV path")
    parser.add_argument(
        "--stat",
        default="mean",
        choices=["mean", "median", "max"],
        help="Aggregation type",
    )
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument(
        "--output-hour",
        default=None,
        help="Output CSV path for per-hour stats",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    result = build_hour_h3_stats(df, args.stat)
    result_hour = build_hour_stats(df, args.stat)

    if args.output is None:
        if args.stat == "median":
            output_path = Path(DEFAULT_OUTPUT_MEDIAN)
        elif args.stat == "max":
            output_path = Path(DEFAULT_OUTPUT_MAX)
        else:
            output_path = Path(DEFAULT_OUTPUT_MEAN)
    else:
        output_path = Path(args.output)

    if args.output_hour is None:
        if args.stat == "median":
            output_hour_path = Path(DEFAULT_OUTPUT_HOUR_MEDIAN)
        elif args.stat == "max":
            output_hour_path = Path(DEFAULT_OUTPUT_HOUR_MAX)
        else:
            output_hour_path = Path(DEFAULT_OUTPUT_HOUR_MEAN)
    else:
        output_hour_path = Path(args.output_hour)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    output_hour_path.parent.mkdir(parents=True, exist_ok=True)
    result_hour.to_csv(output_hour_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Saved: {output_hour_path}")
    print(f"Rows: {len(result)}")
    print(f"Rows: {len(result_hour)}")


if __name__ == "__main__":
    main()
