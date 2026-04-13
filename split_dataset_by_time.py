#!/usr/bin/env python3.9
"""
Split a dataset by time order.
Default behavior:
- Sort by `datetime` ascending
- First 70% rows -> train set
- Last 30% rows -> valid set
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import pandas as pd


@dataclass
class SplitConfig:
    input_file: str
    datetime_col: str
    train_ratio: float
    ascending: bool
    output_train: str
    output_valid: str


def parse_args() -> SplitConfig:
    parser = argparse.ArgumentParser(description="Split CSV dataset by datetime order")
    parser.add_argument(
        "--input-file",
        default="battery_swapping_routing_data.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--datetime-col",
        default="datetime",
        help="Datetime column name",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train ratio in (0,1), e.g. 0.7",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Use descending datetime sort (default is ascending)",
    )
    parser.add_argument(
        "--output-train",
        default="battery_swapping_routing_data_train_time70.csv",
        help="Output CSV path for train split",
    )
    parser.add_argument(
        "--output-valid",
        default="battery_swapping_routing_data_valid_time30.csv",
        help="Output CSV path for valid split",
    )

    args = parser.parse_args()

    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1")

    return SplitConfig(
        input_file=args.input_file,
        datetime_col=args.datetime_col,
        train_ratio=args.train_ratio,
        ascending=not args.descending,
        output_train=args.output_train,
        output_valid=args.output_valid,
    )


def format_time_range(series: pd.Series) -> str:
    if series.empty:
        return "N/A"
    return f"{series.min()} -> {series.max()}"


def split_by_time(cfg: SplitConfig) -> None:
    if not os.path.exists(cfg.input_file):
        raise FileNotFoundError(f"Input file not found: {cfg.input_file}")

    df = pd.read_csv(cfg.input_file)
    if cfg.datetime_col not in df.columns:
        raise KeyError(f"Datetime column not found: {cfg.datetime_col}")

    dt = pd.to_datetime(df[cfg.datetime_col], errors="coerce")
    invalid_count = int(dt.isna().sum())
    if invalid_count > 0:
        raise ValueError(
            f"Found {invalid_count} rows with invalid datetime in column '{cfg.datetime_col}'. "
            "Please clean data first or change datetime column."
        )

    # Keep original row index for stable, explicit ordering after sorting.
    work = df.copy()
    work["__parsed_datetime__"] = dt
    work["__row_id__"] = range(len(work))

    work = work.sort_values(
        by=["__parsed_datetime__", "__row_id__"],
        ascending=[cfg.ascending, True],
        kind="mergesort",
    ).reset_index(drop=True)

    n = len(work)
    split_idx = int(n * cfg.train_ratio)
    if split_idx <= 0 or split_idx >= n:
        raise ValueError(
            "Split index is invalid. Check dataset size and --train-ratio value."
        )

    train_df = work.iloc[:split_idx].drop(columns=["__parsed_datetime__", "__row_id__"])
    valid_df = work.iloc[split_idx:].drop(columns=["__parsed_datetime__", "__row_id__"])

    train_df.to_csv(cfg.output_train, index=False)
    valid_df.to_csv(cfg.output_valid, index=False)

    train_dt = pd.to_datetime(train_df[cfg.datetime_col], errors="coerce")
    valid_dt = pd.to_datetime(valid_df[cfg.datetime_col], errors="coerce")

    print("Split completed.")
    print(f"Input rows: {n}")
    print(f"Train rows: {len(train_df)} ({len(train_df) / n:.2%})")
    print(f"Valid rows: {len(valid_df)} ({len(valid_df) / n:.2%})")
    print(f"Train time range: {format_time_range(train_dt)}")
    print(f"Valid time range: {format_time_range(valid_dt)}")
    print(f"Train file: {cfg.output_train}")
    print(f"Valid file: {cfg.output_valid}")


def main() -> None:
    cfg = parse_args()
    split_by_time(cfg)


if __name__ == "__main__":
    main()
