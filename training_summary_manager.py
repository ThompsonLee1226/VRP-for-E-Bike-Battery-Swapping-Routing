"""Helpers for long-term CSV logging of training runs."""

from __future__ import annotations

import os
from typing import Any, Dict

import pandas as pd


def append_row(csv_path: str, row: Dict[str, Any]) -> None:
    parent_dir = os.path.dirname(csv_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    summary_df = pd.DataFrame([row])
    if os.path.exists(csv_path):
        summary_df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        summary_df.to_csv(csv_path, index=False)


def format_top_features(feature_importance_df: pd.DataFrame, top_n: int = 5) -> str:
    if feature_importance_df.empty:
        return ''
    rows = feature_importance_df.head(top_n)
    return '; '.join(f"{row.feature}:{row.importance:.6f}" for row in rows.itertuples(index=False))


def build_run_summary_row(
    *,
    run_timestamp: str,
    scale_tag: str,
    train_file: str,
    test_file: str,
    output_file: str,
    report_space: str,
    use_log_target: bool,
    split_mode: str,
    train_size: int,
    valid_size: int,
    train_time_range: str,
    valid_time_range: str,
    rent_summary: Dict[str, Any],
    return_summary: Dict[str, Any],
    shared_config: Dict[str, Any],
) -> Dict[str, Any]:
    row = {
        'run_timestamp': run_timestamp,
        'scale_tag': scale_tag,
        'train_file': train_file,
        'test_file': test_file,
        'output_file': output_file,
        'report_metric_space': report_space,
        'use_log_target': use_log_target,
        'split_mode': split_mode,
        'train_size': train_size,
        'valid_size': valid_size,
        'train_time_range': train_time_range,
        'valid_time_range': valid_time_range,
        'rent_best_iteration': rent_summary.get('best_iteration'),
        'rent_validation_rmse_objective': rent_summary.get('best_validation_rmse_objective'),
        'rent_final_validation_rmse': rent_summary.get('final_validation_rmse'),
        'rent_training_seconds': rent_summary.get('training_seconds'),
        'rent_top_features': rent_summary.get('top_features'),
        'return_best_iteration': return_summary.get('best_iteration'),
        'return_validation_rmse_objective': return_summary.get('best_validation_rmse_objective'),
        'return_final_validation_rmse': return_summary.get('final_validation_rmse'),
        'return_training_seconds': return_summary.get('training_seconds'),
        'return_top_features': return_summary.get('top_features'),
    }

    row.update(shared_config)
    return row
