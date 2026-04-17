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
    split_mode: str,
    train_size: int,
    valid_size: int,
    train_time_range: str,
    valid_time_range: str,
    rent_summary: Dict[str, Any],
    return_summary: Dict[str, Any],
    model_type: str,
    shared_config: Dict[str, Any],
) -> Dict[str, Any]:
    row = {
        'run_timestamp': run_timestamp,
        'model_type': model_type,
        'scale_tag': scale_tag,
        'split_mode': split_mode,
        'train_size': train_size,
        'valid_size': valid_size,
        'train_time_range': train_time_range,
        'valid_time_range': valid_time_range,
        'rent_best_iteration': rent_summary.get('best_iteration'),
        'return_best_iteration': return_summary.get('best_iteration'),
        'rent_best_test': rent_summary.get('best_test'),
        'return_best_test': return_summary.get('best_test'),
        'rent_final_metric': rent_summary.get('final_metric', rent_summary.get('final_validation_rmse')),
        'return_final_metric': return_summary.get('final_metric', return_summary.get('final_validation_rmse')),
        'rent_classifier_logloss': rent_summary.get('classifier_logloss'),
        'rent_regressor_poisson_pos': rent_summary.get('regressor_poisson_pos'),
        'return_classifier_logloss': return_summary.get('classifier_logloss'),
        'return_regressor_poisson_pos': return_summary.get('regressor_poisson_pos'),
    }

    row.update(shared_config)
    return row
