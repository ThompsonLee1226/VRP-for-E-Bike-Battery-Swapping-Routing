"""Simple CatBoost batch sweep runner.

Goals:
1. Sweep multiple parameter combinations for CatBoost training.
2. Append parameters + training results into existing training_summary.csv.

This file keeps current project architecture unchanged and only wraps CatBoost_train.py.
"""

from __future__ import annotations

import copy
import itertools
import os
import time
from typing import Any, Dict, Iterable, List, Optional

import CatBoost_train as cb_train
from training_summary_manager import append_row as append_summary_row
from training_summary_manager import build_run_summary_row


# =========================
# Batch run configuration
# =========================
# Change these values directly in this file. No CLI arguments are required.
TRAIN_FILE = cb_train.TRAIN_FILE
TEST_FILE = cb_train.TEST_FILE
RUN_PREFIX = "auto_cb"
TRAIN_SCALE: Optional[int] = None  # None = full-data training; e.g. 20000 for quick test
MAX_RUNS = 0  # 0 = run all parameter combinations


def build_param_grid() -> Dict[str, List[Any]]:
    """Define parameter combinations for sweep.

    Edit only this function when you want to try new parameter values.
    """
    return {
        "learning_rate": [0.015, 0.02, 0.03],
        "depth": [8, 9, 10],
        "l2_leaf_reg": [3.0, 4.0, 5.0],
        "od_wait": [25],
        "iterations": [3000],
    }


def iter_param_combinations(grid: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    """Generate one dict per Cartesian-product combination."""
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def apply_params_to_catboost(params: Dict[str, Any]) -> None:
    """Apply one parameter set to CatBoost_train global config."""
    cb_train.CB_PARAMS["learning_rate"] = params["learning_rate"]
    cb_train.CB_PARAMS["depth"] = params["depth"]
    cb_train.CB_PARAMS["l2_leaf_reg"] = params["l2_leaf_reg"]
    cb_train.CB_PARAMS["od_wait"] = params["od_wait"]
    cb_train.CB_ITERATIONS = params["iterations"]


def run_one(
    run_index: int,
    params: Dict[str, Any],
    train_file: str,
    test_file: str,
    scale: Optional[int],
    run_prefix: str,
) -> None:
    """Train one parameter combination and append result to training_summary.csv."""
    original_cb_params = copy.deepcopy(cb_train.CB_PARAMS)
    original_iterations = cb_train.CB_ITERATIONS

    try:
        apply_params_to_catboost(params)

        run_ts = time.strftime("%Y%m%d_%H%M%S")
        run_id = f"{run_prefix}_{run_ts}_{run_index:03d}"
        run_output_dir = os.path.join(cb_train.TRAINING_RESULTS_DIR, run_id)
        os.makedirs(run_output_dir, exist_ok=True)

        scale_tag = "all" if scale is None else str(scale)

        print("\n" + "=" * 80)
        print(f"Run {run_index}: {run_id}")
        print(f"Scale: {scale_tag}")
        print(f"Params: {params}")
        print("=" * 80)

        df, features = cb_train.load_and_preprocess(train_file, scale=scale)

        rent_model, rent_summary = cb_train.train_model(
            df,
            features,
            target_name="rent",
            scale_tag=scale_tag,
            run_timestamp=run_id,
            run_output_dir=run_output_dir,
        )
        return_model, return_summary = cb_train.train_model(
            df,
            features,
            target_name="return",
            scale_tag=scale_tag,
            run_timestamp=run_id,
            run_output_dir=run_output_dir,
        )

        prediction_file = os.path.join(
            run_output_dir,
            cb_train.PREDICTION_OUTPUT_TEMPLATE.format(scale=scale_tag, ts=run_id),
        )
        cb_train.predict_on_test_data(
            models={"rent": rent_model, "return": return_model},
            feature_cols=features,
            test_file=test_file,
            output_file=prediction_file,
        )

        run_summary_row = build_run_summary_row(
            run_timestamp=run_id,
            scale_tag=scale_tag,
            train_file=train_file,
            test_file=test_file,
            output_file=prediction_file,
            report_space=rent_summary["report_space"],
            use_log_target=cb_train.USE_LOG_TARGET,
            split_mode=cb_train.SPLIT_MODE,
            train_size=rent_summary["train_size"],
            valid_size=rent_summary["valid_size"],
            train_time_range=rent_summary["train_time_range"],
            valid_time_range=rent_summary["valid_time_range"],
            rent_summary=rent_summary,
            return_summary=return_summary,
            shared_config={
                'cb_loss_function': cb_train.CB_PARAMS['loss_function'],
                'cb_eval_metric': cb_train.CB_PARAMS['eval_metric'],
                'cb_learning_rate': cb_train.CB_PARAMS['learning_rate'],
                'cb_depth': cb_train.CB_PARAMS['depth'],
                'cb_l2_leaf_reg': cb_train.CB_PARAMS['l2_leaf_reg'],
                'cb_random_seed': cb_train.CB_PARAMS['random_seed'],
                'cb_task_type': cb_train.CB_PARAMS['task_type'],
                'cb_devices': cb_train.CB_PARAMS['devices'],
                'cb_od_type': cb_train.CB_PARAMS['od_type'],
                'cb_od_wait': cb_train.CB_PARAMS['od_wait'],
                'cb_iterations': cb_train.CB_ITERATIONS,
                'cb_log_eval_period': cb_train.CB_LOG_EVAL_PERIOD,
                'train_valid_test_size': cb_train.TRAIN_VALID_TEST_SIZE,
                'train_valid_random_state': cb_train.TRAIN_VALID_RANDOM_STATE,
                'categorical_features': '|'.join(cb_train.CB_CATEGORICAL_FEATURES),
            },
        )
        append_summary_row(cb_train.TRAINING_SUMMARY_CSV, run_summary_row)
        print(f"Run summary appended to: {cb_train.TRAINING_SUMMARY_CSV}")

    finally:
        # Ensure the next run starts from clean baseline parameters.
        cb_train.CB_PARAMS = original_cb_params
        cb_train.CB_ITERATIONS = original_iterations


def main() -> None:
    """Run sweep: generate parameter combinations and train one by one."""
    grid = build_param_grid()
    combos = list(iter_param_combinations(grid))
    if MAX_RUNS > 0:
        combos = combos[: MAX_RUNS]

    print(f"Total parameter combinations: {len(combos)}")
    print(f"Training summary CSV: {cb_train.TRAINING_SUMMARY_CSV}")
    print(f"Train file: {TRAIN_FILE}")
    print(f"Test file: {TEST_FILE}")
    print(f"Scale: {'all' if TRAIN_SCALE is None else TRAIN_SCALE}")

    for idx, params in enumerate(combos, start=1):
        run_one(
            run_index=idx,
            params=params,
            train_file=TRAIN_FILE,
            test_file=TEST_FILE,
            scale=TRAIN_SCALE,
            run_prefix=RUN_PREFIX,
        )

    print("\nBatch sweep complete.")


if __name__ == "__main__":
    main()
