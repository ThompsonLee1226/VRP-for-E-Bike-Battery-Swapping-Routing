"""Unified batch sweep runner for CatBoost and Hurdle models.

Usage:
    python unified_batch_runner.py
        -> Run batch sweep with model type from training_config.py (AUTO_BATCH_MODEL_TYPE).

    python unified_batch_runner.py --model-type CB
        -> Batch sweep for standard CatBoost model.

    python unified_batch_runner.py --model-type CB_Hurdle
        -> Batch sweep for Hurdle (classifier + regressor) model.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import CatBoost_train as cb_train
import CB_Hurdle_train as hurdle_train
from training_summary_manager import append_row as append_summary_row
from training_summary_manager import build_run_summary_row

try:
    import training_config as cfg
except ImportError:
    cfg = None


def cfg_value(name: str, default: Any) -> Any:
    if cfg is None:
        return default
    return getattr(cfg, name, default)


# =========================
# Batch run configuration
# =========================
TRAIN_FILE = cb_train.TRAIN_FILE
TEST_FILE = cb_train.TEST_FILE
RUN_PREFIX = cfg_value('AUTO_BATCH_RUN_PREFIX', 'auto_cb')
TRAIN_SCALE: Optional[int] = cfg_value('AUTO_BATCH_TRAIN_SCALE', None)
MAX_RUNS = cfg_value('AUTO_BATCH_MAX_RUNS', 0)
DEFAULT_MODEL_TYPE = cfg_value('AUTO_BATCH_MODEL_TYPE', 'CB')


def build_param_grid() -> Dict[str, List[Any]]:
    return copy.deepcopy(cfg_value('AUTO_BATCH_PARAM_GRID', {
        'learning_rate': [0.015, 0.02, 0.03],
        'depth': [8, 9, 10],
        'l2_leaf_reg': [3.0, 4.0, 5.0],
        'od_wait': [25],
        'iterations': [3000],
    }))


def iter_param_combinations(grid: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def apply_params_to_catboost(params: Dict[str, Any]) -> None:
    cb_train.CB_PARAMS["learning_rate"] = params["learning_rate"]
    cb_train.CB_PARAMS["depth"] = params["depth"]
    cb_train.CB_PARAMS["l2_leaf_reg"] = params["l2_leaf_reg"]
    cb_train.CB_PARAMS["od_wait"] = params["od_wait"]
    cb_train.CB_ITERATIONS = params["iterations"]


def apply_params_to_hurdle(params: Dict[str, Any]) -> None:
    for key in ("learning_rate", "depth", "l2_leaf_reg", "od_wait"):
        hurdle_train.CB_CLASSIFIER_PARAMS[key] = params[key]
        hurdle_train.CB_REGRESSOR_PARAMS[key] = params[key]
    hurdle_train.CB_ITERATIONS = params["iterations"]


def run_one_cb(
    run_id: str,
    params: Dict[str, Any],
    train_file: str,
    test_file: str,
    scale: Optional[int],
) -> None:
    original_cb_params = copy.deepcopy(cb_train.CB_PARAMS)
    original_iterations = cb_train.CB_ITERATIONS

    try:
        apply_params_to_catboost(params)
        run_output_dir = os.path.join(cb_train.TRAINING_RESULTS_DIR, run_id)
        os.makedirs(run_output_dir, exist_ok=True)

        scale_tag = "all" if scale is None else str(scale)
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
            split_mode=cb_train.SPLIT_MODE,
            train_size=rent_summary["train_size"],
            valid_size=rent_summary["valid_size"],
            train_time_range=rent_summary["train_time_range"],
            valid_time_range=rent_summary["valid_time_range"],
            rent_summary=rent_summary,
            return_summary=return_summary,
            model_type='CB',
            shared_config={
                'cb_loss_function': cb_train.CB_PARAMS['loss_function'],
                'cb_eval_metric': cb_train.CB_PARAMS['eval_metric'],
                'cb_learning_rate': cb_train.CB_PARAMS['learning_rate'],
                'cb_depth': cb_train.CB_PARAMS['depth'],
                'cb_l2_leaf_reg': cb_train.CB_PARAMS['l2_leaf_reg'],
                'cb_od_wait': cb_train.CB_PARAMS['od_wait'],
                'cb_iterations': cb_train.CB_ITERATIONS,
            },
        )
        append_summary_row(cb_train.TRAINING_SUMMARY_CSV, run_summary_row)
        print(f"Run summary appended to: {cb_train.TRAINING_SUMMARY_CSV}")

    finally:
        cb_train.CB_PARAMS = original_cb_params
        cb_train.CB_ITERATIONS = original_iterations


def run_one_hurdle(
    run_id: str,
    params: Dict[str, Any],
    train_file: str,
    test_file: str,
    scale: Optional[int],
) -> None:
    original_classifier = copy.deepcopy(hurdle_train.CB_CLASSIFIER_PARAMS)
    original_regressor = copy.deepcopy(hurdle_train.CB_REGRESSOR_PARAMS)
    original_iterations = hurdle_train.CB_ITERATIONS

    try:
        apply_params_to_hurdle(params)
        run_output_dir = os.path.join(hurdle_train.TRAINING_RESULTS_DIR, run_id)
        os.makedirs(run_output_dir, exist_ok=True)

        scale_tag = "all" if scale is None else str(scale)
        df, features = hurdle_train.load_and_preprocess(train_file, scale=scale)

        rent_models, rent_summary = hurdle_train.train_model(
            df,
            features,
            target_name="rent",
            scale_tag=scale_tag,
            run_timestamp=run_id,
            run_output_dir=run_output_dir,
        )
        return_models, return_summary = hurdle_train.train_model(
            df,
            features,
            target_name="return",
            scale_tag=scale_tag,
            run_timestamp=run_id,
            run_output_dir=run_output_dir,
        )

        prediction_file = os.path.join(
            run_output_dir,
            hurdle_train.PREDICTION_OUTPUT_TEMPLATE.format(scale=scale_tag, ts=run_id),
        )
        hurdle_train.predict_on_test_data(
            models_dict={"rent": rent_models, "return": return_models},
            feature_cols=features,
            test_file=test_file,
            output_file=prediction_file,
        )

        run_summary_row = build_run_summary_row(
            run_timestamp=run_id,
            scale_tag=scale_tag,
            split_mode=hurdle_train.SPLIT_MODE,
            train_size=rent_summary["train_size"],
            valid_size=rent_summary["valid_size"],
            train_time_range=rent_summary["train_time_range"],
            valid_time_range=rent_summary["valid_time_range"],
            rent_summary=rent_summary,
            return_summary=return_summary,
            model_type='CB_Hurdle',
            shared_config={
                'cb_loss_function': hurdle_train.CB_REGRESSOR_PARAMS['loss_function'],
                'cb_eval_metric': hurdle_train.CB_REGRESSOR_PARAMS['eval_metric'],
                'cb_learning_rate': hurdle_train.CB_REGRESSOR_PARAMS['learning_rate'],
                'cb_depth': hurdle_train.CB_REGRESSOR_PARAMS['depth'],
                'cb_l2_leaf_reg': hurdle_train.CB_REGRESSOR_PARAMS['l2_leaf_reg'],
                'cb_od_wait': hurdle_train.CB_REGRESSOR_PARAMS['od_wait'],
                'cb_iterations': hurdle_train.CB_ITERATIONS,
            },
        )
        append_summary_row(hurdle_train.TRAINING_SUMMARY_CSV, run_summary_row)
        print(f"Run summary appended to: {hurdle_train.TRAINING_SUMMARY_CSV}")

    finally:
        hurdle_train.CB_CLASSIFIER_PARAMS = original_classifier
        hurdle_train.CB_REGRESSOR_PARAMS = original_regressor
        hurdle_train.CB_ITERATIONS = original_iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        choices=["CB", "CB_Hurdle"],
        default=DEFAULT_MODEL_TYPE,
        help="Which model architecture to run",
    )
    args = parser.parse_args()

    grid = build_param_grid()
    combos = list(iter_param_combinations(grid))
    if MAX_RUNS > 0:
        combos = combos[: MAX_RUNS]

    print(f"Total parameter combinations: {len(combos)}")
    print(f"Train file: {TRAIN_FILE}")
    print(f"Test file: {TEST_FILE}")
    print(f"Scale: {'all' if TRAIN_SCALE is None else TRAIN_SCALE}")
    print(f"Model type: {args.model_type}")

    for idx, params in enumerate(combos, start=1):
        run_ts = time.strftime("%Y%m%d_%H%M%S")
        run_id = f"{RUN_PREFIX}_{run_ts}_{idx:03d}"
        print("\n" + "=" * 80)
        print(f"Run {idx}: {run_id}")
        print(f"Params: {params}")
        print("=" * 80)

        if args.model_type == "CB":
            run_one_cb(
                run_id=run_id,
                params=params,
                train_file=TRAIN_FILE,
                test_file=TEST_FILE,
                scale=TRAIN_SCALE,
            )
        else:
            run_one_hurdle(
                run_id=run_id,
                params=params,
                train_file=TRAIN_FILE,
                test_file=TEST_FILE,
                scale=TRAIN_SCALE,
            )

    print("\nBatch sweep complete.")


if __name__ == "__main__":
    main()
