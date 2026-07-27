import os
import random
import argparse
import numpy as np
import pandas as pd

# =============================================================================
# 日期时间配置 —— 在代码最前面预先声明
# =============================================================================
# ★ 选择目标日期时间: 格式 "YYYY/MM/DD HH:00"
#    数据有效范围: 2025/10/23 12:00 ~ 2025/11/10 23:00 (445个可用小时)
#    修改此处即可切换优化目标小时
DEFAULT_TARGET_DATETIME = "2025/10/23 12:00"
DATETIME_RANGE_START = "2025/10/23 12:00"
DATETIME_RANGE_END   = "2025/11/10 23:00"

# ★ 默认预测数据: CB_Hurdle 最优训练结果
#    修改 TRAINING_RESULT_DIR 即可切换训练结果来源
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_RESULT_DIR = os.path.normpath(os.path.join(
    _SCRIPT_DIR, "..", "2_Training", "Training_Results", "20260727_115935"
))
DEFAULT_PREDICTION_FILE = os.path.join(TRAINING_RESULT_DIR, "prediction_CB_Hurdle.csv")


def list_available_hours(file_path=None, start=None, end=None):
    """列出 CSV 文件中所有可用的唯一小时时间戳。

    参数:
        file_path: CSV 文件路径, 默认使用 DEFAULT_PREDICTION_FILE
        start:     起始时间 (str 或 None), 默认 DATETIME_RANGE_START
        end:       结束时间 (str 或 None), 默认 DATETIME_RANGE_END

    返回:
        list[pd.Timestamp]: 排序后的可用小时列表
    """
    if file_path is None:
        file_path = DEFAULT_PREDICTION_FILE
    if start is None:
        start = DATETIME_RANGE_START
    if end is None:
        end = DATETIME_RANGE_END

    df = pd.read_csv(file_path, usecols=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    hours = df["datetime"].dt.floor("h").dropna().unique()
    hours = sorted(hours)

    start_ts = pd.Timestamp(start).floor("h")
    end_ts = pd.Timestamp(end).floor("h")
    hours_in_range = [h for h in hours if start_ts <= h <= end_ts]
    return hours_in_range


def select_random_datetime(file_path=None, start=None, end=None, seed=None):
    """从 CSV 文件的指定时间范围内随机选取一个小时的 datetime。

    参数:
        file_path: CSV 文件路径, 默认使用 DEFAULT_PREDICTION_FILE
        start:     起始时间 (str 或 None), 默认 DATETIME_RANGE_START
        end:       结束时间 (str 或 None), 默认 DATETIME_RANGE_END
        seed:      随机种子 (int 或 None), 用于结果可复现

    返回:
        str: 格式为 "YYYY/MM/DD HH:00" 的日期时间字符串
    """
    if seed is not None:
        random.seed(seed)

    hours = list_available_hours(file_path=file_path, start=start, end=end)
    if not hours:
        raise ValueError(
            f"No available hours found in range [{start}, {end}] "
            f"from file: {file_path or DEFAULT_PREDICTION_FILE}"
        )
    selected = random.choice(hours)
    return selected.strftime("%Y/%m/%d %H:%M")


# =============================================================================
# 验证与快照加载
# =============================================================================

def validate_required_columns(df, required_cols, dataset_name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} missing required columns: {missing}")


def load_prediction_snapshot(file_path, target_datetime=DEFAULT_TARGET_DATETIME):
    """读取指定时间点的全部节点预测数据。"""
    df = pd.read_csv(file_path)
    validate_required_columns(df, ["datetime", "h3"], "prediction file")

    if "rent_pred" not in df.columns and "rent" not in df.columns:
        raise ValueError("prediction file must contain rent_pred or rent")
    if "return_pred" not in df.columns and "return" not in df.columns:
        raise ValueError("prediction file must contain return_pred or return")

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    target_ts = pd.Timestamp(pd.to_datetime(target_datetime)).floor("h")
    hour_mask = df["datetime"].dt.floor("h") == target_ts
    snapshot_df = df.loc[hour_mask].copy()

    if snapshot_df.empty:
        raise ValueError(f"No rows found for target datetime {target_ts}.")

    snapshot_df["h3"] = snapshot_df["h3"].astype(str)
    snapshot_df = snapshot_df.drop_duplicates(subset=["h3"], keep="last").reset_index(drop=True)
    return snapshot_df


def calculate_theta(snapshot_df):
    total_low = snapshot_df["low_power_bike_count"].sum()
    total_soon = snapshot_df["soon_low_power_bike_count"].sum()
    total_normal = snapshot_df["normal_power_bike_count"].sum()

    total_all_bikes = total_soon + total_normal

    if total_all_bikes > 0:
        # 计算全局比例
        theta_soon_global = float(total_soon / total_all_bikes)
        theta_normal_global = float(total_normal / total_all_bikes)
    return theta_soon_global, theta_normal_global


def build_grid_params_from_snapshot(
    snapshot_df,
    theta_soon_global=None,
    theta_normal_global=None,
):
    """把全节点切片整理成 Optimize / Utility 可直接使用的参数字典。"""
    required_cols = [
        "h3",
        "low_power_bike_count",
        "soon_low_power_bike_count",
        "normal_power_bike_count",
    ]
    validate_required_columns(snapshot_df, required_cols, "snapshot data")

    if theta_soon_global is None or theta_normal_global is None:
        theta_soon_global, theta_normal_global = calculate_theta(snapshot_df)

    rent_col = "rent_pred" if "rent_pred" in snapshot_df.columns else "rent"
    return_col = "return_pred" if "return_pred" in snapshot_df.columns else "return"

    grid_params = {}
    for row in snapshot_df.itertuples(index=False):
        node_id = str(getattr(row, "h3"))
        grid_params[node_id] = {
            "n_low": float(getattr(row, "low_power_bike_count")),
            "n_soon": float(getattr(row, "soon_low_power_bike_count")),
            "n_normal": float(getattr(row, "normal_power_bike_count")),
            "theta_soon": float(theta_soon_global),
            "theta_normal": float(theta_normal_global),
            "rho": float(getattr(row, return_col)),
            "lam": float(getattr(row, rent_col)),
            "datetime": pd.Timestamp(getattr(row, "datetime")),
        }

    return grid_params


def prepare_optimize_inputs(
    file_path,
    target_datetime=DEFAULT_TARGET_DATETIME,
    theta_soon_global=None,
    theta_normal_global=None,
):
    """一站式生成 Optimize 所需的 grids 与 grid_params。"""
    snapshot_df = load_prediction_snapshot(file_path, target_datetime=target_datetime)
    if theta_soon_global is None or theta_normal_global is None:
        theta_soon_global, theta_normal_global = calculate_theta(snapshot_df)
    grid_params = build_grid_params_from_snapshot(
        snapshot_df,
        theta_soon_global=theta_soon_global,
        theta_normal_global=theta_normal_global,
    )
    grids = list(grid_params.keys())
    return grids, grid_params, snapshot_df


def generate_offline_utility_matrix(grids, C_max, T_total, P_intervals, grid_params, calc_utility_func, progress_bar=None, y_levels=None):
    """
    预计算静态效用矩阵 Omega，实施连续时间向离散断点转换。

    参数:
    - grids: 区域内格点的ID集合列表
    - C_max: 车载电池最大容量限制
    - T_total: 规划周期总时长
    - P_intervals: 分段逼近的间隔数
    - grid_params: 字典，包含预测得到的各点初始状态 (n_low, n_soon, n_normal) 及流率 (rho, lam)
    - calc_utility_func: 先前实现的函数 calculate_operational_utility
    - y_levels: 可选，离散换电量列表。若为 None 则使用 range(1, C_max+1)

    返回:
    - Omega: 嵌套字典，结构为 Omega[j][y][s]
    - tau_list: 物理时间断点列表
    """
    tau_list = np.linspace(0, T_total, P_intervals + 1).tolist()
    _y_levels = sorted(y_levels) if y_levels is not None else range(1, C_max + 1)
    Omega = {}
    done_steps = 0
    for j in grids:
        Omega[j] = {}
        params = grid_params[j]

        for y in _y_levels:
            Omega[j][y] = {}
            for s, tau_s in enumerate(tau_list):
                utility_val = calc_utility_func(
                    u_j=tau_s,
                    y_j=y,
                    n_low=params["n_low"],
                    n_soon=params["n_soon"],
                    n_normal=params["n_normal"],
                    theta_soon_global=params["theta_soon"],
                    theta_normal_global=params["theta_normal"],
                    rho_j=params["rho"],
                    lam_j=params["lam"],
                    T=T_total,
                )
                Omega[j][y][s] = max(0.0, utility_val)
                done_steps += 1
                if progress_bar is not None:
                    progress_bar.update(done_steps, suffix=f"grid {j}")

    if progress_bar is not None:
        progress_bar.close("utility matrix ready")

    return Omega, tau_list


# -------------------------------------------------------------------------
# 网格间旅行时间计算 (基于经纬度 + Haversine 公式)
# -------------------------------------------------------------------------

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    使用 Haversine 公式计算两点之间的球面距离 (单位: km)。

    参数:
    - lat1, lon1: 起点纬度和经度 (度)
    - lat2, lon2: 终点纬度和经度 (度)

    返回:
    - distance: 两点间的球面距离 (km)
    """
    R = 6371.0  # 地球平均半径 (km)

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return R * c


def extract_grid_coordinates(snapshot_df):
    """
    从快照 DataFrame 中提取每个 grid 的经纬度。

    参数:
    - snapshot_df: 包含 'h3', 'latitude', 'longitude' 列的 DataFrame

    返回:
    - grid_coords: dict, grid_coords[grid_id] = (latitude, longitude)
    """
    validate_required_columns(snapshot_df, ["h3", "latitude", "longitude"], "snapshot for coordinates")

    grid_coords = {}
    for row in snapshot_df.itertuples(index=False):
        grid_id = str(getattr(row, "h3"))
        lat = float(getattr(row, "latitude"))
        lon = float(getattr(row, "longitude"))
        grid_coords[grid_id] = (lat, lon)

    return grid_coords


def calculate_travel_time_matrix(grids, grid_coords, vehicle_speed_kmh=30.0):
    """
    计算网格间的旅行时间矩阵，基于经纬度坐标和恒定车速。

    参数:
    - grids: 格点 ID 列表
    - grid_coords: dict, grid_coords[grid_id] = (latitude, longitude)
    - vehicle_speed_kmh: 车辆恒定行驶速度 (km/h)，默认 30 km/h

    返回:
    - travel_time: dict of dict, travel_time[i][j] 表示从 grid i 到 grid j 的旅行时间 (小时)
    """
    if vehicle_speed_kmh <= 0:
        raise ValueError("vehicle_speed_kmh must be positive.")

    travel_time = {}
    for i in grids:
        travel_time[i] = {}
        lat_i, lon_i = grid_coords[i]
        for j in grids:
            if i == j:
                travel_time[i][j] = 0.0
            else:
                lat_j, lon_j = grid_coords[j]
                dist_km = haversine_distance(lat_i, lon_i, lat_j, lon_j)
                travel_time[i][j] = dist_km / vehicle_speed_kmh

    return travel_time


def build_travel_time_matrix_from_csv(file_path, target_datetime=DEFAULT_TARGET_DATETIME, vehicle_speed_kmh=30.0):
    """
    一站式从 CSV 文件读取数据并构建旅行时间矩阵。

    参数:
    - file_path: CSV 文件路径 (需包含 h3, latitude, longitude 列)
    - target_datetime: 目标时间点
    - vehicle_speed_kmh: 车辆恒定行驶速度 (km/h)，默认 30 km/h

    返回:
    - travel_time: dict of dict, travel_time[i][j] 旅行时间 (小时)
    - grids: 格点 ID 列表
    - grid_coords: dict, 各格点的经纬度
    """
    snapshot_df = load_prediction_snapshot(file_path, target_datetime=target_datetime)
    grids = list(snapshot_df["h3"].astype(str))
    grid_coords = extract_grid_coordinates(snapshot_df)
    travel_time = calculate_travel_time_matrix(grids, grid_coords, vehicle_speed_kmh=vehicle_speed_kmh)
    return travel_time, grids, grid_coords


# =============================================================================
# CLI: 预处理工具入口 —— 支持日期时间选择声明
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre_Process: 从预测 CSV 中提取指定小时的快照数据并输出摘要",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python Pre_Process.py                                          # 使用默认时间 %s
  python Pre_Process.py --datetime "2025/10/28 14:00"            # 指定具体时间
  python Pre_Process.py --random                                 # 随机选取一个可用小时
  python Pre_Process.py --random --seed 42                       # 随机选取 (固定种子)
  python Pre_Process.py --list-hours                             # 列出所有可用小时
  python Pre_Process.py --list-hours --start "2025/10/24" --end "2025/10/27"
  python Pre_Process.py --file path/to/other.csv                 # 指定其他 CSV 文件
        """ % DEFAULT_TARGET_DATETIME,
    )
    parser.add_argument(
        "--file", type=str, default=DEFAULT_PREDICTION_FILE,
        help=f"预测数据 CSV 文件路径 (默认: {os.path.basename(DEFAULT_PREDICTION_FILE)})"
    )
    parser.add_argument(
        "--datetime", type=str, default=None,
        help="目标日期时间, 格式: 'YYYY/MM/DD HH:MM' 或 'YYYY/MM/DD HH:00' (例如 '2025/10/28 14:00')"
    )
    parser.add_argument(
        "--random", action="store_true",
        help="从可用小时中随机选取一个 (范围: %s ~ %s)" % (DATETIME_RANGE_START, DATETIME_RANGE_END)
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="配合 --random 使用, 固定随机种子以实现结果复现"
    )
    parser.add_argument(
        "--start", type=str, default=DATETIME_RANGE_START,
        help=f"随机选取的起始时间 (默认: {DATETIME_RANGE_START})"
    )
    parser.add_argument(
        "--end", type=str, default=DATETIME_RANGE_END,
        help=f"随机选取的结束时间 (默认: {DATETIME_RANGE_END})"
    )
    parser.add_argument(
        "--list-hours", action="store_true",
        help="列出 CSV 中所有可用的小时并退出"
    )

    args = parser.parse_args()

    # =========================================================================
    # 模式 1: 列出所有可用小时
    # =========================================================================
    if args.list_hours:
        hours = list_available_hours(
            file_path=args.file,
            start=args.start,
            end=args.end,
        )
        print(f"文件: {args.file}")
        print(f"时间范围: {args.start} ~ {args.end}")
        print(f"可用小时数: {len(hours)}")
        print("-" * 40)
        for h in hours:
            print(h.strftime("%Y/%m/%d %H:%M"))
        exit(0)

    # =========================================================================
    # 模式 2: 确定目标日期时间
    # =========================================================================
    if args.random:
        target_datetime = select_random_datetime(
            file_path=args.file,
            start=args.start,
            end=args.end,
            seed=args.seed,
        )
        print("=" * 60)
        print("  [随机] 随机选取模式 — 已从可用小时中随机选择")
        if args.seed is not None:
            print(f"  随机种子: {args.seed}")
    elif args.datetime is not None:
        target_datetime = args.datetime
        print("=" * 60)
        print("  [指定] 用户指定日期时间模式")
    else:
        target_datetime = DEFAULT_TARGET_DATETIME
        print("=" * 60)
        print("  [默认] 使用默认日期时间模式")

    print(f"  选择的目标时间: {target_datetime}")
    print(f"  数据文件: {args.file}")
    print("=" * 60)

    # =========================================================================
    # 模式 3: 加载快照并输出摘要
    # =========================================================================
    grids, grid_params, snapshot_df = prepare_optimize_inputs(
        file_path=args.file,
        target_datetime=target_datetime,
    )

    print(f"\n  [OK] 快照加载成功")
    print(f"  Node count: {len(grids)}")
    print(f"  Selected hour: {pd.Timestamp(target_datetime).floor('h')}")
    print(f"\n  前 5 个节点预览:")
    print(snapshot_df[["h3", "datetime", "rent_pred", "return_pred"]].head().to_string(index=False))
