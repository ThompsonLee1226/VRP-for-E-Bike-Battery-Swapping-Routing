"""
=============================================================================
实验配置模块 (Experiment Configuration)
=============================================================================
定义全部 5 个实验组 (M1-M5) 的参数配置, 包括:
  - 各模型的最优求解器配置
  - 对齐对比参数集 (用于公平比较)
  - 消融变体参数 (M2a, M2b)
  - 实例生成器配置

所有实验组共用相同的:
  - 数据文件路径
  - 硬件环境
  - TimeLimit = 1200s
  - C_max = 20, T_total = 1.0h, swap_time_c = 0.02h, speed = 30km/h

=============================================================================
"""

from __future__ import annotations
import os

# =============================================================================
# 路径配置
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ★ 默认数据文件: 从 Pre_Process 导入 CB_Hurdle 最优训练结果
#    若需切换数据源, 修改 Pre_Process.py 中的 TRAINING_RESULT_DIR
from Pre_Process import DEFAULT_PREDICTION_FILE
DEFAULT_DATA_FILE = DEFAULT_PREDICTION_FILE

DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "Optimization_Result_Summary")
DEFAULT_OUTPUT_DIR = os.path.normpath(DEFAULT_OUTPUT_DIR)

# =============================================================================
# 共享基础参数 (所有实验组共用)
# =============================================================================
SHARED_PARAMS = {
    "vehicle_speed_kmh": 30.0,
    "C_max": 20,
    "T_total": 1.0,
    "swap_time_c": 0.02,
    "max_travel_time": 0.2,
    "y_levels": list(range(1, 11)),      # 离散换电量 1~10
    "time_limit_s": 1200,                # 统一 20 分钟时限
}


# =============================================================================
# 实验组定义
# =============================================================================
# 说明:
#   每个 experiment group 包含:
#     - id:            实验代号
#     - name_cn:       中文全称
#     - name_en:       英文全称
#     - main_file:     入口脚本名称
#     - P_intervals:   时间离散段数
#     - mip_gap:       MIP 相对间隙
#     - method:        Gurobi LP 算法 (0=Primal, 2=Barrier)
#     - mip_focus:     MIP 搜索策略
#     - cuts:          割平面强度
#     - heuristics:    启发式时间占比
#     - geo_fencing:   是否启用零效用节点过滤
#     - knn_enabled:   是否启用 KNN 稠密图裁剪
#     - K_neighbors:   KNN 近邻数 (仅 M1/M2 系列)
#     - description:   学术定位说明
#     - ablation_of:   消融目标 (仅 M2 系列)

EXPERIMENT_GROUPS = {

    # =========================================================================
    # M1 — 提出的完整模型 (Proposed Model)
    # =========================================================================
    "M1": {
        "id": "M1",
        "name_cn": "时空图完整模型",
        "name_en": "ST-Graph (Full)",
        "main_file": "main_STGraph.py",
        "optimizer": "optimize_evrp_with_stgraph",
        "P_intervals": 12,
        "mip_gap": 0.01,
        "method": 2,                # Barrier — 适合大规模 LP
        "crossover": 0,
        "mip_focus": 0,             # 均衡
        "cuts": 1,                  # Moderate
        "heuristics": 0.10,
        "symmetry": 0,
        "no_rel_heur_time": 15,
        "pre_dual": 0,
        "pre_passes": 8,
        "node_method": 1,
        "improve_start_time": 0,
        "bar_homogeneous": 1,
        "threads": 8,
        # 空间剪枝
        "geo_fencing": True,
        "knn_enabled": True,
        "K_neighbors": 50,
        # 元信息
        "description": "Proposed model — ST-Graph with dual spatial sparsification",
        "academic_role": "Proposed Method",
    },

    # =========================================================================
    # M2a — 消融: 无 Geo-Fencing
    # =========================================================================
    "M2a": {
        "id": "M2a",
        "name_cn": "时空图—无Geo-Fencing",
        "name_en": "ST-Graph w/o Geo-Fencing",
        "main_file": "main_STGraph.py",
        "optimizer": "optimize_evrp_with_stgraph",
        "P_intervals": 12,
        "mip_gap": 0.01,
        "method": 2,
        "crossover": 0,
        "mip_focus": 0,
        "cuts": 1,
        "heuristics": 0.10,
        "symmetry": 0,
        "no_rel_heur_time": 15,
        "pre_dual": 0,
        "pre_passes": 8,
        "node_method": 1,
        "improve_start_time": 0,
        "bar_homogeneous": 1,
        "threads": 8,
        # 空间剪枝 — 关闭 Geo-Fencing
        "geo_fencing": False,
        "knn_enabled": True,
        "K_neighbors": 50,
        "description": "Ablation: disable Geo-Fencing zero-utility filtering",
        "academic_role": "Ablation Study",
        "ablation_of": "Geo-Fencing",
    },

    # =========================================================================
    # M2b — 消融: 无 KNN 裁剪
    # =========================================================================
    "M2b": {
        "id": "M2b",
        "name_cn": "时空图—无KNN裁剪",
        "name_en": "ST-Graph w/o KNN Pruning",
        "main_file": "main_STGraph.py",
        "optimizer": "optimize_evrp_with_stgraph",
        "P_intervals": 12,
        "mip_gap": 0.01,
        "method": 2,
        "crossover": 0,
        "mip_focus": 0,
        "cuts": 1,
        "heuristics": 0.10,
        "symmetry": 0,
        "no_rel_heur_time": 15,
        "pre_dual": 0,
        "pre_passes": 8,
        "node_method": 1,
        "improve_start_time": 0,
        "bar_homogeneous": 1,
        "threads": 8,
        # 空间剪枝 — 关闭 KNN (设置为极大值, 保留所有满足 max_travel_time 的弧)
        "geo_fencing": True,
        "knn_enabled": False,
        "K_neighbors": 99999,
        "description": "Ablation: disable KNN spatial graph pruning",
        "academic_role": "Ablation Study",
        "ablation_of": "KNN Pruning",
    },

    # =========================================================================
    # M2c — 消融: 完全无剪枝 (可选)
    # =========================================================================
    "M2c": {
        "id": "M2c",
        "name_cn": "时空图—无任何剪枝",
        "name_en": "ST-Graph w/o Any Pruning",
        "main_file": "main_STGraph.py",
        "optimizer": "optimize_evrp_with_stgraph",
        "P_intervals": 12,
        "mip_gap": 0.01,
        "method": 2,
        "crossover": 0,
        "mip_focus": 0,
        "cuts": 1,
        "heuristics": 0.10,
        "symmetry": 0,
        "no_rel_heur_time": 15,
        "pre_dual": 0,
        "pre_passes": 8,
        "node_method": 1,
        "improve_start_time": 0,
        "bar_homogeneous": 1,
        "threads": 8,
        # 空间剪枝 — 全部关闭
        "geo_fencing": False,
        "knn_enabled": False,
        "K_neighbors": 99999,
        "description": "Ablation: disable ALL spatial pruning (extreme baseline)",
        "academic_role": "Ablation Study (Optional)",
        "ablation_of": "All Pruning",
    },

    # =========================================================================
    # M3 — SOS2 分段线性逼近对比模型
    # =========================================================================
    "M3": {
        "id": "M3",
        "name_cn": "SOS2分段线性逼近",
        "name_en": "SOS2-PLA",
        "main_file": "main_SOS2.py",
        "optimizer": "optimize_evrp_with_pla",
        "P_intervals": 5,
        "mip_gap": 0.05,
        "method": 0,                # Primal Simplex — SOS2 模型 Barrier 容易空转
        "crossover": -1,
        "mip_focus": 1,             # 优先找可行解
        "cuts": 1,                  # Moderate
        "heuristics": 0.15,
        "symmetry": 2,
        "no_rel_heur_time": 60,
        "pre_dual": -1,
        "pre_passes": None,
        "node_method": None,
        "improve_start_time": None,
        "bar_homogeneous": None,
        "threads": 8,
        # 空间剪枝
        "geo_fencing": True,
        "knn_enabled": False,
        "K_neighbors": 0,
        "description": "Benchmark — SOS2 piecewise linear approximation + full MTZ",
        "academic_role": "Nonlinear Approximation Benchmark",
    },

    # =========================================================================
    # M4 — Delta 增量逼近对比模型
    # =========================================================================
    "M4": {
        "id": "M4",
        "name_cn": "Delta增量逼近",
        "name_en": "Delta-PLA",
        "main_file": "main_delta.py",
        "optimizer": "optimize_evrp_with_pla_delta",
        "P_intervals": 5,
        "mip_gap": 0.05,
        "method": 0,                # Primal Simplex
        "crossover": -1,
        "mip_focus": 1,
        "cuts": 2,                  # Aggressive — Delta 的 LP 更紧, 可以多切
        "heuristics": 0.15,
        "symmetry": 2,
        "no_rel_heur_time": 120,
        "pre_dual": -1,
        "pre_passes": 5,
        "node_method": None,
        "improve_start_time": None,
        "bar_homogeneous": None,
        "threads": 8,
        # 空间剪枝
        "geo_fencing": True,
        "knn_enabled": False,
        "K_neighbors": 0,
        "description": "Benchmark — Delta incremental approximation + full MTZ",
        "academic_role": "Nonlinear Approximation Benchmark",
    },

    # =========================================================================
    # M5 — MCF 多商品流基准模型
    # =========================================================================
    "M5": {
        "id": "M5",
        "name_cn": "多商品流基准模型",
        "name_en": "MCF-Baseline",
        "main_file": "main_delta_MCF.py",
        "optimizer": "optimize_evrp_with_pla_delta_mcf",
        "P_intervals": 10,          # 更细的离散化以公平对比 M1 的 P=12
        "mip_gap": 0.05,
        "method": 2,                # Barrier — MCF 约束高度同质, Barrier 优势显著
        "crossover": 0,
        "mip_focus": 0,
        "cuts": 0,                  # 关闭割平面 (LazyConstraints 限制)
        "heuristics": 0.10,
        "symmetry": 0,
        "no_rel_heur_time": 15,
        "pre_dual": 0,
        "pre_passes": 8,
        "node_method": 1,
        "improve_start_time": 0,
        "bar_homogeneous": 1,
        "threads": 8,
        "lazy_constraints": 1,      # 启用 Lazy MTZ callback
        # 空间剪枝
        "geo_fencing": True,
        "knn_enabled": False,
        "K_neighbors": 0,
        "description": "Classical baseline — Delta + MCF flow + Lazy MTZ callback",
        "academic_role": "Classical Baseline",
    },
}


# =============================================================================
# 对齐对比参数集 (用于公平比较)
# =============================================================================
# 当审稿人质疑 "M1 的 P=12 比 M3/M4 的 P=5 更细, 不公平" 时,
# 使用以下对齐参数重新运行实验作为敏感性分析。

ALIGNED_PARAMS = {
    # 统一使用 P=10, MIPGap=0.05, TimeLimit=1800
    "P_intervals": 10,
    "mip_gap": 0.05,
    "time_limit_s": 1200,
}


# =============================================================================
# 实例规模配置
# =============================================================================
# 用于从完整数据中采样不同规模的测试实例。
# 注: 具体实现需配合数据采样脚本, 此处定义规模分层。

INSTANCE_SCALES = {
    "small":  {"min_grids": 10,  "max_grids": 30,  "num_instances": 10},
    "medium": {"min_grids": 30,  "max_grids": 80,  "num_instances": 10},
    "large":  {"min_grids": 80,  "max_grids": 200, "num_instances": 5},
    "real":   {"min_grids": 0,   "max_grids": 9999, "num_instances": 1,
               "description": "full real-world scale"},
}


# =============================================================================
# 辅助函数
# =============================================================================
def get_experiment_config(experiment_id: str) -> dict:
    """获取指定实验组的完整配置。"""
    if experiment_id not in EXPERIMENT_GROUPS:
        raise ValueError(f"Unknown experiment ID: {experiment_id}. "
                         f"Available: {list(EXPERIMENT_GROUPS.keys())}")
    cfg = dict(SHARED_PARAMS)
    cfg.update(EXPERIMENT_GROUPS[experiment_id])
    return cfg


def get_experiment_label(experiment_id: str, lang: str = "en") -> str:
    """获取实验组的人类可读标签。"""
    cfg = EXPERIMENT_GROUPS.get(experiment_id, {})
    if lang == "cn":
        return cfg.get("name_cn", experiment_id)
    return cfg.get("name_en", experiment_id)


def list_experiments() -> list[dict]:
    """列出所有实验组的关键信息 (用于打印表格)。"""
    result = []
    for eid, cfg in EXPERIMENT_GROUPS.items():
        result.append({
            "ID": eid,
            "Name": cfg.get("name_en", ""),
            "P": cfg.get("P_intervals", "-"),
            "MIPGap": cfg.get("mip_gap", "-"),
            "Method": cfg.get("method", "-"),
            "Geo-Fence": cfg.get("geo_fencing", False),
            "KNN": cfg.get("knn_enabled", False),
            "K": cfg.get("K_neighbors", "-"),
            "Role": cfg.get("academic_role", ""),
        })
    return result


if __name__ == "__main__":
    # 打印实验组总览
    import pandas as pd
    df = pd.DataFrame(list_experiments())
    print("=" * 100)
    print("实验组配置总览 (Experiment Group Configuration Overview)")
    print("=" * 100)
    print(df.to_string(index=False))
    print()
    print("共享参数 (SHARED_PARAMS):")
    for k, v in SHARED_PARAMS.items():
        print(f"  {k}: {v}")
