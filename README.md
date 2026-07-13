# VRP-for-E-Bike-Battery-Swapping-Routing
A swapping routing problem referring to a fixed region e-bike battery swapping 

# 电动两轮车换电调度路径优化 —— 程序运行手册

> **版本**: v1.0  
> **作者**: 本科研究项目 — E-Bike Battery Swapping Routing Optimization  
> **依赖**: Python 3.8+, Gurobi 10.0+ (需有效 License)

---

## 目录

- [1. 环境准备](#1-环境准备)
- [2. 项目结构](#2-项目结构)
- [3. 快速开始](#3-快速开始)
- [4. 实验组说明](#4-实验组说明)
- [5. 单模型运行](#5-单模型运行)
- [6. 批量实验运行](#6-批量实验运行)
- [7. 输出文件说明](#7-输出文件说明)
- [8. 配置系统](#8-配置系统)
- [9. 消融实验](#9-消融实验)
- [10. 常见问题](#10-常见问题)

---

## 1. 环境准备

### 1.1 Python 依赖

```bash
pip install numpy pandas gurobipy
```

### 1.2 Gurobi License

本项目的所有模型均依赖 Gurobi 求解器，需要有效的 License。

```bash
# 检查 Gurobi 是否可用
python -c "import gurobipy; print(gurobipy.gurobi.version())"
```

如未安装 License，可申请 [Gurobi Academic License](https://www.gurobi.com/academia/academic-program-and-licenses/)（免费，需 `.edu` 邮箱）。

### 1.3 数据文件

默认数据文件位于 `3_Optimization/Grid_Utility_Test.csv`，需包含以下列：

| 列名 | 类型 | 说明 |
|:---|:---|:---|
| `datetime` | datetime | 数据时间戳 |
| `h3` | str | H3 六边形网格 ID |
| `latitude` | float | 网格中心纬度 |
| `longitude` | float | 网格中心经度 |
| `low_power_bike_count` | float | 低电量车辆数 |
| `soon_low_power_bike_count` | float | 即将低电量车辆数 |
| `normal_power_bike_count` | float | 正常电量车辆数 |
| `rent_pred` / `rent` | float | 预测借车速率 |
| `return_pred` / `return` | float | 预测还车速率 |

---

##胆小2. 项目结构

```
3_Optimization/
├── Grid_Utility.py              # 运营效用函数 (含 Soon/Normal 分流计算)
├── Pre_Process.py               # 数据预处理管线 (坐标解析、旅行时间矩阵)
│
├── Optimize_PLA.py              # SOS2-PLA 模型引擎 (M3)
├── Optimize_PLA_Delta.py        # Delta-PLA 模型引擎 (M4)
├── Optimize_PLA_Delta_MCF.py    # Delta-MCF 模型引擎 (M5)
├── Optimize_PLA_STGraph.py      # 时空图 (ST-Graph) 模型引擎 (M1/M2)
│
├── main_STGraph.py              # ★ 主入口 — 时空图完整模型 (M1, M2a/b/c)
├── main_SOS2.py                 # 主入口 — SOS2 对比模型 (M3)
├── main_delta.py                # 主入口 — Delta 对比模型 (M4)
├── main_delta_MCF.py            # 主入口 — MCF 基准模型 (M5)
│
├── experiment_utils.py          # ★ 统一实验工具 (指标采集、结果导出)
├── experiment_config.py         # ★ 实验配置 (7 组实验的参数定义)
├── batch_runner.py              # ★ 批量实验执行器 (一键运行全部实验组)
│
└── Grid_Utility_Test.csv        # 默认输入数据
```

---

## 3. 快速开始

### 3.1 最简运行（单模型）

```bash
cd 3_Optimization

# 运行 M1 — 时空图完整模型 (默认参数)
python main_STGraph.py

# 运行 M3 — SOS2 对比模型
python main_SOS2.py

# 运行 M5 — MCF 基准模型
python main_delta_MCF.py
```

### 3.2 一键运行全部实验组

```bash
# 运行 M1, M2a, M2b, M3, M4, M5 (共 6 组)
python batch_runner.py
```

### 3.3 查看可用实验组

```bash
python batch_runner.py --list
```

输出：
```
  M1     ST-Graph (Full)                      P=12  Gap=0.01  GF=True   KNN=True   [Proposed Method]
  M2a    ST-Graph w/o Geo-Fencing             P=12  Gap=0.01  GF=False  KNN=True   [Ablation Study]
  M2b    ST-Graph w/o KNN Pruning             P=12  Gap=0.01  GF=True   KNN=False  [Ablation Study]
  M2c    ST-Graph w/o Any Pruning             P=12  Gap=0.01  GF=False  KNN=False  [Ablation Study (Optional)]
  M3     SOS2-PLA                             P=5   Gap=0.05  GF=True   KNN=False  [Nonlinear Approximation Benchmark]
  M4     Delta-PLA                            P=5   Gap=0.05  GF=True   KNN=False  [Nonlinear Approximation Benchmark]
  M5     MCF-Baseline                         P=10  Gap=0.05  GF=True   KNN=False  [Classical Baseline]
```

---

## 4. 实验组说明

### 4.1 实验组总览

| 代号 | 中文名称 | 英文名称 | 定位 | 核心差异 |
|:---:|:---|:---|:---|:---|
| **M1** | 时空图完整模型 | ST-Graph (Full) | **Proposed Model** — 本文核心贡献 | P=12, Geo-Fencing ✓, KNN K=50 |
| **M2a** | 时空图—无Geo-Fencing | ST-Graph w/o Geo-Fencing | 消融实验 — 量化零效用过滤的贡献 | P=12, Geo-Fencing ✗, KNN K=50 |
| **M2b** | 时空图—无KNN裁剪 | ST-Graph w/o KNN Pruning | 消融实验 — 量化KNN图裁剪的贡献 | P=12, Geo-Fencing ✓, KNN ✗ |
| **M2c** | 时空图—无任何剪枝 | ST-Graph w/o Any Pruning | 消融实验（可选）— 极端对照 | P=12, Geo-Fencing ✗, KNN ✗ |
| **M3** | SOS2分段线性逼近 | SOS2-PLA | Benchmark — 非线性逼近机制对比 | P=5, SOS2 约束, 全量 MTZ |
| **M4** | Delta增量逼近 | Delta-PLA | Benchmark — 非线性逼近机制对比 | P=5, Delta 阶梯编码, 全量 MTZ |
| **M5** | 多商品流基准模型 | MCF-Baseline | Classical Baseline — 传统骨架体系 | P=10, MCF 流约束, Lazy MTZ Callback |

### 4.2 实验设计逻辑

```
论文对比逻辑链：

  M1 vs M5    → "提出的 ST-Graph 框架 vs 传统 MCF+MTZ 框架"
  
  M1 vs M2a   → "Geo-Fencing 零效用过滤的边际贡献"
  M1 vs M2b   → "KNN 稠密图裁剪的边际贡献"
  M1 vs M2c   → "双重剪枝的联合效应"
  
  M3 vs M4    → "在相同 MTZ 框架下，SOS2-λ vs Delta-δ 的 LP 松弛紧度差异"
  M4 vs M5    → "在相同 Delta 编码下，静态 MTZ vs MCF+Lazy MTZ 的架构差异"
```

### 4.3 注意事项

- **参数说明**: 各模型使用各自的最优 Gurobi 参数配置（通过大量预调参确定）。论文中需在 "Experimental Setup" 小节明确说明这一点。
- **公平比较**: 如需消除参数差异的影响，使用 `--aligned` 模式（见 6.4 节）。
- **运行时长**: 单个模型默认时限为 1800s (30 min)，6 组实验预计总耗时约 3 小时。

---

## 5. 单模型运行

### 5.1 通过 main_*.py 直接运行

每个 `main_*.py` 均可独立运行，支持自定义参数。

#### main_STGraph.py (M1/M2 系列)

```python
from main_STGraph import run_optimization_pipeline

result = run_optimization_pipeline(
    data_file="Grid_Utility_Test.csv",
    vehicle_speed_kmh=30.0,
    C_max=20,              # 车载电池容量
    T_total=1.0,           # 规划周期 (小时)
    P_intervals=12,        # 时间离散段数 (12段 = 每5分钟)
    K_neighbors=50,        # KNN 近邻数
    experiment_id="M1",    # ★ 实验组代号
    geo_fencing=True,      # ★ 是否启用零效用过滤
    knn_enabled=True,      # ★ 是否启用 KNN 裁剪
    verbose=True,
)
```

#### main_SOS2.py (M3)

```python
from main_SOS2 import run_optimization_pipeline

result = run_optimization_pipeline(
    data_file="Grid_Utility_Test.csv",
    P_intervals=5,         # 时间分段数
    BigM=200.0,            # 大M常数 (内部自适应收紧)
    experiment_id="M3",
    verbose=True,
)
```

#### main_delta.py (M4)

```python
from main_delta import run_optimization_pipeline

result = run_optimization_pipeline(
    data_file="Grid_Utility_Test.csv",
    P_intervals=5,
    BigM=200.0,
    experiment_id="M4",
    verbose=True,
)
```

#### main_delta_MCF.py (M5)

```python
from main_delta_MCF import run_optimization_pipeline

result = run_optimization_pipeline(
    data_file="Grid_Utility_Test.csv",
    P_intervals=10,        # 更细的离散化
    BigM=200.0,
    experiment_id="M5",
    verbose=True,
)
```

### 5.2 访问运行结果

```python
# 求解状态
print(result["status"])           # "OPTIMAL" / "TIME_LIMIT" / ...

# 关键指标 (MetricsCollector)
collector = result["collector"]
print(collector.to_dict())        # 全部 40+ 指标

# 路由明细
for step in result["route"]:
    print(f"Grid={step['grid']}, Arrival={step['arrival_time']:.3f}h, "
          f"Swapped={step['y_swapped']}, "
          f"SoonUtil={step.get('utility_soon', 0):.4f}")

# 汇总统计
print(result["summary"])
# {'num_visited': 8, 'total_swaps': 15, 'makespan_hrs': 0.85,
#  'utility_soon': 0.234, 'utility_normal': 0.0, 'utility_low': 1.567}
```

---

## 6. 批量实验运行

### 6.1 基础用法

```bash
# 运行默认 6 组实验 (M1, M2a, M2b, M3, M4, M5)
python batch_runner.py

# 运行指定实验组
python batch_runner.py --groups M1 M3 M5

# 运行指定组 + 可选的 M2c
python batch_runner.py --groups M1 M2a M2b --include-optional
```

### 6.2 指定数据和输出路径

```bash
python batch_runner.py \
    --data "path/to/my_data.csv" \
    --output "path/to/results/" \
    --instance "morning_peak"
```

### 6.3 输出控制

```bash
# 减少终端输出 (仅显示摘要)
python batch_runner.py --quiet

# 包含可选消融组 M2c
python batch_runner.py --include-optional

# 跳过某些实验组
python batch_runner.py --skip M2c
```

### 6.4 对齐参数模式（公平比较）

当需要使用相同的 P 和 MIPGap 进行公平比较时（如回应审稿人质疑）：

```bash
# 所有模型统一使用 P=10, MIPGap=0.05
python batch_runner.py --aligned
```

### 6.5 自定义实验配置

在 Python 中编程式调用：

```python
from batch_runner import BatchRunner
from experiment_config import get_experiment_config

runner = BatchRunner(
    data_file="my_data.csv",
    output_dir="./results",
    verbose=True,
    use_aligned_params=False,  # 各模型使用最优配置
)

# 逐组运行
runner.run_experiment("M1", instance_name="small_01")
runner.run_experiment("M2a", instance_name="small_01")
runner.run_experiment("M5", instance_name="small_01")

# 或一键运行
runner.run_all(
    groups=["M1", "M3", "M5"],
    instance_name="batch_01",
)
```

---

## 7. 输出文件说明

每次运行产生以下文件（存放在 `output_dir` 中）：

### 7.1 单次实验输出

| 文件 | 格式 | 内容 |
|:---|:---|:---|
| `{ID}_{instance}_{timestamp}_metrics.json` | JSON | **全部 40+ 指标**：求解效率、解质量、Soon/Normal分流、时序保真度、模型规模等 |
| `{ID}_{instance}_{timestamp}_summary.csv` | CSV | 同 JSON 内容的单行 CSV 版 |
| `{ID}_{instance}_{timestamp}_bb_progress.csv` | CSV | **B&B 收敛过程**：每 30s 的快照 (runtime, best_obj, best_bound, node_count) |
| `{ID}_{instance}_{timestamp}_route.csv` | CSV | **路由明细**：每个被访问节点的到达时间、换电量、效用分流值 |

### 7.2 批量实验输出

| 文件 | 格式 | 内容 |
|:---|:---|:---|
| `batch_summary_{instance}_{timestamp}_full.csv` | CSV | 所有实验组的全部指标（宽表） |
| `batch_summary_{instance}_{timestamp}_core.csv` | CSV | 所有实验组的核心指标（便于论文制表） |

### 7.3 JSON 指标字段说明

```json
{
  // === 元信息 ===
  "experiment_id": "M1",
  "instance_name": "default",
  "solve_status": "OPTIMAL",

  // === 求解效率 ===
  "cpu_time_s": 45.3,
  "mip_gap_pct": 0.85,
  "best_obj": 2.3456,
  "best_bound": 2.3657,
  "bb_nodes": 1234,
  "first_feasible_time_s": 3.2,
  "root_lp_value": null,

  // === 模型规模 ===
  "num_vars": 15234,
  "num_constrs": 8912,
  "num_binary": 15000,
  "num_continuous": 200,

  // === 解质量 ===
  "objective_value": 2.3456,
  "num_visited_grids": 8,
  "total_swaps": 15,
  "total_travel_time_hrs": 0.45,
  "total_service_time_hrs": 0.30,
  "makespan_hrs": 0.75,
  "utility_per_hour": 3.127,
  "avg_utility_per_grid": 0.293,

  // === Soon/Normal 分流 (★ 论文核心洞察) ===
  "utility_soon": 0.234,
  "utility_normal": 0.0,
  "utility_low": 2.112,
  "soon_ratio": 0.099,

  // === 时序保真度 ===
  "mean_time_deviation_hrs": 0.008,
  "max_time_deviation_hrs": 0.017,
  "nodes_with_deviation": 0,

  // === 空间剪枝 ===
  "original_grid_count": 85,
  "active_grid_count": 42,
  "removed_zero_utility": 43,
  "feasible_arc_count": 1722,

  // === 模型特有 ===
  "num_st_arcs": 20480,
  "lazy_mtz_count": 0
}
```

---

## 8. 配置系统

### 8.1 实验配置结构

所有实验组的参数在 `experiment_config.py` 中统一定义：

```python
EXPERIMENT_GROUPS = {
    "M1": {
        "id": "M1",
        "name_cn": "时空图完整模型",
        "name_en": "ST-Graph (Full)",
        "main_file": "main_STGraph.py",
        "optimizer": "optimize_evrp_with_stgraph",
        "P_intervals": 12,
        "mip_gap": 0.01,
        "method": 2,              # Barrier
        "geo_fencing": True,
        "knn_enabled": True,
        "K_neighbors": 50,
        # ... 完整 Gurobi 参数
    },
    # ... M2a, M2b, M2c, M3, M4, M5
}
```

### 8.2 共享参数

各实验组共用的基础参数：

| 参数 | 值 | 说明 |
|:---|:---|:---|
| `vehicle_speed_kmh` | 30.0 | 服务车恒定速度 |
| `C_max` | 20 | 单次最大载电池数 |
| `T_total` | 1.0 | 规划周期 (小时) |
| `swap_time_c` | 0.02 | 单块换电服务时间 (小时 ≈ 72s) |
| `max_travel_time` | 0.2 | 弧段裁剪阈值 (小时 ≈ 12min) |
| `y_levels` | [1..10] | 离散换电量选项 |
| `time_limit_s` | 1800 | 求解时限 (30min) |

### 8.3 添加新实验组

```python
# 在 experiment_config.py 的 EXPERIMENT_GROUPS 中添加
"MyNewModel": {
    "id": "MyNewModel",
    "name_cn": "我的新模型",
    "name_en": "My New Model",
    "main_file": "main_STGraph.py",
    "optimizer": "optimize_evrp_with_stgraph",
    "P_intervals": 15,
    "mip_gap": 0.02,
    # ... 其他参数
},

# 然后在 batch_runner.py 的 DEFAULT_GROUPS 中添加
DEFAULT_GROUPS = ["M1", "M2a", "M2b", "M3", "M4", "M5", "MyNewModel"]
```

---

## 9. 消融实验

### 9.1 运行消融实验

```bash
# 运行 M1 + 全部消融组
python batch_runner.py --groups M1 M2a M2b --include-optional
```

### 9.2 消融实验参数对照

| 实验组 | Geo-Fencing | KNN | 预期效果 |
|:---|:---:|:---:|:---|
| M1 | ✓ | ✓ (K=50) | 最快、最优 |
| M2a | ✗ | ✓ (K=50) | 网格数更多 → 求解变慢 |
| M2b | ✓ | ✗ (K=∞) | 弧段数更多 → 求解变慢 |
| M2c | ✗ | ✗ (K=∞) | 最慢、模型规模最大 |

### 9.3 通过代码进行消融实验

```python
from main_STGraph import run_optimization_pipeline

# M1 — 完整模型
r1 = run_optimization_pipeline(experiment_id="M1",
    geo_fencing=True, knn_enabled=True, K_neighbors=50)

# M2a — 关闭 Geo-Fencing
r2a = run_optimization_pipeline(experiment_id="M2a",
    geo_fencing=False, knn_enabled=True, K_neighbors=50)

# M2b — 关闭 KNN
r2b = run_optimization_pipeline(experiment_id="M2b",
    geo_fencing=True, knn_enabled=False, K_neighbors=99999)
```

### 9.4 KNN 敏感度分析

```python
for k in [10, 20, 30, 50, 100, 200]:
    r = run_optimization_pipeline(
        experiment_id=f"M1_K{k}",
        geo_fencing=True, knn_enabled=True, K_neighbors=k,
        verbose=False,
    )
    c = r["collector"]
    print(f"K={k:4d}: CPU={c.cpu_time_s:6.1f}s  "
          f"Obj={c.objective_value:.4f}  Gap={c.mip_gap_pct:.2f}%  "
          f"Vars={c.num_vars}")
```

---

## 10. 常见问题

### Q1: Gurobi License 错误

```
gurobipy.GurobiError: No Gurobi license found
```

**解决**: 申请 [Academic License](https://www.gurobi.com/academia/academic-program-and-licenses/) 或检查 License 文件路径。

### Q2: 模型运行超时 (20 min)

默认时限为 1200s。对于大规模实例，可在配置中调整：

```python
# experiment_config.py → SHARED_PARAMS
"time_limit_s": 3600,  # 增加到 60 min
```

或运行时覆盖：

```python
# 注意: time_limit 在 Optimize_*.py 中硬编码，需修改对应文件的 m.setParam('TimeLimit', ...)
```

### Q3: 内存不足

STGraph 模型（M1/M2）在大型实例上可能产生大量时空弧变量。建议：
- 减小 `K_neighbors`（如 K=30）
- 增大 `max_travel_time` 的筛选力度（如 0.15h）
- 使用更粗的时间离散（`P_intervals=6`）

### Q4: 数据文件列名不匹配

```
ValueError: prediction file must contain rent_pred or rent
```

**解决**: 确保 CSV 文件包含 `rent_pred` 或 `rent` 列，以及 `return_pred` 或 `return` 列。

### Q5: 如何查看 B&B 收敛过程

```python
import pandas as pd

# 读取 B&B 进度记录
df = pd.read_csv("Optimization Result/M1_default_20250713_120000_bb_progress.csv")

# 绘制收敛曲线
import matplotlib.pyplot as plt
plt.plot(df["runtime_s"], df["best_obj"], label="Best Objective")
plt.plot(df["runtime_s"], df["best_bound"], label="Best Bound")
plt.xlabel("Runtime (s)")
plt.ylabel("Objective")
plt.legend()
plt.show()
```

### Q6: 如何对比两个模型的解

```python
import json

# 读取两个模型的指标 JSON
with open("results/M1_..._metrics.json") as f:
    m1 = json.load(f)
with open("results/M5_..._metrics.json") as f:
    m5 = json.load(f)

# 计算相对差异
speedup = m5["cpu_time_s"] / max(m1["cpu_time_s"], 1)
obj_gap = (m1["objective_value"] - m5["objective_value"]) / max(m5["objective_value"], 1e-9) * 100

print(f"加速比: {speedup:.1f}x")
print(f"解质量提升: {obj_gap:.1f}%")
print(f"Soon收益占比: M1={m1['soon_ratio']:.1%}  M5={m5['soon_ratio']:.1%}")
```

### Q7: 如何修改求解器参数

所有 Gurobi 参数在各 `Optimize_*.py` 文件中设置（`m.setParam(...)` 调用处）。也可通过 `experiment_config.py` 中的配置项在模型构建时传入。

---

## 附录：快速参考卡片

```bash
# ─── 查看配置 ───────────────────────────────────────
python experiment_config.py                    # 打印参数总览表
python batch_runner.py --list                  # 列出实验组

# ─── 单模型运行 ─────────────────────────────────────
python main_STGraph.py                         # M1 — 时空图完整模型
python main_SOS2.py                            # M3 — SOS2 对比模型
python main_delta.py                           # M4 — Delta 对比模型
python main_delta_MCF.py                       # M5 — MCF 基准模型

# ─── 批量实验 ───────────────────────────────────────
python batch_runner.py                         # 全部 6 组
python batch_runner.py --groups M1 M3 M5       # 指定 3 组
python batch_runner.py --aligned               # 对齐参数公平比较
python batch_runner.py --quiet                 # 减少输出

# ─── 消融实验 ───────────────────────────────────────
python batch_runner.py --groups M1 M2a M2b     # 核心消融
python batch_runner.py --include-optional      # 含 M2c 极端对照
```

---

> **文档维护**: 如发现文档与代码不一致，请以 `experiment_config.py` 中的注释和 `batch_runner.py --list` 的输出为准。
