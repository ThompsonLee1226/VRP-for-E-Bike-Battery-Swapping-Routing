import numpy as np
import gurobipy as gp
from gurobipy import GRB

def optimize_evrp_with_stgraph(
    grids,
    travel_time,
    C_max,
    T_total,
    Omega,
    tau_list,
    swap_time_c,
    progress_tracker=None,
    max_travel_time=0.2,
    y_levels=None,
    K_neighbors=50
):
    """
    基于时间扩展图 (Space-Time Graph, ST-Graph) 架构构建并求解 EVRP 问题。
    
    该架构通过将连续时间变量离散化并嵌入拓扑结构，彻底消除了传统的 MTZ 子回路消除约束、
    大 M 时序同步约束以及复杂的增量式阶梯单调性约束，从根本上解决大模型根节点松弛退化的瓶颈。

    参数:
    - grids: 活跃网格 ID 列表 (已过滤零效用网格)
    - travel_time: 完整旅行时间矩阵 (含 depot=0), 单位: 小时
    - C_max: 车载电池最大容量限制
    - T_total: 规划周期总时长 (小时)
    - Omega: 预计算静态效用矩阵, Omega[j][y][s]
    - tau_list: 物理时间断点列表 (当前设置为 13 个时间点，12 个等长区间)
    - swap_time_c: 单块电池换电服务时间 (小时)
    - progress_tracker: GurobiProgressTracker 实例或 None
    - max_travel_time: 弧段裁剪绝对阈值 (小时)
    - y_levels: 离散换电量列表。若为 None 则使用 range(1, C_max+1)
    - K_neighbors: 每个空间节点的最大近邻限制数 (KNN 稀疏化)
    """
    
    # ==========================================================================
    # 0. 基础常数与离散参数解析
    # ==========================================================================
    depot = 0
    nodes = [depot] + grids
    Y_domain = list(range(1, C_max + 1)) if y_levels is None else sorted(y_levels)
    P = len(tau_list) - 1  # 时间分段数 (当前为 12)
    
    # 右取整时间对齐辅助函数：寻找满足 tau_list[s] >= t_arr 的最小离散步 index
    def get_right_rounded_step(t_arr):
        for s, token_t in enumerate(tau_list):
            if token_t >= t_arr - 1e-7:  # 引入微小扰动规避浮点精度截断
                return s
        return None

    # ==========================================================================
    # 1. 空间邻域剪枝 (KNN + max_travel_time 混合稀疏化)
    # ==========================================================================
    spatial_neighbors = {i: [] for i in nodes}
    for i in nodes:
        # 筛选满足物理距离限制的候选目的地
        candidates = []
        for j in nodes:
            if i != j and travel_time[i][j] <= max_travel_time:
                candidates.append((travel_time[i][j], j))
        
        # 严格按距离由近到远排序，保留前 K_neighbors 个
        candidates.sort(key=lambda x: x[0])
        spatial_neighbors[i] = [tgt for _, tgt in candidates[:K_neighbors]]
        
        # 确保终点 Depot 始终在各节点的决策视野内 (即使略超 max_travel_time 也允许强制返回)
        if i != depot and depot not in spatial_neighbors[i] and travel_time[i][depot] <= T_total:
            spatial_neighbors[i].append(depot)

    # ==========================================================================
    # 2. 时空图网络拓扑生成 (Space-Time Arc Generation)
    # ==========================================================================
    # 时空弧定义格式：(i, s1, j, s2, y) -> 代表在步数 s1 从 i 出发，换电 y 块，在步数 s2 到达 j
    st_arcs = []
    
    # 索引网络流的辅助入弧/出弧查找表，用于极速拼装流守恒方程
    outflow_map = {(n, s): [] for n in nodes for s in range(P + 1)}
    inflow_map  = {(n, s): [] for n in nodes for s in range(P + 1)}
    
    arc_idx = 0
    
    # A. 构造从 Depot 出发的时空弧段 (起始点固定为时间步 0，且不执行换电动作 y=0)
    for j in spatial_neighbors[depot]:
        t_arr = travel_time[depot][j]
        s2 = get_right_rounded_step(t_arr)
        if s2 is not None and s2 <= P:
            arc_info = (depot, 0, j, s2, 0)
            st_arcs.append(arc_info)
            outflow_map[(depot, 0)].append(arc_idx)
            inflow_map[(j, s2)].append(arc_idx)
            arc_idx += 1

    # B. 构造从各个空间网格在任意合法时刻出发的转移时空弧段
    for i in grids:
        for s1 in range(P):  # 在期末时刻 P 出发已无意义
            for j in spatial_neighbors[i]:
                for y_val in Y_domain:
                    # 物理到达时间 = 当前起点时间 + 换电服务耗时 + 空间行驶耗时
                    t_arr = tau_list[s1] + (y_val * swap_time_c) + travel_time[i][j]
                    s2 = get_right_rounded_step(t_arr)
                    
                    # 只有当到达时间在规划周期结束前，才视为合法时空弧
                    if s2 is not None and s2 <= P:
                        arc_info = (i, s1, j, s2, y_val)
                        st_arcs.append(arc_info)
                        outflow_map[(i, s1)].append(arc_idx)
                        inflow_map[(j, s2)].append(arc_idx)
                        arc_idx += 1

    if progress_tracker is None:
        print(f"  [时空图构建] 空间节点: {len(grids)} | 离散时间步: {P}")
        print(f"  [时空图构建] 静态生成有效时空弧段 (决策变量总数): {len(st_arcs)} 条")

    # ==========================================================================
    # 3. 建立 Gurobi 数学模型
    # ==========================================================================
    m = gp.Model("EVRP_SpaceTimeGraph_Engine")

    # 基础求解器性能优化参数配置
    m.setParam('MIPGap', 0.05)
    m.setParam('TimeLimit', 1800)
    m.setParam('Method', 3)              # 并发求解优化（Dual Simplex + Barrier 并行）
    m.setParam('Threads', 8)
    m.setParam('MIPFocus', 1)            # 优先激进寻找高质量可行解
    m.setParam('OutputFlag', 0)          # 接管原生日志，由管线提供精简打印

    # 声明核心决策变量：x[e] = 1 代表车辆启用了第 e 条时空弧段
    x = m.addVars(len(st_arcs), vtype=GRB.BINARY, name="x")

    # ==========================================================================
    # 4. 目标函数设定
    # ==========================================================================
    # 如果时空弧的始发站是网格 i，则该弧直接绑定并变现确定性收益 Omega[i][y][s1]
    obj_expr = gp.quicksum(
        Omega[st_arcs[e][0]][st_arcs[e][4]][st_arcs[e][1]] * x[e]
        for e in range(len(st_arcs)) if st_arcs[e][0] != depot
    )
    m.setObjective(obj_expr, GRB.MAXIMIZE)

    # ==========================================================================
    # 5. 约束条件组装
    # ==========================================================================
    
    # 约束 5a: Depot 出发控制 (车队至多外派 1 辆车)
    m.addConstr(
        gp.quicksum(x[e] for e in outflow_map[(depot, 0)]) <= 1,
        name="DepotDepartureLimit"
    )

    # 约束 5b: Depot 返程闭环 (出发的车必须在时空结束前的任意时间步 s 返回至 Depot)
    m.addConstr(
        gp.quicksum(x[e] for s in range(1, P + 1) for e in inflow_map[(depot, s)]) ==
        gp.quicksum(x[e] for e in outflow_map[(depot, 0)]),
        name="DepotReturnBalance"
    )

    # 约束 5c: 全局网格时空流守恒 (进入某一网格时空点 (j, s) 的流量等于离开该点的流量)
    for j in grids:
        for s in range(P + 1):
            m.addConstr(
                gp.quicksum(x[e] for e in inflow_map[(j, s)]) ==
                gp.quicksum(x[e] for e in outflow_map[(j, s)]),
                name=f"ST_FlowConservation_{j}_{s}"
            )

    # 约束 5d: 空间物理唯一性访问约束 (为了杜绝车辆在同一空间节点反复刷分，物理节点至多被访问一次)
    for j in grids:
        m.addConstr(
            gp.quicksum(x[e] for s in range(P + 1) for e in outflow_map[(j, s)]) <= 1,
            name=f"SpatialGridUniqueVisit_{j}"
        )

    # 约束 5e: 全局车载容量边界控制 (整条时空链路消耗的换电总量不能突破最大车载限制)
    m.addConstr(
        gp.quicksum(st_arcs[e][4] * x[e] for e in range(len(st_arcs))) <= C_max,
        name="GlobalBatteryCapacityLimit"
    )

    # ==========================================================================
    # 6. 执行优化求解与指针挂载
    # ==========================================================================
    def _progress_callback(model, where):
        if progress_tracker is None:
            return
        if where == GRB.Callback.MIP:
            runtime = model.cbGet(GRB.Callback.RUNTIME)
            best_obj = model.cbGet(GRB.Callback.MIP_OBJBST)
            best_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
            node_count = model.cbGet(GRB.Callback.MIP_NODCNT)
            progress_tracker.record(runtime, best_obj, best_bound, node_count)

    # 单次运行，无需在外部切分多阶段循环
    m.optimize(_progress_callback)

    # 将时空图的映射元数据及变量集注入模型，由主程序 main_STGraph 解析生成路由 CSV
    m._st_arcs = st_arcs
    m._x = x
    m._grids = grids
    m._outflow_map = outflow_map
    m._inflow_map = inflow_map
    m._P_intervals = P

    return m