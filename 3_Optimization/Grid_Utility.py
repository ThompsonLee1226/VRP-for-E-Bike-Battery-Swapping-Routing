import numpy as np

def calc_quality_weighted_inventory(t, t_start, m_soon, m_normal, theta_soon_global, theta_normal_global, rho, lam):
    """
    计算在任意时间 t 的质量加权库存 W(t) = N_normal(t) + 0.8 * N_soon(t)
    基于论文中的单状态解析解(式 3.11 与 3.12)进行线性组合推导
    """
    dt = t - t_start
    if dt <= 0:
        return m_normal + 0.8 * m_soon
        
    # 定义全局质量期望基准 W^global
    w_global = theta_normal_global + 0.8 * theta_soon_global
    # 计算当前初始状态下的总可用库存 M_0
    M0 = m_soon + m_normal
    
    if M0 <= 0:
        return 0.0

    if np.isclose(rho, lam):
        # 情况 B: 供需守恒系统 (式 3.12)
        w_start = m_normal + 0.8 * m_soon
        return w_global * M0 + (w_start - w_global * M0) * np.exp(-lam / M0 * dt)
    else:
        # 情况 A: 非守恒系统 (式 3.11)
        w_start = m_normal + 0.8 * m_soon
        total_inv_t = M0 + (rho - lam) * dt
        # 规避因截断引起的微小数值下溢
        if total_inv_t <= 0:
            return 0.0
        
        term_deviation = (w_start - w_global * M0) * (M0 / total_inv_t) ** (lam / (rho - lam))
        term_quota = w_global * total_inv_t
        return term_deviation + term_quota

def calculate_operational_utility(u_j, y_j, n_low, n_soon, n_normal, 
                                  theta_soon_global, theta_normal_global, 
                                  rho_j, lam_j, T):
    """
    根据论文 Section 4.2 质量调整后的服务效用框架，计算格点 j 的边际干预效用 U_j(u_j, y_j)
    
    参数说明:
    u_j: 服务车到达格点 j 的时间
    y_j: 换电数量
    n_low, n_soon, n_normal: 格点 j 在 t=0 时刻各状态车的初始库存
    theta_soon_global, theta_normal_global: 全局车况分布比例
    rho_j, lam_j: 格点 j 的用户流入(还车)与流出(借车)速率
    T: 规划周期长度
    """
    
   # -------------------------------------------------------------------------
    # 0. 全局比例强行刨除 low 状态并进行归一化互补处理
    # -------------------------------------------------------------------------
    sum_serviceable_theta = theta_soon_global + theta_normal_global
    
    # 强制使流入系统的 serviceable 车辆比例之和等于 1 (互补)
    theta_soon_pure = theta_soon_global / sum_serviceable_theta
    theta_normal_pure = theta_normal_global / sum_serviceable_theta
    
    # 相应地，流入系统的总速率中，也应当只计算 active（soon+normal）部分的有效流入率
    # 这样才能完美匹配分母只有 soon+normal 的 ODE 动态
    rho_j_pure = rho_j * sum_serviceable_theta

    # -------------------------------------------------------------------------
    # 1. 基础常数与自然动态边界计算 (后续计算全部替换为纯净无low参数)
    # -------------------------------------------------------------------------
    N_j0 = n_soon + n_normal
    
    if lam_j > rho_j_pure:
        t_0 = N_j0 / (lam_j - rho_j_pure)
    else:
        t_0 = float('inf')
        
    tau_0 = min(t_0, T)
    
    if u_j >= T or u_j >= tau_0:
        return 0.0

    # -------------------------------------------------------------------------
    # 2. 计算到达时刻 u_j 瞬间的自然系统状态 (使用 rho_j_pure)
    # -------------------------------------------------------------------------
    N_low_u = n_low 
    
    if np.isclose(rho_j_pure, lam_j):
        N_soon_u = theta_soon_pure * N_j0 + (n_soon - theta_soon_pure * N_j0) * np.exp(-lam_j / N_j0 * u_j)
        N_normal_u = theta_normal_pure * N_j0 + (n_normal - theta_normal_pure * N_j0) * np.exp(-lam_j / N_j0 * u_j)
    else:
        linear_term = N_j0 + (rho_j_pure - lam_j) * u_j
        if linear_term <= 1e-12:
            return 0.0
        scale_fact = (N_j0 / linear_term) ** (lam_j / (rho_j_pure - lam_j))
        N_soon_u = (n_soon - theta_soon_pure * N_j0) * scale_fact + theta_soon_pure * linear_term
        N_normal_u = (n_normal - theta_normal_pure * N_j0) * scale_fact + theta_normal_pure * linear_term

    N_soon_u = max(0.0, N_soon_u)
    N_normal_u = max(0.0, N_normal_u)

    # -------------------------------------------------------------------------
    # 3. 执行换电操作 (保持原样)
    # -------------------------------------------------------------------------
    swapped_low = min(y_j, N_low_u)
    remaining_y = max(0, y_j - N_low_u)
    swapped_soon = min(remaining_y, N_soon_u)
    
    delta_swap = swapped_low * 1.0 + swapped_soon * 0.2
    
    tilde_n_soon = N_soon_u - swapped_soon
    tilde_n_normal = N_normal_u + swapped_low + swapped_soon
    tilde_N_j0 = tilde_n_soon + tilde_n_normal 

    # -------------------------------------------------------------------------
    # 4. 计算换电后的新边界 (使用 rho_j_pure)
    # -------------------------------------------------------------------------
    if lam_j > rho_j_pure:
        tilde_t_0 = u_j + tilde_N_j0 / (lam_j - rho_j_pure)
    else:
        tilde_t_0 = float('inf')
        
    tilde_tau_0 = min(tilde_t_0, T)

    # -------------------------------------------------------------------------
    # 5. 期末质量评估 (调用时传入纯净互补比例与修正后的流入率)
    # -------------------------------------------------------------------------
    if np.isfinite(t_0) and np.isclose(tau_0, t_0):
        w_tau_0 = 0.0
    else:
        w_tau_0 = calc_quality_weighted_inventory(
            tau_0, 0.0, n_soon, n_normal, 
            theta_soon_pure, theta_normal_pure, rho_j_pure, lam_j
        )
        
    if np.isfinite(tilde_t_0) and np.isclose(tilde_tau_0, tilde_t_0):
        w_tilde_tau_0 = 0.0
    else:
        w_tilde_tau_0 = calc_quality_weighted_inventory(
            tilde_tau_0, u_j, tilde_n_soon, tilde_n_normal, 
            theta_soon_pure, theta_normal_pure, rho_j_pure, lam_j
        )
        
    residual_penalty = w_tilde_tau_0 - w_tau_0
    utility_j = delta_swap - residual_penalty

    return max(0.0, utility_j)


def calculate_operational_utility_with_split(u_j, y_j, n_low, n_soon, n_normal,
                                             theta_soon_global, theta_normal_global,
                                             rho_j, lam_j, T):
    """
    与 calculate_operational_utility 完全相同的逻辑，但返回分流效用字典:
        {"total": 总效用, "soon": Soon电池贡献, "normal": Normal电池贡献, "low": Low电池贡献}

    分流逻辑:
      - low  电池贡献: 换电质量增益 1.0 × swapped_low → 全部记入 low
      - soon 电池贡献: 换电质量增益 0.2 × swapped_soon → 全部记入 soon
      - 期末剩余惩罚 (residual_penalty) 按 Soon/Normal 对质量加权库存的贡献比例分摊
    """
    # -----------------------------------------------------------------------
    # 0. 全局比例归一化 (与原始函数完全一致)
    # -----------------------------------------------------------------------
    sum_serviceable_theta = theta_soon_global + theta_normal_global
    if sum_serviceable_theta <= 0:
        return {"total": 0.0, "soon": 0.0, "normal": 0.0, "low": 0.0}

    theta_soon_pure = theta_soon_global / sum_serviceable_theta
    theta_normal_pure = theta_normal_global / sum_serviceable_theta
    rho_j_pure = rho_j * sum_serviceable_theta

    # -----------------------------------------------------------------------
    # 1. 基础常数与边界 (与原始完全一致)
    # -----------------------------------------------------------------------
    N_j0 = n_soon + n_normal

    if lam_j > rho_j_pure:
        t_0 = N_j0 / (lam_j - rho_j_pure)
    else:
        t_0 = float('inf')

    tau_0 = min(t_0, T)

    if u_j >= T or u_j >= tau_0:
        return {"total": 0.0, "soon": 0.0, "normal": 0.0, "low": 0.0}

    # -----------------------------------------------------------------------
    # 2. 到达时刻自然系统状态
    # -----------------------------------------------------------------------
    N_low_u = n_low

    if np.isclose(rho_j_pure, lam_j):
        N_soon_u = theta_soon_pure * N_j0 + (n_soon - theta_soon_pure * N_j0) * np.exp(-lam_j / N_j0 * u_j)
        N_normal_u = theta_normal_pure * N_j0 + (n_normal - theta_normal_pure * N_j0) * np.exp(-lam_j / N_j0 * u_j)
    else:
        linear_term = N_j0 + (rho_j_pure - lam_j) * u_j
        if linear_term <= 1e-12:
            return {"total": 0.0, "soon": 0.0, "normal": 0.0, "low": 0.0}
        scale_fact = (N_j0 / linear_term) ** (lam_j / (rho_j_pure - lam_j))
        N_soon_u = (n_soon - theta_soon_pure * N_j0) * scale_fact + theta_soon_pure * linear_term
        N_normal_u = (n_normal - theta_normal_pure * N_j0) * scale_fact + theta_normal_pure * linear_term

    N_soon_u = max(0.0, N_soon_u)
    N_normal_u = max(0.0, N_normal_u)

    # -----------------------------------------------------------------------
    # 3. 执行换电操作 + 分摊质量增益
    # -----------------------------------------------------------------------
    swapped_low = min(y_j, N_low_u)
    remaining_y = max(0, y_j - N_low_u)
    swapped_soon = min(remaining_y, N_soon_u)
    # 注: 若 y_j > N_low_u + N_soon_u, 多余的部分理论上不会发生（受限于物理库存）
    # 此处保留与原始函数一致的逻辑: 只计算实际换电量

    # 直接质量增益分流
    delta_low = swapped_low * 1.0       # Low→Normal: 质量提升 1.0
    delta_soon = swapped_soon * 0.2     # Soon→Normal: 质量提升 0.2
    delta_swap_total = delta_low + delta_soon

    tilde_n_soon = N_soon_u - swapped_soon
    tilde_n_normal = N_normal_u + swapped_low + swapped_soon
    tilde_N_j0 = tilde_n_soon + tilde_n_normal

    # -----------------------------------------------------------------------
    # 4. 换电后新边界
    # -----------------------------------------------------------------------
    if lam_j > rho_j_pure:
        tilde_t_0 = u_j + tilde_N_j0 / (lam_j - rho_j_pure)
    else:
        tilde_t_0 = float('inf')

    tilde_tau_0 = min(tilde_t_0, T)

    # -----------------------------------------------------------------------
    # 5. 期末质量评估
    # -----------------------------------------------------------------------
    if np.isfinite(t_0) and np.isclose(tau_0, t_0):
        w_tau_0 = 0.0
    else:
        w_tau_0 = calc_quality_weighted_inventory(
            tau_0, 0.0, n_soon, n_normal,
            theta_soon_pure, theta_normal_pure, rho_j_pure, lam_j
        )

    if np.isfinite(tilde_t_0) and np.isclose(tilde_tau_0, tilde_t_0):
        w_tilde_tau_0 = 0.0
    else:
        w_tilde_tau_0 = calc_quality_weighted_inventory(
            tilde_tau_0, u_j, tilde_n_soon, tilde_n_normal,
            theta_soon_pure, theta_normal_pure, rho_j_pure, lam_j
        )

    residual_penalty = w_tilde_tau_0 - w_tau_0

    # -----------------------------------------------------------------------
    # 6. 将 residual_penalty 按 delta_swap 的来源比例分摊到 soon/low
    #    (normal 在到达时刻不产生直接换电收益，但承担 residual_penalty 的分摊)
    # -----------------------------------------------------------------------
    total_utility = max(0.0, delta_swap_total - residual_penalty)

    if delta_swap_total > 1e-12:
        # 按直接增益比例分摊净效用
        soon_share = delta_soon / delta_swap_total
        low_share = delta_low / delta_swap_total
    else:
        soon_share = 0.0
        low_share = 0.0

    utility_soon = total_utility * soon_share
    utility_low = total_utility * low_share
    # Normal 电池在到达时没有直接换电收益，但承担 residual_penalty 后的
    # 剩余效用按比例分配，因此 normal 直接分量为 0
    utility_normal = 0.0

    return {
        "total": round(total_utility, 8),
        "soon": round(utility_soon, 8),
        "normal": round(utility_normal, 8),
        "low": round(utility_low, 8),
    }