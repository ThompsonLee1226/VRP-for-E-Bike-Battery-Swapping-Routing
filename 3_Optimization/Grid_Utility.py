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