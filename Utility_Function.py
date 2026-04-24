import numpy as np

def utility(rent_rate, return_rate, u, y, n_low, n_soon, n_normal, T=1.0):
    """
    Utility 计算函数：输入需求与车辆状态，直接输出效用值。
    
    参数 (支持单个数字，也支持传入 Numpy 数组进行批量计算):
    - rent_rate   : 借车需求率 (\lambda)
    - return_rate : 还车到达率 (\mu)
    - u           : 调度车到达该格点的时间
    - y           : 计划在该格点换电的数量
    - n_low       : 初始亏电车数量
    - n_soon      : 初始即将亏电车数量
    - n_normal    : 初始满电车数量
    - T           : 规划总时长 (默认 1 小时)
    
    返回:
    - utility     : 最终计算出的 Utility 值
    """
    # 统一格式化为 numpy 数组，确保数学计算的稳定性
    lam = np.asarray(rent_rate, dtype=float)
    mu = np.asarray(return_rate, dtype=float)
    u = np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=float)
    n_low = np.asarray(n_low, dtype=float)
    n_soon = np.asarray(n_soon, dtype=float)
    n_normal = np.asarray(n_normal, dtype=float)

    # --- 因子 1: instant-jump (\Delta B) ---
    delta_B = np.minimum(y, n_low) * 0.8 + np.minimum(n_soon, np.maximum(0, y - n_low)) * 0.5

    # --- 因子 2: conversion-rate (W) ---
    N_u = n_soon + n_normal + np.minimum(n_low, y) + (mu - lam) * u
    N_T = n_soon + n_normal + np.minimum(n_low, y) + (mu - lam) * T
    
    # 设定极小值底线，防止除以 0 的数学错误
    N_u = np.clip(N_u, 1e-6, None)
    N_T = np.clip(N_T, 1e-6, None)

    W = np.zeros_like(lam)
    mask_eq = np.isclose(mu, lam, atol=1e-5) # 平稳情况 \mu == \lambda
    mask_neq = ~mask_eq                      # 非平稳情况 \mu != \lambda

    if np.any(mask_neq):
        exponent = lam[mask_neq] / (mu[mask_neq] - lam[mask_neq])
        ratio = np.clip(N_u[mask_neq] / N_T[mask_neq], 1e-6, None)
        W[mask_neq] = 1.0 - np.power(ratio, exponent)

    if np.any(mask_eq):
        W[mask_eq] = 1.0 - np.exp(- (lam[mask_eq] / N_u[mask_eq]) * (T - u[mask_eq]))

    W = np.clip(W, 0.0, 1.0)

    # --- 因子 3: stock-out-risk (R) ---
    N_serv = np.maximum(0, n_soon + n_normal - lam * u)
    expected_demand = lam * (T - u)

    R = np.zeros_like(lam)
    mask_pos = expected_demand > 1e-6
    if np.any(mask_pos):
        R[mask_pos] = np.maximum(0, (expected_demand[mask_pos] - N_serv[mask_pos]) / expected_demand[mask_pos])
        
    R = np.clip(R, 0.0, 1.0)

    # --- 最终汇总 ---
    utility = delta_B * W * R

    # 如果输入的是单个数字，则返回干净的浮点数；如果是列表，则返回数组
    return float(utility) if utility.ndim == 0 else utility