import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Utility_Function import utility
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

OUTPUT_DIR = "utility_viz_random_hour"

def visualize_utility_landscape(T=1.0, u=0.5, y=10, n_low=10, n_soon=5, n_normal=2, max_rate=30):
    """
    同时绘制 2D 热力图和 3D 曲面图，探究 lambda 和 mu 对 Utility 的影响
    """
    # 1. 构建 X-Y 网格数据
    rent_vals = np.linspace(0, max_rate, 100)   # X 轴: \lambda
    return_vals = np.linspace(0, max_rate, 100) # Y 轴: \mu
    Rent_grid, Return_grid = np.meshgrid(rent_vals, return_vals)

    # 2. 向量化计算 Z 轴 (Utility)
    Z_utility = utility(
        rent_rate=Rent_grid,
        return_rate=Return_grid,
        u=u,
        y=y,
        n_low=n_low,
        n_soon=n_soon,
        n_normal=n_normal,
        T=T,
    )

    # 3. 初始化画布 (1行2列)
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(f'Utility Landscape (T={T}, u={u}, y={y}, n_low={n_low}, n_soon={n_soon})', fontsize=16)

    # --- 方法 1: 2D 热力等高线图 (Heatmap / Contourf) ---
    ax1 = fig.add_subplot(1, 2, 1)
    # 使用 contourf 绘制平滑的热力等高线，viridis 是经典的色带
    contour = ax1.contourf(Rent_grid, Return_grid, Z_utility, levels=50, cmap='viridis')
    ax1.set_title('Method 1: 2D Heatmap', fontsize=14)
    ax1.set_xlabel('Rent Rate ($\lambda$)', fontsize=12)
    ax1.set_ylabel('Return Rate ($\mu$)', fontsize=12)
    # 叠加等高线标尺
    cbar = fig.colorbar(contour, ax=ax1)
    cbar.set_label('Utility', rotation=270, labelpad=15)

    # --- 方法 2: 3D 曲面图 (Surface Plot) ---
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    # 绘制三维曲面
    surf = ax2.plot_surface(Rent_grid, Return_grid, Z_utility, cmap='plasma', edgecolor='none', alpha=0.9)
    ax2.set_title('Method 2: 3D Surface Plot', fontsize=14)
    ax2.set_xlabel('Rent Rate ($\lambda$)', fontsize=10)
    ax2.set_ylabel('Return Rate ($\mu$)', fontsize=10)
    ax2.set_zlabel('Utility', fontsize=10)
    # 调整视角以便更好地观察山峰和谷底
    ax2.view_init(elev=30, azim=225)
    fig.colorbar(surf, ax=ax2, shrink=0.7, pad=0.1, label='Utility')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88) # 给主标题留出空间
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    stamp = f"T{T}_u{u}_y{y}_nlow{n_low}_nsoon{n_soon}_nnorm{n_normal}"
    fig.savefig(output_path / f"utility_landscape_{stamp}.png", dpi=220)
    plt.show()

if __name__ == "__main__":
    # 调用绘图函数，你可以随意修改这里的参数来观察曲面的变化
    visualize_utility_landscape(
        T=1.0,       # 规划总时长
        u=0.5,       # 调度车到达时间
        y=3,         # 计划换电数量
        n_low=5,     # 严重亏电车
        n_soon=5,    # 即将亏电车
        n_normal=2,  # 满电车
        max_rate=40  # 坐标轴最大观察速率
    )