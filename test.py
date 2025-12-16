import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# === 1. 核心动力学模型 (基于 Zhang et al. 的二体传播) ===
MU = 3.986004418e14  # 地球引力常数 (m^3/s^2)


def solve_kepler_batch(M, e, tol=1e-8):
    """
    向量化求解开普勒方程 E - e*sin(E) = M
    使用牛顿迭代法，支持数组输入
    """
    E = M.copy()
    for _ in range(10):
        f_val = E - e * np.sin(E) - M
        f_der = 1 - e * np.cos(E)
        delta = f_val / f_der
        E -= delta
        if np.all(np.abs(delta) < tol):
            break
    return E


def generate_tdrd_surface(a0, e0, f0, dv, target_time, n_beta=30, n_alpha=60):
    """
    生成固定时间可达域的曲面点云
    原理：论文中 TDRD 的边界是由所有可能的脉冲方向映射得到的封闭曲面
    """
    # 1. 初始状态 (标量)
    p0 = a0 * (1 - e0 ** 2)
    r0_mag = p0 / (1 + e0 * np.cos(f0))

    # 初始速度 (径向 vr, 切向 vt)
    v0_r = np.sqrt(MU / p0) * e0 * np.sin(f0)
    v0_t = np.sqrt(MU / p0) * (1 + e0 * np.cos(f0))

    # 2. 构建脉冲方向网格 (alpha, beta)
    beta_vec = np.linspace(-np.pi / 2 + 0.1, np.pi / 2 - 0.1, n_beta)
    alpha_vec = np.linspace(0, 2 * np.pi, n_alpha)

    B, A = np.meshgrid(beta_vec, alpha_vec)
    B = B.flatten()
    A = A.flatten()

    # 3. 计算施加脉冲后的速度 (LVLH 坐标系)
    dv_r = dv * np.cos(B) * np.cos(A)
    dv_t = dv * np.cos(B) * np.sin(A)
    dv_n = dv * np.sin(B)

    v1_r = v0_r + dv_r
    v1_t = v0_t + dv_t
    v1_n = dv_n
    v1_sq = v1_r ** 2 + v1_t ** 2 + v1_n ** 2

    # 4. 批量计算转移轨道要素
    inv_a1 = 2.0 / r0_mag - v1_sq / MU
    valid_mask = inv_a1 > 1e-10

    inv_a1 = inv_a1[valid_mask]
    v1_r = v1_r[valid_mask]
    v1_t = v1_t[valid_mask]
    v1_n = v1_n[valid_mask]

    a1 = 1.0 / inv_a1
    n1 = np.sqrt(MU / a1 ** 3)

    h_x = np.zeros_like(v1_r)
    h_y = -r0_mag * v1_n
    h_z = r0_mag * v1_t
    h1_sq = h_x ** 2 + h_y ** 2 + h_z ** 2

    p1 = h1_sq / MU
    term = 1 - p1 / a1
    term = np.maximum(term, 0)
    e1 = np.sqrt(term)

    # 5. 计算初始真近点角 f1_0
    rv_dot = r0_mag * v1_r
    cos_f0 = (p1 / r0_mag - 1.0) / (e1 + 1e-12)
    sin_f0 = rv_dot * np.sqrt(p1 / MU) / (p1 * e1 + 1e-12)
    f1_0 = np.arctan2(sin_f0, cos_f0)

    # 6. 传播到目标时间
    term_sqrt = np.sqrt((1 - e1) / (1 + e1))
    E0 = 2 * np.arctan(term_sqrt * np.tan(f1_0 / 2))
    M0 = E0 - e1 * np.sin(E0)

    M1 = M0 + n1 * target_time
    E1 = solve_kepler_batch(M1, e1)

    term_sqrt_inv = 1.0 / term_sqrt
    f1_1 = 2 * np.arctan(term_sqrt_inv * np.tan(E1 / 2))
    d_theta = f1_1 - f1_0

    # 7. 计算终端位置 (转移平面内)
    r_final_mag = p1 / (1 + e1 * np.cos(f1_1))

    # 8. 坐标变换
    delta_i = np.arctan2(v1_n, v1_t)

    x_orb = r_final_mag * np.cos(d_theta)
    y_orb = r_final_mag * np.sin(d_theta)
    z_orb = np.zeros_like(x_orb)

    x_final = x_orb
    y_final = y_orb * np.cos(delta_i) - z_orb * np.sin(delta_i)
    z_final = y_orb * np.sin(delta_i) + z_orb * np.cos(delta_i)

    return x_final, y_final, z_final


# === 主程序 ===
import plotly.graph_objects as go
import numpy as np

# ... (保留之前的 MU 常量和 generate_tdrd_surface 函数定义) ...

# === 主程序 ===
if __name__ == "__main__":
    # 1. 设置参数 (同前)
    r_geo = 42164.0 * 1000
    e_geo = 0.0
    f0 = 0.0
    dv = 50.0
    t_target = 12 * 3600

    # 2. 计算数据 (同前)
    import time

    start = time.time()
    x, y, z = generate_tdrd_surface(r_geo, e_geo, f0, dv, t_target, n_beta=40, n_alpha=80)
    print(f"计算耗时: {time.time() - start:.4f}s")

    # 3. 计算标称位置
    n_geo = np.sqrt(MU / r_geo ** 3)
    theta_nom = n_geo * t_target
    x_nom = r_geo * np.cos(theta_nom)
    y_nom = r_geo * np.sin(theta_nom)

    # === Plotly 绘图部分 ===
    # 单位转换为 km
    x_km, y_km, z_km = x / 1e3, y / 1e3, z / 1e3

    fig = go.Figure()

    # (1) 绘制可达域点云
    fig.add_trace(go.Scatter3d(
        x=x_km, y=y_km, z=z_km,
        mode='markers',
        marker=dict(
            size=2,
            color=z_km,  # 颜色随 Z 轴变化
            colorscale='Viridis',  # 颜色映射
            opacity=0.6
        ),
        name='可达域 (Reachable Domain)'
    ))

    # (2) 绘制标称位置 (红星)
    fig.add_trace(go.Scatter3d(
        x=[x_nom / 1e3], y=[y_nom / 1e3], z=[0],
        mode='markers',
        marker=dict(size=10, color='red', symbol='diamond'),
        name='标称位置 (Nominal)'
    ))

    # (3) 绘制初始位置
    fig.add_trace(go.Scatter3d(
        x=[r_geo / 1e3], y=[0], z=[0],
        mode='markers',
        marker=dict(size=6, color='black', symbol='circle'),
        name='初始位置 (Start)'
    ))

    # (4) 设置布局 (关键：锁定比例，否则球会变扁)
    fig.update_layout(
        title=f"GEO Satellite TDRD (t={t_target / 3600}h, dv={dv}m/s)",
        scene=dict(
            xaxis_title='Radial (km)',
            yaxis_title='Along-Track (km)',
            zaxis_title='Cross-Track (km)',
            aspectmode='data'  # 强制 xyz 比例一致
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()