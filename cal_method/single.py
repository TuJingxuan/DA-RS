import numpy as np
from scipy.optimize import newton, minimize_scalar
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import brentq
# --- 常量定义 ---
MU = 3.986e14  # 地球引力常数 (m^3/s^2)

# --- 辅助函数：开普勒方程求解 ---
def solve_kepler(M, e, tol=1e-9):
    """求解 E - e*sin(E) = M"""
    E = M if e < 0.8 else np.pi
    for _ in range(50):
        f_val = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        if abs(f_prime) < 1e-10: break # 防止除零
        E_new = E - f_val / f_prime
        if abs(E_new - E) < tol:
            return E_new
        E = E_new
    return E

def get_mean_anomaly(f, e):
    """从真近点角 f 计算平近点角 M，处理多圈情况"""
    term = np.sqrt((1 - e) / (1 + e)) * np.tan(f / 2.0)
    E = 2.0 * np.arctan(term)
    # 圈数修正，保持连续性
    k = np.floor((f + np.pi) / (2 * np.pi))
    E_adj = E + k * 2 * np.pi
    return E_adj - e * np.sin(E_adj)

# --- 核心模块1: 牛顿法求解 Alpha (Eq. 25 & Appendix) ---
def solve_alpha_roots_newton(theta_val, beta, t_target, r0_norm, p0, e0, f0, dv_max):
    """
    最稳健的求解器：不依赖导数，使用区间扫描 + Brent 方法。
    专门处理 t_target 接近周期或多圈的情况。
    
    参数:
        theta_val: 目标真近点角增量 (Delta f) [rad]
        beta: 轨道面旋转角 [rad]
        t_target: 目标飞行时间 [s]
    """
    
    # --- 1. 预计算常量 ---
    term_beta = (MU / p0) * ((1 + e0 * np.cos(f0))**2) * (np.sin(beta)**2)
    val_inside = dv_max**2 - term_beta
    dv_M = np.sqrt(max(0, val_inside))
    
    v0_r = np.sqrt(MU/p0) * e0 * np.sin(f0)
    v0_t = np.sqrt(MU/p0) * (1 + e0 * np.cos(f0))
    
    # --- 2. 残差函数：核心在于“自动对齐圈数” ---
    def time_residual(alpha):
        sin_a, cos_a = np.sin(alpha), np.cos(alpha)
        
        # A. 状态计算
        v1_x = v0_r + dv_M * cos_a
        v1_y = v0_t * np.cos(beta) + dv_M * sin_a
        v1_sq = v1_x**2 + v1_y**2
        
        h = r0_norm * v1_y
        energy = v1_sq / 2.0 - MU / r0_norm
        a_sem = -MU / (2.0 * energy)
        
        # 保护：非椭圆轨道直接返回大残差
        if a_sem <= 0: return 1e5
        
        n = np.sqrt(MU / a_sem**3)
        
        # 偏心率
        rv_dot = r0_norm * v1_x
        mu_inv = 1.0 / MU
        ex = mu_inv * ((v1_sq - MU/r0_norm)*r0_norm - rv_dot*v1_x)
        ey = mu_inv * ((v1_sq - MU/r0_norm)*0       - rv_dot*v1_y)
        e = np.sqrt(ex**2 + ey**2)
        
        # 真近点角 f1_1
        p = h**2 / MU
        cos_f1 = np.clip((p/r0_norm - 1.0) / (e + 1e-9), -1.0, 1.0)
        sin_f1 = v1_x * np.sqrt(p/MU) / (e + 1e-9)
        f1_1 = np.arctan2(sin_f1, cos_f1)
        
        # B. 目标真近点角 f1_2
        f1_2 = f1_1 + theta_val
        
        # C. 计算平近点角 M0, M1
        # 辅助函数：将 f 转为 E
        def f_to_E(f, ecc):
            return 2.0 * np.arctan(np.sqrt((1-ecc)/(1+ecc)) * np.tan(f/2.0))
        
        E0 = f_to_E(f1_1, e)
        E1_raw = f_to_E(f1_2, e)
        
        # Kepler Eq: M = E - e sin E
        M0 = E0 - e * np.sin(E0)
        M1_raw = E1_raw - e * np.sin(E1_raw)
        
        # D. 关键步骤：圈数对齐 (Automatic Revolution Matching)
        # 我们计算出的 dM_raw 范围在 [-2pi, 2pi] 之间
        dM_raw = M1_raw - M0
        
        # 目标应该是多少？
        M_target_total = n * t_target
        
        # 找到一个整数 k，使得 (dM_raw + k*2pi) 最接近 M_target_total
        # 这就是这一步的核心：我们不关心它转了几圈，我们强行让它去匹配目标时间
        # 从而消除 "6600s 是 0 圈还是 1 圈" 的歧义
        k = np.round((M_target_total - dM_raw) / (2 * np.pi))
        
        dM_final = dM_raw + k * 2 * np.pi
        
        # 计算时间 t
        t_calc = dM_final / n
        
        return t_calc - t_target

    # --- 3. 求解策略：扫描 + Brent ---
    roots = []
    
    # 扫描密度：100点通常足够捕获 1-2 个根的区间
    # 如果函数极其震荡，可以增加到 180 或 360
    alphas = np.linspace(-np.pi, np.pi, 100)
    residuals = np.array([time_residual(a) for a in alphas])
    
    # 寻找符号反转区间 (Bracketing)
    for i in range(len(alphas) - 1):
        y1 = residuals[i]
        y2 = residuals[i+1]
        
        if y1 * y2 < 0: # 发现跨越零点
            try:
                # Brentq 只要区间此时有根，必收敛
                root = brentq(time_residual, alphas[i], alphas[i+1], xtol=1e-6)
                # 归一化输出
                root = (root + np.pi) % (2 * np.pi) - np.pi
                roots.append(root)
            except ValueError:
                # 极罕见情况：区间内有奇点导致符号判定错误
                continue
    
    # --- 4. 兜底策略：处理相切 (Tangency) ---
    # 如果扫描没发现根，但残差最小值很小，说明是边界相切
    # 这在 RD 边界求解时非常常见
    if not roots:
        min_idx = np.argmin(np.abs(residuals))
        min_val = np.abs(residuals[min_idx])
        
        # 如果残差小于 10秒 (相对于 6600s 误差 < 0.2%)，我们可以认为这就是根
        # 或者是两个极其接近的根
        if min_val < 10.0:
            best_alpha = alphas[min_idx]
            # 尝试在这个点附近微调
            # 使用 minimize_scalar 在局部找最小值，看是否能降到 0
            from scipy.optimize import minimize_scalar
            
            # 定义局部搜索范围
            window = 0.2 # rad
            bnds = (best_alpha - window, best_alpha + window)
            
            # 最小化残差平方
            res_opt = minimize_scalar(lambda x: time_residual(x)**2, bounds=bnds, method='bounded')
            
            # 如果优化后的时间误差小于 1s，认为找到解
            if np.sqrt(res_opt.fun) < 1.0:
                root = (res_opt.x + np.pi) % (2 * np.pi) - np.pi
                roots.append(root)

    return roots

# --- 核心模块2: 求解 Theta 范围 (Eq. 20) ---
def solve_theta_range(beta, t_prop, r0_norm, p0, e0, f0, dv_max):
    """
    修正版 Algorithm 2 Step 2a: 
    增加对 Delta f 的圈数修正，防止出现负值或不连续跳变。
    """
    
    # 辅助：计算无缠绕的 Delta f
    def get_unwrapped_delta_f(alpha):
        # 1. 计算状态 (同前)
        term_beta = (MU / p0) * ((1 + e0 * np.cos(f0))**2) * (np.sin(beta)**2)
        val = dv_max**2 - term_beta
        if val < 0: val = 0
        dv_M = np.sqrt(val)
        
        v0_r = np.sqrt(MU/p0) * e0 * np.sin(f0)
        v0_t = np.sqrt(MU/p0) * (1 + e0 * np.cos(f0))
        v1_x = v0_r + dv_M * np.cos(alpha)
        v1_y = v0_t * np.cos(beta) + dv_M * np.sin(alpha)
        
        v1_sq = v1_x**2 + v1_y**2
        h = r0_norm * v1_y
        energy = v1_sq / 2.0 - MU / r0_norm
        a = -MU / (2.0 * energy)
        
        if a <= 0: return 1e9 # 保护非椭圆
        n = np.sqrt(MU / a**3)
        
        # e
        rv_dot = r0_norm * v1_x
        mu_inv = 1.0/MU
        ex = mu_inv * ((v1_sq - MU/r0_norm)*r0_norm - rv_dot*v1_x)
        ey = mu_inv * ((v1_sq - MU/r0_norm)*0 - rv_dot*v1_y)
        e = np.sqrt(ex**2 + ey**2)
        
        # 起始 f1
        p = h**2/MU
        cos_f1 = np.clip((p/r0_norm - 1.0)/e, -1.0, 1.0)
        sin_f1 = v1_x * np.sqrt(p/MU)/e
        f1_1 = np.arctan2(sin_f1, cos_f1)
        
        # 2. 通过平近点角 M 推算圈数 (核心修正)
        # E0
        term0 = np.sqrt((1 - e) / (1 + e)) * np.tan(f1_1 / 2.0)
        E0 = 2.0 * np.arctan(term0)
        M0 = E0 - e * np.sin(E0)
        
        # M1 = M0 + n*t
        M1 = M0 + n * t_prop
        
        # 反解 E1
        E1 = solve_kepler(M1, e)
        
        # 3. 计算 f2 (带圈数)
        term1 = np.sqrt((1 + e) / (1 - e)) * np.tan(E1 / 2.0)
        f1_2_principal = 2.0 * np.arctan(term1) # 这是 (-pi, pi)
        
        # 估算 f 的总增量应该和 M 的总增量近似
        # M_diff = n * t_prop
        # f_diff_approx ≈ M_diff
        # 我们需要找到 k，使得 (f1_2_principal - f1_1 + k*2pi) ≈ M_diff
        
        raw_diff = f1_2_principal - f1_1
        M_diff = n * t_prop
        
        k = np.round((M_diff - raw_diff) / (2 * np.pi))
        
        delta_f = raw_diff + k * 2 * np.pi
        
        return delta_f

    # 优化求解
    # 注意：由于 delta_f 是连续函数，我们应该能得到平滑的 min/max
    res_min = minimize_scalar(get_unwrapped_delta_f, bounds=(-np.pi, np.pi), method='bounded')
    theta_min = res_min.fun
    
    res_max = minimize_scalar(lambda x: -get_unwrapped_delta_f(x), bounds=(-np.pi, np.pi), method='bounded')
    theta_max = -res_max.fun # 取反回正
    
    return theta_min, theta_max

# --- 计算位置矢量模长 rf (Eq. 10) ---
def calculate_rf(alpha, theta, beta, r0_norm, p0, e0, f0, dv_max):
    """
    根据给定的 alpha 和 theta，计算对应的 rf
    """
    term_beta = (MU / p0) * ((1 + e0 * np.cos(f0))**2) * (np.sin(beta)**2)
    val = dv_max**2 - term_beta
    if val < 0: val = 0
    dv_M = np.sqrt(val)
    
    v0_r = np.sqrt(MU/p0) * e0 * np.sin(f0)
    v0_t = np.sqrt(MU/p0) * (1 + e0 * np.cos(f0))
    v1_x = v0_r + dv_M * np.cos(alpha)
    v1_y = v0_t * np.cos(beta) + dv_M * np.sin(alpha)
    h = r0_norm * v1_y # h标量
    
    # Eq. (10): rf = h^2 / (mu(1-cos theta) + h*vy*cos theta - h*vx*sin theta)
    # 注意: 这里的 theta 是相对角，公式中的 vx, vy 是 S1 系下的 v1_x, v1_y
    # 论文 Eq. 10 分母: mu(1-cos theta) + h*v1y*cos theta - h*v1x*sin theta
    # 修正: 论文公式可能有误或者使用了特定推导，通常极坐标下 r = p / (1 + e cos(f_new))
    # 我们这里直接用轨道公式 r = p / (1 + e cos(f1 + theta)) 最稳妥
    
    v1_sq = v1_x**2 + v1_y**2
    energy = v1_sq / 2.0 - MU / r0_norm
    # e 向量
    rv_dot = r0_norm * v1_x
    mu_inv = 1.0/MU
    ex = mu_inv * ((v1_sq - MU/r0_norm)*r0_norm - rv_dot*v1_x)
    ey = mu_inv * ((v1_sq - MU/r0_norm)*0 - rv_dot*v1_y)
    e = np.sqrt(ex**2 + ey**2)
    p = h**2/MU
    
    # 当前真近点角
    cos_f1 = np.clip((p/r0_norm - 1.0)/e, -1.0, 1.0)
    sin_f1 = v1_x * np.sqrt(p/MU)/e
    f1_1 = np.arctan2(sin_f1, cos_f1)
    
    f1_2 = f1_1 + theta
    rf = p / (1 + e * np.cos(f1_2))
    return rf

# --- Algorithm 2 主程序 ---
def algorithm_2_main(a0, e0, f0, dv_max, t1, N1=10, N3=20):
    """
    执行算法2
    返回: point_cloud list [(x, y, z), ...] 用于绘图
    """
    p0 = a0 * (1 - e0**2)
    r0_norm = p0 / (1 + e0 * np.cos(f0))
    
    # 1. 计算 beta_max [Eq. 8, 9]
    # term = sqrt(p/mu) * dv / (1 + e cos f)
    term_b = np.sqrt(p0/MU) * dv_max / (1 + e0 * np.cos(f0))
    if term_b > 1.0: term_b = 1.0
    beta_max = np.arcsin(term_b)
    
    print(f"Algorithm 2 Start: Beta_max = {np.degrees(beta_max):.2f} deg")
    
    results = [] # 存储结果 [beta, theta, rf_min, rf_max]
    
    # 2. 遍历 beta
    for i in range(N1+1):
        beta = (i / N1) * beta_max
        
        # Step 2a: Solve [theta_1, theta_2] 
        theta_min, theta_max = solve_theta_range(beta, t1, r0_norm, p0, e0, f0, dv_max)
        
        # Step 2b: 遍历 theta
        if theta_max < theta_min: continue
        
        theta_list = np.linspace(theta_min, theta_max, N3 + 1)[1:-1] # 去掉端点
        
        for theta in theta_list:
            # Step 2c: 求解 alpha (Newton法) [cite: 261-266]
            roots = solve_alpha_roots_newton(theta, beta, t1, r0_norm, p0, e0, f0, dv_max)
            #print(len(roots))
            if not roots: continue
            
            r_vals = []
            for alpha in roots:
                # Step 2d: 计算 rf
                rf = calculate_rf(alpha, theta, beta, r0_norm, p0, e0, f0, dv_max)
                r_vals.append(rf)
            
            if r_vals:
                rf_min = min(r_vals)
                rf_max = max(r_vals)
                results.append((beta, theta, rf_min, rf_max))
                # 对称性处理 (-beta)
                if beta > 1e-6:
                    results.append((-beta, theta, rf_min, rf_max))

    return results

def plot_envelope(results, f0, r0_norm):
    """简单的3D可视化结果"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    xs, ys, zs = [], [], []
    
    # 将球坐标 (r, theta, beta) 转换回笛卡尔坐标画图
    # 注意: 这里的坐标系是相对初始位置 r0 的
    # 简单起见，画出 S1 系下的点
    for (beta, theta, r_min, r_max) in results:
        # 这里的 theta 是 delta_f，也就是相对于 r0 的角度
        # 在 S1 系中:
        # x = r cos(theta)
        # y = r sin(theta) * cos(beta)  <-- 近似几何
        # z = r sin(theta) * sin(beta)
        
        # 绘制 min点
        xs.append(r_min * np.cos(theta))
        ys.append(r_min * np.sin(theta) * np.cos(beta))
        zs.append(r_min * np.sin(theta) * np.sin(beta))
        
        # 绘制 max点
        xs.append(r_max * np.cos(theta))
        ys.append(r_max * np.sin(theta) * np.cos(beta))
        zs.append(r_max * np.sin(theta) * np.sin(beta))

    ax.scatter(xs, ys, zs, s=1, c='b', alpha=0.5, label='RD Envelope')
    ax.scatter([r0_norm], [0], [0], c='r', marker='*', s=100, label='Initial Pos')
    
    ax.set_xlabel('Radial (m)')
    ax.set_ylabel('Transverse (m)')
    ax.set_zlabel('Normal (m)')
    ax.legend()
    plt.show()

# --- 运行示例 ---
if __name__ == "__main__":
    # 参数设置 (参考论文 Table 1) [cite: 386]
    a0_val = 41266 * 1000  # m
    e0_val = 0.0
    f0_val = np.radians(0.0)
    dv_val = 100.0          # m/s
    t1_val = 3600.0*40          # s
    
    # 运行算法
    data = algorithm_2_main(a0_val, e0_val, f0_val, dv_val, t1_val, N1=10, N3=15)
    
    # 简单的可视化
    p0_val = a0_val * (1 - e0_val**2)
    r0_n = p0_val / (1 + e0_val * np.cos(f0_val))
    plot_envelope(data, f0_val, r0_n)