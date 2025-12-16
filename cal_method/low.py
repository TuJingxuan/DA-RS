import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

# =============================================================================
# 1. 全局常数与基础类
# =============================================================================
MU = 398600.4418  # 地球引力常数 (km^3/s^2)

class Orbit:
    def __init__(self, p, e, f_end):
        self.p = p
        self.e = e
        self.f_end = f_end  # 上一阶段结束时的真近点角

class Impulse:
    def __init__(self, mag, alpha):
        self.mag = mag
        self.alpha = alpha

class Node:
    def __init__(self, eta, phi):
        self.eta = eta
        self.phi = phi

# =============================================================================
# 2. 核心数学工具 (对应论文公式)
# =============================================================================

def Mz(theta):
    """对应论文 Eq. (6) 中的 Mz 矩阵"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, s, 0],
        [-s, c, 0],
        [0, 0, 1]
    ])

def Mx(beta):
    """对应论文 Eq. (6) 中的 Mx 矩阵"""
    c, s = np.cos(beta), np.sin(beta)
    return np.array([
        [1, 0, 0],
        [0, c, s],
        [0, -s, c]
    ])

# =============================================================================
# 3. 物理传播引擎 (The Physics Engine)
#    严格实现 Eq. 4 - Eq. 14
# =============================================================================
def propagate_chain(orbit0, impulses, nodes, stages):
    """
    输入:
        orbit0: 初始轨道
        impulses: 脉冲列表 (List of Impulse objects)
        nodes: 目标节点列表 (List of Node objects)
        stages: 传播级数 (N 或 N-1)
    输出:
        final_info: 包含最终节点全局坐标的字典
        r_final: 最终半径
        c_vec: 不等式约束列表 (对应 Eq. 9, value >= 0 表示满足)
    """
    c_vec = []
    # R_cum 代表 R_{0(i-2)}，即从 Global 到当前 Local 的变换矩阵
    R_cum = np.eye(3)
    curr_orbit = Orbit(orbit0.p, orbit0.e, orbit0.f_end)
    
    r_final = 0.0
    final_info = {'eta_global': 0.0, 'phi_global': 0.0}

    for k in range(stages):
        # 1. 获取当前级的目标方向 (在全局坐标系中)
        eta_g = nodes[k].eta
        phi_g = nodes[k].phi
        
        # 2. 坐标转换 Global -> Local (Eq. 4)
        # u_local = R_cum * u_global
        u_g = np.array([
            np.cos(phi_g) * np.cos(eta_g),
            np.cos(phi_g) * np.sin(eta_g),
            np.sin(phi_g)
        ])
        u_l = R_cum @ u_g  # Matrix multiplication
        
        r_norm = np.linalg.norm(u_l)
        if r_norm < 1e-9: r_norm = 1e-9
        
        # 计算局部坐标系下的 eta_pi, phi_pi
        phi_p = np.arcsin(np.clip(u_l[2] / r_norm, -1.0, 1.0))
        eta_p = np.arctan2(u_l[1], u_l[0])
        
        # 3. 几何关系解算 (Eq. 7) -> beta, theta
        f_prev = curr_orbit.f_end
        delta = eta_p - f_prev
        
        # 处理共面奇点 (sin(delta) ~ 0)
        if abs(np.sin(delta)) < 1e-5:
            beta = 0.0
            theta = abs(delta)
        else:
            # tan(phi) = sin(delta) * tan(beta)
            beta = np.arctan(np.tan(phi_p) / np.sin(delta))
            # cos(theta) = cos(delta) * cos(phi)
            cos_theta = np.cos(delta) * np.cos(phi_p)
            theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            
        # 4. 可达性检查 (Eq. 8 & 9)
        # 计算维持轨道面改变所需的最小 Delta V (Out-of-plane component)
        dv_total = impulses[k].mag
        p, e = curr_orbit.p, curr_orbit.e
        
        # v_transverse^2 = (mu/p) * (1 + e*cos(f))^2
        v_trans_sq = (MU / p) * (1 + e * np.cos(f_prev))**2
        dv_out_sq = v_trans_sq * (np.sin(beta)**2)
        
        # 约束: Available^2 - Required^2 >= 0
        c_val = dv_total**2 - dv_out_sq
        c_vec.append(c_val)
        
        # 计算面内剩余 Delta V (用于更新轨道形状)
        if c_val < 0:
            dv_M = 0.0 # 物理不可行，设为0以保证数学运算不报错，依靠约束惩罚
        else:
            dv_M = np.sqrt(c_val)
            
        # 5. 速度更新 (Eq. 10)
        alpha = impulses[k].alpha
        # vx: Radial, vy: Transverse
        vx = np.sqrt(MU/p)*e*np.sin(f_prev) + dv_M * np.cos(alpha)
        vy = np.sqrt(MU/p)*(1+e*np.cos(f_prev))*np.cos(beta) + dv_M * np.sin(alpha)
        
        # 6. 轨道根数更新 (Eq. 11, 12)
        # Angular momentum h
        h_new = (p / (1 + e * np.cos(f_prev))) * vy # r * vy
        p_new = h_new**2 / MU
        
        # Eccentricity e
        r_curr = p / (1 + e * np.cos(f_prev))
        term_e = (vy - MU/h_new)**2
        e_new = (r_curr * vy / MU) * np.sqrt(term_e + vx**2)
        
        # True Anomaly f_start (Eq. 11)
        cos_f1 = (h_new * vy - MU) / (MU * e_new)
        sin_f1 = (h_new * vx) / (MU * e_new)
        f_start = np.arctan2(sin_f1, cos_f1)
        
        f_end_new = f_start + theta # 下一节点的真近点角
        
        # 7. 计算下一节点半径 (Eq. 13)
        r_next = p_new / (1 + e_new * np.cos(f_end_new))
        
        # 8. 更新旋转矩阵 R_cum (Eq. 5)
        # R_new = Mz(-f_new) * Mz(theta) * Mx(beta) * Mz(f_prev) * R_old
        # 这里的变换矩阵 T 是将 Old Local 转换到 New Local
        T = Mz(-f_end_new) @ Mz(theta) @ Mx(beta) @ Mz(f_prev)
        R_cum = T @ R_cum
        
        # 更新状态用于下一次迭代
        curr_orbit.p = p_new
        curr_orbit.e = e_new
        curr_orbit.f_end = f_end_new
        
        # 记录最后一次传播的结果
        if k == stages - 1:
            r_final = r_next
            
            # 反算最终位置的 Global 坐标 (用于 Step B 的约束检查)
            # r_local = [r * cos(f), r * sin(f), 0]
            # r_global = R_cum.T * r_local
            r_vec_local = np.array([
                r_next * np.cos(f_end_new),
                r_next * np.sin(f_end_new),
                0.0
            ])
            r_vec_global = R_cum.T @ r_vec_local
            
            final_info['eta_global'] = np.arctan2(r_vec_global[1], r_vec_global[0])
            norm_r = np.linalg.norm(r_vec_global)
            final_info['phi_global'] = np.arcsin(np.clip(r_vec_global[2]/norm_r, -1.0, 1.0))
            
            # 归一化 eta 到 0~2pi
            if final_info['eta_global'] < 0:
                final_info['eta_global'] += 2*np.pi
                
    return final_info, r_final, c_vec

# =============================================================================
# 4. 变量管理 (Mapping Optimization Variables <-> Physical Model)
#    对应 Eq. 21
# =============================================================================
def decode_X(X, N, Dv_max, include_alpha_N):
    impulses = []
    nodes = []
    
    idx = 0
    rem_dv = Dv_max
    
    # 第一次脉冲 (Impulse 1)
    alpha1 = X[idx]; idx += 1
    eps1 = X[idx]; idx += 1
    
    dv1 = rem_dv * eps1
    rem_dv -= dv1
    impulses.append(Impulse(dv1, alpha1))
    
    # 中间节点和脉冲 (Node 2..N, Impulse 2..N-1)
    for k in range(N - 1):
        # Node k position
        eta = X[idx]; idx += 1
        phi = X[idx]; idx += 1
        nodes.append(Node(eta, phi))
        
        # Impulse k+1 (除了最后一次)
        if k < N - 2:
            alpha = X[idx]; idx += 1
            eps = X[idx]; idx += 1
            dv = rem_dv * eps
            rem_dv -= dv
            impulses.append(Impulse(dv, alpha))
            
    # 最后一次脉冲 (Impulse N)
    if include_alpha_N:
        alpha_N = X[idx] # Step B 包含此变量
        impulses.append(Impulse(rem_dv, alpha_N)) # 使用剩余全部 Dv
    else:
        impulses.append(Impulse(0.0, 0.0)) # Step A 不优化最后一次方向
        
    return impulses, nodes

def get_bounds(num_vars, include_alpha_N):
    bounds = []
    
    # Impulse 1: alpha, eps
    bounds.append((-np.pi, np.pi)) 
    bounds.append((0.01, 0.99))    
    
    current_idx = 2
    
    while current_idx < num_vars:
        # Check if last variable is alpha_N
        if current_idx == num_vars - 1 and include_alpha_N:
            bounds.append((-np.pi, np.pi))
            break
            
        # Node: eta, phi
        bounds.append((0.0, 2*np.pi))  
        bounds.append((-np.pi, np.pi)) 
        current_idx += 2
        if current_idx >= num_vars: break
        
        # Check if Step A ends here (no more impulses)
        if not include_alpha_N and current_idx >= num_vars: break
        
        # Impulse: alpha, eps
        if current_idx < num_vars - 1: 
             bounds.append((-np.pi, np.pi)) 
             bounds.append((0.01, 0.99))    
             current_idx += 2
    return bounds

def generate_smart_guess(num_vars, include_alpha_N, eta_target):
    # 生成“聪明”的初始猜测，防止求解器一开始就陷入不可行区域
    # 策略：线性分布经度，共面飞行，均匀分配脉冲
    
    # 加入微小扰动打破对称性
    x0 = np.random.uniform(-0.01, 0.01, num_vars)
    bounds = get_bounds(num_vars, include_alpha_N)
    
    eta_indices = []
    for i in range(num_vars):
        lb, ub = bounds[i]
        
        if abs(lb - 0.01) < 1e-6: # Epsilon -> 0.5
            x0[i] = 0.5 
        elif abs(lb) < 1e-6 and abs(ub - 2*np.pi) < 1e-6: # Eta
            eta_indices.append(i)
        elif abs(lb + np.pi) < 1e-6: # Alpha/Phi -> 0 (Coplanar)
            x0[i] = 0.0 
            
    # 线性分布 eta: 从 0 到 eta_target
    n_nodes = len(eta_indices)
    for k, idx in enumerate(eta_indices):
        x0[idx] = eta_target * ((k + 1) / n_nodes)
        
    return x0

# =============================================================================
# 5. 优化目标与约束 (Objectives & Constraints)
# =============================================================================

def objective_phi(x, N, Dv_max, orbit0):
    # Eq. 28: Maximize |phi_N|
    return -abs(x[-1])

def objective_r(x, N, Dv_max, orbit0, sign, target_node):
    # Eq. 32: Min/Max r_f
    impulses, nodes = decode_X(x, N, Dv_max, True)
    # Step B 需要到达指定的目标点，将其加入节点列表
    nodes.append(target_node)
    _, r_f, _ = propagate_chain(orbit0, impulses, nodes, N)
    return sign * r_f

# =============================================================================
# 6. 主算法 (Algorithm 2)
# =============================================================================
def solve_mird():
    print("开始计算 MIRD (Algorithm 2 - Python)...")
    start_time = time.time()
    
    # 1. 初始条件
    p0 = 10000 * (1 - 0.2**2)
    e0 = 0.2
    f0 = 0.0
    orbit0 = Orbit(p0, e0, f0)
    
    Dv_max = 0.3
    N = 3
    
    n1 = 100  # 经度采样数
    n2 = 9  # 纬度采样数
    
    mird_data = [] # Store [eta, phi, r_min, r_max]
    
    # 2. 外层循环 (Eta)
    for i in range(n1):
        eta_f = (i / n1) * 2 * np.pi
        
        # ----------------------------------------------------
        # Step A: Solve Phi_max (Eq. 28)
        # ----------------------------------------------------
        num_vars_A = 2 + (N-2)*4 + 2
        x0_A = generate_smart_guess(num_vars_A, False, eta_f)
        bounds_A = get_bounds(num_vars_A, False)
        
        # 约束: 1. 物理可达 (c>=0)  2. 最终经度匹配 (eq=0)
        def con_physics_A(x):
            impulses, nodes = decode_X(x, N, Dv_max, False)
            _, _, c_vec = propagate_chain(orbit0, impulses, nodes, N-1)
            return np.array(c_vec)
            
        def con_geom_A(x):
            impulses, nodes = decode_X(x, N, Dv_max, False)
            final_info, _, _ = propagate_chain(orbit0, impulses, nodes, N-1)
            return np.sin((final_info['eta_global'] - eta_f)/2) # sin(diff/2) handles 0/2pi wrap
            
        constraints_A = [
            {'type': 'ineq', 'fun': con_physics_A},
            {'type': 'eq', 'fun': con_geom_A}
        ]
        
        res_A = minimize(objective_phi, x0_A, args=(N, Dv_max, orbit0),
                         method='SLSQP', bounds=bounds_A, constraints=constraints_A,
                         options={'ftol': 1e-4, 'disp': False, 'maxiter': 300})
        
        phi_fmax = 0.0
        if res_A.success or (hasattr(res_A, 'fun') and not np.isnan(res_A.fun)):
            phi_fmax = -res_A.fun
        
        # 物理保护: 0.3km/s 造成的倾角改变有限，超过 20deg 必为数值异常
        if phi_fmax > 0.35: phi_fmax = 0.05
        if phi_fmax < 1e-4: phi_fmax = 1e-4
        
        # 保存 Step A 结果用于热启动 Step B
        x_opt_A = res_A.x
        
        # ----------------------------------------------------
        # Step B: Solve R_min/max (Eq. 32)
        # ----------------------------------------------------
        num_vars_B = 2 + (N-1)*4 + 1
        x_base = generate_smart_guess(num_vars_B, True, eta_f)
        # 热启动
        len_common = min(len(x_base), len(x_opt_A))
        x_base[:len_common] = x_opt_A[:len_common]
        
        nodes_found = 0
        
        for j in range(n2 + 1):
            phi_f = (j / n2) * phi_fmax
            target_node = Node(eta_f, phi_f) # 固定目标点
            
            # 约束: 1. 物理可达  2. 最终经纬度匹配
            def con_physics_B(x):
                impulses, nodes = decode_X(x, N, Dv_max, True)
                nodes.append(target_node) # Append target explicitly
                _, _, c_vec = propagate_chain(orbit0, impulses, nodes, N)
                return np.array(c_vec)
            
            def con_geom_B(x):
                impulses, nodes = decode_X(x, N, Dv_max, True)
                nodes.append(target_node)
                final_info, _, _ = propagate_chain(orbit0, impulses, nodes, N)
                return np.array([
                    np.sin((final_info['eta_global'] - eta_f)/2),
                    final_info['phi_global'] - phi_f
                ])
                
            constraints_B = [
                {'type': 'ineq', 'fun': con_physics_B},
                {'type': 'eq', 'fun': con_geom_B}
            ]
            bounds_B = get_bounds(num_vars_B, True)
            
            # 针对 Max/Min 设置不同的 Alpha 初值
            x0_max = x_base.copy(); x0_max[-1] = 0.0   # Alpha ~ 0 (Speed up)
            x0_min = x_base.copy(); x0_min[-1] = np.pi # Alpha ~ Pi (Slow down)
            
            # Solve Max R
            res_max = minimize(objective_r, x0_max, args=(N, Dv_max, orbit0, -1, target_node),
                               method='SLSQP', bounds=bounds_B, constraints=constraints_B, options={'ftol': 1e-3})
            
            # Solve Min R
            res_min = minimize(objective_r, x0_min, args=(N, Dv_max, orbit0, 1, target_node),
                               method='SLSQP', bounds=bounds_B, constraints=constraints_B, options={'ftol': 1e-3})
            
            r_max, r_min = np.nan, np.nan
            if res_max.success or (hasattr(res_max, 'fun') and res_max.fun < 0): r_max = -res_max.fun
            if res_min.success or (hasattr(res_min, 'fun') and res_min.fun > 0): r_min = res_min.fun
            
            # 数据填补 (如果一边算出一边没算出)
            if np.isnan(r_max) and not np.isnan(r_min): r_max = r_min
            if np.isnan(r_min) and not np.isnan(r_max): r_min = r_max
            
            # 剔除异常值
            if r_max > 50000: r_max = np.nan
            
            if not np.isnan(r_max):
                mird_data.append([eta_f, phi_f, r_min, r_max])
                nodes_found += 1
                if phi_f > 1e-5: # 对称性
                    mird_data.append([eta_f, -phi_f, r_min, r_max])
                    nodes_found += 1
                    
        print(f"Eta: {np.degrees(eta_f):5.1f} deg | Phi_max: {np.degrees(phi_fmax):5.2f} deg | Nodes: {nodes_found}")

    print(f"Done. Time: {time.time()-start_time:.2f}s")
    
    # 3. 绘图
    if len(mird_data) > 0:
        data = np.array(mird_data)
        eta = data[:,0]
        phi = data[:,1]
        r_min, r_max = data[:,2], data[:,3]
        
        # Spherical to Cartesian (S_p0)
        # x = r cos(phi) cos(eta) ...
        
        x_max = r_max * np.cos(phi) * np.cos(eta)
        y_max = r_max * np.cos(phi) * np.sin(eta)
        z_max = r_max * np.sin(phi)
        
        x_min = r_min * np.cos(phi) * np.cos(eta)
        y_min = r_min * np.cos(phi) * np.sin(eta)
        z_min = r_min * np.sin(phi)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(x_max, y_max, z_max, c='r', s=2, alpha=0.5, label='Max Boundary')
        ax.scatter(x_min, y_min, z_min, c='b', s=2, alpha=0.5, label='Min Boundary')
        
        # 绘制参考地球
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        xe = 6378 * np.cos(u) * np.sin(v)
        ye = 6378 * np.sin(u) * np.sin(v)
        ze = 6378 * np.cos(v)
        ax.plot_wireframe(xe, ye, ze, color="cyan", alpha=0.2, label='Earth')
        
        # 设置等比例尺 (防止球看起来扁)
        max_range = np.array([x_max.max()-x_max.min(), y_max.max()-y_max.min(), z_max.max()-z_max.min()]).max() / 2.0
        mid_x = (x_max.max()+x_max.min()) * 0.5
        mid_y = (y_max.max()+y_max.min()) * 0.5
        mid_z = (z_max.max()+z_max.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title(f'MIRD Envelope (N={N})')
        ax.legend()
        plt.show()
    else:
        print("未找到有效数据，请检查 Dv_max 是否过小。")

if __name__ == "__main__":
    solve_mird()