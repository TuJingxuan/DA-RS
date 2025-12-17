import numpy as np
from scipy.optimize import minimize
from math import sin, cos, tan, sqrt, asin, acos, pi, atan2
import matplotlib.pyplot as plt
import time

class MIRDSolver:
    """
    论文算法复现: Multi-Impulse Reachable Domain (Algorithm 2)
    """
    def __init__(self, mu=398600.4418):
        self.mu = mu

    # ---------------------------------------------------------
    # 1. 基础旋转矩阵 (Rotation Matrices)
    # ---------------------------------------------------------
    def _rot_x(self, angle):
        """绕 X 轴旋转 (对应倾角改变 beta)"""
        c, s = cos(angle), sin(angle)
        return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])

    def _rot_z(self, angle):
        """绕 Z 轴旋转 (对应面内角度变化 f, eta)"""
        c, s = cos(angle), sin(angle)
        return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

    # ---------------------------------------------------------
    # 2. 动力学传播 (Dynamics, Eq. 8-13)
    # ---------------------------------------------------------
    def propagate_impulse(self, p, e, f, dv, alpha, beta):
        """
        执行单次脉冲机动，计算新轨道要素
        """
        # [Eq. 8] 物理可行性检查: dv 必须足以支付平面改变
        v_t_coeff = sqrt(self.mu / p) * (1 + e * cos(f))
        dv_perp_sq = (v_t_coeff * sin(beta))**2
        
        # 如果 dv 不够改变倾角，返回失败
        if dv**2 < dv_perp_sq - 1e-12:
            return None, None, None, None, False
        
        # 计算剩余的面内 dv 分量
        dv_in_plane = sqrt(max(0.0, dv**2 - dv_perp_sq))
        
        # [Eq. 10] 计算脉冲后的速度矢量
        v_r = sqrt(self.mu / p) * e * sin(f)
        v_new_r = v_r + dv_in_plane * cos(alpha)
        # 注意: 切向速度在新平面上的投影需乘 cos(beta)
        v_new_t = v_t_coeff * cos(beta) + dv_in_plane * sin(alpha)
        
        # 反解新轨道要素
        r = p / (1 + e * cos(f))
        h_new = r * v_new_t
        p_new = (h_new**2) / self.mu
        
        v_sq = v_new_r**2 + v_new_t**2
        energy = v_sq / 2 - self.mu / r
        
        # 保护: 避免抛物线/双曲线
        if abs(energy) < 1e-9: 
            e_new = 1.0 
        else:
            a_new = -self.mu / (2 * energy)
            e_new = sqrt(max(0.0, 1 - p_new / a_new))
            
        # 计算新轨道的真近点角 f_new
        f_new = atan2(h_new * v_new_r / self.mu, p_new / r - 1)
        if f_new < 0: f_new += 2 * pi
            
        return p_new, e_new, f_new, h_new, True

    # ---------------------------------------------------------
    # 3. 末端极值解 (SIRD Analytical Solution)
    # ---------------------------------------------------------
    def get_optimal_sird_limit(self, p, e, dv, eta_target):
        """
        计算最后一次脉冲在目标方向 eta_target 处的最大可达纬度。
        """
        # 解析极值条件: cos(eta - f) = -e * cos(eta)
        rhs = np.clip(-e * cos(eta_target), -1.0, 1.0)
        delta = acos(rhs)
        
        # 两个候选脉冲位置 (SIRD 的最优施加点)
        f_candidates = [eta_target - delta, eta_target + delta]
        max_phi = 0.0
        
        for f_imp in f_candidates:
            v_t = sqrt(self.mu / p) * (1 + e * cos(f_imp))
            
            # 全力改变倾角: sin(beta_max/2) = dv / 2v
            ratio = dv / (2 * v_t)
            beta_max = pi if ratio >= 1.0 else 2 * asin(ratio)
            
            # 球面三角: sin(phi) = sin(u) * sin(beta)
            sin_phi = sin(eta_target - f_imp) * sin(beta_max)
            max_phi = max(max_phi, abs(asin(np.clip(sin_phi, -1.0, 1.0))))
            
        return max_phi

    # ---------------------------------------------------------
    # 4. 目标函数 (Objective Function, Eq. 28)
    # ---------------------------------------------------------
    def objective_func(self, X, init_orbit, eta_f, dv_max, N):
        """
        计算 phi_f (返回负值以供 minimize 最小化)
        """
        p, e, f, _, _, _ = init_orbit
        R_acc = np.eye(3) # 累积旋转矩阵
        dv_rem = dv_max
        
        idx = 0
        # --- 遍历前 N-1 次脉冲 ---
        for k in range(1, N):
            if k == 1:
                # 第一次脉冲: 固定 f0=0, 仅优化 alpha, eps
                alpha, eps = X[idx], X[idx+1]
                idx += 2
                f_maneuver = f # f0
                beta = 0.0 # 初始面内机动
            else:
                # 后续脉冲: eta, lam, alpha, eps
                eta, lam, alpha, eps = X[idx:idx+4]
                idx += 4
                f_maneuver = eta
                
                # [Eq. 19/21] 松弛约束: lambda -> beta
                dv_step = eps * dv_rem
                v_t_approx = sqrt(self.mu/p)*(1+e*cos(f_maneuver))
                beta_limit = pi/2 if dv_step >= v_t_approx else asin(dv_step/v_t_approx)
                beta = lam * beta_limit 
            
            # 动力学传播
            dv_step = eps * dv_rem
            dv_rem -= dv_step
            p_next, e_next, f_next_start, _, valid = self.propagate_impulse(p, e, f_maneuver, dv_step, alpha, beta)
            
            if not valid: return 10.0 # 惩罚值
            
            # 更新旋转矩阵 R_{0->i}
            # 顺序: Mz(f_next) * Mx(beta) * Mz(-f_current)
            R_step = self._rot_z(f_next_start) @ self._rot_x(beta) @ self._rot_z(-f_maneuver)
            R_acc = R_step @ R_acc
            
            p, e, f = p_next, e_next, f_next_start

        # --- 终端计算 ---
        # [Eq. 23] 计算目标点在最终轨道系中的方位角 eta_pf
        R_inv = R_acc.T
        vec_eta = np.array([cos(eta_f), sin(eta_f), 0])
        vec_proj = R_inv @ vec_eta
        eta_pf = atan2(vec_proj[1], vec_proj[0])
        
        # [Eq. 21] 计算最后一次脉冲的能力边界 (SIRD Limit)
        phi_limit = self.get_optimal_sird_limit(p, e, dv_rem, eta_pf)
        
        # [Eq. 27] 极大化 phi_f
        A = R_inv[0, 2] * cos(eta_pf) + R_inv[1, 2] * sin(eta_pf)
        B = R_inv[2, 2]
        
        # 寻找驻点: tan(phi) = B/A
        phi_star = atan2(B, A)
        
        # 检查候选点 (边界 + 驻点)
        candidates = [phi_limit, -phi_limit]
        for phi_c in [phi_star, phi_star + pi, phi_star - pi]:
            if abs(phi_c) <= phi_limit + 1e-9:
                candidates.append(phi_c)
                
        # 计算最大值
        max_phi_f = max([asin(np.clip(A*cos(p_)+B*sin(p_), -1, 1)) for p_ in candidates])
        
        return -max_phi_f

    # ---------------------------------------------------------
    # 5. 主求解器 (Algorithm 2 Main Loop)
    # ---------------------------------------------------------
    def solve_phi_max(self, orbit_elements, dv_max, N, n1=50):
        # 变量边界
        bounds = [(-pi, pi), (0.001, 0.999)] # k=1
        for _ in range(2, N):
            bounds.extend([(0, 2*pi), (-1.0, 1.0), (-pi, pi), (0.01, 0.99)]) # k>1
            
        results = []
        
        # 网格化初值 (Grid Search) + 热启动 (Warm Start)
        grid_guesses = [
            [0.0, 0.5, pi, 0.9, 0.0, 0.5],    # 远地点, 正倾角
            [0.0, 0.5, pi, -0.9, 0.0, 0.5],   # 远地点, 负倾角
            [0.0, 0.5, 0.0, 0.9, 0.0, 0.5],   # 近地点
            [0.0, 0.5, pi/2, 0.9, 0.0, 0.5]   # 中点
        ]
        last_best_x = None # 用于热启动
        
        print(f"Starting MIRD Calculation (N={N}, Points={n1})...")
        t_start = time.time()

        for i in range(n1):
            eta_f = (i / n1) * 2 * pi
            global_best_phi = -1e9
            best_x_curr = None
            
            # 构建初值列表: 热启动 + 网格搜索
            current_guesses = []
            if last_best_x is not None:
                current_guesses.append(last_best_x)
            current_guesses.extend(grid_guesses)
            
            for x0_vals in current_guesses:
                x0 = np.array(x0_vals)
                # 补全 N>3 的情况
                if len(x0) < len(bounds): 
                    extra = [pi, 0.5, 0.0, 0.5] * (N - 3)
                    x0 = np.concatenate([x0, extra])[:len(bounds)]
                
                try:
                    res = minimize(
                        self.objective_func, x0, 
                        args=(orbit_elements, eta_f, dv_max, N), 
                        method='SLSQP', 
                        bounds=bounds, 
                        tol=1e-9, # 高精度以消除锯齿
                        options={'maxiter': 200}
                    )
                    
                    val = -res.fun
                    if val > global_best_phi:
                        global_best_phi = val
                        best_x_curr = res.x
                except: continue
            
            # 更新热启动种子
            if best_x_curr is not None:
                last_best_x = best_x_curr
                
            results.append((eta_f, global_best_phi))
            
            if (i+1) % 10 == 0: 
                print(f"Progress: {i+1}/{n1} | Phi_max: {global_best_phi:.5f}")

        print(f"Done. Time: {time.time()-t_start:.2f}s")
        return results

# -----------------------------------------------------------
# 运行示例 (Case 6)
# -----------------------------------------------------------
if __name__ == "__main__":
    mu = 3.986004418e14
    a = 10000e3
    e = 0.2
    p = a * (1 - e**2)
    # f0 固定为 0
    orbit0 = (p, e, 0.0, 0.0, 0.0, 0.0) 
    
    solver = MIRDSolver(mu)
    res = solver.solve_phi_max(orbit0, dv_max=100, N=3, n1=100)
    
    # 绘图
    data = np.array(res)
    eta_vals = data[:, 0]
    phi_vals = data[:, 1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(eta_vals, phi_vals, 'b-', linewidth=2, label=r'MIRD $\phi_{fmax}$')
    plt.title(r'Fig 6 Reproduction: $\phi_{fmax}$ vs $\eta_f$ (N=3)')
    plt.xlabel(r'Azimuth $\eta_f$ (rad)')
    plt.ylabel(r'Max Latitude $\phi_{fmax}$ (rad)')
    plt.xlim(0, 2*pi)
    plt.xticks(np.linspace(0, 2*pi, 5), ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.show()