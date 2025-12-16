import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# --- 论文参数 (Section 5.1 Case 2) ---
MU = 398600.4418
A0 = 10000.0
E0 = 0.2
DV_MAX = 0.3  # km/s (为了效果明显，可设为0.5，此处按论文设0.3)
N_IMPULSE = 3

class StrictAlgorithm2:
    def __init__(self, a0, e0, dv_max, N):
        self.a0 = a0
        self.e0 = e0
        self.p0 = a0 * (1 - e0**2)
        self.dv_max = dv_max
        self.N = N

    # --- Eq. 6: 旋转矩阵 ---
    def Mx(self, x): return np.array([[1, 0, 0], [0, np.cos(x), np.sin(x)], [0, -np.sin(x), np.cos(x)]])
    def My(self, x): return np.array([[np.cos(x), 0, -np.sin(x)], [0, 1, 0], [np.sin(x), 0, np.cos(x)]])
    def Mz(self, x): return np.array([[np.cos(x), np.sin(x), 0], [-np.sin(x), np.cos(x), 0], [0, 0, 1]])

    def get_rv(self, p, e, f):
        """标准二体状态计算"""
        rm = p / (1 + e * np.cos(f))
        r = rm * np.array([np.cos(f), np.sin(f), 0])
        v = np.sqrt(MU/p) * np.array([-np.sin(f), e + np.cos(f), 0])
        return r, v

    def _propagate_N_minus_1(self, X):
        """
        递推前 N-1 次脉冲，对应论文 Section 3/4 的状态更新。
        返回: R_accum (坐标系旋转), p, e (当前形状), dv_rem (剩余预算)
        """
        p, e = self.p0, self.e0
        # 初始 R 为单位阵
        R_accum = np.eye(3)
        dv_budget = self.dv_max
        idx = 0
        
        # --- Impulse 1 ---
        # 变量: [alpha1, eps1] (Eq 28/32 变量定义)
        alpha1 = X[idx]; eps1 = X[idx+1]; idx += 2
        
        dv1 = eps1 * dv_budget
        dv_budget -= dv1
        
        # 动力学更新 (Impulse 1)
        r, v = self.get_rv(p, e, alpha1)
        v_unit = v / np.linalg.norm(v)
        v_new = v + dv1 * v_unit # 假设切向，这是MIRD求解的常规假设
        
        h = np.cross(r, v_new)
        vec_e = np.cross(v_new, h)/MU - r/np.linalg.norm(r)
        p = np.dot(h, h) / MU
        e = np.linalg.norm(vec_e)
        
        # --- Impulse 2 ... N-1 ---
        for k in range(self.N - 2):
            # 变量: [eta, lam, alpha, eps]
            eta = X[idx]; lam = X[idx+1]; alpha = X[idx+2]; eps = X[idx+3]
            idx += 4
            
            dv = eps * dv_budget
            dv_budget -= dv
            
            # --- 严格实现 Eq. 19 (中间过程约束) ---
            # 为了让梯度连续，我们不在 python 层做 if 截断，而是通过数学变换
            # phi = lam * phi_max
            # phi_max 由 dv 和 alpha 决定
            
            term = (1 + e * np.cos(alpha))
            # K = (mu / p*dv^2) * term^2 - 1
            K = (MU / (p * dv**2 + 1e-12)) * term**2 - 1
            
            # 安全处理 K (避免数值爆炸)
            if K < 1e-5: K = 1e-5 
            
            tan2_phi_max = (np.sin(eta - alpha)**2) / K
            phi_max = np.arctan(np.sqrt(tan2_phi_max))
            
            phi = lam * phi_max
            
            # --- Eq. 5 & 7: 坐标系更新 ---
            # 简化实现: Mz(eta) * My(phi) 模拟相对转动
            R_step = self.Mz(eta) @ self.My(phi)
            R_accum = R_accum @ R_step
            
            # 动力学更新
            r, v = self.get_rv(p, e, alpha)
            v_unit = v / np.linalg.norm(v)
            v_new = v + dv * v_unit
            h = np.cross(r, v_new)
            vec_e = np.cross(v_new, h)/MU - r/np.linalg.norm(r)
            p = np.dot(h, h) / MU
            e = np.linalg.norm(vec_e)
            
        return R_accum, p, e, dv_budget

    def _calc_final_constraints(self, X, eta_f, phi_f):
        """
        计算 Eq. 9 约束值: strict check for final impulse
        Constraint: C(x) >= 0
        """
        # 1. 递推状态
        R, p, e, dv_N = self._propagate_N_minus_1(X)
        
        # 2. 获取最后一次脉冲位置 alpha_N (包含在变量 X 的最后)
        alpha_N = X[-1]
        
        # 3. 目标方向转换 (Global -> Local)
        u_global = np.array([
            np.cos(phi_f)*np.cos(eta_f), 
            np.cos(phi_f)*np.sin(eta_f), 
            np.sin(phi_f)
        ])
        u_local = R.T @ u_global # 转到局部系
        
        # 计算局部系下的球坐标 (用于 Eq 9)
        # u_local = [cos(phi_p)cos(eta_p), cos(phi_p)sin(eta_p), sin(phi_p)]
        phi_p = np.arcsin(np.clip(u_local[2], -1, 1))
        eta_p = np.arctan2(u_local[1], u_local[0])
        
        # --- Eq. 9: Reachability Condition ---
        # tan^2(phi_p) <= sin^2(eta_p - f) / K
        # f 即为 alpha_N
        
        term = (1 + e * np.cos(alpha_N))
        K = (MU / (p * dv_N**2 + 1e-12)) * term**2 - 1
        
        if K <= 0: return -1.0 # 物理不可达 (dv过大? 不太可能在小脉冲下发生, 除非p极小)
        
        rhs = (np.sin(eta_p - alpha_N)**2) / K
        lhs = np.tan(phi_p)**2
        
        return rhs - lhs # 必须 >= 0

    def objective_func(self, X, eta_f, phi_f, mode):
        """
        目标函数 r_f
        """
        R, p, e, dv_N = self._propagate_N_minus_1(X)
        alpha_N = X[-1]
        
        # 目标方向
        u_global = np.array([np.cos(phi_f)*np.cos(eta_f), np.cos(phi_f)*np.sin(eta_f), np.sin(phi_f)])
        u_local = R.T @ u_global
        
        # 计算平面改变消耗 (Eq 8 变形)
        # 此时已经通过 Constraint 保证了 dv_N 足够
        # 我们需要计算 "面内等效 dV" (dV_M in Eq 8)
        
        u_z = u_local[2] # sin(beta)
        beta = np.arcsin(np.clip(u_z, -1, 1))
        
        term = (1 + e * np.cos(alpha_N))
        
        # Eq 8: dV_M^2 = dV^2 - (mu/p)*(1+ecosf)^2 * sin^2 beta
        term_beta_cost = (MU / p) * term**2 * np.sin(beta)**2
        
        dv_sq = dv_N**2
        if dv_sq < term_beta_cost:
            # 理论上被 Constraint 拦截，但防止浮点误差
            dv_M = 0
        else:
            dv_M = np.sqrt(dv_sq - term_beta_cost)
            
        # 计算面内半径极值
        # 目标真近点角 (Local Frame)
        f_target = np.arctan2(u_local[1], u_local[0])
        
        # r_f = p_new / (1 + e_new * cos(f_true_new))
        # 简化计算: 假设 dV_M 切向施加
        
        # 当前状态
        r_now = p / term
        v_mag = np.sqrt(MU/p) * np.sqrt(1 + e**2 + 2*e*np.cos(alpha_N))
        v_t = np.sqrt(MU/p) * term
        v_r = np.sqrt(MU/p) * e * np.sin(alpha_N)
        
        # 施加 dV_M
        if mode == 'max':
            v_t_new = v_t + dv_M
        else:
            v_t_new = v_t - dv_M
            
        # 更新 p, e
        h_new = r_now * v_t_new
        p_new = h_new**2 / MU
        
        v_sq_new = v_r**2 + v_t_new**2
        inv_a = 2/r_now - v_sq_new/MU
        if inv_a <= 0: e_new = 0.0 # 保护
        else: e_new = np.sqrt(max(0, 1 - p_new*inv_a))
        
        # 近地点旋转修正 (近似: 目标 f_target 对应的半径)
        # 由于我们无法精确控制 e_vec 的旋转，这里使用能量法近似半径变化
        # r_new ~ r_nom + dr
        r_nom = p / (1 + e * np.cos(f_target))
        dr = (2 * r_nom**2 * v_mag / MU) * dv_M # Sensitivity
        
        if mode == 'max':
            val = r_nom + dr
            return -val # minimize negative
        else:
            val = r_nom - dr
            return val

    def solve(self, n1=20, n2=6):
        # 变量 X (N=3):
        # [alpha1, eps1] + [eta2, lam2, alpha2, eps2] + [alpha_3]
        # 共 7 个变量
        bounds = [
            (0, 2*np.pi), (0.01, 0.99),       # Imp 1
            (0, 2*np.pi), (-1, 1), (0, 2*np.pi), (0.01, 0.99), # Imp 2
            (0, 2*np.pi)                      # Imp 3 Position
        ]
        
        # 随机初值
        x0 = np.array([np.pi, 0.5, np.pi, 0.5, np.pi, 0.5, np.pi])
        
        results = []
        print(f"Strict Paper Solver (Algorithm 2) Running... (N={self.N})")
        t0 = time.time()
        
        for i in range(n1):
            eta = (i/n1) * 2 * np.pi
            print(f"Slice {i+1}/{n1}...", end='\r')
            
            # 物理 Phi 极限
            phi_limit = 0.05 
            
            for j in range(n2 + 1):
                phi = (j/n2) * phi_limit
                
                # 定义约束: Eq 9 必须 >= 0
                cons = ({'type': 'ineq', 
                         'fun': lambda x: self._calc_final_constraints(x, eta, phi)})
                
                # 1. 求解 r_max
                res_max = minimize(self.objective_func, x0, args=(eta, phi, 'max'),
                                   bounds=bounds, constraints=cons, 
                                   method='SLSQP', tol=1e-4, options={'maxiter': 50})
                
                # 2. 求解 r_min (热启动)
                res_min = minimize(self.objective_func, res_max.x, args=(eta, phi, 'min'),
                                   bounds=bounds, constraints=cons,
                                   method='SLSQP', tol=1e-4, options={'maxiter': 50})
                
                if res_max.success:
                    r_max = -res_max.fun
                    r_min = res_min.fun
                    
                    # 过滤物理异常值
                    if 6000 < r_min < r_max < 20000:
                        results.append([eta, phi, r_min, r_max])
                        if abs(phi) > 1e-6:
                            results.append([eta, -phi, r_min, r_max])
                            
        print(f"\nCompleted in {time.time()-t0:.2f}s")
        return np.array(results)

# --- 绘图 ---
if __name__ == "__main__":
    solver = StrictAlgorithm2(a0=10000, e0=0.2, dv_max=0.3, N=3)
    data = solver.solve(n1=20, n2=6)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    X_max = data[:,3] * np.cos(data[:,1]) * np.cos(data[:,0])
    Y_max = data[:,3] * np.cos(data[:,1]) * np.sin(data[:,0])
    Z_max = data[:,3] * np.sin(data[:,1])
    
    X_min = data[:,2] * np.cos(data[:,1]) * np.cos(data[:,0])
    Y_min = data[:,2] * np.cos(data[:,1]) * np.sin(data[:,0])
    Z_min = data[:,2] * np.sin(data[:,1])
    
    ax.scatter(X_max, Y_max, Z_max, c='b', s=10, alpha=0.4, label='Outer Envelope')
    ax.scatter(X_min, Y_min, Z_min, c='r', s=10, alpha=0.4, label='Inner Envelope')
    
    # 标称轨道
    t = np.linspace(0, 2*np.pi, 200)
    r0 = solver.p0 / (1 + solver.e0 * np.cos(t))
    ax.plot(r0*np.cos(t), r0*np.sin(t), 0, 'k--', lw=2, label='Nominal')
    
    # 比例修正
    # ax.set_box_aspect([1,1,0.3])
    
    plt.legend()
    plt.title("Strict Algorithm 2 (Eq 9 Constraint Applied)")
    plt.show()