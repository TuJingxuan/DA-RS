import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
from mpl_toolkits.mplot3d import Axes3D
import time

class OrbitalGameAnalyzer:
    def __init__(self):
        # GEO 轨道参数
        self.mu = 3.986004418e14  # 地球引力常数 m^3/s^2
        self.r_geo = 42164137.0   # GEO 轨道半径 (m)
        self.n = np.sqrt(self.mu / self.r_geo**3) # 平均角速度 (rad/s)
        
    def get_cw_stm(self, t):
        """计算 C-W 状态转移矩阵 Phi(t)"""
        n = self.n
        nt = n * t
        c = np.cos(nt)
        s = np.sin(nt)
        
        phi = np.zeros((6, 6))
        # 简化版 C-W STM (行: r_x, r_y, r_z, v_x, v_y, v_z)
        phi[0,0] = 4 - 3*c;      phi[0,1] = 0; phi[0,2] = 0; phi[0,3] = s/n;          phi[0,4] = 2*(1-c)/n;      phi[0,5] = 0
        phi[1,0] = 6*(s-nt);     phi[1,1] = 1; phi[1,2] = 0; phi[1,3] = 2*(c-1)/n;    phi[1,4] = (4*s - 3*nt)/n; phi[1,5] = 0
        phi[2,0] = 0;            phi[2,1] = 0; phi[2,2] = c; phi[2,3] = 0;            phi[2,4] = 0;              phi[2,5] = s/n
        
        # 速度行这里省略详细计算，因为我们主要关注最终位置 r_final
        # 但为了完整性，如果需要计算终端速度约束，需要补全。
        # 这里只计算对位置的影响矩阵 H_k (即 Phi 的右上角 3x3)
        return phi

    def calculate_boundary(self, X0, total_fuel, max_pulse, dt, total_time, num_dirs=200):
        """
        利用贪心算法快速计算可达域的边界顶点
        X0: 初始相对状态 (6,)
        """
        steps = int(total_time / dt)
        if steps == 0: return np.array([X0[:3]])
        
        t_span = np.arange(1, steps + 1) * dt
        
        # 1. 预计算无控轨迹 (Zero-Effort Miss)
        # 所有的脉冲都是叠加在无控轨迹之上的
        phi_final = self.get_cw_stm(total_time)
        r_nominal = (phi_final @ X0)[:3] # 无控漂移后的位置
        
        # 2. 预计算每个时间步的脉冲对最终位置的影响矩阵 B_eff
        # r_final = r_nominal + sum( Phi(tf - tk) * [0; u_k] )
        # 也就是利用 STM 的右上角块
        H_stack = np.zeros((steps, 3, 3))
        for i, t in enumerate(t_span):
            time_to_go = total_time - (i * dt) # 假设在第 i 步施加脉冲
            phi = self.get_cw_stm(time_to_go)
            H_stack[i] = phi[0:3, 3:6] # B矩阵部分
            
        # 3. 在单位球面上采样方向向量 L (用于寻找边界)
        # 使用 Fibonacci Sphere 采样
        boundary_points = []
        phi_golden = np.pi * (3. - np.sqrt(5.)) 
        
        idx_array = np.arange(num_dirs)
        y = 1 - (idx_array / float(num_dirs - 1)) * 2 
        radius = np.sqrt(1 - y * y)
        theta = phi_golden * idx_array
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        L_dirs = np.column_stack((x, y, z)) # (num_dirs, 3)

        # 4. 针对每个方向 L，分配燃料 (贪心策略/PMP)
        # 计算效率矩阵: E = H_stack.T * L
        # (Steps, 3, 3) dot (3,) -> (Steps, 3)
        for L in L_dirs:
            # 计算每个时间步在该方向上的“效能” (Projected Gradient)
            eff_vecs = np.einsum('ijk,j->ik', H_stack, L) 
            eff_norms = np.linalg.norm(eff_vecs, axis=1)
            
            # 按效能排序
            sorted_indices = np.argsort(-eff_norms)
            
            fuel_left = total_fuel
            delta_v_sum = np.zeros(3) # 累计的位置偏移量
            
            for idx in sorted_indices:
                if fuel_left <= 0: break
                
                # 施加脉冲
                pulse = min(max_pulse, fuel_left)
                
                if eff_norms[idx] > 1e-12:
                    # 最优脉冲方向与效能向量同向
                    direction = eff_vecs[idx] / eff_norms[idx]
                    # 计算该脉冲产生的最终位移: H * u
                    # 这里利用等效性：disp = direction * pulse * norm(H*u_dir) ?? 
                    # 不，直接用 H * u 更稳
                    u_vec = direction * pulse
                    displacement = H_stack[idx] @ u_vec
                    
                    delta_v_sum += displacement
                    fuel_left -= pulse
            
            # 最终位置 = 无控位置 + 脉冲产生的位移
            boundary_points.append(r_nominal + delta_v_sum)
            
        return np.array(boundary_points)

    def check_coverage(self, pursuer_pts, evader_pts):
        """
        判断覆盖率
        方法：构建 Pursuer 的 Delaunay 三角剖分，检查 Evader 的顶点是否都在内部
        """
        try:
            # 1. 构建追踪者的几何体 (凸包/三角剖分)
            # Delaunay 本质上实现了论文中的“三角剖分”和“区域判断”
            hull = Delaunay(pursuer_pts)
            
            # 2. 检查逃跑者所有边界点是否在单纯形内
            # find_simplex 返回 -1 表示在外部
            simplex_indices = hull.find_simplex(evader_pts)
            
            total_points = len(evader_pts)
            inside_points = np.sum(simplex_indices >= 0)
            
            ratio = inside_points / total_points
            return ratio, (ratio >= 0.999) # 浮点容差
            
        except Exception as e:
            # 极少数情况（如点共面）可能导致 Delaunay 失败，返回未覆盖
            print(f"Geometry Error: {e}")
            return 0.0, False

def run_simulation():
    analyzer = OrbitalGameAnalyzer()
    
    # --- 1. 随机生成初始状态 ---
    # 限制在一个合理的相对范围内 (例如 +/- 20km)，否则 10h 肯定追不上
    pos_range = 20e3  # 20 km
    vel_range = 0.5   # 0.5 m/s
    
    # 相对状态 X0 = [rx, ry, rz, vx, vy, vz]
    # 假设追踪者在原点 (0,0,0)，只随机生成逃跑者的相对位置
    X0_evader = np.random.uniform(-1, 1, 6)
    X0_evader[0:3] *= pos_range
    X0_evader[3:6] *= vel_range
    
    print(f"--- 初始化 ---")
    print(f"初始相对位置 (km): {X0_evader[0:3]/1000}")
    print(f"初始相对速度 (m/s): {X0_evader[3:6]}")
    
    # --- 2. 参数设置 ---
    P_FUEL = 200.0; P_PULSE = 1.5
    E_FUEL = 100.0; E_PULSE = 0.5
    DT = 60.0 # 60s
    T_MIN = 0.0
    T_MAX = 10 * 3600.0 # 10 hours
    T_LIMIT_SEARCH = 20 # 二分法迭代次数
    
    found_time = None
    final_p_pts = None
    final_e_pts = None
    
    # --- 3. 二分法搜索 ---
    print(f"\n--- 开始二分法搜索 (Max T={T_MAX/3600}h) ---")
    
    low = T_MIN
    high = T_MAX
    best_coverage = 0.0
    
    for i in range(T_LIMIT_SEARCH):
        mid = (low + high) / 2.0
        
        # 极短时间跳过 (物理上不可能)
        if mid < 600: 
            low = mid
            continue

        # A. 计算追踪者可达域 (边界点)
        # 追踪者从原点出发去追，但在相对坐标系里，
        # 等价于追踪者在 (0,0) 产生能力包络，看是否包住“逃跑者从X0出发的包络”
        # 注意：这里我们计算的是绝对位移能力。
        # 实际上：判断 (Pursuer_Reach_Set) 是否包含 (Evader_Reach_Set_from_X0)
        
        # 追踪者 (从 0,0 出发的能力)
        p_pts = analyzer.calculate_boundary(np.zeros(6), P_FUEL, P_PULSE, DT, mid, num_dirs=300)
        
        # 逃跑者 (从 X0_evader 出发)
        e_pts = analyzer.calculate_boundary(X0_evader, E_FUEL, E_PULSE, DT, mid, num_dirs=300)
        
        # B. 覆盖检测
        ratio, is_covered = analyzer.check_coverage(p_pts, e_pts)
        
        print(f"Iter {i+1}: T = {mid/3600:.2f} h, Coverage = {ratio*100:.1f}%")
        
        if is_covered:
            found_time = mid
            final_p_pts = p_pts
            final_e_pts = e_pts
            high = mid # 尝试更短的时间
        else:
            best_coverage = max(best_coverage, ratio)
            low = mid # 需要更多时间
            
        # 精度截断
        if (high - low) < 60.0: # 精度 1 分钟
            break
            
    # --- 4. 结果展示 ---
    if found_time:
        print(f"\n>>> 成功! 在 T = {found_time/3600:.3f} 小时 实现 100% 覆盖。")
        
        # 绘图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 画追踪者 (绿色半透明凸包)
        try:
            hull_p = ConvexHull(final_p_pts)
            for s in hull_p.simplices:
                ax.plot_trisurf(final_p_pts[:,0], final_p_pts[:,1], final_p_pts[:,2], triangles=[s], color='g', alpha=0.1)
        except:
            pass # 如果点太少画不出凸包
        ax.scatter(final_p_pts[:,0], final_p_pts[:,1], final_p_pts[:,2], c='g', s=5, label='Pursuer Reachable Set')
        
        # 画逃跑者 (红色点云)
        ax.scatter(final_e_pts[:,0], final_e_pts[:,1], final_e_pts[:,2], c='r', s=10, label='Evader Reachable Set')
        
        # 画原点和初始位置
        ax.scatter(0,0,0, c='k', marker='x', s=100, label='Pursuer Start')
        # 逃跑者当前时刻的中心位置（无控漂移点）
        center_e = np.mean(final_e_pts, axis=0)
        ax.scatter(center_e[0], center_e[1], center_e[2], c='b', marker='*', s=100, label='Evader Center')

        ax.set_xlabel('Radial (m)')
        ax.set_ylabel('Along-Track (m)')
        ax.set_zlabel('Normal (m)')
        ax.legend()
        plt.title(f"Capture Possible at T={found_time/3600:.2f}h\n(Green covers Red)")
        plt.show()
        
    else:
        print(f"\n>>> 失败。在 10 小时内无法实现 100% 覆盖。")
        print(f"最大覆盖率达到: {best_coverage*100:.1f}%")
        print("可能原因：初始距离过远，或逃跑者利用相位漂移逃出了追踪者的燃料范围。")

if __name__ == "__main__":
    run_simulation()