import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tqdm
# 引入计算几何库
from scipy.spatial import Delaunay, ConvexHull, QhullError

# 假设 orbitx 库可用
try:
    from orbitx import coe2rv, rv_from_r0v0
except ImportError:
    # 简单的 fallback 防止报错（如果你本地有 orbitx 可以忽略这段）
    print("Warning: 'orbitx' library not found. Using placeholder functions.")
    def coe2rv(x): return np.zeros(6)
    def rv_from_r0v0(rv, dt): return rv

# ==========================================
# Part 1: MIRD 计算核心算法 (保持不变)
# ==========================================

def constrained_dv_distribution(total_dv, steps, max_step_dv):
    factors = np.random.random(steps)
    dvs = (factors / np.sum(factors)) * total_dv
    
    for _ in range(100):
        mask_exceed = dvs > max_step_dv
        if not np.any(mask_exceed):
            break
        excess = np.sum(dvs[mask_exceed] - max_step_dv)
        dvs[mask_exceed] = max_step_dv
        mask_valid = ~mask_exceed
        if np.sum(mask_valid) > 0:
            factors_valid = np.random.random(np.sum(mask_valid))
            dvs[mask_valid] += (factors_valid / np.sum(factors_valid)) * excess
        else:
            break
    return dvs

def get_random_directions(n):
    vecs = np.random.normal(size=(n, 3))
    norms = np.linalg.norm(vecs, axis=1)[:, np.newaxis]
    return vecs / norms

def get_constrained_mird_point(initial_state_rv, total_dv, steps, step_limit_dv, dt):
    try:
        dv_mags = constrained_dv_distribution(total_dv, steps, step_limit_dv)
    except Exception as e:
        return initial_state_rv[:3]
    
    dv_dirs = get_random_directions(steps)
    dvs_vectors = dv_dirs * dv_mags[:, np.newaxis]
    
    curr_rv = initial_state_rv.copy()
    for i in range(steps):
        curr_rv[3:6] += dvs_vectors[i]
        curr_rv = rv_from_r0v0(curr_rv, dt)
        
    return curr_rv[:3]

# ==========================================
# Part 2: 初始场景生成 (保持不变)
# ==========================================

def generate_initial_scenario(seed=42):
    np.random.seed(seed)

    num_p = 4
    base_sma = 42166.3
    dist_cap = 40.0
    e_init_dist_min_offset = 80.0
    e_init_dist_max_offset = 81.0
    current_sma_perturb_km = 0.5
    
    # 注意：你原本设置 E=0.100, P=0.150。为了测试"逃逸"效果，
    # 如果 E 的能力比 P 小很多，可能全是红色（被包围）。
    # 如果想看到绿色点，可以适当增加 e_init_dv 或减少 p_init_dv。
    p_init_dv = 0.200  
    e_init_dv = 0.100  

    ecc = 0.0
    inc = 0.0 
    raan, argp = 0.0, 0.0
    
    agents_data = {}

    ta_eva = np.random.uniform(0.0, 2 * np.pi)
    
    eva_sma = base_sma
    if current_sma_perturb_km > 0:
        perturb_km = np.random.uniform(-current_sma_perturb_km, current_sma_perturb_km)
        eva_sma += perturb_km
    
    e_rv = coe2rv(np.array([eva_sma, ecc, inc, raan, argp, ta_eva*180/np.pi]))
    agents_data['e_0'] = {'rv': e_rv, 'dv': e_init_dv, 'type': 'evader'}
    
    inner_dist = dist_cap + e_init_dist_min_offset
    outer_dist = dist_cap + e_init_dist_max_offset
    
    num_forward = num_p // 2 + num_p % 2
    num_backward = num_p // 2
    directions = ([1.0] * num_forward) + ([-1.0] * num_backward)
    np.random.shuffle(directions)

    for i in range(num_p):
        agent_id = f'p_{i}'
        target_dist = np.random.uniform(inner_dist, outer_dist)
        direction = directions[i]
        angle_offset = (target_dist / base_sma) * direction
        ta_pur = (ta_eva + angle_offset) % (2 * np.pi)
        
        pur_sma = base_sma
        if current_sma_perturb_km > 0:
            perturb_km = np.random.uniform(-current_sma_perturb_km, current_sma_perturb_km)
            pur_sma += perturb_km

        p_rv = coe2rv(np.array([pur_sma, ecc, inc, raan, argp, ta_pur*180/np.pi]))
        agents_data[agent_id] = {'rv': p_rv, 'dv': p_init_dv, 'type': 'pursuer'}

    print(f"Scenario generated: 1 Evader, {num_p} Pursuers around SMA {base_sma} km.")
    return agents_data, base_sma

# ==========================================
# Part 3: 新增 - 包络分析函数
# ==========================================

def analyze_containment(pursuer_points_dict, evader_points):
    """
    判断 evader_points 是否被 pursuer_points_dict 中所有点云的并集所包含。
    """
    # 1. 为每个追击者构建 Delaunay 三角剖分
    p_hulls = []
    for p_id, p_pts in pursuer_points_dict.items():
        try:
            # Delaunay 用于判断点是否在凸包内
            hull = Delaunay(p_pts)
            p_hulls.append(hull)
        except QhullError:
            print(f"Warning: Could not compute Hull for {p_id} (points might be coplanar).")
        except ValueError:
            pass

    if not p_hulls or len(evader_points) == 0:
        return False, 0.0, evader_points

    # 2. 检查每个逃逸点
    escaped_points = []
    contained_points = []
    
    for e_pt in evader_points:
        is_inside_any = False
        for hull in p_hulls:
            # find_simplex >= 0 表示点在单纯形内部
            if hull.find_simplex(e_pt) >= 0:
                is_inside_any = True
                break 
        
        if is_inside_any:
            contained_points.append(e_pt)
        else:
            escaped_points.append(e_pt)
            
    escaped_points = np.array(escaped_points)
    contained_points = np.array(contained_points)
    
    total = len(evader_points)
    coverage_ratio = len(contained_points) / total if total > 0 else 0
    is_fully_contained = (len(escaped_points) == 0)
    
    return is_fully_contained, coverage_ratio, escaped_points, contained_points

# ==========================================
# Part 4: 主程序
# ==========================================

if __name__ == "__main__":
    # --- 1. 设置仿真参数 ---
    N_SAMPLES_PER_AGENT = 500  
    TOTAL_TIME_HOURS = 2.5     
    STEPS = 150                
    
    TOTAL_TIME_SEC = TOTAL_TIME_HOURS * 3600
    DT = TOTAL_TIME_SEC / STEPS 
    STEP_DV_LIMIT_KM_S = 1.5e-3 

    print(f"Simulation Configuration:")
    print(f"  Forecast Time: {TOTAL_TIME_HOURS} hours")
    print(f"  Steps: {STEPS}, DT: {DT:.1f} s")
    print("-" * 30)

    # --- 2. 生成初始场景 ---
    agents_data, center_sma = generate_initial_scenario(seed=100)

    # --- 3. 计算所有 MIRD 并存储 (先不画) ---
    print("\nComputing MIRD for all agents...")
    
    pursuer_clouds = {} # 存储追击者点云 {id: points}
    evader_cloud = []   # 存储逃逸者点云
    
    sorted_agent_ids = sorted(agents_data.keys())
    
    for agent_id in sorted_agent_ids:
        agent_info = agents_data[agent_id]
        r0v0 = agent_info['rv']
        print(r0v0)
        total_dv = agent_info['dv']
        agent_type = agent_info['type']
        
        mird_points = []
        for _ in tqdm.tqdm(range(N_SAMPLES_PER_AGENT), desc=f"  Calculating {agent_id}", leave=False):
            final_pos = get_constrained_mird_point(
                initial_state_rv=r0v0,
                total_dv=total_dv,
                steps=STEPS,
                step_limit_dv=STEP_DV_LIMIT_KM_S,
                dt=DT
            )
            mird_points.append(final_pos)
        mird_points = np.array(mird_points)
        
        if agent_type == 'pursuer':
            pursuer_clouds[agent_id] = mird_points
        else:
            evader_cloud = mird_points # 假设只有一个 evader

    # --- 4. 进行包含分析 ---
    print("\nAnalyzing Containment...")
    # 如果 evader_cloud 是 list，转为 numpy
    evader_cloud = np.array(evader_cloud)
    
    is_contained, ratio, escaped_pts, contained_pts = analyze_containment(pursuer_clouds, evader_cloud)
    
    print(f"Containment Result: {'[SUCCESS] Full Capture' if is_contained else '[FAIL] Escape Possible'}")
    print(f"Coverage Ratio: {ratio:.2%}")
    print(f"Escaped Points: {len(escaped_pts)} / {len(evader_cloud)}")

    # --- 5. 绘图 ---
    print("\nPlotting...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    legend_handles = {}

    # A. 画追击者 (蓝色)
    for p_id, points in pursuer_clouds.items():
        # 点云
        scatter_p = ax.scatter(points[:,0], points[:,1], points[:,2], 
                               c='blue', s=5, alpha=0.1, edgecolors='none')
        legend_handles['Pursuer Reachable'] = scatter_p
        
        # (可选) 画凸包线框让体积感更强
        try:
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'b-', lw=0.5, alpha=0.05)
        except: pass

    # B. 画逃逸者 (区分被包围和未被包围)
    # 1. 被包围的点 (红色)
    if len(contained_pts) > 0:
        scatter_e_in = ax.scatter(contained_pts[:,0], contained_pts[:,1], contained_pts[:,2], 
                                  c='red', s=5, alpha=0.3, label='Trapped')
        legend_handles['Evader (Trapped)'] = scatter_e_in
    
    # 2. 逃逸的点 (绿色高亮 'x')
    if len(escaped_pts) > 0:
        scatter_e_out = ax.scatter(escaped_pts[:,0], escaped_pts[:,1], escaped_pts[:,2], 
                                   c="#082C08", s=20, marker='x', alpha=1.0, label='Escaped')
        legend_handles['Evader (Escaped)'] = scatter_e_out

    # C. 画初始位置
    for agent_id, info in agents_data.items():
        pos = info['rv'][:3]
        c = 'k' if info['type'] == 'pursuer' else 'm' # 追击者黑点，逃逸者紫点
        m = '^' if info['type'] == 'pursuer' else '*'
        ax.scatter([pos[0]], [pos[1]], [pos[2]], c=c, marker=m, s=100, zorder=10)

    # --- 6. 图表设置 ---
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title(f'Reachability & Containment Analysis (Tf={TOTAL_TIME_HOURS}h)\n'
                 f'Coverage: {ratio:.1%} | Escaped: {len(escaped_pts)} points')

    # [重要] 设置坐标轴等比例 (Equal Aspect Ratio)
    # 不设置的话，3D凸包看起来会变形，导致误判位置关系
    all_points = np.vstack([p for p in pursuer_clouds.values()] + [evader_cloud])
    x_lim = [np.min(all_points[:,0]), np.max(all_points[:,0])]
    y_lim = [np.min(all_points[:,1]), np.max(all_points[:,1])]
    z_lim = [np.min(all_points[:,2]), np.max(all_points[:,2])]
    
    max_range = np.array([x_lim[1]-x_lim[0], y_lim[1]-y_lim[0], z_lim[1]-z_lim[0]]).max() / 2.0
    mid_x = np.mean(x_lim)
    mid_y = np.mean(y_lim)
    mid_z = np.mean(z_lim)
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend(handles=list(legend_handles.values()), labels=list(legend_handles.keys()), loc='best')
    
    plt.show()