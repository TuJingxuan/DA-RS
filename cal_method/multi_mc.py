from orbitx import coe2rv, rv_from_r0v0
import numpy as np
import random
import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 核心参数设置 ---
COE = [41266, 0.0, 0, 0, 0, 0]  # 初始轨道 a=10000km, e=0.2
TOTAL_DV = 0.3       # 总能力: 0.3 km/s (300 m/s)
STEP_LIMIT = 0.001  # 单步限制: 1.5 m/s = 0.0015 km/s
TOTAL_TIME = 10 * 3600  # 总时间 10小时
STEPS = 600          # 步数 (每分钟一次)
DT = 60              # 步长 (秒)

r0v0 = coe2rv(COE)


# --- 算法 1: 约束型随机分配 (削峰填谷) ---
def constrained_dv_distribution(total_dv, steps, max_step_dv):
    # 1. 初始随机分配
    factors = np.random.random(steps)
    dvs = (factors / np.sum(factors)) * total_dv
    
    # 2. 迭代修正
    for _ in range(50): # 迭代几次以确保收敛
        mask_exceed = dvs > max_step_dv
        if not np.any(mask_exceed):
            break
        print(1)
        # 计算溢出总量
        excess = np.sum(dvs[mask_exceed] - max_step_dv)
        
        # 削峰
        dvs[mask_exceed] = max_step_dv
        
        # 填谷 (分配给未超标的部分)
        mask_valid = ~mask_exceed
        if np.sum(mask_valid) > 0:
            factors_valid = np.random.random(np.sum(mask_valid))
            dvs[mask_valid] += (factors_valid / np.sum(factors_valid)) * excess
            
    return dvs

# --- 辅助函数：随机方向 ---
def get_random_directions(n):
    vecs = np.random.normal(size=(n, 3))
    norms = np.linalg.norm(vecs, axis=1)[:, np.newaxis]
    return vecs / norms

# --- MIRD 计算 (受限多脉冲) ---
def get_constrained_mird_point(initial_state,total_dv=TOTAL_DV, steps=STEPS, step_limit=STEP_LIMIT):
    # 1. 获取满足 1.5m/s 限制的速度序列
    try:
        dv_mags = constrained_dv_distribution(total_dv, steps, step_limit)
    except Exception:
        return initial_state[:3] # 异常回退
    
    # 2. 随机方向
    dv_dirs = get_random_directions(steps)
    dvs = dv_dirs * dv_mags[:, np.newaxis]
    
    # 3. 动力学递推
    curr_rv = initial_state.copy()
    for i in range(steps):
        curr_rv[3:6] += dvs[i]
        curr_rv = rv_from_r0v0(curr_rv, DT)
        
    return curr_rv[:3]


# --- 绘图辅助：标称轨道 ---
def get_nominal_orbit(coe, points=200):
    a, e = coe[0], coe[1]
    p = a * (1 - e**2)
    thetas = np.linspace(0, 2*np.pi, points)
    rs = p / (1 + e * np.cos(thetas))
    xs = rs * np.cos(thetas)
    ys = rs * np.sin(thetas)
    zs = np.zeros_like(xs)
    return xs, ys, zs

# --- 主程序 ---
if __name__ == "__main__":
    N_SAMPLES = 1000 # 样本数
    
    print(f"Running Simulation ({N_SAMPLES} samples)...")
    
    # 1. 计算受限 MIRD1
    total_dv1=300e-3  # 300 m/s
    steps1=600
    step_limit1=1.5e-3  # 1.5 m/s
    r0v01=coe2rv([41266, 0.0, 0, 0, 0, 0])
    mird_points1 = []
    print("Computing Constrained Multi-Impulse1...")
    for _ in tqdm.tqdm(range(N_SAMPLES)):
        mird_points1.append(get_constrained_mird_point(r0v01, total_dv=total_dv1, steps=steps1, step_limit=step_limit1))
    mird_points1 = np.array(mird_points1)
    
    # 2. 计算受限 MIRD2
    total_dv2=100e-3  # 300 m/s
    steps2=600
    step_limit2=1e-3  # 1.5 m/s
    r0v02=coe2rv([41266, 0.0, 0, 0, 0, 0])
    mird_points2 = []
    print("Computing Constrained Multi-Impulse2...")
    for _ in tqdm.tqdm(range(N_SAMPLES)):
        mird_points2.append(get_constrained_mird_point(r0v02, total_dv=total_dv2, steps=steps2, step_limit=step_limit2))
    mird_points2 = np.array(mird_points2)
    # 3. 绘图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    
    # 画多脉冲 (蓝色核心)
    ax.scatter(mird_points1[:,0], mird_points1[:,1], mird_points1[:,2], 
               c='r', s=2, alpha=0.8, label=f'MIRD (Step <= {STEP_LIMIT*1000} m/s)')
    ax.scatter(mird_points2[:,0], mird_points2[:,1], mird_points2[:,2], 
               c='b', s=2, alpha=0.8, label=f'MIRD (Step <= {STEP_LIMIT*1000} m/s)')
    # 画标称轨道
    nom_x, nom_y, nom_z = get_nominal_orbit(COE)
    ax.plot(nom_x, nom_y, nom_z, 'k--', lw=2, label='Nominal Orbit')
    ax.scatter([r0v01[0]], [r0v01[1]], [r0v01[2]], c='r', marker='x', s=50, label='Start1')
    ax.scatter([r0v02[0]], [r0v02[1]], [r0v02[2]], c='b', marker='x', s=50, label='Start1')
    
    
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.legend()
    
    plt.show()