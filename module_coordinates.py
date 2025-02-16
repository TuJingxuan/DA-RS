import numpy as np
import math

def cz(th):
    """
    Rotation around the z-axis
    :param th: rotation angle
    :return: rotation matrix
    """
    C = np.matrix([[math.cos(th), math.sin(th), 0], [-math.sin(th), math.cos(th), 0], [0, 0, 1]])
    return C

def cx(th):
    """
    Rotation around the x-axis
    :param th: rotation angle
    :return: rotation matrix
    """
    C = np.matrix([[1, 0, 0], [0, math.cos(th), math.sin(th)], [0, -math.sin(th), math.cos(th)]])
    return C

def E2C(Ux, a, e, i, omega, w, M):
    """
    Calculate the position and velocity vectors using orbital elements
    :param Ux: Gravitational constant
    :param a: Semi-major axis (kilometers)
    :param e: Orbital eccentricity (non-dimensional)
    :param i: Orbital inclination (radians)
    :param omega: Right ascension of ascending node (radians)
    :param w: Argument of perigee (radians)
    :param M: Mean anomaly (radians)
    :return: Cartesian elements
    """
    E1 = M
    while True:
        E2 = E1 - (E1 - e * math.sin(E1) - M) / (1 - e * math.cos(E1))
        if (abs(E2 - E1) < 1e-7):
            break
        else:
            E1 = E2
    E = E2
    f = 2 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2), math.sqrt(1 - e) * math.cos(E / 2))
    if f < 0:
        f = f + 2 * math.pi

    nr = a * (1 - e**2) / (1 + e * math.cos(f))
    u = w + f
    C = cz(w) * cx(i) * cz(omega)
    P = C.T * np.matrix([[1], [0], [0]])
    Q = C.T * np.matrix([[0], [1], [0]])
    r = nr * (math.cos(f) * P + math.sin(f) * Q)
    tmp = math.sqrt(Ux / a / (1 - e**2))
    v = tmp * (-math.sin(f) * P + (e + math.cos(f)) * Q)
    r = np.array([r[0, 0], r[1, 0], r[2, 0]])
    v = np.array([v[0, 0], v[1, 0], v[2, 0]])
    x_vector = np.array([r[0], r[1], r[2], v[0], v[1], v[2]])
    x = {
        "r": r,
        "v": v,
        "vector": x_vector,
    }
    return x

def C2E(U, x):
    """
    Calculate the orbital elements using the position and velocity vectors
    :param U: Gravitational constant
    :param x: Cartesian elements
    :return: Six orbital elements
    """
    rx = x[0]
    ry = x[1]
    rz = x[2]
    vx = x[3]
    vy = x[4]
    vz = x[5]

    r = np.array([rx, ry, rz])
    v = np.array([vx, vy, vz])
    nr = np.linalg.norm(r)
    nv = np.linalg.norm(v)
    qq = nr * nv**2 / U
    a = nr / (2.0 - qq) # 计算半长轴
    if abs(a) > 1e+12:
        a = -1e+12

    if a > 0:
        # 椭圆形轨道
        esinE = math.sqrt(1 / U / a) * np.dot(r, v)
        ecosE = 1 - nr / a
        e = math.sqrt(esinE**2 + ecosE**2) # Eccentricity
        E = math.atan2(esinE, ecosE)
        if E < 0:
            E = E + 2 * math.pi
        M = E - esinE
        f = 2 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2), math.sqrt(1 - e) * math.cos(E / 2))
        if f < 0:
            f = f + 2 * math.pi

        i = math.acos((r[0] * v[1] - r[1] * v[0]) / math.sqrt(U * a * (1 - e**2))) # calculate inclination
        if i < 1e-8:
            # 赤道轨道
            l = math.atan2(r[1], r[0])
            if l < 0:
                l = l + 2 * math.pi # 求历元真黄经
            if e > 1e-8:
                # 倾角为零但不是圆轨道
                type = 21
                lamda = l - f # 近拱点黄经
                if lamda < 0:
                    lamda = lamda + 2 * math.pi # 求历元真黄经
                output = {
                    "type": type,
                    "a": a,
                    "e": e,
                    "i": i,
                    "lamda": lamda,
                    "M": M,
                }
            else:
                # 赤道圆轨道
                type = 11
                output = {
                    "type": type,
                    "a": a,
                    "e": e,
                    "i": i,
                    "l": l,
                }
        else:
            # 非赤道轨道
            sinOmega = (r[1] * v[2] - r[2] * v[1]) / math.sqrt(U * a * (1 - e**2)) / math.sin(i)
            cosOmega = (r[0] * v[2] - r[2] * v[0]) / math.sqrt(U * a * (1 - e**2)) / math.sin(i)
            omega = math.atan2(r[1] * v[2] - r[2] * v[1], r[0] * v[2] - r[2] * v[0])
            if omega < 0:
                omega = omega + 2 * math.pi # 求升交点经度
            sinU = r[2] / nr / math.sin(i)
            cosU = r[1] / nr * sinOmega + r[0] / nr * cosOmega
            phase = math.atan2(sinU, cosU)
            if phase < 0:
                phase = phase + 2 * math.pi # 求相角
            if e < 1e-8:
                # 圆轨道
                type = 12
                output = {
                    "type": type,
                    "a": a,
                    "e": e,
                    "i": i,
                    "omega": omega,
                    "u": phase,
                }
            else:
                w = phase - f # 计算近拱点角距
                if w < 0:
                    w = w + 2 * math.pi
                type = 22
                output = {
                    "type": type,
                    "a": a,
                    "e": e,
                    "i": i,
                    "omega": omega,
                    "w": w,
                    "f": f,
                    "phase": phase,
                    "M": M,
                    "E": E,
                }
    else:
        h = np.cross(r, v)
        nh = np.linalg.norm(h)
        n = np.cross(np.array([0,0,1]), h)
        ee = ((nv**2 - U / nr) * r - np.dot(r, v) * v) / U
        e = np.linalg.norm(ee)
        i = math.acos(h[2] / nh)
        if np.dot(r, v) > 0:
            f = math.acos(np.dot(ee, r) / e / nr)
        else:
            f = 2 * math.pi - math.acos(np.dot(ee, r) / e / nr)
        if i > 1e-8:
            if n[1] > 0:
                omega = math.acos(n[0] / np.linalg.norm(n))
            else:
                omega = 2 * math.pi - math.acos(n[0] / np.linalg.norm(n))
            if ee[2] > 0:
                w = math.acos(np.dot(n, ee) / np.linalg.norm(n) / e)
            else:
                w = 2 * math.pi - math.acos(np.dot(n, ee) / np.linalg.norm(n) / e)
            if a < -1e12:
                # 抛物线轨道
                D = math.sqrt(a * (1 - e**2)) * math.tan(f) # 抛物线偏近点角
                type = 32
                output = {
                    "type": type,
                    "a": 1e+12,
                    "e": e,
                    "i": i,
                    "omega": omega,
                    "w": w,
                    "D": D,
                }
            else:
                coshF = (e + math.cos(f)) / (1 + e * math.cos(f))
                F = math.log(coshF + math.sqrt(coshF**2 - 1))
                type = 42
                output = {
                    "type": type,
                    "a": a,
                    "e": e,
                    "i": i,
                    "omega": omega,
                    "w": w,
                    "F": F,
                }
        else:
            # 赤道平面内
            l = math.atan2(r[1], r[0])
            if l < 0:
                l = l + 2 * math.pi # 求历元真黄经
            lamda = l - f # 近拱点黄经
            if a < -1e12:
                # 抛物线轨道
                D = math.sqrt(a * (1 - e**2)) * math.tan(f) # 抛物线偏近点角
                type = 31
                output = {
                    "type": type,
                    "a": 1e+12,
                    "e": e,
                    "i": 0,
                    "lamda": lamda,
                    "D": D,
                }
            else:
                coshF = (e + math.cos(f)) / (1 + e * math.cos(f))
                F = math.log(coshF + math.sqrt(coshF**2 - 1))
                type = 41
                output = {
                    "type": type,
                    "a": a,
                    "e": e,
                    "i": 0,
                    "lamda": lamda,
                    "F": F,
                }
    return output