#import daceypy_import_helper  # noqa: F401
from typing import Callable, Type
import numpy as np
import math
from daceypy import DA, RK, array, integrator
from jplephem.spk import SPK  # JPL ephemeris

#import daceypy_import_helper  # noqa: F401
from typing import Callable, Type
import numpy as np
import math
from daceypy import DA, RK, array, integrator
from jplephem.spk import SPK  # JPL ephemeris

def day_JD(year, month, day, hour, minute, second):
    """
    Calculate the Julian date
    :param year: Year
    :param month: Month
    :param day: Day
    :param hour: Hour
    :param minute: Minute
    :param second: Second
    :return: Julian date
    """
    JD = 367 * year - np.floor(1.75 * (year + np.floor((month + 9) / 12))) \
         + np.floor(275 * month / 9) + day + 1721013.5 + hour / 24 \
         + minute / (24 * 60) + second / (24 * 3600)
    return JD

def aJ2_dynamics(r: np.array) -> np.array:
    """
    Calculate the Earth's non-spherical perturbations of J2
    :param r: Position vector of the spacecraft relative the Earth center
    :return: Non-spherical perturbation acceleration
    """
    miuE = 398600.435436096  # gravitational constant of the Earth (obtained from SPICE)
    x = r[0]
    y = r[1]
    z = r[2]
    nr = math.sqrt(x**2 + y**2 + z**2)
    J2 = 1.08262668e-3
    Re = 6378.1366
    aj2x = miuE * x * Re**2 / nr**5 * J2 * (-1.5 + 7.5 * (z / nr)**2)
    aj2y = miuE * y * Re**2 / nr**5 * J2 * (-1.5 + 7.5 * (z / nr)**2)
    aj2z = miuE * z * Re**2 / nr**5 * J2 * (-4.5 + 7.5 * (z / nr)**2)
    aj2 = np.array([aj2x, aj2y, aj2z])
    return aj2

def aJ2(r: array) -> array:
    """
    Calculate the Earth's non-spherical perturbations of J2 (under the framework of DA)
    :param r: Position vector of the spacecraft relative the Earth center
    :return: Non-spherical perturbation acceleration
    """
    miuE = 398600.435436096  # gravitational constant of the Earth (obtained from SPICE)
    x: array = r[0:1]
    y: array = r[1:2]
    z: array = r[2:3]
    pos: array = r[:3]
    r = pos.vnorm()
    J2 = 1.08262668e-3
    Re = 6378.1366
    aj2x = miuE * x * Re**2 / r**5 * J2 * (-1.5 + 7.5 * (z / r)**2)
    aj2y = miuE * y * Re**2 / r**5 * J2 * (-1.5 + 7.5 * (z / r)**2)
    aj2z = miuE * z * Re**2 / r**5 * J2 * (-4.5 + 7.5 * (z / r)**2)
    aj2 = aj2x.concat(aj2y).concat(aj2z)
    return aj2

def aJ2J3J4_dynamics(r):
    """
    Calculate the Earth's non-spherical perturbations of J2, J3, and J4
    :param r: Position vector of the spacecraft relative the Earth center
    :return: Non-spherical perturbation acceleration
    """
    miuE = 398600.435436096  # gravitational constant of the Earth (obtained from SPICE)
    x = r[0]
    y = r[1]
    z = r[2]
    nr = math.sqrt(x**2 + y**2 + z**2)
    J2 = 1.08262668e-3
    J3 = -2.532656485e-6
    J4 = -1.61962159e-6
    Re = 6378.1366
    aj234x = miuE * x * Re**2 / nr**5 * J2 * (-1.5 + 7.5 * (z / nr)**2) \
           + miuE * x * z * Re**3 / nr**7 * J3 * (-7.5 + 17.5 * (z / nr)**2) \
           + (5 / 8) * miuE * x * Re**4 / nr**7 * J4 * (3 - 42 * (z / nr)**2 + 63 * (z / nr)**4)
    aj234y = miuE * y * Re**2 / nr**5 * J2 * (-1.5 + 7.5 * (z / nr)**2) \
           + miuE * y * z * Re**3 / nr**7 * J3 * (-7.5 + 17.5 * (z / nr)**2) \
           + (5 / 8) * miuE * y * Re**4 / nr**7 * J4 * (3 - 42 * (z / nr)**2 + 63 * (z / nr)**4)
    aj234z = miuE * z * Re**2 / nr**5 * J2 * (-4.5 + 7.5 * (z / nr)**2) \
           + miuE * Re**3 / nr**5 * J3 * (1.5 - 15 * (z / nr)**2 + 17.5 * (z / nr)**4) \
           + (5 / 8) * miuE * z * Re**4 / nr**7 * J4 * (15 - 70 * (z / nr)**2 + 63 * (z / nr)**4)
    aj234 = np.array([aj234x, aj234y, aj234z])
    return aj234

def third_body_dynamics(miu: float, r: np.array, rd: np.array) -> np.array:
    """
    Calculate the perturbation acceleration of a third body
    :param miu: Gravitational constant of the third body
    :param r: Position vector of the spacecraft
    :param rd: Position vector of the third body
    :return: Third-body acceleration
    """
    d = rd - r
    D = np.linalg.norm(d)
    Rd = np.linalg.norm(rd)
    perturbation_x = miu * (d[0] / D ** 3 - rd[0] / Rd ** 3)
    perturbation_y = miu * (d[1] / D ** 3 - rd[1] / Rd ** 3)
    perturbation_z = miu * (d[2] / D ** 3 - rd[2] / Rd ** 3)
    perturbation = np.array([perturbation_x, perturbation_y, perturbation_z])
    return perturbation

def third_body(miu: float, r: array, rd: np.array) -> array:
    """
    Calculate the perturbation acceleration of a third body (under the framework of DA)
    :param miu: Gravitational constant of the third body
    :param r: Position vector of the spacecraft
    :param rd: Position vector of the third body
    :return: Third-body acceleration
    """
    d: array = rd - r
    D = d.vnorm()
    Rd = np.linalg.norm(rd)
    px: array = miu * (d[0:1] / D ** 3 - rd[0] / Rd**3)
    py: array = miu * (d[1:2] / D ** 3 - rd[1] / Rd ** 3)
    pz: array = miu * (d[2:3] / D ** 3 - rd[2] / Rd ** 3)
    acc = px.concat(py).concat(pz)
    return acc

def Earth_ephemeris(JD: float):
    """
    Calculate the state of the Earth relative to the Sun
    :param JD: Julian date
    :return: State of the Earth relative to the Sun
    """
    kernel = SPK.open('de432s.bsp')
    pos_EM_Sun, vel_EM_Sun = kernel[0, 3].compute_and_differentiate(JD)
    pos_Earth_EM, vel_Earth_EM = kernel[3, 399].compute_and_differentiate(JD)
    pos_Sun_EM = -pos_EM_Sun
    vel_Sun_EM = -vel_EM_Sun
    pos_Sun_Earth = pos_Sun_EM - pos_Earth_EM
    vel_Sun_Earth = (vel_Sun_EM - vel_Earth_EM) / 86400.0
    pos_Earth_Sun = -pos_Sun_Earth
    vel_Earth_Sun = -vel_Sun_Earth
    x_Earth_Sun = np.array([
        pos_Earth_Sun[0], pos_Earth_Sun[1], pos_Earth_Sun[2], vel_Earth_Sun[0], vel_Earth_Sun[1], vel_Earth_Sun[2],
    ])
    state_Earth_Sun = {
        "pos": pos_Earth_Sun,
        "vel": vel_Earth_Sun,
        "vector": x_Earth_Sun,
    }
    return state_Earth_Sun

def Moon_ephemeris(JD: float):
    """
    Calculate the state of the Moon relative to the Earth
    :param JD: Julian date
    :return: State of the Moon relative to the Earth
    """
    kernel = SPK.open('de432s.bsp')
    pos_Earth_EM, vel_Earth_EM = kernel[3, 399].compute_and_differentiate(JD)
    pos_Moon_EM, vel_Moon_EM = kernel[3, 301].compute_and_differentiate(JD)
    pos_Moon_Earth = pos_Moon_EM - pos_Earth_EM
    vel_Moon_Earth = (vel_Moon_EM - vel_Earth_EM) / 86400.0
    x_Moon_Earth = np.array([
        pos_Moon_Earth[0], pos_Moon_Earth[1], pos_Moon_Earth[2], vel_Moon_Earth[0], vel_Moon_Earth[1], vel_Moon_Earth[2]
    ])
    state_Moon_Earth = {
        "pos": pos_Moon_Earth,
        "vel": vel_Moon_Earth,
        "vector": x_Moon_Earth,
    }
    return state_Moon_Earth

def TBP_dynamics(t: float, x: np.array) -> np.array:
    """
    Two-body dynamics (without time scaling)
    :param t: time epoch [t0, tf]
    :param x: orbital state
    :return: dx: state derivatives
    """
    miuE = 398600.435436096  # gravitational constant of the Earth (obtained from SPICE)
    pos: np.array = x[:3]
    vel: np.array = x[3:]
    r = np.linalg.norm(pos)
    acc: np.array = -miuE * pos / (r ** 3)
    dx = np.concatenate((vel, acc))
    return dx

def TBP(x: array, t: float) -> array:
    """
    Two-body dynamics without time scaling (under the framework of DA)
    :param x: orbital state
    :param t: time epoch [t0, tf]
    :return: dx: state derivatives
    """
    miuE = 398600.435436096  # gravitational constant of the Earth (obtained from SPICE)
    pos: array = x[:3]
    vel: array = x[3:]
    r = pos.vnorm()
    acc: array = -miuE * pos / (r ** 3)
    dx = vel.concat(acc)
    return dx

def TBP_time(x: array, tau: float, t0: DA, tf: DA) -> array:
    """
    Two-body dynamics with time scaling (under the framework of DA)
    :param x: orbital state
    :param tau: time epoch [0, 1]
    :param t0: initial epoch
    :param tf: final epoch
    :return: dx: state derivatives
    """
    # input time tau is normalized. To retrieve t: tau * (tf - t0)
    # RHS of ODE must be multiplied by (tf - t0) to scale
    # t is computed but useless in case of autonomous dynamics
    miuE = 398600.435436096  # gravitational constant of the Earth
    t = tau * (tf - t0)  # epoch t is not employed in the two-body dynamics
    pos: array = x[:3]
    vel: array = x[3:]
    r = pos.vnorm()
    acc: array = -miuE * pos / (r ** 3)
    dx = (tf - t0) * (vel.concat(acc))
    return dx

def TBP_J2_dynamics(t: float, x: np.array) -> np.array:
    """
    Two-body & J2 dynamics (without time scaling)
    :param t: time epoch [t0, tf]
    :param x: orbital state
    :return: dx: state derivatives
    """
    miuE = 398600.435436096  # gravitational constant of the Earth (obtained from SPICE)
    pos: np.array = x[:3]
    vel: np.array = x[3:]
    r = np.linalg.norm(pos)
    acc: np.array = -miuE * pos / (r ** 3) + aJ2_dynamics(pos)
    dx = np.concatenate((vel, acc))
    return dx

def TBP_J2(x: array, t: float) -> array:
    """
    Two-body & J2 dynamics without time scaling (under the framework of DA)
    :param x: orbital state
    :param t: time epoch [t0, tf]
    :return: dx: state derivatives
    """
    miuE = 398600.435436096  # gravitational constant of the Earth (obtained from SPICE)
    pos: array = x[:3]
    vel: array = x[3:]
    r = pos.vnorm()
    acc: array = -miuE * pos / (r ** 3) + aJ2(pos)
    dx = vel.concat(acc)
    return dx

def TBP_J2_time(x: array, tau: float, t0: DA, tf: DA) -> array:
    """
    Two-body & J2 dynamics with time scaling (under the framework of DA)
    :param x: orbital state
    :param tau: time epoch [0, 1]
    :param t0: initial epoch
    :param tf: final epoch
    :return: dx: state derivatives
    """
    # input time tau is normalized. To retrieve t: tau * (tf - t0)
    # RHS of ODE must be multiplied by (tf - t0) to scale
    # t is computed but useless in case of autonomous dynamics
    miuE = 398600.435436096  # gravitational constant of the Earth
    t = tau * (tf - t0)  # epoch t is not employed in the two-body dynamics
    pos: array = x[:3]
    vel: array = x[3:]
    r = pos.vnorm()
    acc: array = -miuE * pos / (r ** 3) + aJ2(pos)
    dx = (tf - t0) * (vel.concat(acc))
    return dx

def TBP_J2_SM_dynamics(t: float, x: np.array, JD0: float) -> np.array:
    """
    Two-body & J2 & Sun & Moon dynamics (without time scaling)
    :param t: time epoch [t0, tf]
    :param x: orbital state
    :param JD0: Initial Julian date
    :return: dx: state derivatives
    """
    """Parameters"""
    miuE = 398600.435436096  # gravitational constant of the Earth (all obtained from SPICE)
    miuS = 132712440041.939  # gravitational constant of the Sun
    miuM = 4902.80006616380  # gravitational constant of the Moon
    JD = JD0 + t / 86400.0
    """Third-body perturbations"""
    pos: np.array = x[:3]
    vel: np.array = x[3:]
    rS = -Earth_ephemeris(JD)["pos"]
    rM = Moon_ephemeris(JD)["pos"]
    accS = third_body_dynamics(miuS, pos, rS)
    accM = third_body_dynamics(miuM, pos, rM)
    r = np.linalg.norm(pos)
    acc: np.array = -miuE * pos / (r ** 3) + aJ2_dynamics(pos) + accS + accM
    dx = np.concatenate((vel, acc))
    return dx

def TBP_J2_SM(t: float, x: array, JD0: float) -> array:
    """
    Two-body & J2 & Sun & Moon dynamics without time scaling (under the framework of DA)
    :param t: time epoch [t0, tf]
    :param x: orbital state
    :param JD0: Initial Julian date
    :return: dx: state derivatives
    """
    """Parameters"""
    miuE = 398600.435436096  # gravitational constant of the Earth (all obtained from SPICE)
    miuS = 132712440041.939  # gravitational constant of the Sun
    miuM = 4902.80006616380  # gravitational constant of the Moon
    JD: float = JD0 + t / 86400.0
    """Third-body perturbations"""
    pos: array = x[:3]
    vel: array = x[3:]
    rS = -Earth_ephemeris(JD)["pos"]
    rM = Moon_ephemeris(JD)["pos"]
    accS = third_body(miuS, pos, rS)  # DA
    accM = third_body(miuM, pos, rM)  # DA
    r = pos.vnorm()
    acc: array = -miuE * pos / (r ** 3) + aJ2(pos) + accS + accM
    dx = vel.concat(acc)
    return dx

def TBP_J2_SM_time(x: array, tau: float, t0: DA, tf: DA, JD0: float) -> array:
    """
    Two-body & J2 & Sun & Moon dynamics with time scaling (under the framework of DA)
    :param x: orbital state
    :param tau: time epoch [0, 1]
    :param t0: initial epoch
    :param tf: final epoch
    :param JD0: Initial Julian date
    :return: dx: state derivatives
    """
    # input time tau is normalized. To retrieve t: tau * (tf - t0)
    # RHS of ODE must be multiplied by (tf - t0) to scale
    # t is computed but useless in case of autonomous dynamics
    """Parameters"""
    miuE = 398600.435436096  # gravitational constant of the Earth (all obtained from SPICE)
    miuS = 132712440041.939  # gravitational constant of the Sun
    miuM = 4902.80006616380  # gravitational constant of the Moon
    t = t0 + (tf - t0) * tau
    JD: np.array = JD0 + t.cons() / 86400.0
    JD: float = JD[0]
    """Third-body perturbations"""
    pos: array = x[:3]
    vel: array = x[3:]
    rS = -Earth_ephemeris(JD)["pos"]
    rM = Moon_ephemeris(JD)["pos"]
    accS = third_body(miuS, pos, rS)  # DA
    accM = third_body(miuM, pos, rM)  # DA
    r = pos.vnorm()
    acc: array = -miuE * pos / (r ** 3) + aJ2(pos) + accS + accM
    dx = (tf - t0) * (vel.concat(acc))
    return dx

def CRTBP_dynamics(t: float, y: np.array, mu: float) -> np.array:
    """
    CRTBP dynamics (without time scaling)
    :param t: time epoch [t0, tf]
    :param y: orbital state
    :param mu: non-dimensional gravitational constant
    :return: state derivatives
    """
    r1 = math.sqrt((mu + y[0]) ** 2 + (y[1]) ** 2 + (y[2]) ** 2)
    r2 = math.sqrt((1 - mu - y[0]) ** 2 + (y[1]) ** 2 + (y[2]) ** 2)
    m1 = 1 - mu
    m2 = mu
    dydt = np.array([
        y[3],
        y[4],
        y[5],
        y[0] + 2 * y[4] + m1 * (-mu - y[0]) / (r1 ** 3) + m2 * (1 - mu - y[0]) / (r2 ** 3),
        y[1] - 2 * y[3] - m1 * (y[1]) / (r1 ** 3) - m2 * y[1] / (r2 ** 3),
        -m1 * y[2] / (r1 ** 3) - m2 * y[2] / (r2 ** 3)
    ])
    return dydt

def CRTBP(y: array, t: float, mu: float) -> np.array:
    """
    CRTBP dynamics (under the framework of DA)
    :param y: orbital state
    :param t: time epoch [t0, tf]
    :param mu: non-dimensional gravitational constant
    :return: state derivatives
    """
    r1 = ((mu + y[0]) ** 2 + (y[1]) ** 2 + (y[2]) ** 2).sqrt()
    r2 = ((1 - mu - y[0]) ** 2 + (y[1]) ** 2 + (y[2]) ** 2).sqrt()
    m1 = 1 - mu
    m2 = mu
    pos: array = y[:3]
    vel: array = y[3:]
    px: array = y[0] + 2 * y[4] + m1 * (-mu - y[0]) / (r1 ** 3) + m2 * (1 - mu - y[0]) / (r2 ** 3)
    py: array = y[1] - 2 * y[3] - m1 * (y[1]) / (r1 ** 3) - m2 * y[1] / (r2 ** 3)
    pz: array = -m1 * y[2] / (r1 ** 3) - m2 * y[2] / (r2 ** 3)
    acc = array.zeros(3)
    acc[0] = px
    acc[1] = py
    acc[2] = pz
    dydt = vel.concat(acc)
    return dydt

def CRTBP_time(y: array, t: float, mu: float, t0: DA, tf: DA) -> np.array:
    """
    CRTBP dynamics (under the framework of DA)
    :param y: orbital state
    :param t: time epoch [t0, tf]
    :param mu: non-dimensional gravitational constant
    :param t0: initial epoch
    :param tf: final epoch
    :return: state derivatives
    """
    r1 = ((mu + y[0]) ** 2 + (y[1]) ** 2 + (y[2]) ** 2).sqrt()
    r2 = ((1 - mu - y[0]) ** 2 + (y[1]) ** 2 + (y[2]) ** 2).sqrt()
    m1 = 1 - mu
    m2 = mu
    pos: array = y[:3]
    vel: array = y[3:]
    px: array = y[0] + 2 * y[4] + m1 * (-mu - y[0]) / (r1 ** 3) + m2 * (1 - mu - y[0]) / (r2 ** 3)
    py: array = y[1] - 2 * y[3] - m1 * (y[1]) / (r1 ** 3) - m2 * y[1] / (r2 ** 3)
    pz: array = -m1 * y[2] / (r1 ** 3) - m2 * y[2] / (r2 ** 3)
    acc = array.zeros(3)
    acc[0] = px
    acc[1] = py
    acc[2] = pz
    dydt = (tf - t0) * vel.concat(acc)
    return dydt