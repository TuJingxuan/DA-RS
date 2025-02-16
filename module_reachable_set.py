import daceypy_import_helper  # noqa: F401
from typing import Callable, Type
import numpy as np
import math
import scipy
import random
import alphashape
import os
from multiprocessing import Pool
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from daceypy import DA, RK, array, integrator, ADS
from module_integrator import RK78, base_propagation, advanced_propagation
from module_orbit_dynamics import TBP_dynamics, TBP_J2_dynamics, CRTBP_dynamics
from module_plot import plot_envelope_equation, plot_envelope_points
from module_envelope_equation_map import envelope_equation_partial_map_inversion
import time
import warnings
warnings.filterwarnings("ignore")

miuE = 398600.435436096
Re = 6378.1366
RelTol = 10 ** -12
AbsTol = 10 ** -12

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def Schmidt_orthogonalization(a):
    """Schmidt orthogonalization"""
    m, n = len(a), len(a[0])
    if m < n:
        a = a.T
        m, n = len(a), len(a[0])
    b = np.zeros([m, n])
    b[:, 0] = a[:, 0]
    for ik in range(n - 1):
        i = ik + 1
        for j in range(i):
            b[:, i] = b[:, i] - np.dot(a[:, i], b[:, j]) / np.dot(b[:, j], b[:, j]) * b[:, j]
        b[:, i] = b[:, i] + a[:, i]
    for k in range(n):
        b[:, k] = b[:, k] / np.linalg.norm(b[:, k])
    return b

def poincare_plane_direction(
        x: np.array,
) -> np.array:
    """
    Compute the directions of a Poincare plane
    :param x: orbital state
    :return: unit directions
    """
    direction = np.zeros([3, 3])
    r = x[:3]
    v = x[3:]
    h = np.cross(r, v)
    d1 = v / np.linalg.norm(v)  # direction along the velocity vector
    d2 = h / np.linalg.norm(h)  # direction along the angular moment
    d3 = np.cross(d2, v) / np.linalg.norm(np.cross(d2, v))
    direction[0] = d1
    direction[1] = d3
    direction[2] = d2
    return direction.T

def generate_perimeter(
        Ns: int,
) -> np.array:
    """Generate the grid points for initial domains"""
    xgrid = np.linspace(-1, 1, Ns)
    ygrid = np.linspace(-1, 1, Ns)
    lb = np.ones((Ns, 2))
    xb = 1
    yb = 1
    lb[:, 0] = lb[:, 0] * (-xb)
    lb[:, 1] = yb * ygrid
    rb = np.ones((Ns, 2))
    rb[:, 0] = rb[:, 0] * xb
    rb[:, 1] = -yb * ygrid
    bb = np.ones((Ns, 2))
    bb[:, 0] = -xb * xgrid
    bb[:, 1] = bb[:, 1] * -yb
    tb = np.ones((Ns, 2))
    tb[:, 0] = xb * xgrid
    tb[:, 1] = tb[:, 1] * yb
    return tb, rb, bb, lb

def reachable_set_propagation(
        domain_0: ADS,
        x0: np.array,
        xf: np.array,
        dv: float,
        t0: array,
        tf: array,
        f: Callable[[array, float, array, array], array]
) -> ADS:
    """
    Propagate the reachable sets using adaptive domain split (ADS)
    :param domain_0: initial domain
    :param x0: initial state
    :param dv: maximal velocity increment
    :param t0: initial epoch
    :param tf: final epoch
    :param f: dynamics
    :return: final ADS domain
    """
    """Define two angles"""
    alpha = domain_0.manifold[0]
    beta = domain_0.manifold[1]
    """Define initial state"""
    x0_ = array.zeros(6) + x0
    dvx = dv * alpha.cos() * beta.cos()
    dvy = dv * alpha.cos() * beta.sin()
    dvz = dv * alpha.sin()
    x0_[3] += dvx  # add maneuver
    x0_[4] += dvy
    x0_[5] += dvz
    """Propagate the nominal orbit"""
    RHS = lambda x, tau: f(x, tau, t0, tf + domain_0.manifold[2])
    cart = RK78(x0_, 0.0, 1.0, RHS)[:3]  # only consider the position reachable set
    """Perform partial map inversion"""
    T = poincare_plane_direction(xf)
    DirMap = array([
        T[0, 0] * cart[0] + T[1, 0] * cart[1] + T[2, 0] * cart[2],
        T[0, 1] * cart[0] + T[1, 1] * cart[1] + T[2, 1] * cart[2],
        T[0, 2] * cart[0] + T[1, 2] * cart[1] + T[2, 2] * cart[2],
    ])
    yp = T[0, 0] * xf[0] + T[1, 0] * xf[1] + T[2, 0] * xf[2]
    DirMap_xp = DirMap[0] - yp  # fix the y-axis coordinate
    alpha_ = DA(1)
    beta_ = DA(2)
    AugMap = array([
        alpha_, beta_, DirMap_xp,
    ])
    ParInvMap = AugMap.invert()
    ParInvMap = ParInvMap.plug(3, 0)
    ParMap = DirMap.eval(ParInvMap)
    PoincareMap = array([
        ParMap[1] + 1e-16 * DA(3).sin(),
        ParMap[2] + 1e-16 * DA(3).sin(),
    ])
    """Perform adaptive domain split"""
    return ADS(domain_0.box, domain_0.nsplit, PoincareMap)

def reachable_set_map(
        x0: np.array,
        t0: float,
        tf: float,
        xf: np.array,
        dv: float,
        f: Callable[[array, float, array, array], array],
        toll: float,
) -> list:
    """Calculate the reachable map using the adaptive domain split"""
    """Initialize DA"""
    DA.init(6, 3)
    DA.setEps(1e-32)
    """Define ADS"""
    init_array = array([
        math.pi / 2 * DA(1), math.pi * DA(2), DA(3),
    ])
    init_domain = ADS(init_array, [])
    init_list = [init_domain]
    final_lists = []
    final_list = init_list.copy()
    final_lists.append(final_list)  # add also initial domains
    Nmax = 10
    """Begin ADS"""
    start = time.time()
    with DA.cache_manager():  # optional, for efficiency
        final_list = ADS.eval(
            final_list, toll, Nmax,
            lambda domain: reachable_set_propagation(domain, x0, xf, dv, array([t0]), array([tf]), f))
        final_lists.append(final_list)
    time_cost = time.time() - start
    print("Computational cost: %.4f" % time_cost)
    """Return results"""
    return final_lists

def single_reachable_set_initial_box(
        domain: ADS,
        Ns: int,
) -> np.array:
    """Calculate the initial boxes"""
    """Generate the grid points"""
    tb, rb, bb, lb = generate_perimeter(Ns)
    """Propagate the boundary points"""
    Map = domain.manifold
    Box = domain.box
    perimeter_guess = np.concatenate((tb, rb, bb, lb))  # [-1, 1]
    final_manifold = np.zeros((2, perimeter_guess.shape[0]))
    initial_box = np.zeros((2, perimeter_guess.shape[0]))
    for i in range(perimeter_guess.shape[0]):
        final_manifold[:, i] = Map.eval([perimeter_guess[i, 0], perimeter_guess[i, 1]])
        initial_box[:, i] = Box.eval([perimeter_guess[i, 0], perimeter_guess[i, 1]])[:2]  # only alpha and beta
    """Return data"""
    return initial_box, final_manifold, perimeter_guess.T

def single_reachable_set_map_envelope_solve(
        domain: ADS,
        Ns: int,
) -> np.array:
    """Determine the envelope of a single domain by solving the envelope equation"""
    start = time.time()
    """Generate the grid points"""
    tb, rb, bb, lb = generate_perimeter(Ns)
    perimeter_solved = np.concatenate((tb, rb, bb, lb))  # [-1, 1]
    """Solve the conditions of envelope"""
    Map = domain.manifold
    """Determine the envelope equation map"""
    derivatives = array([
        Map[0].deriv(1), Map[1].deriv(1), Map[0].deriv(2), Map[1].deriv(2),
    ])
    envelope_equation = derivatives[0] * derivatives[3] - derivatives[1] * derivatives[2]
    envelope_bound = envelope_equation.bound()
    """Determine whether exists a solution"""
    if envelope_bound[0] <= 0 <= envelope_bound[1]:
        if_sol = 1
        """Determine the envelope by solving the envelope equation"""
        for i in range(len(perimeter_solved)):
            root = solve_envelope_equation(envelope_equation, perimeter_solved[i])
            if (root >= 0) and (root <= 1):
                perimeter_solved[i] *= root
    else:
        if_sol = 0  # doesn't contain a solution
    """Collect all four bounds"""
    final_envelope_solved = np.zeros((2, perimeter_solved.shape[0]))
    for i in range(perimeter_solved.shape[0]):
        final_envelope_solved[:, i] = Map.eval([
            perimeter_solved[i, 0], perimeter_solved[i, 1], 0,
        ])
    time_cost = time.time() - start
    """Return data"""
    return final_envelope_solved, perimeter_solved.T, if_sol, time_cost

def single_reachable_set_map_envelope_analytical(
        domain: ADS,
        Ns: int,
        N_anchor_point: int,
) -> np.array:
    """Determine the envelope of a single domain using a analytical way (based on DA series)"""
    start = time.time()
    """Generate the grid points"""
    tb, rb, bb, lb = generate_perimeter(Ns)
    perimeter_analytical = np.concatenate((tb, rb, bb, lb))  # [-1, 1]
    tb, rb, bb, lb = generate_perimeter(N_anchor_point)
    perimeter_anchor = np.concatenate((tb, rb, bb, lb))  # [-1, 1]
    perimeter_anchor = np.unique(perimeter_anchor, axis=0)
    """Solve the conditions of envelope"""
    Map = domain.manifold
    """Determine the envelope equation map"""
    derivatives = array([
        Map[0].deriv(1), Map[1].deriv(1), Map[0].deriv(2), Map[1].deriv(2),
    ])
    envelope_equation = derivatives[0] * derivatives[3] - derivatives[1] * derivatives[2]
    envelope_bound = envelope_equation.bound()
    """Determine whether exists a solution"""
    if envelope_bound[0] <= 0 <= envelope_bound[1]:
        if_sol = 1
        """Then determine the envelope using a analytical way (based on the DA series)"""
        envelope_map = []
        points = np.zeros([len(perimeter_anchor), 2])
        if_envelope_map = np.zeros([len(perimeter_anchor)])
        """First solve the envelope equation for anchor point"""
        for i in range(len(perimeter_anchor)):
            point = np.array([
                math.atan2(
                    perimeter_anchor[i, 1],
                    perimeter_anchor[i, 0]
                ),
                np.linalg.norm(perimeter_anchor[i]),
            ])  # anchor point
            root = solve_envelope_equation(envelope_equation, perimeter_anchor[i])
            if (root >= 0) and (root <= 1):
                perimeter_anchor[i] *= root
                point[1] *= root
                if_envelope_map[i] = 1
                envelope_map.append(
                    envelope_equation_partial_map_inversion(
                        envelope_equation=envelope_equation,
                        point=point,
                    )
                )
                points[i] = point
            else:
                envelope_map.append(DA(1))  # append a useless thing
                points[i] = point
        """Then predict the envelope points using the DA map"""
        for i in range(len(perimeter_analytical)):
            theta = math.atan2(perimeter_analytical[i, 1], perimeter_analytical[i, 0])
            r = np.linalg.norm(perimeter_analytical[i])
            "find the nearest anchor point"
            flag = abs(theta - points[:, 0])  # find the nearest anchor point
            for k in range(len(perimeter_anchor)):
                if flag[k] >= 2 * math.pi:
                    flag[k] -= (2 * math.pi)
            # id = np.where(flag == flag.min())[0][0]
            point_id = np.argsort(flag)
            if if_envelope_map[point_id[0]] == 1:
                point_id = point_id[0]
            else:
                if if_envelope_map[point_id[1]] == 1:
                    point_id = point_id[1]
                else:
                    point_id = point_id[0]
            if if_envelope_map[point_id] == 1:
                point = points[point_id]
                theta -= point[0]
                if theta > math.pi:
                    theta -= 2 * math.pi
                if theta < -math.pi:
                    theta += 2 * math.pi
                root = envelope_map[point_id].eval(
                    np.array([
                        theta, 0, 0,
                    ])
                ) + point[1]
                if (root > 0) and (root <= r):
                    perimeter_analytical[i] *= (root / r)
    else:
        if_sol = 0  # doesn't contain a solution
    """Collect all four bounds"""
    final_envelope_analytical = np.zeros((2, perimeter_analytical.shape[0]))
    for i in range(perimeter_analytical.shape[0]):
        final_envelope_analytical[:, i] = Map.eval([
            perimeter_analytical[i, 0], perimeter_analytical[i, 1], 0,
        ])
    time_cost = time.time() - start
    """Return data"""
    return final_envelope_analytical, perimeter_analytical.T, perimeter_anchor.T, if_sol, time_cost

def reachable_set_map_envelope(
        final_list: list,
        Ns: int,
        N_anchor_point: int,
        if_solved: bool,
        if_only_output: bool,
) -> np.array:
    """Determine the envelope of the angle-only measurement"""
    """Generate data for recording"""
    initial_box = np.zeros([len(final_list), 2, (Ns * 4)])
    final_manifold = np.zeros([len(final_list), 2, (Ns * 4)])
    final_envelope = np.zeros([len(final_list), 4, (Ns * 4)])
    final_perimeter = np.zeros([len(final_list), 6, (Ns * 4)])
    final_perimeter_anchor = np.zeros([len(final_list), 2, (N_anchor_point * 4 - 4)])
    if_contain_sol = np.zeros([len(final_list)])
    time_cost = np.zeros([len(final_list), 2])
    """Generate the envelopes"""
    for k in range(len(final_list)):  # for each sub-domain
        box, manifold, perimeter_guess = single_reachable_set_initial_box(final_list[k], Ns)
        final_envelope_solved, perimeter_solved, if_sol, time_cost_solve = single_reachable_set_map_envelope_solve(
            domain=final_list[k],
            Ns=Ns,
        )
        final_envelope_analytical, perimeter_analytical, perimeter_anchor, _, time_cost_analytical = single_reachable_set_map_envelope_analytical(
            domain=final_list[k],
            Ns=Ns,
            N_anchor_point=N_anchor_point,
        )
        initial_box[k] = box
        final_manifold[k] = manifold
        final_envelope[k, :2] = final_envelope_solved
        final_envelope[k, 2:] = final_envelope_analytical
        final_perimeter[k, :2] = perimeter_guess
        final_perimeter[k, 2:4] = perimeter_solved
        final_perimeter[k, 4:] = perimeter_analytical
        final_perimeter_anchor[k] = perimeter_anchor
        if_contain_sol[k] = if_sol
        time_cost[k] = np.array([
            time_cost_solve, time_cost_analytical,
        ])
    """Return data"""
    if if_solved:
        envelope_points = final_envelope[:, :2, :]
    else:
        envelope_points = final_envelope[:, 2:, :]
    if if_only_output:
        """If only output efficient solution (DA-based envelope)"""
        return initial_box, final_manifold, final_envelope[:, 2:, :]
    else:
        return initial_box, final_manifold, final_envelope, final_perimeter, final_perimeter_anchor, envelope_points, if_contain_sol, time_cost

def combine_envelope(
        envelope_points: np.array,
) -> np.array:
    """Combine the envelopes of all sub-domains"""
    """Normalize the envelope points"""
    lb = np.min(envelope_points, axis=0)
    ub = np.max(envelope_points, axis=0)
    envelope_points = (envelope_points - lb) / (ub - lb)
    alpha_r = 0.01
    """Determine the envelope"""
    alpha_shape = alphashape.alphashape(envelope_points, alpha=alpha_r)
    envelope_points = np.array(alpha_shape.exterior.coords)
    """Map the envelope points"""
    envelope_points = envelope_points * (ub - lb) + lb
    """Return results"""
    return envelope_points

def envelope_equation_func(
        x: float,
        envelope_equation: array,
        point: np.array,
):
    """
    Solve the envelope equation
    :param x: search step
    :param envelope_equation: envelope equation map
    :param point: reference point
    :return: value of the envelope equation
    """
    fval = envelope_equation.eval(
        x * np.array([
            point[0], point[1], 0,
        ])
    )
    return fval

def solve_envelope_equation(
        envelope_equation: array,
        point: np.array,
) -> float:
    """
    Solve the envelope equation
    :param envelope_equation: envelope equation map
    :param point: reference point
    :return: solution
    """
    """Define initial guess"""
    initial_guess = np.array([0.0, 0.5, 1.0])
    root_data = np.array([-1.0, -1.0, -1.0])  # type is float (or all elements become zero!)
    """Traverse all initial guess"""
    for k in range(len(initial_guess)):
        func = lambda x: envelope_equation_func(x, envelope_equation, point)
        root, _, ier, _ = fsolve(func, x0=initial_guess[k], full_output=True)
        if ier == 1:  # find a solution
            if (root > 0) and (root <= 1):
                root_data[k] = root
                break
    """Return data"""
    return float(np.max(root_data))


def event_y(
        t: float,
        X: np.array,
        T: np.array,
        yp0: float,
) -> np.array:
    yp = T[0, 0] * X[0] + T[1, 0] * X[1] + T[2, 0] * X[2]
    y = yp - yp0
    return y

def generate_monte_carlo_points(
        x0: np.array,
        t0: float,
        tf: float,
        xf: np.array,
        N: int,
        dv: float,
        final_list: list,
        f: Callable[[float, np.array], np.array],
) -> np.array:
    """Generate Monte Carlo points"""
    setup_seed(0)
    DIM = 2
    points = np.zeros([2, len(final_list), N, DIM])
    points_all = list()
    """Traverse all domain"""
    for k in range(len(final_list)):
        print("======== Domain: %d ========" % k)
        Box = final_list[k].box
        Map = final_list[k].manifold
        """For each domain generate N points"""
        for i in range(N):
            print("======== Point: %d-%d ========" % (k, i))
            input = np.array([
                np.random.uniform(low=-1.0, high=1.0, size=None),
                np.random.uniform(low=-1.0, high=1.0, size=None),
            ])
            angle = Box.eval(input)
            dx0 = np.array([
                0,
                0,
                0,
                dv * math.cos(angle[0]) * math.cos(angle[1]),
                dv * math.cos(angle[0]) * math.sin(angle[1]),
                dv * math.sin(angle[0]),
            ])
            """Propagate the neighbouring orbit"""
            T = poincare_plane_direction(xf)
            yp0 = T[0, 0] * xf[0] + T[1, 0] * xf[1] + T[2, 0] * xf[2]
            event_func = lambda t, x: event_y(t, x, T, yp0)
            sol = solve_ivp(f, [t0, 1.5 * tf], x0 + dx0, args=(), method='RK45',
                            t_eval=None, max_step=np.inf, events=[event_func], rtol=RelTol, atol=AbsTol)
            xp = sol.y_events[0]
            tp = sol.t_events[0]
            dtp = abs(tp - tf)
            id = np.where(dtp == dtp.min())[0][0]
            xp = xp[id]
            points[0, k, i, 0] = T[0, 1] * xp[0] + T[1, 1] * xp[1] + T[2, 1] * xp[2]  # xp
            points[0, k, i, 1] = T[0, 2] * xp[0] + T[1, 2] * xp[1] + T[2, 2] * xp[2]  # zp
            """Predict the orbit using the ADS"""
            xp_pred = Map.eval(input)
            points[1, k, i, 0] = xp_pred[0]  # xp
            points[1, k, i, 1] = xp_pred[1]  # zp
            """Re-propagate the orbits"""
            t_eval = np.linspace(t0, tp[id], 101)
            sol = solve_ivp(f, [t0, tp[id]], x0 + dx0, args=(), method='RK45',
                            t_eval=t_eval, max_step=np.inf, rtol=RelTol, atol=AbsTol)
            points_all.append(sol.y.T)
    """Return results"""
    points_all = np.array(points_all)
    return points, points_all

def generate_monte_carlo_points_all(
        x0: np.array,
        t0: float,
        tf: float,
        N: int,
        dv: float,
        dynamics: str,
        Nt: int,
        t_series: np.array,
        y_series: np.array,
) -> np.array:
    """Generate Monte Carlo points"""
    setup_seed(0)
    DIM = 6
    points = np.zeros([N, Nt + 1, DIM])
    points_series = np.zeros([len(t_series), N, 2])
    """Generate random data for the MC simulation"""
    rand_data = np.zeros([N, 2])
    for k in range(N):
        rand_data[k] = np.array([
            np.random.uniform(low=-1.0, high=1.0, size=None),
            np.random.uniform(low=-1.0, high=1.0, size=None),
        ])
    """MC simulation"""
    pool = Pool(processes=os.cpu_count() // 3 * 2)
    result = pool.map(
        generate_monte_carlo_points_all_func,
        [_build_input_par(
            i, rand_data, x0, t0, tf, dv, Nt, t_series, y_series, dynamics
        ) for i in range(N)]
    )
    pool.close()
    for i in range(len(result)):
        p, ps = result[i]
        points[i] = p
        for k in range(len(t_series)):
            points_series[k, i, 0] = ps[k, 0]
            points_series[k, i, 1] = ps[k, 1]
    pool.join()
    """Return data"""
    return points, points_series, rand_data

def _build_input_par(
        i: int,
        rand_data: np.array,
        x0: np.array,
        t0: float,
        tf: float,
        dv: float,
        Nt: int,
        t_series: np.array,
        y_series: np.array,
        dynamics: str,
):
    """Generate data for multi-processing calculation"""
    rd = rand_data[i]
    return i, rd, x0, t0, tf, dv, Nt, t_series, y_series, dynamics

def generate_monte_carlo_points_all_func(d):
    """Function for propagate the orbits in MC simulation"""
    i, rd, x0, t0, tf, dv, Nt, t_series, y_series, dynamics = d
    print("======== Point: %d ========" % i)
    point_series = np.zeros([len(t_series), 2])
    """Define dynamics"""
    if dynamics == "TBP_J2":
        f = lambda t, x: TBP_J2_dynamics(t, x)
    elif dynamics == "CRTBP":
        mu = 0.0121505839705277
        f = lambda t, x: CRTBP_dynamics(t, x, mu)
    else:
        f = lambda t, x: TBP_dynamics(t, x)
    """Propagate the orbits"""
    angle = np.array([
        rd[0] * math.pi / 2,  # alpha
        rd[1] * math.pi,  # beta
    ])
    dx0 = np.array([
        0,
        0,
        0,
        dv * math.cos(angle[0]) * math.cos(angle[1]),
        dv * math.cos(angle[0]) * math.sin(angle[1]),
        dv * math.sin(angle[0]),
    ])
    """Propagate the neighbouring orbit"""
    tk = t_series[-1]
    xk = y_series[-1]
    T = poincare_plane_direction(xk)
    yp0 = T[0, 0] * xk[0] + T[1, 0] * xk[1] + T[2, 0] * xk[2]
    event_func = lambda t, x: event_y(t, x, T, yp0)
    sol = solve_ivp(f, [t0, 1.5 * tf], x0 + dx0, args=(), method='RK45',
                    t_eval=None, max_step=np.inf, events=[event_func], rtol=RelTol, atol=AbsTol)
    tp = sol.t_events[0]
    dtp = abs(tp - tk)
    id = np.where(dtp == dtp.min())[0][0]
    t_eval = np.linspace(t0, tp[id], Nt + 1)
    sol = solve_ivp(f, [t0, tp[id]], x0 + dx0, args=(), method='RK45',
                    t_eval=t_eval, max_step=np.inf, rtol=RelTol, atol=AbsTol)
    point = sol.y.T
    """Get the plane states"""
    for k in range(len(t_series)):
        tk = t_series[k]
        xk = y_series[k]
        T = poincare_plane_direction(xk)
        yp0 = T[0, 0] * xk[0] + T[1, 0] * xk[1] + T[2, 0] * xk[2]
        event_func = lambda t, x: event_y(t, x, T, yp0)
        sol = solve_ivp(f, [t0, 1.5 * tk], x0 + dx0, args=(), method='RK45',
                        t_eval=None, max_step=np.inf, events=[event_func], rtol=RelTol, atol=AbsTol)
        xp = sol.y_events[0]
        tp = sol.t_events[0]
        dtp = abs(tp - tk)
        id = np.where(dtp == dtp.min())[0][0]
        xp = xp[id]
        point_series[k, 0] = T[0, 1] * xp[0] + T[1, 1] * xp[1] + T[2, 1] * xp[2]  # xp
        point_series[k, 1] = T[0, 2] * xp[0] + T[1, 2] * xp[1] + T[2, 2] * xp[2]  # zp
    """Return results"""
    return point, point_series

if __name__ == "__main__":
    print("Hello world!")
