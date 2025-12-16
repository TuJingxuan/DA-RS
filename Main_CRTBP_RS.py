#import daceypy_import_helper  # noqa: F401
from typing import Callable, Type
import numpy as np
import math
import scipy
from scipy.integrate import solve_ivp
import scipy.io as scio
from daceypy import DA, RK, array, integrator, ADS
from module_orbit_dynamics import CRTBP_dynamics, CRTBP_time
from module_reachable_set import reachable_set_map, reachable_set_map_envelope, generate_monte_carlo_points, \
    generate_monte_carlo_points_all, combine_envelope, poincare_plane_direction
import time
import warnings
warnings.filterwarnings("ignore")

miuE = 398600.435436096
Re = 6378.1366
RelTol = 10 ** -12
AbsTol = 10 ** -12

def generate_scenario(
        N: float,
        f: Callable[[float, np.array], np.array],
        Nt: int,
):
    """Generate scenario parameters for the simulation"""
    """Parameters setting"""
    data = scipy.io.loadmat("./data/NRHO_Stable_Data.mat")
    # data = scipy.io.loadmat("./data/NRHO_9_2_Data.mat")
    period = data["period"][0, 0]
    x0 = data["x0"].T[0]
    """Propagate the nominal orbit"""
    t0 = 0.0
    tf = period * N
    t_eval = np.linspace(t0, tf, Nt + 2)
    sol = solve_ivp(f, [t0, tf], x0, args=(), method='RK45', t_eval=t_eval,
                    max_step=np.inf, rtol=RelTol, atol=AbsTol)
    xf = sol.y.T[-1, :]  # final state of the nominal trajectory
    T = np.zeros([len(sol.y.T), 3, 3])
    for k in range(len(sol.y.T)):
        xfk = sol.y.T[k]
        T[k] = poincare_plane_direction(xfk)
    """Nominal orbit"""
    t_eval = np.linspace(t0, tf, 101)
    sol_ = solve_ivp(f, [t0, tf], x0, args=(), method='RK45', t_eval=t_eval,
                    max_step=np.inf, rtol=RelTol, atol=AbsTol)
    nominal_orbit = sol_.y.T  # final state of the nominal trajectory
    """Return data"""
    return x0, t0, tf, xf, sol, T, nominal_orbit

def single_epoch_RS_envelope(
        N: float,
        dv: float,
        f: Callable[[float, np.array], np.array],
        ft: Callable[[array, float, array, array], array],
        if_save: bool,
):
    """Determine the envelope of a single-epoch RS"""
    """Parameters setting"""
    Ns = 51
    N_anchor_point = 6
    if N <= 0.15:
        tol = 1e-6
    else:
        tol = 1e-5
    """Generate simulated scenario"""
    x0, t0, tf, xf, _, T, nominal_orbit = generate_scenario(N, f, 0)
    """Compute the high-order map"""
    time_cost = np.zeros([2])
    start = time.time()
    final_lists = reachable_set_map(x0, t0, tf, xf, dv, ft, tol)
    final_list = final_lists[-1]  # the map at the final epoch
    """Determine the envelope points"""
    initial_box, final_manifold, final_envelope, final_perimeter, final_perimeter_anchor, envelope_points, if_contain_sol, time_cost_envelope = reachable_set_map_envelope(
        final_list=final_list,
        Ns=Ns,
        N_anchor_point=N_anchor_point,
        if_solved=False,  # do not use the solved envelope
        if_only_output=False,  # output all results
    )
    time_cost[0] = time.time() - start
    """Combine the envelope points using alpha-based method"""
    envelope_points = envelope_points.transpose(1, 0, 2).reshape(2, -1).T
    envelope_points = combine_envelope(envelope_points)
    """Generate MC points"""
    Nmc = 100
    start = time.time()
    points, points_all = generate_monte_carlo_points(x0, t0, tf, xf, Nmc, dv, final_list, f)
    time_cost[1] = time.time() - start
    """Save data"""
    if if_save:
        file_name = "./data/NRHO_Stable_RS_single_case.mat"
        # file_name = "./data/NRHO_9_2_RS_single_case.mat"
        scio.savemat(
            file_name, {
                "initial_box": initial_box,
                "final_manifold": final_manifold,
                "final_envelope": final_envelope,
                "final_perimeter": final_perimeter,
                "final_perimeter_anchor": final_perimeter_anchor,
                "envelope_points": envelope_points,
                "if_contain_sol": if_contain_sol,
                "points": points,
                "points_all": points_all,
                "nominal_orbit": nominal_orbit,
                "xf": xf,
                "T": T[-1],
                "time_cost": time_cost,
                "time_cost_envelope": time_cost_envelope,
            },
        )
    """Return results"""
    return final_lists

def single_epoch_RS_envelope_tol(
        N: float,
        dv: float,
        f: Callable[[float, np.array], np.array],
        ft: Callable[[array, float, array, array], array],
        if_save: bool,
):
    """Determine the envelope of a single-epoch RS"""
    """Parameters setting"""
    Ns = 51
    N_anchor_point = 6
    """Case 1"""
    tol = 1e-5
    x0, t0, tf, xf, _, T, nominal_orbit = generate_scenario(N, f, 0)
    final_lists = reachable_set_map(x0, t0, tf, xf, dv, ft, tol)
    final_list = final_lists[-1]  # the map at the final epoch
    initial_box5, _, final_envelope = reachable_set_map_envelope(
        final_list=final_list,
        Ns=Ns,
        N_anchor_point=N_anchor_point,
        if_solved=False,  # do not use the solved envelope
        if_only_output=True,  # output only analytical results
    )
    envelope_points = final_envelope.transpose(1, 0, 2).reshape(2, -1).T
    envelope_points5 = combine_envelope(envelope_points)
    """Case 2"""
    tol = 1e-6
    x0, t0, tf, xf, _, T, nominal_orbit = generate_scenario(N, f, 0)
    final_lists = reachable_set_map(x0, t0, tf, xf, dv, ft, tol)
    final_list = final_lists[-1]  # the map at the final epoch
    initial_box6, _, final_envelope = reachable_set_map_envelope(
        final_list=final_list,
        Ns=Ns,
        N_anchor_point=N_anchor_point,
        if_solved=False,  # do not use the solved envelope
        if_only_output=True,  # output only analytical results
    )
    envelope_points = final_envelope.transpose(1, 0, 2).reshape(2, -1).T
    envelope_points6 = combine_envelope(envelope_points)
    """Generate MC points"""
    Nmc = 100
    points, points_all = generate_monte_carlo_points(x0, t0, tf, xf, Nmc, dv, final_list, f)
    """Save data"""
    if if_save:
        scio.savemat(
            "./data/NRHO_Stable_RS_single_case_tol.mat", {
                "initial_box5": initial_box5,
                "envelope_points5": envelope_points5,
                "initial_box6": initial_box6,
                "envelope_points6": envelope_points6,
                "points": points,
                "points_all": points_all,
                "nominal_orbit": nominal_orbit,
                "xf": xf,
                "T": T[-1],
            },
        )
    """Return results"""
    return initial_box5, initial_box6, envelope_points5, envelope_points6

def time_series_RS_envelope(
        N: float,
        dv: float,
        f: Callable[[float, np.array], np.array],
        ft: Callable[[array, float, array, array], array],
        if_save: bool,
        Nt: int,
):
    """Determine the envelope of time-series RS"""
    """Parameters setting"""
    Ns = 51
    N_anchor_point = 6
    """Generate simulated scenario"""
    x0, t0, tf, _, sol, T, nominal_orbit = generate_scenario(N, f, Nt - 1)
    envelope_points_data = np.zeros([Nt, 1000, 2])
    envelope_points_index = np.zeros([Nt])
    n_split = np.zeros([Nt])
    for k in range(Nt):
        print("======== Epoch: %d ========" % k)
        Nk = N * (k + 1) / Nt
        if Nk <= 0.15:
            tol = 1e-6
        else:
            tol = 1e-5
        tfk = sol.t[k + 1]
        xfk = sol.y.T[k + 1]
        """Compute the high-order map"""
        final_lists = reachable_set_map(x0, t0, tfk, xfk, dv, ft, tol)
        final_list = final_lists[-1]  # the map at the final epoch
        """Determine the envelope points"""
        n_split[k] = len(final_list)
        initial_box, final_manifold, final_envelope = reachable_set_map_envelope(
            final_list=final_list,
            Ns=Ns,
            N_anchor_point=N_anchor_point,
            if_solved=False,  # do not use the solved envelope
            if_only_output=True,  # output only analytical results
        )
        """Combine the envelope points using alpha-based method"""
        envelope_points = final_envelope.transpose(1, 0, 2).reshape(2, -1).T
        envelope_points = combine_envelope(envelope_points)
        """Save data"""
        envelope_points_index[k] = len(envelope_points)
        envelope_points_data[k, 0:len(envelope_points), :] = envelope_points
    """Generate MC points"""
    Nmc = 10000
    points, points_series, _ = generate_monte_carlo_points_all(x0, t0, tf, Nmc, dv, "CRTBP", int(Nt * 5),
                                                               sol.t[1:], sol.y.T[1:])
    """Save data"""
    if if_save:
        scio.savemat(
            "./data/NRHO_Stable_RS_time_series.mat", {
                "envelope_points_index": envelope_points_index,
                "envelope_points_data": envelope_points_data,
                "points": points,
                "points_series": points_series,
                "nominal_orbit": nominal_orbit,
                "t_series": sol.t[1:],
                "x_series": sol.y.T[1:],
                "T": T,
                "n_split": n_split,
            },
        )
    """Return results"""
    return envelope_points_index, envelope_points_data, points

def main():
    """Main function"""
    """Parameters setting"""
    mu = 0.0121505839705277
    unitV = 1.02454629434750
    dv = 1e-2 / unitV
    N = 1.0  # number of revolution
    # N = 0.5  # number of revolution
    f = lambda t, x: CRTBP_dynamics(t, x, mu)
    ft = lambda x, t, t0, tf: CRTBP_time(x, t, mu, t0, tf)
    if_save = True
    """Single case analysis"""
    single_epoch_RS_envelope(N, dv, f, ft, if_save)
    """Single case comparison"""
    # N = 0.1  # number of revolution
    # single_epoch_RS_envelope_tol(N, dv, f, ft, if_save)
    """Time-series analysis"""
    # Nt = 100
    # time_series_RS_envelope(N, dv, f, ft, if_save, Nt)

if __name__ == "__main__":
    main()
