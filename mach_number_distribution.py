import numpy as np

from mlutil import Progress
from root_finder_mach_number import root_finder_mach_number


def local_diameter(x, g):
    x = np.asarray(x, dtype=float)
    Li, Lc, Lt, Ld, _Ls, Di, Dt, De = g

    x1 = Li
    x2 = Li + Lc
    x3 = x2 + Lt
    x4 = x3 + Ld

    D = np.empty_like(x)

    inlet = x < x1
    converging = (x >= x1) & (x < x2)
    throat = (x >= x2) & (x < x3)
    diverging = (x >= x3) & (x < x4)
    standoff = x >= x4

    D[inlet] = Di
    D[converging] = ((Dt - Di) / Lc) * (x[converging] - Li) + Di
    D[throat] = Dt
    D[diverging] = ((De - Dt) / Ld) * (x[diverging] - x3) + Dt
    D[standoff] = De

    return D


def normal_shock(M1, gamma):
    """Normal-shock relations: returns (M2, p2/p1, p02/p01)."""
    M2 = np.sqrt((M1 ** 2 + 2 / (gamma - 1))
                 / ((2 * gamma / (gamma - 1)) * M1 ** 2 - 1))
    p_ratio = (2 * gamma * M1 ** 2 - (gamma - 1)) / (gamma + 1)
    # p0 = p * (1 + (gamma-1)/2 * M^2)^(gamma/(gamma-1)) on each side
    p0_ratio = p_ratio * ((1 + (gamma - 1) / 2 * M2 ** 2)
                          / (1 + (gamma - 1) / 2 * M1 ** 2)) ** (gamma / (gamma - 1))
    return M2, p_ratio, p0_ratio


def mach_number_distribution(nozzle_geometry, x, gamma, M0sub, M0super, tol,
                             bow_shock, verbose=True, on_progress=None):
    g = np.asarray(nozzle_geometry, dtype=float)
    x = np.asarray(x, dtype=float)

    Mach = np.zeros(x.shape)
    At = np.pi * (g[6] / 2) ** 2

    throat_start = g[0] + g[1]
    throat_end = throat_start + g[2]
    exit_pos = throat_end + g[3]

    D = local_diameter(x, g)
    A = np.pi * (D / 2) ** 2

    throat_end_i = int(np.argmin(np.abs(x - throat_end)))

    total_steps = x.size
    prog = Progress('Mach solve', total_steps, verbose, on_progress)

    for i in range(total_steps):
        is_supersonic = x[i] >= throat_end

        if i == 0:
            M0 = M0sub
        elif i == throat_end_i:
            M0 = M0super
        elif throat_start <= x[i] < throat_end:
            M0 = 1.001
        else:
            M0 = Mach[i - 1]

        Mach[i] = root_finder_mach_number(gamma, tol, M0, A[i], At, is_supersonic)

        prog.update(i + 1)

    prog.done()

    exit_index = int(np.argmin(np.abs(x - exit_pos)))
    shock_index = None

    if bow_shock:
        mach_exit = Mach[exit_index]
        L_shock = 0.143 * g[7] * np.exp(3.24 / mach_exit ** 2)
        M_shock, _, _ = normal_shock(mach_exit, gamma)
        shock_index = int(np.argmin(np.abs(x - (x[-1] - L_shock))))
        Mach[shock_index:] = np.linspace(M_shock, 0.0, x.size - shock_index)

    return Mach, exit_index, shock_index
