import numpy as np
from scipy.optimize import brentq


def root_finder_mach_number(gamma, tol, M0, A, At, is_supersonic=None):
    if is_supersonic is None:
        is_supersonic = M0 > 1

    n = (gamma + 1) / (2 * (gamma - 1))
    ar_target = A / At

    if abs(ar_target - 1) < 1e-4:
        return 1.0

    def f(M):
        return (1.0 / M) * ((2 / (gamma + 1)) * (1 + ((gamma - 1) / 2) * M ** 2)) ** n - ar_target

    if is_supersonic:
        lower, upper = 1 + 1e-6, 60.0
    else:
        lower, upper = 1e-6, 1 - 1e-6

    flo = f(lower)
    fhi = f(upper)

    if np.sign(flo) == np.sign(fhi):
        if is_supersonic:
            upper = 500.0
            fhi = f(upper)
        if np.sign(flo) == np.sign(fhi):
            return M0

    try:
        return brentq(f, lower, upper, xtol=tol)
    except (ValueError, RuntimeError):
        return M0
