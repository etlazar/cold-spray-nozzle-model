import numpy as np

from mlutil import Progress


def _cd(Rep):
    if Rep == 0:
        return np.inf             # Stokes limit; the drag force itself -> 0
    return (24.0 / Rep) * (1 + 0.15 * Rep ** 0.687) + 0.42 / (1 + 4.25e4 * Rep ** -1.16)


def rk4_vp(gas_density, v, dp, h, x, viscosity, p_density, vp0, verbose=True,
           on_progress=None):
    x = np.asarray(x, dtype=float)
    Rep = np.zeros(x.shape)
    Cd = np.zeros(x.shape)
    vp = np.zeros(x.shape)
    vp[0] = vp0

    h = h * 1e-3
    mp = p_density * (4 / 3) * np.pi * (dp / 2) ** 3
    Ap = np.pi * (dp / 2) ** 2

    def dvp(Cd_, rho, v_, vp_):
        # Zero slip (e.g. vp0 = gas velocity): no drag, and avoids inf * 0
        if v_ == vp_:
            return 0.0
        return (Cd_ * rho * Ap) / (2 * mp * vp_) * abs(v_ - vp_) * (v_ - vp_)

    N = x.size - 1
    prog = Progress('Particle velocity', N, verbose, on_progress)

    for i in range(N):
        Rep[i] = gas_density[i] * abs(v[i] - vp[i]) * dp / viscosity[i]
        Cd[i] = _cd(Rep[i])

        gas_density_half = (gas_density[i] + gas_density[i + 1]) / 2
        v_half = (v[i] + v[i + 1]) / 2
        viscosity_half = (viscosity[i] + viscosity[i + 1]) / 2
        gas_density_full = gas_density[i + 1]
        v_full = v[i + 1]
        viscosity_full = viscosity[i + 1]

        k1 = dvp(Cd[i], gas_density[i], v[i], vp[i])

        vp_k2 = vp[i] + 0.5 * h * k1
        Rep_half = gas_density_half * abs(v_half - vp_k2) * dp / viscosity_half
        k2 = dvp(_cd(Rep_half), gas_density_half, v_half, vp_k2)

        vp_k3 = vp[i] + 0.5 * h * k2
        Rep_half3 = gas_density_half * abs(v_half - vp_k3) * dp / viscosity_half
        k3 = dvp(_cd(Rep_half3), gas_density_half, v_half, vp_k3)

        vp_k4 = vp[i] + h * k3
        Rep_full = gas_density_full * abs(v_full - vp_k4) * dp / viscosity_full
        k4 = dvp(_cd(Rep_full), gas_density_full, v_full, vp_k4)

        vp[i + 1] = vp[i] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        prog.update(i + 1)

    prog.done()

    Rep[-1] = gas_density[-1] * abs(v[-1] - vp[-1]) * dp / viscosity[-1]
    Cd[-1] = _cd(Rep[-1])

    return Rep, Cd, vp
