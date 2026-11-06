import numpy as np

from mlutil import Progress


def rk4_tp(cp, dp, hp, temp, p_temp0, x, h, p_density, vp, verbose=True,
           on_progress=None):
    x = np.asarray(x, dtype=float)
    Tp = np.zeros(x.shape)
    Tp[0] = p_temp0

    mp = p_density * (4 / 3) * np.pi * (dp / 2) ** 3
    Ap = np.pi * dp ** 2

    def dTp(hp_, temp_, Tp_, vp_):
        return (hp_ / (cp * vp_)) * (Ap / mp) * (temp_ - Tp_)

    h = h * 1e-3
    N = x.size - 1
    prog = Progress('Particle temperature', N, verbose, on_progress)

    for i in range(N):
        temp_half = (temp[i] + temp[i + 1]) / 2
        hp_half = (hp[i] + hp[i + 1]) / 2
        vp_half = (vp[i] + vp[i + 1]) / 2

        k1 = dTp(hp[i], temp[i], Tp[i], vp[i])
        k2 = dTp(hp_half, temp_half, Tp[i] + 0.5 * h * k1, vp_half)
        k3 = dTp(hp_half, temp_half, Tp[i] + 0.5 * h * k2, vp_half)
        k4 = dTp(hp[i + 1], temp[i + 1], Tp[i] + h * k3, vp[i + 1])

        Tp[i + 1] = Tp[i] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        prog.update(i + 1)

    prog.done()
    return Tp
