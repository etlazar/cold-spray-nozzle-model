import numpy as np


def find_property(prop, val0, var, constants=None):
    var = np.asarray(var, dtype=float)

    if prop == 'temp':
        gamma = constants[0]
        return val0 / (1 + (gamma - 1) / 2 * var ** 2)

    if prop == 'pressure':
        gamma = constants[0]
        return val0 / (1 + (gamma - 1) / 2 * var ** 2) ** (gamma / (gamma - 1))

    if prop == 'density':
        gamma = constants[0]
        return val0 / (1 + (gamma - 1) / 2 * var ** 2) ** (1 / (gamma - 1))

    if prop == 'velocity':
        gamma, R = constants[0], constants[1]
        temp = val0 / (1 + (gamma - 1) / 2 * var ** 2)
        return var * np.sqrt(gamma * R * temp)

    if prop == 'viscosity':
        ref_temp, Cs = constants[0], constants[1]
        return val0 * ((ref_temp + Cs) / (var + Cs)) * (var / ref_temp) ** 1.5

    if prop == 'kg':
        return 1e-4 * np.exp(
            0.94306384 * np.log(var)
            + 0.12279898e3 / var
            - 0.11839435e5 / var ** 2
            - 0.10668773
        )

    if prop == 'Pr':
        return (constants * var[0]) / var[1]

    if prop == 'Nu':
        return 2 + 0.6 * var[0] ** 0.5 * var[1] ** (1 / 3)

    if prop == 'hp':
        return var[0] * (var[1] / constants)

    raise ValueError(f'Unknown property: {prop}')
