import time
from dataclasses import dataclass, field

import numpy as np

from find_property import find_property
from mach_number_distribution import mach_number_distribution, normal_shock
from mlutil import colon
from rk4_tp import rk4_tp
from rk4_vp import rk4_vp

PSI_TO_PA = 6894.757
P_ATM = 101325.0              # Pa
VP0_REST = 0.1                # m/s, particles start (almost) at rest


@dataclass
class Params:
    # Nozzle Geometry
    Li: float = 5.0               # mm
    Lc: float = 17.02             # mm
    Lt: float = 2.29              # mm
    Ld: float = 150.62            # mm
    Ls: float = 25.0              # mm
    Di: float = 9.52              # mm
    Dt: float = 1.98              # mm
    De: float = 3.81              # mm

    # Gas Parameters
    pressure0: float = 200.0      # psig (1 atm is added)
    temp0: float = 200.0          # C
    gamma: float = 1.4
    cp: float = 1.04              # kJ/kg·K
    R: float = 296.8
    Cs: float = 107.0
    ref_temp: float = 273.0
    ref_viscosity: float = 1.66e-5

    # Powder Parameters
    p_density: float = 7310.0     # kg/m^3
    dp10: float = 2.0             # μm
    dp50: float = 10.0            # μm
    dp90: float = 30.0            # μm
    psi: float = 0.5
    vp0: float | None = VP0_REST  # m/s; None starts particles at the inlet gas velocity
    p_temp0: float = 25.0         # C
    cp_par: float = 227.0

    # Solver Parameters
    h: float = 0.001              # mm
    int_M0: float = 0.03
    post_throat_M0: float = 1.05
    tol: float = 1e-5
    bow_shock: bool = True
    # Exit dimension used in Billig's bow-shock standoff correlation:
    # 'diameter' (De) or 'radius' (De / 2)
    standoff_length_scale: str = 'diameter'


@dataclass
class Results:
    x: np.ndarray
    M: np.ndarray
    temp: np.ndarray
    pressure: np.ndarray
    gas_density: np.ndarray
    v: np.ndarray
    viscosity: np.ndarray
    kg: np.ndarray
    Pr: np.ndarray
    nozzle_geometry: np.ndarray
    exit_index: int
    shock_index: int | None       # first grid point behind the bow shock
    particles: dict = field(default_factory=dict)
    pressure0: float | None = None    # absolute stagnation pressure used [Pa]


def solve(p=None, verbose=True, progress=None):
    """progress (optional): callable(fraction, message) for GUI status updates."""
    if p is None:
        p = Params()

    n_stages = 2 + 2 * 3

    def report(stage, msg):
        if progress is not None:
            progress(stage / n_stages, msg)

    def sub_report(stage, msg):
        """callback(fraction within this stage) for the per-step solver loops."""
        if progress is None:
            return None
        return lambda f: progress((stage + f) / n_stages, msg)

    t_start = time.perf_counter()

    # Consolidations, Calculations, and Conversions
    pressure0 = p.pressure0 * PSI_TO_PA + P_ATM   # gauge -> absolute
    temp0 = p.temp0 + 273.15
    dp = {'D10': p.dp10 * 1e-6, 'D50': p.dp50 * 1e-6, 'D90': p.dp90 * 1e-6}
    p_temp0 = p.p_temp0 + 273.15
    gas_density0 = pressure0 / (p.R * temp0)
    cp = p.cp * 1e3               # kJ/kg·K -> J/kg·K

    nozzle_geometry = np.array([p.Li, p.Lc, p.Lt, p.Ld, p.Ls, p.Di, p.Dt, p.De])
    Ltotal = nozzle_geometry[:5].sum()
    x = colon(0.0, p.h, Ltotal)

    # Find the mach number distribution
    report(0, 'Solving Mach distribution...')
    M, exit_index, shock_index = mach_number_distribution(
        nozzle_geometry, x, p.gamma, p.int_M0, p.post_throat_M0, p.tol,
        p.bow_shock, verbose, sub_report(0, 'Solving Mach distribution...'),
        p.standoff_length_scale)

    # Find the properties of the gas
    report(1, 'Computing gas properties...')
    temp = find_property('temp', temp0, M, [p.gamma])
    pressure = find_property('pressure', pressure0, M, [p.gamma])
    gas_density = find_property('density', gas_density0, M, [p.gamma])
    v = find_property('velocity', temp0, M, [p.gamma, p.R])
    viscosity = find_property('viscosity', p.ref_viscosity, temp, [p.ref_temp, p.Cs])

    # The normal shock is not isentropic: T0 is conserved but p0 drops, so
    # everything behind the shock decelerates isentropically from p02
    if shock_index is not None:
        _, _, p0_ratio = normal_shock(M[exit_index], p.gamma)
        post = slice(shock_index, None)
        pressure[post] = find_property('pressure', pressure0 * p0_ratio, M[post], [p.gamma])
        gas_density[post] = find_property('density', gas_density0 * p0_ratio, M[post], [p.gamma])

    # Calculate the Properties need to solve the heat transfer ode
    kg = find_property('kg', 0, temp)
    Pr = find_property('Pr', 0, [viscosity, kg], cp)

    # dvp/dx ~ 1/vp, so starting near rest makes the first steps stiff;
    # vp0 = None starts the particles at the local gas velocity instead
    vp0 = v[0] if p.vp0 is None else p.vp0

    particles = {}
    for k, (name, d) in enumerate(dp.items()):
        if verbose:
            print(f'--- {name} (dp = {d * 1e6:g} um) ---')

        # Find the velocity of the particles
        msg = f'Particle velocity: {name}...'
        report(2 + 2 * k, msg)
        Rep, Cd, vp = rk4_vp(gas_density, v, d, p.h, x, viscosity,
                             p.p_density, vp0, verbose, sub_report(2 + 2 * k, msg))
        Nu = find_property('Nu', 0, [Rep, Pr])
        hp = find_property('hp', 0, [Nu, kg], d)

        # Finds temp of the particles
        msg = f'Particle temperature: {name}...'
        report(3 + 2 * k, msg)
        Tp = rk4_tp(p.cp_par, d, hp, temp, p_temp0, x, p.h, p.p_density, vp, verbose,
                    sub_report(3 + 2 * k, msg))

        particles[name] = {'dp': d, 'Rep': Rep, 'Cd': Cd, 'vp': vp,
                           'Nu': Nu, 'hp': hp, 'Tp': Tp}

    report(n_stages, 'Done')
    if verbose:
        print(f'Total time: {time.perf_counter() - t_start:.2f} s')

    return Results(x=x, M=M, temp=temp, pressure=pressure,
                   gas_density=gas_density, v=v, viscosity=viscosity,
                   kg=kg, Pr=Pr, nozzle_geometry=nozzle_geometry,
                   exit_index=exit_index, shock_index=shock_index,
                   particles=particles, pressure0=pressure0)


def plot_results(r, h=0.001):
    import matplotlib.pyplot as plt

    from plot_nozzle import plot_nozzle

    x = r.x
    labels = list(r.particles)

    # Plot Nozzle
    plot_nozzle(r.nozzle_geometry, h)

    # Plot and Compare gas and particle properties
    _, ax = plt.subplots()
    ax.plot(x, r.temp, label='Gas')
    for name in labels:
        ax.plot(x, r.particles[name]['Tp'], label=f'Particle {name}')
    ax.set_title('Particle and Gas Tempature Across the Nozzle')
    ax.set_ylabel('Tempature [K]')
    ax.set_xlabel('Axial Postion [mm]')
    ax.set_xlim(x[0], x[-1])
    ax.legend()

    _, ax = plt.subplots()
    ax.plot(x, r.v, label='Gas')
    for name in labels:
        ax.plot(x, r.particles[name]['vp'], label=f'Particle {name}')
    ax.set_title('Gas and Particle Velocity Across the Nozzle')
    ax.set_ylabel('Velocity [m/s]')
    ax.set_xlabel('Axial Position [mm]')
    ax.set_xlim(x[0], x[-1])
    ax.legend()

    _, ax = plt.subplots()
    ax.plot(x, r.pressure)
    ax.set_title('Gas Pressure')
    ax.set_ylabel('Pressure [Pa]')
    ax.set_xlabel('Axial Postion [mm]')
    ax.set_xlim(x[0], x[-1])

    _, ax = plt.subplots()
    ax.plot(x, r.M)
    ax.set_title('Mach Number')
    ax.set_ylabel('Mach Number')
    ax.set_xlabel('Axial Position [mm]')
    ax.set_xlim(x[0], x[-1])
    ax.grid(True)

    plt.show()


if __name__ == '__main__':
    params = Params()
    results = solve(params)
    plot_results(results, params.h)
