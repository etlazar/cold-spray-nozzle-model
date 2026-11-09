"""Unit tests for the 1D isentropic model. Run with:  python -m unittest test_model"""
import unittest
from dataclasses import replace

import numpy as np

from find_property import find_property
from mach_number_distribution import ShockPlacementError, normal_shock
from root_finder_mach_number import MachSolveError, root_finder_mach_number
from solver import P_ATM, PSI_TO_PA, Params, solve

GAMMA = 1.4
R = 296.8
P0 = 200 * PSI_TO_PA     # Pa
T0 = 200 + 273.15        # K
RHO0 = P0 / (R * T0)
M_EXIT = 2.859


def isentropic_state(M, p0=P0, rho0=RHO0):
    T = find_property('temp', T0, M, [GAMMA])
    p = find_property('pressure', p0, M, [GAMMA])
    rho = find_property('density', rho0, M, [GAMMA])
    u = find_property('velocity', T0, M, [GAMMA, R])
    return T, p, rho, u


class TestIsentropicExit(unittest.TestCase):
    def test_exit_state(self):
        T, p, rho, u = isentropic_state(M_EXIT)
        self.assertAlmostEqual(T, 179.6, delta=0.1)
        self.assertAlmostEqual(p, 46440, delta=50)
        self.assertAlmostEqual(rho, 0.871, delta=0.001)
        self.assertAlmostEqual(u, 781, delta=0.5)

    def test_ideal_gas_consistency(self):
        T, p, rho, _ = isentropic_state(M_EXIT)
        self.assertAlmostEqual(p / (rho * R * T), 1.0, places=10)


class TestNormalShock(unittest.TestCase):
    def setUp(self):
        self.M2, self.p_ratio, self.p0_ratio = normal_shock(M_EXIT, GAMMA)
        _, self.p1, self.rho1, self.u1 = isentropic_state(M_EXIT)
        # Downstream state: T0 conserved, decelerate isentropically from p02
        _, self.p2, self.rho2, self.u2 = isentropic_state(
            self.M2, P0 * self.p0_ratio, RHO0 * self.p0_ratio)

    def test_downstream_mach(self):
        self.assertAlmostEqual(self.M2, 0.484, delta=0.001)

    def test_stagnation_pressure_ratio(self):
        self.assertAlmostEqual(self.p0_ratio, 0.370, delta=0.001)

    def test_static_pressure_jump(self):
        self.assertAlmostEqual(self.p2 / self.p1, self.p_ratio, places=10)
        self.assertAlmostEqual(self.p2, 4.35e5, delta=1e3)

    def test_downstream_density(self):
        self.assertAlmostEqual(self.rho2, 3.24, delta=0.01)

    def test_mass_conservation(self):
        self.assertAlmostEqual(self.rho1 * self.u1, 680.5, delta=0.5)
        self.assertAlmostEqual(self.rho2 * self.u2 / (self.rho1 * self.u1), 1.0, places=10)

    def test_momentum_conservation(self):
        lhs = self.p1 + self.rho1 * self.u1 ** 2
        rhs = self.p2 + self.rho2 * self.u2 ** 2
        self.assertAlmostEqual(rhs / lhs, 1.0, places=10)


class TestPrandtl(unittest.TestCase):
    def test_nitrogen_prandtl(self):
        cp = Params().cp * 1e3       # J/kg·K, as converted in solve()
        temp = np.linspace(300, 470, 18)
        mu = find_property('viscosity', 1.66e-5, temp, [273, 107])
        kg = find_property('kg', 0, temp)
        Pr = find_property('Pr', 0, [mu, kg], cp)
        np.testing.assert_allclose(Pr, 0.70, atol=0.03)


class TestSolver(unittest.TestCase):
    """End-to-end check on a coarse grid, which leaves the gas state unchanged."""

    @classmethod
    def setUpClass(cls):
        cls.r = solve(replace(Params(), h=0.01), verbose=False)

    def test_exit_matches_isentropic(self):
        i = self.r.exit_index
        self.assertAlmostEqual(self.r.M[i], M_EXIT, delta=0.001)
        self.assertAlmostEqual(self.r.pressure[i], 46440, delta=50)

    def test_post_shock_state(self):
        i = self.r.shock_index
        self.assertAlmostEqual(self.r.M[i], 0.484, delta=0.001)
        self.assertAlmostEqual(self.r.temp[i] - 273.15, 178.8, delta=0.1)
        self.assertAlmostEqual(self.r.pressure[i], 4.35e5, delta=1e3)
        self.assertAlmostEqual(self.r.gas_density[i], 3.24, delta=0.01)
        e = self.r.exit_index
        flux_ratio = (self.r.gas_density[i] * self.r.v[i]) / (self.r.gas_density[e] * self.r.v[e])
        self.assertAlmostEqual(flux_ratio, 1.0, places=6)

    def test_prandtl_everywhere(self):
        self.assertTrue(np.all((self.r.Pr > 0.65) & (self.r.Pr < 0.80)))


class TestSmallParticleTracksGas(unittest.TestCase):
    """A 0.1 um particle has a tiny relaxation time, so up to the nozzle exit it
    should follow the gas. It is very stiff: h = 0.001 mm with vp0 = gas velocity
    is needed for the temperature RK4 to stay stable."""

    @classmethod
    def setUpClass(cls):
        p = replace(Params(), dp10=0.1, dp50=0.1, dp90=0.1, vp0=None,
                    h=0.001, bow_shock=False)
        cls.p = p
        cls.r = solve(p, verbose=False)
        cls.d = cls.r.particles['D10']

    def test_tracks_gas_to_exit(self):
        r, d, p = self.r, self.d, self.p
        x = r.x
        # Skip the initial Tp relaxation and the sonic kinks at the throat ends,
        # where dv/dx of the quasi-1D gas solution is unbounded
        window = (x >= p.Li + p.Lc - 1) & (x <= p.Li + p.Lc + p.Lt + 1)
        s = (x >= 0.05) & (np.arange(x.size) <= r.exit_index) & ~window
        rel_dv = np.abs(d['vp'][s] - r.v[s]) / r.v[s]
        dT = np.abs(d['Tp'][s] - r.temp[s])
        self.assertLess(rel_dv.max(), 0.02)
        self.assertLess(dT.max(), 0.5)

    def test_exit_state(self):
        e = self.r.exit_index
        self.assertAlmostEqual(self.d['vp'][e] / self.r.v[e], 1.0, delta=5e-4)
        self.assertAlmostEqual(self.d['Tp'][e], self.r.temp[e], delta=0.1)


class TestMachSolveErrors(unittest.TestCase):
    def test_root_finder_raises(self):
        # A/At < 1 has no solution on either branch
        for supersonic in (False, True):
            with self.assertRaises(MachSolveError):
                root_finder_mach_number(GAMMA, 1e-5, 1.5, 0.5, 1.0, supersonic)

    def test_solve_reports_position(self):
        # Exit narrower than the throat: the diverging section has A/At < 1
        with self.assertRaises(MachSolveError) as cm:
            solve(replace(Params(), De=1.5, h=0.1), verbose=False)
        msg = str(cm.exception)
        self.assertIn('x =', msg)
        self.assertIn('A/At', msg)


class TestShockPlacement(unittest.TestCase):
    def test_standoff_shorter_than_shock_raises(self):
        # L_shock = 0.143 * 3.81 * exp(3.24 / 2.859^2) ~ 0.81 mm > Ls
        with self.assertRaises(ShockPlacementError):
            solve(replace(Params(), Ls=0.5, h=0.01), verbose=False)

    def test_no_shock_no_check(self):
        r = solve(replace(Params(), Ls=0.5, h=0.01, bow_shock=False), verbose=False)
        self.assertIsNone(r.shock_index)

    def test_radius_halves_standoff(self):
        rd = solve(replace(Params(), h=0.01), verbose=False)
        rr = solve(replace(Params(), h=0.01, standoff_length_scale='radius'), verbose=False)
        L_d = rd.x[-1] - rd.x[rd.shock_index]
        L_r = rr.x[-1] - rr.x[rr.shock_index]
        self.assertAlmostEqual(L_d, 0.81, delta=0.01)
        self.assertAlmostEqual(L_r, L_d / 2, delta=0.01)

    def test_bad_length_scale(self):
        with self.assertRaises(ValueError):
            solve(replace(Params(), h=0.1, standoff_length_scale='width'), verbose=False)


class TestGaugePressure(unittest.TestCase):
    def test_gauge_adds_one_atmosphere(self):
        p = replace(Params(), h=0.05, vp0=None)
        r_abs = solve(p, verbose=False)
        r_g = solve(replace(p, pressure_is_gauge=True), verbose=False)
        self.assertAlmostEqual(r_abs.pressure0, 200 * 6894.757, places=6)
        self.assertAlmostEqual(r_g.pressure0 - r_abs.pressure0, P_ATM, places=6)
        # Isentropic Mach distribution does not depend on p0
        np.testing.assert_array_equal(r_g.M, r_abs.M)


if __name__ == '__main__':
    unittest.main()
