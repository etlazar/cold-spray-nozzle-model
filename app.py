"""Desktop GUI for the 1D isentropic cold spray nozzle model.

Run with:  python app.py
"""
import csv
import sys
import time
import traceback
from dataclasses import replace

import matplotlib
import numpy as np
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QScrollArea, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from mach_number_distribution import ShockPlacementError
from plot_nozzle import plot_nozzle
from root_finder_mach_number import MachSolveError
from solver import VP0_REST, Params, solve

# (attribute on Params, label, unit, min, max, decimals, step)
INPUTS = {
    'Gas': [
        ('pressure0', 'Gas pressure', 'psig', 0.1, 1e4, 2, 10),
        ('temp0', 'Gas temperature', '°C', -200, 3000, 2, 10),
        ('cp', 'Gas cp', 'kJ/kg·K', 0.01, 100, 4, 0.01),
    ],
    'Powder': [
        ('p_density', 'Particle density', 'kg/m³', 1, 1e5, 1, 100),
        ('dp10', 'D10', 'µm', 0.01, 1e4, 3, 1),
        ('dp50', 'D50', 'µm', 0.01, 1e4, 3, 1),
        ('dp90', 'D90', 'µm', 0.01, 1e4, 3, 1),
        ('cp_par', 'Particle cp', 'J/kg·K', 1, 1e5, 2, 10),
    ],
    'Geometry / Solver': [
        ('Li', 'Inlet length', 'mm', 0.001, 1e4, 3, 1),
        ('Ls', 'Standoff distance', 'mm', 0.001, 1e4, 3, 1),
        ('h', 'Step size', 'mm', 1e-5, 10, 5, 0.001),
    ],
}

PARTICLES = ['D10', 'D50', 'D90']

# Allowed overshoot of the particle temperature bounds before flagging instability
TP_TOL = 1.0   # K


def unstable_particles(r, p):
    """Names of particles whose solution is non-physical.

    Explicit RK4 goes unstable for small particles when the step is too large.
    Physically vp stays in [0, ~v_max] and Tp between the coldest of (its start,
    the gas) and the stagnation temperature T0.
    """
    v_max = np.nanmax(r.v)
    Tp_min = min(np.nanmin(r.temp), p.p_temp0 + 273.15) - TP_TOL
    Tp_max = p.temp0 + 273.15 + TP_TOL

    def is_unstable(d):
        vp, Tp = d['vp'], d['Tp']
        if not (np.all(np.isfinite(vp)) and np.all(np.isfinite(Tp))):
            return True
        return (vp.min() < 0 or vp.max() > 1.5 * v_max
                or Tp.min() < Tp_min or Tp.max() > Tp_max)

    return [n for n in PARTICLES if is_unstable(r.particles[n])]


def fmt_time(seconds):
    m, s = divmod(int(round(seconds)), 60)
    return f'{m}:{s:02d}'


class SolverWorker(QObject):
    progress = Signal(float, str)
    finished = Signal(object, object)   # (results, params)
    failed = Signal(str, str)           # (title, message)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            r = solve(self.params, verbose=False,
                      progress=lambda f, m: self.progress.emit(f, m))
            self.finished.emit(r, self.params)
        except MachSolveError as e:
            self.failed.emit('Mach solve failed', str(e))
        except ShockPlacementError as e:
            self.failed.emit('Bow shock placement', str(e))
        except Exception:
            self.failed.emit('Solver error', traceback.format_exc())


class PlotTab(QWidget):
    def __init__(self):
        super().__init__()
        self.figure = Figure(layout='constrained')
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QVBoxLayout(self)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas)

    def new_axes(self):
        self.figure.clear()
        return self.figure.add_subplot()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('1D Isentropic Cold Spray Model')
        self.resize(1400, 850)
        self.thread = None
        self.worker = None
        self.last = None   # (results, params)

        defaults = Params()
        self.fields = {}

        # ---- Left panel: inputs ----
        left = QWidget()
        left_layout = QVBoxLayout(left)
        for group_name, rows in INPUTS.items():
            box = QGroupBox(group_name)
            form = QFormLayout(box)
            for attr, label, unit, lo, hi, dec, step in rows:
                spin = QDoubleSpinBox()
                spin.setRange(lo, hi)
                spin.setDecimals(dec)
                spin.setSingleStep(step)
                spin.setValue(getattr(defaults, attr))
                spin.setSuffix(f'  {unit}')
                spin.setKeyboardTracking(False)
                self.fields[attr] = spin
                if attr == 'pressure0':
                    spin.setToolTip('Gauge pressure; 1 atm is added for the absolute value.')
                form.addRow(label, spin)
            if group_name == 'Powder':
                self.vp0_gas = QCheckBox('Start particles at inlet gas velocity')
                self.vp0_gas.setToolTip(
                    f'Unchecked: particles start at {VP0_REST:g} m/s.')
                self.vp0_gas.setChecked(defaults.vp0 is None)
                form.addRow(self.vp0_gas)
            if group_name == 'Geometry / Solver':
                self.bow_shock = QCheckBox('Use bow shock')
                self.bow_shock.setChecked(defaults.bow_shock)
                form.addRow(self.bow_shock)
            left_layout.addWidget(box)

        self.run_btn = QPushButton('Run')
        self.run_btn.setMinimumHeight(36)
        self.run_btn.clicked.connect(self.run)
        self.reset_btn = QPushButton('Reset to defaults')
        self.reset_btn.clicked.connect(self.reset)
        left_layout.addWidget(self.run_btn)
        left_layout.addWidget(self.reset_btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.status = QLabel('Ready')
        self.status.setWordWrap(True)
        left_layout.addWidget(self.progress)
        left_layout.addWidget(self.status)
        left_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(320)

        # ---- Right panel: plots + substrate table ----
        self.tabs = QTabWidget()
        self.plot_tabs = {}
        for name in ['Nozzle', 'Temperature', 'Velocity', 'Pressure', 'Mach']:
            tab = PlotTab()
            self.plot_tabs[name] = tab
            self.tabs.addTab(tab, name)

        # Gas values at the nozzle exit (and bow shock) beside the particle
        # values at the substrate
        self.gas_table = QTableWidget()
        self.particle_table = QTableWidget()
        for table in (self.gas_table, self.particle_table):
            table.setEditTriggers(QTableWidget.NoEditTriggers)
            table.verticalHeader().setVisible(False)
            table.verticalHeader().setDefaultSectionSize(22)
        self.export_btn = QPushButton('Export reported values (CSV)')
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_csv)
        self.export_profile_btn = QPushButton('Export full nozzle profile (CSV)')
        self.export_profile_btn.setEnabled(False)
        self.export_profile_btn.clicked.connect(self.export_profile_csv)

        gas_box = QGroupBox('Gas properties')
        QVBoxLayout(gas_box).addWidget(self.gas_table)
        particle_box = QGroupBox('Particles at the substrate')
        QVBoxLayout(particle_box).addWidget(self.particle_table)

        table_box = QWidget()
        tb_layout = QVBoxLayout(table_box)
        tb_layout.setContentsMargins(0, 0, 0, 0)
        tables = QHBoxLayout()
        tables.addWidget(gas_box, 2)
        tables.addWidget(particle_box, 3)
        tb_layout.addLayout(tables)
        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(self.export_profile_btn)
        buttons.addWidget(self.export_btn)
        tb_layout.addLayout(buttons)

        right = QSplitter(Qt.Vertical)
        right.addWidget(self.tabs)
        right.addWidget(table_box)
        right.setSizes([560, 290])

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(right)
        splitter.setSizes([340, 1060])
        self.setCentralWidget(splitter)

    # ---------------------------------------------------------------
    def read_params(self):
        values = {attr: spin.value() for attr, spin in self.fields.items()}
        values['bow_shock'] = self.bow_shock.isChecked()
        values['vp0'] = None if self.vp0_gas.isChecked() else VP0_REST
        return replace(Params(), **values)

    def reset(self):
        defaults = Params()
        for attr, spin in self.fields.items():
            spin.setValue(getattr(defaults, attr))
        self.bow_shock.setChecked(defaults.bow_shock)
        self.vp0_gas.setChecked(defaults.vp0 is None)

    def run(self):
        p = self.read_params()
        if not (p.dp10 <= p.dp50 <= p.dp90):
            QMessageBox.warning(self, 'Check inputs',
                                'Expected D10 ≤ D50 ≤ D90.')
            return
        # Step size (mm) must not exceed the smallest particle diameter (µm)
        dp_min = min(p.dp10, p.dp50, p.dp90)
        if dp_min * 1e-3 < p.h:
            QMessageBox.warning(
                self, 'Step size too large',
                f'The smallest particle ({dp_min:g} µm) is smaller than the '
                f'step size ({p.h * 1e3:g} µm). Use a step size of at most '
                f'{dp_min * 1e-3:g} mm.')
            return

        self.run_btn.setEnabled(False)
        self.progress.setValue(0)
        self.t_start = time.perf_counter()

        self.thread = QThread()
        self.worker = SolverWorker(p)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.start()

    def on_progress(self, frac, msg):
        self.progress.setValue(int(frac * 1000))
        elapsed = time.perf_counter() - self.t_start
        # Wait for a little progress before extrapolating the remaining time
        if frac > 0.01:
            remaining = elapsed * (1 - frac) / frac
            self.progress.setFormat(
                f'%p%  —  {fmt_time(elapsed)} elapsed, ~{fmt_time(remaining)} left')
        else:
            self.progress.setFormat(f'%p%  —  {fmt_time(elapsed)} elapsed, estimating...')
        self.status.setText(msg)

    def on_failed(self, title, msg):
        self.run_btn.setEnabled(True)
        self.status.setText(f'Error: {title}')
        QMessageBox.critical(self, title, msg)

    def on_finished(self, r, p):
        self.run_btn.setEnabled(True)
        self.last = (r, p)
        elapsed = time.perf_counter() - self.t_start
        self.progress.setFormat(f'%p%  —  finished in {fmt_time(elapsed)}')
        self.status.setText(f'Done — {r.x.size:,} grid points')
        self.draw_plots(r, p)
        self.fill_table(r)
        self.export_btn.setEnabled(True)
        self.export_profile_btn.setEnabled(True)

        unstable = unstable_particles(r, p)
        if unstable:
            self.status.setText(f'Unstable solution for {", ".join(unstable)}')
            QMessageBox.warning(
                self, 'Numerical instability',
                f'The particle solution for {", ".join(unstable)} is unstable '
                f'at a step size of {p.h:g} mm (non-physical velocity or '
                'temperature). Reduce the step size (e.g. ≤ 0.005 mm) and rerun.')

    # ---------------------------------------------------------------
    def draw_plots(self, r, p):
        x = r.x

        def finish(tab, ax, title, ylabel, legend=False):
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Axial Position [mm]')
            ax.set_xlim(x[0], x[-1])
            ax.grid(True, alpha=0.3)
            if legend:
                ax.legend()
            tab.canvas.draw_idle()

        tab = self.plot_tabs['Nozzle']
        ax = tab.new_axes()
        plot_nozzle(r.nozzle_geometry, max(p.h, (x[-1] - x[0]) / 2000), ax=ax)
        tab.canvas.draw_idle()

        tab = self.plot_tabs['Temperature']
        ax = tab.new_axes()
        ax.plot(x, r.temp, label='Gas')
        for name in PARTICLES:
            ax.plot(x, r.particles[name]['Tp'], label=f'Particle {name}')
        finish(tab, ax, 'Particle and Gas Temperature Across the Nozzle',
               'Temperature [K]', legend=True)

        tab = self.plot_tabs['Velocity']
        ax = tab.new_axes()
        ax.plot(x, r.v, label='Gas')
        for name in PARTICLES:
            ax.plot(x, r.particles[name]['vp'], label=f'Particle {name}')
        finish(tab, ax, 'Gas and Particle Velocity Across the Nozzle',
               'Velocity [m/s]', legend=True)

        tab = self.plot_tabs['Pressure']
        ax = tab.new_axes()
        ax.plot(x, r.pressure)
        finish(tab, ax, 'Gas Pressure', 'Pressure [Pa]')

        tab = self.plot_tabs['Mach']
        ax = tab.new_axes()
        ax.plot(x, r.M)
        finish(tab, ax, 'Mach Number', 'Mach Number')

    @staticmethod
    def gas_locations(r):
        """(column label, grid index) for the nozzle exit and, if present, the bow shock."""
        locs = [('Nozzle exit', r.exit_index)]
        if r.shock_index is not None:
            locs.append(('Bow shock', r.shock_index))
        return locs

    def gas_rows(self, r):
        """Rows of (quantity, unit, value at each gas location)."""
        idx = [i for _, i in self.gas_locations(r)]
        quantities = [
            ('Axial position', 'mm', r.x),
            ('Mach number', '-', r.M),
            ('Gas temperature', '°C', r.temp - 273.15),
            ('Gas pressure', 'Pa', r.pressure),
            ('Gas density', 'kg/m³', r.gas_density),
            ('Gas velocity', 'm/s', r.v),
            ('Gas viscosity', 'Pa·s', r.viscosity),
            ('Gas thermal conductivity (kg)', 'W/m·K', r.kg),
            ('Prandtl number', '-', r.Pr),
        ]
        return [(label, unit, *arr[idx]) for label, unit, arr in quantities]

    def particle_rows(self, r):
        """Rows of (quantity, unit, D10, D50, D90) at the last grid point."""
        quantities = [
            ('Particle diameter', 'µm', lambda d: d['dp'] * 1e6),
            ('Particle velocity', 'm/s', lambda d: d['vp'][-1]),
            ('Particle temperature', '°C', lambda d: d['Tp'][-1] - 273.15),
            ('Particle Reynolds number', '-', lambda d: d['Rep'][-1]),
            ('Drag coefficient', '-', lambda d: d['Cd'][-1]),
            ('Nusselt number', '-', lambda d: d['Nu'][-1]),
            ('Heat transfer coeff. (hp)', 'W/m²·K', lambda d: d['hp'][-1]),
        ]
        return [(label, unit, *[fn(r.particles[n]) for n in PARTICLES])
                for label, unit, fn in quantities]

    def fill_table(self, r):
        gas_headers = [name for name, _ in self.gas_locations(r)]
        self._fill(self.gas_table, ['Quantity', 'Unit', *gas_headers], self.gas_rows(r))
        self._fill(self.particle_table, ['Quantity', 'Unit', *PARTICLES],
                   self.particle_rows(r))

    @staticmethod
    def _fill(table, headers, rows):
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                text = f'{val:.6g}' if isinstance(val, (int, float)) or hasattr(val, 'dtype') else str(val)
                item = QTableWidgetItem(text)
                if j >= 2:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                table.setItem(i, j, item)
        hh = table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for j in range(1, len(headers)):
            hh.setSectionResizeMode(j, QHeaderView.ResizeToContents)

    def export_csv(self):
        if self.last is None:
            return
        r, p = self.last
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export reported values', 'reported_values.csv', 'CSV (*.csv)')
        if not path:
            return
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['Inputs'])
            for rows in INPUTS.values():
                for attr, label, unit, *_ in rows:
                    w.writerow([label, unit, getattr(p, attr)])
            w.writerow(['Absolute gas pressure', 'Pa', f'{r.pressure0:.8g}'])
            w.writerow(['Initial particle velocity', 'm/s',
                        'inlet gas velocity' if p.vp0 is None else p.vp0])
            w.writerow(['Use bow shock', '', p.bow_shock])
            w.writerow(['Bow shock standoff length scale', '', p.standoff_length_scale])
            w.writerow([])
            fmt = lambda row: [f'{v:.8g}' if not isinstance(v, str) else v for v in row]
            w.writerow(['Gas properties'])
            w.writerow(['Quantity', 'Unit', *[name for name, _ in self.gas_locations(r)]])
            for row in self.gas_rows(r):
                w.writerow(fmt(row))
            w.writerow([])
            w.writerow(['Particles at the substrate'])
            w.writerow(['Quantity', 'Unit', *PARTICLES])
            for row in self.particle_rows(r):
                w.writerow(fmt(row))
        self.status.setText(f'Saved {path}')

    @staticmethod
    def profile_columns(r):
        """(header, array) for every quantity at every grid point along the nozzle."""
        cols = [
            ('Axial position [mm]', r.x),
            ('Mach number [-]', r.M),
            ('Gas temperature [K]', r.temp),
            ('Gas pressure [Pa]', r.pressure),
            ('Gas density [kg/m^3]', r.gas_density),
            ('Gas velocity [m/s]', r.v),
            ('Gas viscosity [Pa.s]', r.viscosity),
            ('Gas thermal conductivity [W/m.K]', r.kg),
            ('Prandtl number [-]', r.Pr),
        ]
        for n in PARTICLES:
            d = r.particles[n]
            cols += [
                (f'{n} particle velocity [m/s]', d['vp']),
                (f'{n} particle temperature [K]', d['Tp']),
                (f'{n} particle Reynolds number [-]', d['Rep']),
                (f'{n} drag coefficient [-]', d['Cd']),
                (f'{n} Nusselt number [-]', d['Nu']),
                (f'{n} heat transfer coeff. [W/m^2.K]', d['hp']),
            ]
        return cols

    def export_profile_csv(self):
        if self.last is None:
            return
        r, _ = self.last
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export full nozzle profile', 'nozzle_profile.csv', 'CSV (*.csv)')
        if not path:
            return
        cols = self.profile_columns(r)
        self.status.setText(f'Writing {r.x.size:,} rows...')
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            np.savetxt(path, np.column_stack([arr for _, arr in cols]),
                       fmt='%.8g', delimiter=',', comments='',
                       header=','.join(name for name, _ in cols))
        finally:
            QApplication.restoreOverrideCursor()
        self.status.setText(f'Saved {path}')


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
