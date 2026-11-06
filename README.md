# 1D Isentropic Cold Spray Nozzle Model
**Neils Lab** | **Co-op 2026** | **Ethan Lazar**

Desktop app that solves quasi-1D isentropic flow through a converging-diverging cold spray nozzle (with an optional normal/bow shock at the substrate) and the velocity and temperature of D10/D50/D90 powder particles carried by the gas.

---
## Setup
**Create and activate** the conda environment:
```bash
conda env create -f environment.yml
conda activate nozzle
```
**Run** the app:
```bash
python app.py
```
**_or_** run without the GUI (default inputs, plots only):
```bash
python solver.py
```
**Test:**
```bash
python -m unittest test_model   # slow, runs the full solver
```

## Using the App
1. Set gas, powder and geometry/solver inputs on the left and press **Run**. At the default 0.001 mm step a run takes a few minutes.
2. Results appear in the plot tabs (Nozzle, Temperature, Velocity, Pressure, Mach) and the tables below.
3. Export:
   - **Reported values (CSV)**: inputs, gas properties at the exit / bow shock, particle properties at the substrate.
   - **Full nozzle profile (CSV)**: every quantity at every grid point (~200,000 rows, ~50 MB, temperatures in K).

**Note:** if the step size is too large for small particles, the RK4 solver goes unstable; the app warns you and suggests a smaller step.

## Files
```
├── app.py                        # GUI (PySide6 + matplotlib)
├── solver.py                     # Params (inputs/defaults) and solve()
├── mach_number_distribution.py   # Mach number along the nozzle, normal shock
├── root_finder_mach_number.py    # Mach number root finder
├── find_property.py              # isentropic relations, gas/particle properties
├── rk4_vp.py, rk4_tp.py          # RK4 particle velocity and temperature
├── plot_nozzle.py, mlutil.py     # nozzle drawing and helpers
├── test_model.py                 # unit tests
└── environment.yml               # conda environment (pinned versions)
```
