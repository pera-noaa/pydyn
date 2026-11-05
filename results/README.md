# results/

Small tracked outputs of the `analysis/` scripts, each named in the `docs/cfl_headroom.md`
section that reports it. The runs that made them, with machine, job and model sha, are in
`runs/MANIFEST.md`. Nothing here is read by the tests.

| Files | Made by | Reported in |
|---|---|---|
| `cfl3x2_2019_cap_replay.csv` | `analysis/cfl_cap_replay.py` | `docs/cfl_headroom.md` §8 |
| `cfl3x2nam1x1_2019-{01,04,07,10}_cap_replay.csv` | `analysis/cfl_cap_replay_zoom.py` | §11 |
| `step_accuracy_3x2.csv`, `step_accuracy_6x4.csv` | `analysis/step_accuracy.py` | §9 |
| `step_accuracy_3x2_2019-10_*.csv`, `step_accuracy_6x4_xyzv.csv`, `step_accuracy_3x2_2019-09s_*.csv` | `analysis/step_accuracy.py` | §10 |
| `step_accuracy_3x2_2019-10e_*.csv` | `analysis/step_accuracy.py` | §11 |
| `obs_departure_3x2_2019-10f{,-point}.csv` | `analysis/obs_departure.py` | §12, §13 |
| `obs_residual_3x2_2019-10f{,-point}.csv` | `analysis/obs_residual.py` | §14 |
