# EN3330 Assignment 03: Lagrangian Duality

This repository contains the simulation code and the LaTeX report for the problem `min x^2  s.t.  x >= 1`. It covers the Lagrangian, the dual function, the primal function, the saddle point and weak duality.

| Path | Contents |
|------|----------|
| `src/duality_simulation.py` | Generates all figures and runs the numerical checks |
| `figures/` | Generated figures (PDF for the report, PNG for preview) |
| `results/numerical_checks.txt` | Output of the numerical checks |
| `report/report.tex` | LaTeX source of the report |

## Run

```bash
python3 src/duality_simulation.py                 # needs numpy, matplotlib
cd report && latexmk -pdf report.tex              # builds report/report.pdf
```
