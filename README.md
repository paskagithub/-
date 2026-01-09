# Optimal Transport Experiment Suite

## Overview
This project runs **14 mandatory optimal transport experiments** and writes the results into three CSV files. The experiments cover linear programming (LP) solvers, an ADMM baseline, and entropic OT solvers, matching the report tables referenced below.

## Installation
Minimum requirements:
- Python 3.10+
- `numpy`
- `pandas` (for post-processing and table creation)

Optional solver backends:
- `gurobipy` (Gurobi)
- `mosek` (MOSEK Fusion)

> Note: Gurobi and MOSEK require valid licenses. If unavailable, the runner will skip them gracefully with `status=NO_SOLVER`.

## How to Run
```bash
python run_all.py --threads 1 --seed 0
```

## Outputs and Report Tables
The runner always writes CSVs to `results/`:
- `results/results_lp.csv` → Tables 4.1 and 4.2
- `results/results_admm.csv` → Table 4.3
- `results/results_eot.csv` → Tables 5.1, 5.2, 5.3

## Notes
- Solver licensing: if `gurobipy` or `mosek` cannot be used (missing package or license), those runs return `status=NO_SOLVER` and still appear in the CSVs.
- No external dataset downloads are used. Instance **D** is constructed in a DOTmark-like fashion directly in code.
