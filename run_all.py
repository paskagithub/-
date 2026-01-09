import argparse
import csv
import os
from collections import Counter
from typing import Any

import numpy as np

import admm_ot
import dual_grad_eot
import lp_gurobi
import lp_mosek
import ot_instances
import sinkhorn


LP_COLUMNS = [
    "instance",
    "m",
    "n",
    "solver",
    "method",
    "build_time_s",
    "solve_time_s",
    "total_time_s",
    "objective",
    "r_row_l1",
    "r_col_l1",
    "r_nn",
    "status",
    "error_msg",
]

ADMM_COLUMNS = [
    "instance",
    "m",
    "n",
    "rho",
    "tol",
    "max_iter",
    "iters_used",
    "total_time_s",
    "objective",
    "r_row_l1",
    "r_col_l1",
    "r_nn",
    "status",
    "error_msg",
]

EOT_COLUMNS = [
    "instance",
    "m",
    "n",
    "algorithm",
    "epsilon",
    "tol",
    "max_iter",
    "iters",
    "time_s",
    "final_marg_error",
    "objective_eot",
    "status",
    "error_msg",
]


def _write_csv(path: str, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, np.nan) for col in columns})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OT experiments and write CSV results.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--time_limit", type=float, default=None)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    instance_s = ot_instances.build_instance_S(seed=args.seed)
    instance_d = ot_instances.build_instance_D(seed=args.seed)

    lp_rows: list[dict[str, Any]] = []
    admm_rows: list[dict[str, Any]] = []
    eot_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()

    def add_status(result: dict[str, Any]) -> None:
        status_counts[result.get("status", "UNKNOWN")] += 1

    lp_specs = [
        (instance_s, "SIMPLEX", lp_gurobi.solve_ot_lp_gurobi),
        (instance_s, "INTERIOR_POINT", lp_gurobi.solve_ot_lp_gurobi),
        (instance_s, "SIMPLEX", lp_mosek.solve_ot_lp_mosek),
        (instance_s, "INTERIOR_POINT", lp_mosek.solve_ot_lp_mosek),
        (instance_d, "SIMPLEX", lp_gurobi.solve_ot_lp_gurobi),
        (instance_d, "INTERIOR_POINT", lp_gurobi.solve_ot_lp_gurobi),
        (instance_d, "SIMPLEX", lp_mosek.solve_ot_lp_mosek),
        (instance_d, "INTERIOR_POINT", lp_mosek.solve_ot_lp_mosek),
    ]

    for inst, method, solver_fn in lp_specs:
        result = solver_fn(
            inst["C"],
            inst["mu"],
            inst["nu"],
            method,
            threads=args.threads,
            time_limit=args.time_limit,
            verbose=args.verbose,
        )
        add_status(result)
        row = {
            "instance": inst["instance"],
            "m": inst["m"],
            "n": inst["n"],
        }
        row.update(result)
        lp_rows.append(row)

    admm_specs = [
        (instance_s, "S"),
        (instance_d, "D"),
    ]

    for inst, label in admm_specs:
        result = admm_ot.solve_ot_admm(
            inst["C"],
            inst["mu"],
            inst["nu"],
            rho=5.0,
            tol=1e-6,
            max_iter=5000,
            proj_iters=10,
            verbose=args.verbose,
        )
        add_status(result)
        row = {
            "instance": label,
            "m": inst["m"],
            "n": inst["n"],
        }
        row.update(result)
        admm_rows.append(row)

    eot_specs = [
        (instance_d, "SINKHORN", 0.05),
        (instance_d, "SINKHORN", 0.01),
        (instance_d, "DUAL_GRAD", 0.05),
        (instance_d, "DUAL_GRAD", 0.01),
    ]

    for inst, algorithm, epsilon in eot_specs:
        if algorithm == "SINKHORN":
            result = sinkhorn.solve_eot_sinkhorn(
                inst["C"],
                inst["mu"],
                inst["nu"],
                epsilon=epsilon,
                tol=1e-6,
                max_iter=20000,
                stabilization=True,
                verbose=args.verbose,
            )
        else:
            result = dual_grad_eot.solve_eot_dual_grad(
                inst["C"],
                inst["mu"],
                inst["nu"],
                epsilon=epsilon,
                tol=1e-6,
                max_iter=50000,
                verbose=args.verbose,
            )
        add_status(result)
        row = {
            "instance": inst["instance"],
            "m": inst["m"],
            "n": inst["n"],
        }
        row.update(result)
        eot_rows.append(row)

    _write_csv(os.path.join(args.out_dir, "results_lp.csv"), LP_COLUMNS, lp_rows)
    _write_csv(os.path.join(args.out_dir, "results_admm.csv"), ADMM_COLUMNS, admm_rows)
    _write_csv(os.path.join(args.out_dir, "results_eot.csv"), EOT_COLUMNS, eot_rows)

    print("Summary status counts:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()
