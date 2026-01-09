import time
from typing import Any

import numpy as np

import metrics

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as exc:  # pragma: no cover - handled at runtime
    gp = None
    GRB = None
    _GUROBI_IMPORT_ERROR = exc
else:
    _GUROBI_IMPORT_ERROR = None


def _base_result(method: str) -> dict:
    return {
        "solver": "GUROBI",
        "method": method,
        "build_time_s": 0.0,
        "solve_time_s": 0.0,
        "total_time_s": 0.0,
        "objective": np.nan,
        "r_row_l1": np.nan,
        "r_col_l1": np.nan,
        "r_nn": np.nan,
        "status": "ERROR",
        "error_msg": None,
    }


def solve_ot_lp_gurobi(
    C: np.ndarray,
    mu: np.ndarray,
    nu: np.ndarray,
    method: str,
    threads: int = 1,
    time_limit: float | None = None,
    verbose: bool = False,
) -> dict:
    method = method.upper()
    if method not in {"SIMPLEX", "INTERIOR_POINT"}:
        raise ValueError("method must be 'SIMPLEX' or 'INTERIOR_POINT'.")

    result = _base_result(method)

    if gp is None or GRB is None:
        result["status"] = "NO_SOLVER"
        result["error_msg"] = str(_GUROBI_IMPORT_ERROR or "gurobipy not available")
        return result

    C = np.asarray(C, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    nu = np.asarray(nu, dtype=np.float64)

    if C.ndim != 2:
        raise ValueError("C must be a 2D array.")
    if mu.ndim != 1 or nu.ndim != 1:
        raise ValueError("mu and nu must be 1D arrays.")
    m, n = C.shape
    if mu.shape[0] != m or nu.shape[0] != n:
        raise ValueError("mu and nu must match C dimensions.")

    total_start = time.perf_counter()
    try:
        build_start = time.perf_counter()
        model = gp.Model()
        model.Params.OutputFlag = 1 if verbose else 0
        model.Params.Threads = int(threads)
        model.Params.Method = 1 if method == "SIMPLEX" else 2
        if time_limit is not None:
            model.Params.TimeLimit = float(time_limit)

        pi = model.addMVar((m, n), lb=0.0, name="pi")
        model.addMConstr(pi.sum(axis=1), "=", mu, name="row_sum")
        model.addMConstr(pi.sum(axis=0), "=", nu, name="col_sum")
        model.setObjective((C * pi).sum(), GRB.MINIMIZE)
        model.update()
        result["build_time_s"] = time.perf_counter() - build_start

        solve_start = time.perf_counter()
        model.optimize()
        result["solve_time_s"] = time.perf_counter() - solve_start
        result["total_time_s"] = time.perf_counter() - total_start

        if model.Status == GRB.OPTIMAL:
            result["status"] = "OPTIMAL"
        elif model.Status == GRB.TIME_LIMIT:
            result["status"] = "TIME_LIMIT"
        elif model.Status == GRB.INFEASIBLE:
            result["status"] = "INFEASIBLE"
        else:
            result["status"] = "ERROR"

        pi_value = None
        if model.SolCount > 0 and model.Status in {GRB.OPTIMAL, GRB.TIME_LIMIT}:
            pi_value = pi.X.astype(np.float64)
            res = metrics.residuals(mu, nu, pi_value)
            result["r_row_l1"] = res["r_row_l1"]
            result["r_col_l1"] = res["r_col_l1"]
            result["r_nn"] = res["r_nn"]
            result["objective"] = metrics.ot_objective(C, pi_value)
            result["pi"] = pi_value

    except gp.GurobiError as exc:
        result["status"] = "ERROR"
        result["error_msg"] = str(exc)
        result["total_time_s"] = time.perf_counter() - total_start
    except Exception as exc:
        result["status"] = "ERROR"
        result["error_msg"] = str(exc)
        result["total_time_s"] = time.perf_counter() - total_start

    if result["error_msg"] is None and result["status"] == "ERROR":
        result["error_msg"] = "Unknown solver error"

    return result
