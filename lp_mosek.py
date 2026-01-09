import sys
import time

import numpy as np

import metrics

try:
    import mosek
    from mosek.fusion import Domain, Expr, Model, Matrix
except Exception as exc:  # pragma: no cover - handled at runtime
    mosek = None
    Domain = None
    Expr = None
    Model = None
    Matrix = None
    _MOSEK_IMPORT_ERROR = exc
else:
    _MOSEK_IMPORT_ERROR = None


def _base_result(method: str) -> dict:
    return {
        "solver": "MOSEK",
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


def _set_mosek_params(model: Model, method: str, threads: int, time_limit: float | None) -> None:
    if method == "SIMPLEX":
        model.setSolverParam("optimizer", "freeSimplex")
    if threads is not None:
        model.setSolverParam("numThreads", int(threads))
    if time_limit is not None:
        model.setSolverParam("timeLimit", float(time_limit))


def solve_ot_lp_mosek(
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

    if Model is None:
        result["status"] = "NO_SOLVER"
        result["error_msg"] = str(_MOSEK_IMPORT_ERROR or "mosek not available")
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
        model = Model()
        if verbose:
            model.setLogHandler(sys.stdout)
        _set_mosek_params(model, method, threads, time_limit)

        pi = model.variable("pi", [m, n], Domain.greaterThan(0.0))
        model.constraint("row_sum", Expr.sum(pi, 1), Domain.equalsTo(mu))
        model.constraint("col_sum", Expr.sum(pi, 0), Domain.equalsTo(nu))
        model.objective("objective", mosek.fusion.ObjectiveSense.Minimize, Expr.sum(Expr.mul(Matrix.dense(C), pi)))
        result["build_time_s"] = time.perf_counter() - build_start

        solve_start = time.perf_counter()
        model.solve()
        result["solve_time_s"] = time.perf_counter() - solve_start
        result["total_time_s"] = time.perf_counter() - total_start

        prob_status = model.getProblemStatus()
        if prob_status == mosek.fusion.ProblemStatus.PrimalAndDualFeasible:
            result["status"] = "OPTIMAL"
        elif prob_status == mosek.fusion.ProblemStatus.PrimalFeasible:
            result["status"] = "TIME_LIMIT"
        elif prob_status == mosek.fusion.ProblemStatus.PrimalInfeasible:
            result["status"] = "INFEASIBLE"
        else:
            result["status"] = "ERROR"

        sol_status = model.getPrimalSolutionStatus()
        if sol_status in {
            mosek.fusion.SolutionStatus.Optimal,
            mosek.fusion.SolutionStatus.NearOptimal,
            mosek.fusion.SolutionStatus.Feasible,
            mosek.fusion.SolutionStatus.NearFeasible,
        }:
            pi_value = np.array(pi.level(), dtype=np.float64).reshape(m, n)
            res = metrics.residuals(mu, nu, pi_value)
            result["r_row_l1"] = res["r_row_l1"]
            result["r_col_l1"] = res["r_col_l1"]
            result["r_nn"] = res["r_nn"]
            result["objective"] = metrics.ot_objective(C, pi_value)
            result["pi"] = pi_value

        model.dispose()
    except Exception as exc:
        message = str(exc)
        if "license" in message.lower():
            result["status"] = "NO_SOLVER"
        else:
            result["status"] = "ERROR"
        result["error_msg"] = message
        result["total_time_s"] = time.perf_counter() - total_start

    if result["error_msg"] is None and result["status"] == "ERROR":
        result["error_msg"] = "Unknown solver error"

    return result
