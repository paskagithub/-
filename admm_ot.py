import time

import numpy as np

import metrics


def _project_marginals(z: np.ndarray, mu: np.ndarray, nu: np.ndarray, iters: int) -> np.ndarray:
    m, n = z.shape
    for _ in range(iters):
        row_sums = z.sum(axis=1)
        z += ((mu - row_sums) / n)[:, None]
        col_sums = z.sum(axis=0)
        z += ((nu - col_sums) / m)[None, :]
    row_sums = z.sum(axis=1)
    z += ((mu - row_sums) / n)[:, None]
    return z


def solve_ot_admm(
    C: np.ndarray,
    mu: np.ndarray,
    nu: np.ndarray,
    rho: float = 5.0,
    tol: float = 1e-6,
    max_iter: int = 5000,
    proj_iters: int = 10,
    verbose: bool = False,
) -> dict:
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

    z = np.outer(mu, nu).astype(np.float64)
    u = np.zeros_like(z)

    start = time.perf_counter()
    status = "MAX_ITER"
    error_msg = None
    iters_used = 0

    try:
        for k in range(1, max_iter + 1):
            pi = np.maximum(0.0, z - u - C / rho)
            Q = pi + u
            z = _project_marginals(Q, mu, nu, proj_iters)
            u = u + pi - z

            err = metrics.marginal_error(mu, nu, z)
            if verbose and (k == 1 or k % 100 == 0 or err <= tol):
                print(f"iter={k} err={err:.3e}")
            if err <= tol:
                status = "CONVERGED"
                iters_used = k
                break
        else:
            iters_used = max_iter
    except Exception as exc:
        status = "ERROR"
        error_msg = str(exc)
        iters_used = k if "k" in locals() else 0

    total_time_s = time.perf_counter() - start

    res = metrics.residuals(mu, nu, z)
    pi_min = float(np.min(pi)) if "pi" in locals() else np.nan
    r_nn = max(0.0, -pi_min) if "pi" in locals() else np.nan

    return {
        "rho": float(rho),
        "tol": float(tol),
        "max_iter": int(max_iter),
        "iters_used": int(iters_used),
        "total_time_s": total_time_s,
        "objective": metrics.ot_objective(C, z),
        "r_row_l1": res["r_row_l1"],
        "r_col_l1": res["r_col_l1"],
        "r_nn": r_nn,
        "status": status,
        "error_msg": error_msg,
    }
