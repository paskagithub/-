import time

import numpy as np

import metrics


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    stabilized = a - a_max
    return np.squeeze(a_max, axis=axis) + np.log(np.sum(np.exp(stabilized), axis=axis))


def solve_eot_sinkhorn(
    C: np.ndarray,
    mu: np.ndarray,
    nu: np.ndarray,
    epsilon: float,
    tol: float = 1e-6,
    max_iter: int = 20000,
    stabilization: bool = True,
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
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    start = time.perf_counter()
    status = "MAX_ITER"
    error_msg = None
    iters = 0
    pi = None

    try:
        if stabilization:
            log_u = np.zeros(m, dtype=np.float64)
            log_v = np.zeros(n, dtype=np.float64)
            log_mu = np.log(mu + 1e-300)
            log_nu = np.log(nu + 1e-300)
            log_K = -C / float(epsilon)

            for k in range(1, max_iter + 1):
                log_u = log_mu - _logsumexp(log_K + log_v[None, :], axis=1)
                log_v = log_nu - _logsumexp(log_K.T + log_u[None, :], axis=1)

                log_pi = log_u[:, None] + log_K + log_v[None, :]
                pi = np.exp(log_pi)
                err = metrics.marginal_error(mu, nu, pi)

                if verbose and (k == 1 or k % 100 == 0 or err <= tol):
                    print(f"iter={k} err={err:.3e}")
                if err <= tol:
                    status = "CONVERGED"
                    iters = k
                    break
            else:
                iters = max_iter
        else:
            K = np.exp(-C / float(epsilon)).astype(np.float64)
            u = np.ones(m, dtype=np.float64) / m
            v = np.ones(n, dtype=np.float64) / n

            for k in range(1, max_iter + 1):
                u = mu / (K @ v + 1e-300)
                v = nu / (K.T @ u + 1e-300)
                pi = (u[:, None] * K) * v[None, :]
                err = metrics.marginal_error(mu, nu, pi)

                if verbose and (k == 1 or k % 100 == 0 or err <= tol):
                    print(f"iter={k} err={err:.3e}")
                if err <= tol:
                    status = "CONVERGED"
                    iters = k
                    break
            else:
                iters = max_iter

    except Exception as exc:
        status = "ERROR"
        error_msg = str(exc)

    time_s = time.perf_counter() - start
    final_marg_error = metrics.marginal_error(mu, nu, pi) if pi is not None else np.nan
    objective_eot = metrics.eot_objective(C, pi, epsilon) if pi is not None else np.nan

    return {
        "algorithm": "SINKHORN",
        "epsilon": float(epsilon),
        "tol": float(tol),
        "max_iter": int(max_iter),
        "iters": int(iters),
        "time_s": time_s,
        "final_marg_error": float(final_marg_error),
        "objective_eot": float(objective_eot),
        "status": status,
        "error_msg": error_msg,
    }
