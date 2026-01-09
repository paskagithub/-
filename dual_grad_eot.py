import time

import numpy as np

import metrics


def solve_eot_dual_grad(
    C: np.ndarray,
    mu: np.ndarray,
    nu: np.ndarray,
    epsilon: float,
    tol: float = 1e-6,
    max_iter: int = 50000,
    eta0: float = 1.0,
    beta: float = 0.5,
    c: float = 1e-4,
    eta_min: float = 1e-12,
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

    alpha = np.zeros(m, dtype=np.float64)
    beta_vec = np.zeros(n, dtype=np.float64)

    start = time.perf_counter()
    status = "MAX_ITER"
    error_msg = None
    iters = 0
    pi = None

    try:
        for k in range(1, max_iter + 1):
            E = (alpha[:, None] + beta_vec[None, :] - C) / float(epsilon)
            max_E = np.max(E)
            E_shift = E - max_E
            pi_shift = np.exp(E_shift)
            scale = np.exp(max_E)
            pi = pi_shift * scale

            err = metrics.marginal_error(mu, nu, pi)
            if verbose and (k == 1 or k % 100 == 0 or err <= tol):
                print(f"iter={k} err={err:.3e}")
            if err <= tol:
                status = "CONVERGED"
                iters = k
                break

            g_alpha = mu - pi.sum(axis=1)
            g_beta = nu - pi.sum(axis=0)
            grad_norm_sq = float(np.sum(g_alpha ** 2) + np.sum(g_beta ** 2))

            dual_current = float(alpha @ mu + beta_vec @ nu - epsilon * scale * np.sum(pi_shift))

            eta = float(eta0)
            while eta >= eta_min:
                alpha_new = alpha + eta * g_alpha
                beta_new = beta_vec + eta * g_beta
                E_new = (alpha_new[:, None] + beta_new[None, :] - C) / float(epsilon)
                max_E_new = np.max(E_new)
                E_new_shift = E_new - max_E_new
                pi_new_shift = np.exp(E_new_shift)
                scale_new = np.exp(max_E_new)
                dual_new = float(alpha_new @ mu + beta_new @ nu - epsilon * scale_new * np.sum(pi_new_shift))

                if dual_new >= dual_current + c * eta * grad_norm_sq:
                    alpha = alpha_new
                    beta_vec = beta_new
                    break
                eta *= beta

            if eta < eta_min:
                status = "STEP_TOO_SMALL"
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
        "algorithm": "DUAL_GRAD",
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
