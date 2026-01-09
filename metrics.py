import numpy as np


def _validate_matrix_shapes(C: np.ndarray, pi: np.ndarray) -> tuple[int, int]:
    if C.ndim != 2 or pi.ndim != 2:
        raise ValueError("C and pi must be 2D arrays.")
    if C.shape != pi.shape:
        raise ValueError("C and pi must have matching shapes.")
    return C.shape


def _validate_marginals(mu: np.ndarray, nu: np.ndarray, m: int, n: int) -> None:
    if mu.ndim != 1 or nu.ndim != 1:
        raise ValueError("mu and nu must be 1D arrays.")
    if mu.shape[0] != m or nu.shape[0] != n:
        raise ValueError("mu and nu must match pi dimensions.")


def ot_objective(C: np.ndarray, pi: np.ndarray) -> float:
    _validate_matrix_shapes(C, pi)
    return float(np.sum(C.astype(np.float64) * pi.astype(np.float64)))


def residuals(mu: np.ndarray, nu: np.ndarray, pi: np.ndarray) -> dict:
    if pi.ndim != 2:
        raise ValueError("pi must be a 2D array.")
    m, n = pi.shape
    _validate_marginals(mu, nu, m, n)

    pi = pi.astype(np.float64)
    mu = mu.astype(np.float64)
    nu = nu.astype(np.float64)

    row_sum = pi.sum(axis=1)
    col_sum = pi.sum(axis=0)
    r_row_l1 = float(np.sum(np.abs(row_sum - mu)))
    r_col_l1 = float(np.sum(np.abs(col_sum - nu)))
    min_pi = float(np.min(pi))
    r_nn = max(0.0, -min_pi)

    return {
        "r_row_l1": r_row_l1,
        "r_col_l1": r_col_l1,
        "r_nn": r_nn,
        "row_sum": row_sum,
        "col_sum": col_sum,
        "min_pi": min_pi,
    }


def eot_objective(C: np.ndarray, pi: np.ndarray, epsilon: float) -> float:
    _validate_matrix_shapes(C, pi)
    C = C.astype(np.float64)
    pi = pi.astype(np.float64)
    entropy = np.sum(pi * (np.log(pi + 1e-300) - 1.0))
    return float(np.sum(C * pi) + float(epsilon) * entropy)


def marginal_error(mu: np.ndarray, nu: np.ndarray, pi: np.ndarray) -> float:
    if pi.ndim != 2:
        raise ValueError("pi must be a 2D array.")
    m, n = pi.shape
    _validate_marginals(mu, nu, m, n)
    row_sum = pi.astype(np.float64).sum(axis=1)
    col_sum = pi.astype(np.float64).sum(axis=0)
    mu = mu.astype(np.float64)
    nu = nu.astype(np.float64)
    return float(np.sum(np.abs(row_sum - mu)) + np.sum(np.abs(col_sum - nu)))
