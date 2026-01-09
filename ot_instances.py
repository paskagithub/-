import numpy as np


def _normalize_histogram(hist: np.ndarray) -> np.ndarray:
    hist = np.maximum(hist, 0.0)
    total = float(hist.sum())
    if total <= 0.0:
        raise ValueError("Histogram must have positive mass.")
    hist = hist / total
    if not np.all(hist >= 0.0):
        raise AssertionError("Histogram has negative entries after normalization.")
    if not np.isclose(hist.sum(), 1.0, atol=1e-12):
        raise AssertionError("Histogram does not sum to 1 after normalization.")
    return hist


def _pairwise_squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_norm = (x ** 2).sum(axis=1, keepdims=True)
    y_norm = (y ** 2).sum(axis=1, keepdims=True)
    return x_norm + y_norm.T - 2.0 * x @ y.T


def build_instance_S(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    m = n = 200
    x = rng.random((m, 2), dtype=np.float64)
    y = rng.random((n, 2), dtype=np.float64)
    C = _pairwise_squared_distances(x, y).astype(np.float64)
    mu = _normalize_histogram(rng.random(m, dtype=np.float64))
    nu = _normalize_histogram(rng.random(n, dtype=np.float64))
    meta = {
        "seed": seed,
        "dim": 2,
        "description": "Synthetic geometric OT instance",
    }
    return {
        "instance": "S",
        "C": C,
        "mu": mu,
        "nu": nu,
        "m": m,
        "n": n,
        "meta": meta,
    }


def _gaussian_blobs(grid: np.ndarray, centers: np.ndarray, weights: np.ndarray, sigma: float) -> np.ndarray:
    diffs = grid[:, None, :] - centers[None, :, :]
    dist2 = (diffs ** 2).sum(axis=2)
    blobs = np.exp(-dist2 / (2.0 * sigma ** 2))
    return (blobs * weights[None, :]).sum(axis=1)


def build_instance_D(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    grid_size = 16
    m = n = grid_size * grid_size
    coords_1d = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    xx, yy = np.meshgrid(coords_1d, coords_1d, indexing="xy")
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)

    sigma = 0.08
    centers_mu = rng.random((3, 2), dtype=np.float64)
    centers_nu = rng.random((3, 2), dtype=np.float64)
    weights_mu = rng.random(3, dtype=np.float64)
    weights_mu = weights_mu / weights_mu.sum()
    weights_nu = rng.random(3, dtype=np.float64)
    weights_nu = weights_nu / weights_nu.sum()

    mu_raw = _gaussian_blobs(grid, centers_mu, weights_mu, sigma)
    nu_raw = _gaussian_blobs(grid, centers_nu, weights_nu, sigma)
    mu = _normalize_histogram(mu_raw)
    nu = _normalize_histogram(nu_raw)

    C = _pairwise_squared_distances(grid, grid).astype(np.float64)

    meta = {
        "seed": seed,
        "dim": 2,
        "sigma": sigma,
        "centers_mu": centers_mu,
        "centers_nu": centers_nu,
        "weights_mu": weights_mu,
        "weights_nu": weights_nu,
        "grid_size": grid_size,
        "description": "DOTmark-like histogram OT instance",
    }

    return {
        "instance": "D",
        "C": C,
        "mu": mu,
        "nu": nu,
        "m": m,
        "n": n,
        "meta": meta,
    }
