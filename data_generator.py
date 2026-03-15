import numpy as np


def generate_cells(
    n_cells: int,
    mu_C: float,
    sigma_C: float,
    mu_R: float,
    sigma_R: float,
    seed: int | None = None,
):
    rng = np.random.default_rng(seed)

    C = rng.normal(mu_C, sigma_C, size=n_cells)
    R = rng.normal(mu_R, sigma_R, size=n_cells)

    # truncate to positive values
    C = np.maximum(C, 1e-6)
    R = np.maximum(R, 1e-6)

    X = np.column_stack([C, R])
    return X