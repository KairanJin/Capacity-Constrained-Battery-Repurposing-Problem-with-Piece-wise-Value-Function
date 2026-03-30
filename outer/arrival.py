# outer/arrival.py
from __future__ import annotations
import numpy as np


def gaussian_arrival_generator(
    rng: np.random.Generator,
    n_arrivals: int,
    mu_C: float,
    sigma_C: float,
    mu_R: float,
    sigma_R: float,
) -> np.ndarray:
    """
    Generate one future arrival batch A_t.
    Output shape: (n_arrivals, 2)
    """
    C = rng.normal(mu_C, sigma_C, size=n_arrivals)
    R = rng.normal(mu_R, sigma_R, size=n_arrivals)

    C = np.maximum(C, 1e-6)
    R = np.maximum(R, 1e-6)

    return np.column_stack([C, R]).astype(float)