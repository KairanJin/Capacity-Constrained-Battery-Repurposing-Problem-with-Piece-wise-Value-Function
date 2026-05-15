# heuristics/_grasp_stats.py
import numpy as np


class GroupStats:
    """Incremental tracker for group centroid, delta, and phi. O(d) updates."""
    __slots__ = ('sum_vec', 'sum_sq', 'w_dot_sum', 'count')

    def __init__(self):
        self.sum_vec = None   # np.ndarray, shape (d,)
        self.sum_sq = 0.0     # scalar
        self.w_dot_sum = 0.0  # scalar
        self.count = 0        # int

    def add(self, cell_idx, X, X_sq_norms, w_dot_X):
        if self.count == 0:
            self.sum_vec = X[cell_idx].copy()
        else:
            self.sum_vec += X[cell_idx]
        self.sum_sq += X_sq_norms[cell_idx]
        self.w_dot_sum += w_dot_X[cell_idx]
        self.count += 1

    def remove(self, cell_idx, X, X_sq_norms, w_dot_X):
        self.sum_vec -= X[cell_idx]
        self.sum_sq -= X_sq_norms[cell_idx]
        self.w_dot_sum -= w_dot_X[cell_idx]
        self.count -= 1

    @property
    def centroid(self):
        return self.sum_vec / self.count

    @property
    def delta(self):
        n = self.count
        return self.sum_sq / n - (self.sum_vec @ self.sum_vec) / (n * n)

    @property
    def phi(self):
        return self.w_dot_sum / self.count

    def partial_score(self, lambda_penalty):
        if self.count == 0:
            return 0.0
        d = self.delta if self.count >= 2 else 0.0
        return self.phi - lambda_penalty * d

    def clone(self):
        st = GroupStats()
        st.sum_vec = self.sum_vec.copy() if self.sum_vec is not None else None
        st.sum_sq = self.sum_sq
        st.w_dot_sum = self.w_dot_sum
        st.count = self.count
        return st


def precompute_arrays(X, w):
    """O(N*d). Returns (X_sq_norms, w_dot_X) for O(1) lookups."""
    X_sq_norms = np.sum(X ** 2, axis=1)
    w_dot_X = X @ np.asarray(w)
    return X_sq_norms, w_dot_X


def sq_dist_to_centroid(cell_idx, centroid, X_sq_norms, X):
    """||X[c] - mu||^2 = ||X[c]||^2 - 2*X[c].mu + ||mu||^2. O(d)."""
    mu_sq = centroid @ centroid
    return X_sq_norms[cell_idx] - 2.0 * (X[cell_idx] @ centroid) + mu_sq
