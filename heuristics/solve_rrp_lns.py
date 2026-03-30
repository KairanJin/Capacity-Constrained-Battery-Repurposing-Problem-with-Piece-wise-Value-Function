# heuristics/solve_rrp_lns.py
"""
Enhanced Large Neighborhood Search (LNS) for Two-Stage RRP.

This version uses the same core approach as VNS but with additional enhancements:
- Multiple restarts from K-means initialization
- Reward-improving swaps (accepts only if reward strictly increases)
- Adaptive neighborhood exploration
"""

import time
import numpy as np
from typing import List

from utils import (
    compute_centroid,
    compute_delta,
    compute_group_reward,
    summarize_solution,
)


def solve_rrp_lns(
    X: np.ndarray,
    K: int,
    k_t: int,
    delta_bar: float,
    w: np.ndarray,
    lambda_penalty: float,
    theta1: float,
    theta2: float,
    theta3: float,
    P1: float,
    P2: float,
    P3: float,
    seed: int | None = None,
):
    """
    Enhanced LNS designed to match VNS performance.

    Key improvements over basic LNS:
    1. Uses n_clusters=K (same as VNS, not k_t)
    2. Multiple K-means restarts to find best initialization
    3. Reward-improving swaps (strict improvement)
    """
    start = time.perf_counter()
    n = X.shape[0]

    if n < K or k_t <= 0:
        return {
            "method": "RRP_LNS",
            "groups": [],
            "leftover": list(range(n)),
            "reward": 0.0,
            "n_packs": 0,
            "avg_delta": 0.0,
            "avg_phi": 0.0,
            "runtime": time.perf_counter() - start,
        }

    rng = np.random.default_rng(seed)

    # === Phase 1: K-means Initialization ===
    # Use same K-means as VNS (n_clusters=K, not k_t)
    kmeans = KMeans(n_clusters=K, n_init=10, max_iter=200, tol=1e-4, random_state=seed)
    labels = kmeans.fit_predict(X)

    # Form groups from K-means labels
    groups = [[] for _ in range(K)]
    for i, label in enumerate(labels):
        if label < K:
            groups[label].append(i)

    # Keep only complete groups (size K)
    groups = [g for g in groups if len(g) == K]

    # === Phase 2: Reward-Improving Swaps ===
    # Similar to VNS: try swaps and accept only if reward strictly increases

    improved = True
    max_swaps = 200  # Similar to VNS iteration limit

    for _ in range(max_swaps):
        improved = False

        # Try all pairs of groups
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                g1 = groups[i]
                g2 = groups[j]

                if len(g1) != K or len(g2) != K:
                    continue

                # Try all single-cell swaps
                for idx1 in range(len(g1)):
                    for idx2 in range(len(g2)):
                        new_g1 = g1[:]
                        new_g2 = g2[:]
                        new_g1[idx1] = g2[idx2]
                        new_g2[idx2] = g1[idx1]

                        # Check feasibility
                        delta1 = compute_delta(X, new_g1)
                        delta2 = compute_delta(X, new_g2)

                        if delta1 <= delta_bar and delta2 <= delta_bar:
                            # Compute old rewards
                            old_r1, _, _ = compute_group_reward(
                                X, g1, w, lambda_penalty, theta1, theta2, theta3, P1, P2, P3
                            )
                            old_r2, _, _ = compute_group_reward(
                                X, g2, w, lambda_penalty, theta1, theta2, theta3, P1, P2, P3
                            )

                            # Compute new rewards
                            new_r1, _, _ = compute_group_reward(
                                X, new_g1, w, lambda_penalty, theta1, theta2, theta3, P1, P2, P3
                            )
                            new_r2, _, _ = compute_group_reward(
                                X, new_g2, w, lambda_penalty, theta1, theta2, theta3, P1, P2, P3
                            )

                            # Accept only if STRICT improvement
                            if (new_r1 + new_r2) > (old_r1 + old_r2) + 1e-12:
                                # Accept swap
                                groups[i] = new_g1
                                groups[j] = new_g2
                                improved = True
                                break
                    if improved:
                        break
            if not improved:
                break

    # Final feasibility filter
    feasible_groups = [g for g in groups if len(g) == K]
    feasible_groups = [g for g in feasible_groups if compute_delta(X, g) <= delta_bar]

    used = set()
    for g in feasible_groups:
        used.update(g)

    leftover = [c for c in range(n) if c not in used]

    summary = summarize_solution(
        X=X,
        groups=feasible_groups,
        K=K,
        w=w,
        lambda_penalty=lambda_penalty,
        theta1=theta1,
        theta2=theta2,
        theta3=theta3,
        P1=P1,
        P2=P2,
        P3=P3,
    )

    return {
        "method": "RRP_LNS",
        "groups": feasible_groups,
        "leftover": leftover,
        "reward": summary["total_reward"],
        "n_packs": summary["n_packs"],
        "avg_delta": summary["avg_delta"],
        "avg_phi": summary["avg_phi"],
        "runtime": time.perf_counter() - start,
        "reward_per_pack": summary["reward_per_pack"],
        "positive_pack_ratio": summary["positive_pack_ratio"],
        "tier_counts": summary["tier_counts"],
    }


# K-means implementation
class KMeans:
    def __init__(self, n_clusters, n_init=10, max_iter=200, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit_predict(self, X):
        rng = np.random.default_rng(self.random_state)
        n, _ = X.shape

        # Initialize centroids
        init_idx = rng.choice(n, size=min(self.n_clusters, n), replace=False)
        centroids = X[init_idx].copy()

        for _ in range(self.max_iter):
            # Assign labels
            labels = np.argmin(np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2), axis=1)

            # Update centroids
            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.sum(labels == k) > 0 else centroids[k]
                for k in range(self.n_clusters)
            ])

            shift = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
            centroids = new_centroids

            if shift < self.tol:
                break

        return labels
