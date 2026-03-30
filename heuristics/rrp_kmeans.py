# heuristics/rrp_kmeans.py
import time
import numpy as np

from utils import (
    compute_centroid,
    compute_delta,
    compute_phi,
    compute_group_reward,
)


def _group_reward_if_feasible(
    X,
    group,
    K,
    delta_bar,
    w,
    lambda_penalty,
    theta1,
    theta2,
    theta3,
    P1,
    P2,
    P3,
):
    if len(group) != K:
        return -np.inf, None, None
    delta = compute_delta(X, group)
    if delta > delta_bar:
        return -np.inf, None, delta

    reward, phi, delta = compute_group_reward(
        X, group, w, lambda_penalty, theta1, theta2, theta3, P1, P2, P3
    )
    return reward, phi, delta


def _assign_to_nearest_nonfull(X, centers, K):
    n = X.shape[0]
    k = len(centers)

    groups = [[] for _ in range(k)]
    leftover = []

    for i in range(n):
        dists = np.sum((centers - X[i]) ** 2, axis=1)
        order = np.argsort(dists)

        assigned = False
        for j in order:
            if len(groups[j]) < K:
                groups[j].append(i)
                assigned = True
                break

        if not assigned:
            leftover.append(i)

    return groups, leftover


def _update_centers(X, groups, centers):
    new_centers = centers.copy()
    for j, g in enumerate(groups):
        if len(g) > 0:
            new_centers[j] = compute_centroid(X, g)
    return new_centers


def _recluster_incomplete_groups(X, groups, K, seed=0):
    incomplete_cells = []
    full_groups = []

    for g in groups:
        if len(g) == K:
            full_groups.append(g)
        else:
            incomplete_cells.extend(g)

    if len(incomplete_cells) < K:
        return groups

    n_new = len(incomplete_cells) // K
    if n_new == 0:
        return groups

    subX = X[incomplete_cells]
    rng = np.random.default_rng(seed)
    init_idx = rng.choice(len(incomplete_cells), size=n_new, replace=False)
    centers = subX[init_idx].copy()

    new_groups_local, rem_local = _assign_to_nearest_nonfull(subX, centers, K)

    new_groups = []
    for g in full_groups:
        new_groups.append(g)

    for g in new_groups_local:
        mapped = [incomplete_cells[idx] for idx in g]
        new_groups.append(mapped)

    leftover = [incomplete_cells[idx] for idx in rem_local]
    if leftover:
        new_groups.append(leftover)

    return new_groups


def _try_reward_improving_swap(
    X,
    groups,
    K,
    delta_bar,
    w,
    lambda_penalty,
    theta1,
    theta2,
    theta3,
    P1,
    P2,
    P3,
):
    """
    在满组之间做 swap。
    接受条件：交换后两个组都仍可行，且两组 reward 之和严格增加。
    """
    full_group_ids = [idx for idx, g in enumerate(groups) if len(g) == K]
    if len(full_group_ids) <= 1:
        return groups, False

    # 只在 full groups 上构建 membership
    group_of = {}
    for gid in full_group_ids:
        for cell in groups[gid]:
            group_of[cell] = gid

    for z in range(X.shape[0]):
        if z not in group_of:
            continue

        r = group_of[z]
        g_r = groups[r]

        mu_r = compute_centroid(X, g_r)
        dist_r = float(np.sum((X[z] - mu_r) ** 2))

        # 候选组先按“更近”排序，但真正接受依据是 reward increase
        candidate_groups = []
        for r2 in full_group_ids:
            if r2 == r:
                continue
            mu_r2 = compute_centroid(X, groups[r2])
            dist_r2 = float(np.sum((X[z] - mu_r2) ** 2))
            if dist_r2 < dist_r:
                candidate_groups.append((dist_r2, r2))

        candidate_groups.sort(key=lambda x: x[0])

        old_reward_r, _, _ = _group_reward_if_feasible(
            X, g_r, K, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3
        )

        for _, r2 in candidate_groups:
            g_r2 = groups[r2]
            old_reward_r2, _, _ = _group_reward_if_feasible(
                X, g_r2, K, delta_bar, w, lambda_penalty,
                theta1, theta2, theta3, P1, P2, P3
            )

            best_gain = 0.0
            best_swap_cell = None
            best_g1 = None
            best_g2 = None

            for zh in g_r2:
                new_g1 = [zh if x == z else x for x in g_r]
                new_g2 = [z if x == zh else x for x in g_r2]

                new_reward_r, _, _ = _group_reward_if_feasible(
                    X, new_g1, K, delta_bar, w, lambda_penalty,
                    theta1, theta2, theta3, P1, P2, P3
                )
                new_reward_r2, _, _ = _group_reward_if_feasible(
                    X, new_g2, K, delta_bar, w, lambda_penalty,
                    theta1, theta2, theta3, P1, P2, P3
                )

                if np.isfinite(new_reward_r) and np.isfinite(new_reward_r2):
                    gain = (new_reward_r + new_reward_r2) - (old_reward_r + old_reward_r2)
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best_swap_cell = zh
                        best_g1 = new_g1
                        best_g2 = new_g2

            if best_swap_cell is not None:
                groups[r] = best_g1
                groups[r2] = best_g2
                return groups, True

    return groups, False


def solve_rrp_kmeans(
    X: np.ndarray,
    K: int,
    k_t: int,
    delta_bar: float,
    L1: int,
    L2: int,
    tol: float,
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
    start = time.perf_counter()

    n = X.shape[0]
    if n < K or k_t <= 0:
        return {
            "method": "RRP_KMEANS",
            "groups": [],
            "leftover": list(range(n)),
            "reward": 0.0,
            "n_packs": 0,
            "avg_delta": 0.0,
            "avg_phi": 0.0,
            "runtime": time.perf_counter() - start,
        }

    rng = np.random.default_rng(seed)
    init_idx = rng.choice(n, size=k_t, replace=False)
    centers = X[init_idx].copy()

    groups = [[] for _ in range(k_t)]
    leftover = list(range(n))

    # Stage 1: SSE-style initialization
    for _ in range(L1):
        groups, leftover = _assign_to_nearest_nonfull(X, centers, K)
        new_centers = _update_centers(X, groups, centers)

        shift = np.max(np.linalg.norm(new_centers - centers, axis=1))
        centers = new_centers
        if shift < tol:
            break

    # regroup incomplete groups
    while True:
        incomplete_total = sum(len(g) for g in groups if len(g) < K)
        if incomplete_total < K:
            break
        new_groups = _recluster_incomplete_groups(X, groups, K, seed=seed if seed is not None else 0)
        if len(new_groups) == len(groups):
            break
        groups = new_groups

    # Stage 2: reward-improving swap
    for _ in range(L2):
        groups, improved = _try_reward_improving_swap(
            X, groups, K, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3
        )
        if not improved:
            break

    # final feasibility filter
    feasible_groups = []
    used = set()

    rewards = []
    phis = []
    deltas = []

    for g in groups:
        if len(g) != K:
            continue
        delta = compute_delta(X, g)
        if delta <= delta_bar:
            reward, phi, delta = compute_group_reward(
                X, g, w, lambda_penalty, theta1, theta2, theta3, P1, P2, P3
            )
            feasible_groups.append(g)
            used.update(g)
            rewards.append(reward)
            phis.append(phi)
            deltas.append(delta)

    leftover = [i for i in range(n) if i not in used]

    return {
        "method": "RRP_KMEANS",
        "groups": feasible_groups,
        "leftover": leftover,
        "reward": float(np.sum(rewards)) if rewards else 0.0,
        "n_packs": len(feasible_groups),
        "avg_delta": float(np.mean(deltas)) if deltas else 0.0,
        "avg_phi": float(np.mean(phis)) if phis else 0.0,
        "runtime": time.perf_counter() - start,
    }