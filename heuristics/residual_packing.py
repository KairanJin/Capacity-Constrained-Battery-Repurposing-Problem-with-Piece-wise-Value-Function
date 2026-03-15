# heuristics/residual_packing.py
import itertools
import numpy as np

from utils import compute_group_reward, compute_delta


def _evaluate_group(
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
        return {
            "feasible": False,
            "reward": -np.inf,
            "phi": -np.inf,
            "delta": np.inf,
        }

    delta = compute_delta(X, group)
    if delta > delta_bar:
        return {
            "feasible": False,
            "reward": -np.inf,
            "phi": -np.inf,
            "delta": delta,
        }

    reward, phi, delta = compute_group_reward(
        X, group, w, lambda_penalty, theta1, theta2, theta3, P1, P2, P3
    )
    return {
        "feasible": True,
        "reward": reward,
        "phi": phi,
        "delta": delta,
    }


def _cell_quality(X, cell, w):
    return float(np.dot(w, X[cell]))


def _find_best_residual_pack(
    X,
    leftover,
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
    seed_candidate_limit=12,
    neighbor_candidate_limit=12,
    min_accept_reward=0.0,
):
    """
    在 leftover 中找一个最好的正收益 feasible pack。
    返回:
        best_group, best_info
    若不存在则返回 (None, None)
    """
    if len(leftover) < K:
        return None, None

    # 优先从高质量 cell 开始作为 seed
    ranked_seeds = sorted(
        leftover,
        key=lambda i: _cell_quality(X, i, w),
        reverse=True
    )[:min(seed_candidate_limit, len(leftover))]

    best_group = None
    best_info = None
    best_reward = min_accept_reward

    leftover_set = set(leftover)

    for seed in ranked_seeds:
        others = [c for c in leftover if c != seed]

        # 从 seed 的邻域挑候选
        scored_neighbors = []
        for c in others:
            dist = float(np.sum((X[c] - X[seed]) ** 2))
            score = -dist + 0.05 * _cell_quality(X, c, w)
            scored_neighbors.append((score, c))
        scored_neighbors.sort(reverse=True)

        candidate_pool = [c for _, c in scored_neighbors[:min(neighbor_candidate_limit, len(scored_neighbors))]]

        if len(candidate_pool) < K - 1:
            continue

        for comb in itertools.combinations(candidate_pool, K - 1):
            g = sorted([seed] + list(comb))
            info = _evaluate_group(
                X, g, K, delta_bar, w, lambda_penalty,
                theta1, theta2, theta3, P1, P2, P3
            )
            if info["feasible"] and info["reward"] > best_reward:
                best_reward = info["reward"]
                best_group = g
                best_info = info

    return best_group, best_info


def residual_pack_repair(
    X,
    groups,
    leftover,
    K,
    k_t,
    delta_bar,
    w,
    lambda_penalty,
    theta1,
    theta2,
    theta3,
    P1,
    P2,
    P3,
    min_accept_reward=0.0,
    seed_candidate_limit=12,
    neighbor_candidate_limit=12,
    max_rounds=100,
):
    """
    Residual Packing Phase:
    repeatedly extract positive-profit feasible packs from leftover only.

    Parameters
    ----------
    groups : list[list[int]]
        Existing packs. Will be preserved.
    leftover : list[int]
        Remaining cells.
    k_t : int
        Maximum number of packs allowed in total.
    min_accept_reward : float
        Only accept a new pack if reward > min_accept_reward.

    Returns
    -------
    new_groups, new_leftover
    """
    groups = [sorted(g) for g in groups]
    leftover = sorted(leftover)

    rounds = 0
    while rounds < max_rounds:
        rounds += 1

        if len(groups) >= k_t:
            break
        if len(leftover) < K:
            break

        best_group, best_info = _find_best_residual_pack(
            X=X,
            leftover=leftover,
            K=K,
            delta_bar=delta_bar,
            w=w,
            lambda_penalty=lambda_penalty,
            theta1=theta1,
            theta2=theta2,
            theta3=theta3,
            P1=P1,
            P2=P2,
            P3=P3,
            seed_candidate_limit=seed_candidate_limit,
            neighbor_candidate_limit=neighbor_candidate_limit,
            min_accept_reward=min_accept_reward,
        )

        if best_group is None:
            break

        groups.append(best_group)
        used = set(best_group)
        leftover = [c for c in leftover if c not in used]

    return groups, leftover