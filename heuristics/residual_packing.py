# heuristics/residual_packing.py
import numpy as np

from utils import compute_delta, piecewise_value
from heuristics._grasp_stats import GroupStats, sq_dist_to_centroid


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

    mu = X[group].mean(axis=0)
    phi = float(np.dot(w, mu))
    value = piecewise_value(phi, theta1, theta2, theta3, P1, P2, P3)
    reward = value - lambda_penalty * delta
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
    neighbor_candidate_limit=20,
    min_accept_reward=0.0,
    X_sq_norms=None,
    w_dot_X=None,
):
    """
    Find the best positive-profit feasible pack from leftover.
    Uses greedy construction with multiple random restarts instead of
    exhaustive combination enumeration.
    """
    if len(leftover) < K:
        return None, None

    use_incremental = X_sq_norms is not None and w_dot_X is not None

    ranked_seeds = sorted(
        leftover,
        key=lambda i: w_dot_X[i] if use_incremental else _cell_quality(X, i, w),
        reverse=True
    )[:min(seed_candidate_limit, len(leftover))]

    best_group = None
    best_info = None
    best_reward = min_accept_reward

    for seed in ranked_seeds:
        others = [c for c in leftover if c != seed]

        if use_incremental:
            seed_sq = X_sq_norms[seed]
            scored_neighbors = []
            for c in others:
                dist = X_sq_norms[c] + seed_sq - 2.0 * (X[c] @ X[seed])
                score = -dist + 0.05 * w_dot_X[c]
                scored_neighbors.append((score, c))
        else:
            scored_neighbors = []
            for c in others:
                dist = float(np.sum((X[c] - X[seed]) ** 2))
                score = -dist + 0.05 * _cell_quality(X, c, w)
                scored_neighbors.append((score, c))

        scored_neighbors.sort(reverse=True)
        candidate_pool = [c for _, c in scored_neighbors[:min(neighbor_candidate_limit, len(scored_neighbors))]]

        if len(candidate_pool) < K - 1:
            continue

        # Greedy construction with multiple restarts
        n_greedy_tries = min(10, len(candidate_pool))
        for try_idx in range(n_greedy_tries):
            if try_idx == 0:
                ordered = candidate_pool[:]
            else:
                ordered = list(
                    np.random.default_rng(try_idx * 1000 + hash(seed) % 10000)
                    .choice(candidate_pool, size=len(candidate_pool), replace=False)
                )

            if use_incremental:
                group = [seed]
                stats = GroupStats()
                stats.add(seed, X, X_sq_norms, w_dot_X)
                remaining = ordered[:]

                for step in range(K - 1):
                    best_gain = -np.inf
                    best_cell = None
                    for c in remaining:
                        temp = stats.clone()
                        temp.add(c, X, X_sq_norms, w_dot_X)
                        gain = temp.partial_score(lambda_penalty)
                        if gain > best_gain:
                            best_gain = gain
                            best_cell = c

                    if best_cell is None:
                        break
                    group.append(best_cell)
                    stats.add(best_cell, X, X_sq_norms, w_dot_X)
                    remaining.remove(best_cell)

                if len(group) == K:
                    g = sorted(group)
                    delta = stats.delta
                    if delta <= delta_bar:
                        phi = stats.phi
                        value = piecewise_value(phi, theta1, theta2, theta3, P1, P2, P3)
                        reward = value - lambda_penalty * delta
                        if reward > best_reward:
                            best_reward = reward
                            best_group = g
                            best_info = {"feasible": True, "reward": reward,
                                         "phi": phi, "delta": delta}
            else:
                # Fallback: greedy without incremental stats
                group = [seed]
                remaining = ordered[:]
                for step in range(K - 1):
                    best_gain = -np.inf
                    best_cell = None
                    for c in remaining:
                        test_group = group + [c]
                        if len(test_group) >= 2:
                            mu = X[test_group].mean(axis=0)
                            delta = float(np.mean(np.sum((X[test_group] - mu) ** 2, axis=1)))
                            phi = float(np.dot(w, mu))
                            gain = phi - lambda_penalty * delta
                        else:
                            gain = float(np.dot(w, X[c]))
                        if gain > best_gain:
                            best_gain = gain
                            best_cell = c
                    if best_cell is None:
                        break
                    group.append(best_cell)
                    remaining.remove(best_cell)

                if len(group) == K:
                    g = sorted(group)
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
    neighbor_candidate_limit=20,
    max_rounds=100,
    X_sq_norms=None,
    w_dot_X=None,
):
    """
    Residual Packing Phase:
    repeatedly extract positive-profit feasible packs from leftover only.
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
            X_sq_norms=X_sq_norms,
            w_dot_X=w_dot_X,
        )

        if best_group is None:
            break

        groups.append(best_group)
        used = set(best_group)
        leftover = [c for c in leftover if c not in used]

    return groups, leftover