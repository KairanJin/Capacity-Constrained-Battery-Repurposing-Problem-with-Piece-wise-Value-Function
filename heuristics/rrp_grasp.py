# heuristics/rrp_grasp.py
import time
import itertools
import numpy as np

from utils import compute_centroid, compute_delta, compute_group_reward, summarize_solution
from heuristics.residual_packing import residual_pack_repair

# =========================================================
# Basic evaluation
# =========================================================

def _evaluate_partial_group(
    X: np.ndarray,
    group: list[int],
    w: np.ndarray,
) -> tuple[float, float]:
    """
    For a partial group, return:
    - phi_partial = w^T mu_G
    - delta_partial = mean squared deviation
    Used only for greedy guidance.
    """
    if len(group) == 0:
        return 0.0, 0.0
    mu = compute_centroid(X, group)
    phi = float(np.dot(w, mu))
    delta = compute_delta(X, group) if len(group) >= 2 else 0.0
    return phi, delta


def _evaluate_full_group(
    X: np.ndarray,
    group: list[int],
    K: int,
    delta_bar: float,
    w: np.ndarray,
    lambda_penalty: float,
    theta1: float,
    theta2: float,
    theta3: float,
    P1: float,
    P2: float,
    P3: float,
):
    """
    Full evaluation for a completed pack.
    """
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


def _group_reward_only(
    X: np.ndarray,
    group: list[int],
    K: int,
    delta_bar: float,
    w: np.ndarray,
    lambda_penalty: float,
    theta1: float,
    theta2: float,
    theta3: float,
    P1: float,
    P2: float,
    P3: float,
) -> float:
    info = _evaluate_full_group(
        X, group, K, delta_bar, w, lambda_penalty,
        theta1, theta2, theta3, P1, P2, P3
    )
    return info["reward"] if info["feasible"] else -np.inf


# =========================================================
# Construction phase
# =========================================================

def _partial_score(
    X: np.ndarray,
    group: list[int],
    w: np.ndarray,
    lambda_penalty: float,
) -> float:
    """
    Greedy score for a partial group.
    We cannot use the exact stepwise reward before reaching size K,
    so use a smooth proxy:
        score = phi_partial - lambda * delta_partial
    """
    phi, delta = _evaluate_partial_group(X, group, w)
    return phi - lambda_penalty * delta


def _marginal_gain_partial(
    X: np.ndarray,
    group: list[int],
    cell: int,
    w: np.ndarray,
    lambda_penalty: float,
) -> float:
    old_score = _partial_score(X, group, w, lambda_penalty)
    new_score = _partial_score(X, group + [cell], w, lambda_penalty)
    return new_score - old_score


def _build_one_group_grasp(
    X: np.ndarray,
    available: list[int],
    K: int,
    w: np.ndarray,
    lambda_penalty: float,
    rng: np.random.Generator,
    rcl_size: int,
) -> list[int]:
    """
    Build one group using GRASP construction:
    choose from RCL according to marginal gain.
    """
    if len(available) < K:
        return []

    # Start from the best-quality seed among a small candidate set
    qualities = [(float(np.dot(w, X[i])), i) for i in available]
    qualities.sort(reverse=True)
    seed_pool = [i for _, i in qualities[:min(rcl_size, len(qualities))]]
    first = int(rng.choice(seed_pool))

    group = [first]
    remaining = [i for i in available if i != first]

    while len(group) < K and len(remaining) > 0:
        gains = []
        for i in remaining:
            mg = _marginal_gain_partial(X, group, i, w, lambda_penalty)
            gains.append((mg, i))

        gains.sort(reverse=True, key=lambda x: x[0])

        # RCL from top candidates
        rcl = [i for _, i in gains[:min(rcl_size, len(gains))]]
        chosen = int(rng.choice(rcl))
        group.append(chosen)
        remaining.remove(chosen)

    return sorted(group) if len(group) == K else []


def _construction_phase(
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
    rng: np.random.Generator,
    rcl_size: int,
    max_group_attempts: int,
):
    """
    Iteratively construct up to k_t feasible groups.
    """
    available = list(range(X.shape[0]))
    groups = []

    attempts = 0
    while len(available) >= K and len(groups) < k_t and attempts < max_group_attempts:
        attempts += 1

        g = _build_one_group_grasp(
            X=X,
            available=available,
            K=K,
            w=w,
            lambda_penalty=lambda_penalty,
            rng=rng,
            rcl_size=rcl_size,
        )
        if len(g) < K:
            break

        info = _evaluate_full_group(
            X, g, K, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3
        )

        if info["feasible"] and info["reward"] > 0:
            groups.append(g)
            used = set(g)
            available = [i for i in available if i not in used]
        else:
            # avoid repeatedly rebuilding the exact same bad group:
            # remove one low-contribution cell from candidate consideration for this attempt
            qualities = [(float(np.dot(w, X[i])), i) for i in g]
            qualities.sort()
            worst = qualities[0][1]
            if worst in available:
                available.remove(worst)

    leftover = sorted(available)
    return groups, leftover


# =========================================================
# Local search
# =========================================================

def _swap_first_improvement(
    X: np.ndarray,
    groups: list[list[int]],
    K: int,
    delta_bar: float,
    w: np.ndarray,
    lambda_penalty: float,
    theta1: float,
    theta2: float,
    theta3: float,
    P1: float,
    P2: float,
    P3: float,
    group_candidate_limit: int,
    cell_candidate_limit: int,
) -> bool:
    """
    1-1 swap between groups, first improvement.
    """
    if len(groups) <= 1:
        return False

    # prioritize low-reward groups
    scored = []
    for idx, g in enumerate(groups):
        r = _group_reward_only(
            X, g, K, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3
        )
        scored.append((r, idx))
    scored.sort()  # low reward first

    cand_group_ids = [idx for _, idx in scored[:min(group_candidate_limit, len(scored))]]

    for a in cand_group_ids:
        g1 = groups[a]
        mu1 = compute_centroid(X, g1)
        d1 = [(float(np.sum((X[c] - mu1) ** 2)), pos) for pos, c in enumerate(g1)]
        d1.sort(reverse=True)
        idxs1 = [pos for _, pos in d1[:min(cell_candidate_limit, len(d1))]]

        for b in range(len(groups)):
            if b == a:
                continue
            g2 = groups[b]
            mu2 = compute_centroid(X, g2)
            d2 = [(float(np.sum((X[c] - mu2) ** 2)), pos) for pos, c in enumerate(g2)]
            d2.sort(reverse=True)
            idxs2 = [pos for _, pos in d2[:min(cell_candidate_limit, len(d2))]]

            old_sum = _group_reward_only(
                X, g1, K, delta_bar, w, lambda_penalty,
                theta1, theta2, theta3, P1, P2, P3
            ) + _group_reward_only(
                X, g2, K, delta_bar, w, lambda_penalty,
                theta1, theta2, theta3, P1, P2, P3
            )

            for i in idxs1:
                for j in idxs2:
                    new_g1 = g1[:]
                    new_g2 = g2[:]
                    new_g1[i], new_g2[j] = new_g2[j], new_g1[i]

                    new_sum = _group_reward_only(
                        X, new_g1, K, delta_bar, w, lambda_penalty,
                        theta1, theta2, theta3, P1, P2, P3
                    ) + _group_reward_only(
                        X, new_g2, K, delta_bar, w, lambda_penalty,
                        theta1, theta2, theta3, P1, P2, P3
                    )

                    if new_sum > old_sum + 1e-12:
                        groups[a] = sorted(new_g1)
                        groups[b] = sorted(new_g2)
                        return True

    return False


def _leftover_replace_first_improvement(
    X: np.ndarray,
    groups: list[list[int]],
    leftover: list[int],
    K: int,
    delta_bar: float,
    w: np.ndarray,
    lambda_penalty: float,
    theta1: float,
    theta2: float,
    theta3: float,
    P1: float,
    P2: float,
    P3: float,
    group_candidate_limit: int,
    cell_candidate_limit: int,
    leftover_candidate_limit: int,
) -> bool:
    """
    Replace one outlier cell in a group with one promising leftover cell.
    """
    if len(leftover) == 0 or len(groups) == 0:
        return False

    scored = []
    for idx, g in enumerate(groups):
        r = _group_reward_only(
            X, g, K, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3
        )
        scored.append((r, idx))
    scored.sort()
    cand_group_ids = [idx for _, idx in scored[:min(group_candidate_limit, len(scored))]]

    for a in cand_group_ids:
        g = groups[a]
        old_reward = _group_reward_only(
            X, g, K, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3
        )

        mu = compute_centroid(X, g)
        d = [(float(np.sum((X[c] - mu) ** 2)), pos) for pos, c in enumerate(g)]
        d.sort(reverse=True)
        idxs = [pos for _, pos in d[:min(cell_candidate_limit, len(d))]]

        # promising leftover cells
        left_scores = []
        for c in leftover:
            score = float(np.dot(w, X[c])) - 0.05 * float(np.sum((X[c] - mu) ** 2))
            left_scores.append((score, c))
        left_scores.sort(reverse=True)
        cand_left = [c for _, c in left_scores[:min(leftover_candidate_limit, len(left_scores))]]

        for i in idxs:
            out_cell = g[i]
            for in_cell in cand_left:
                new_g = g[:]
                new_g[i] = in_cell

                new_reward = _group_reward_only(
                    X, new_g, K, delta_bar, w, lambda_penalty,
                    theta1, theta2, theta3, P1, P2, P3
                )

                if new_reward > old_reward + 1e-12:
                    groups[a] = sorted(new_g)
                    leftover.remove(in_cell)
                    leftover.append(out_cell)
                    leftover.sort()
                    return True

    return False


def _local_search(
    X: np.ndarray,
    groups: list[list[int]],
    leftover: list[int],
    K: int,
    delta_bar: float,
    w: np.ndarray,
    lambda_penalty: float,
    theta1: float,
    theta2: float,
    theta3: float,
    P1: float,
    P2: float,
    P3: float,
    max_local_iter: int,
    group_candidate_limit: int,
    cell_candidate_limit: int,
    leftover_candidate_limit: int,
):
    for _ in range(max_local_iter):
        improved = False

        if _swap_first_improvement(
            X, groups, K, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3,
            group_candidate_limit, cell_candidate_limit
        ):
            improved = True
            continue

        if _leftover_replace_first_improvement(
            X, groups, leftover, K, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3,
            group_candidate_limit, cell_candidate_limit, leftover_candidate_limit
        ):
            improved = True
            continue

        if not improved:
            break

    return groups, leftover


# =========================================================
# Main solver
# =========================================================

def solve_rrp_grasp(
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
    n_starts: int = 20,
    rcl_size: int = 4,
    max_group_attempts: int = 200,
    max_local_iter: int = 30,
    group_candidate_limit: int = 6,
    cell_candidate_limit: int = 2,
    leftover_candidate_limit: int = 10,
):
    start = time.perf_counter()
    n = X.shape[0]

    if n < K or k_t <= 0:
        return {
            "method": "RRP_GRASP",
            "groups": [],
            "leftover": list(range(n)),
            "reward": 0.0,
            "n_packs": 0,
            "avg_delta": 0.0,
            "avg_phi": 0.0,
            "runtime": time.perf_counter() - start,
        }

    rng = np.random.default_rng(seed)
    best_groups = []
    best_leftover = list(range(n))
    best_reward = -np.inf

    for _ in range(n_starts):
        groups, leftover = _construction_phase(
            X=X,
            K=K,
            k_t=k_t,
            delta_bar=delta_bar,
            w=w,
            lambda_penalty=lambda_penalty,
            theta1=theta1,
            theta2=theta2,
            theta3=theta3,
            P1=P1,
            P2=P2,
            P3=P3,
            rng=rng,
            rcl_size=rcl_size,
            max_group_attempts=max_group_attempts,
        )

        groups, leftover = residual_pack_repair(
            X=X,
            groups=groups,
            leftover=leftover,
            K=K,
            k_t=k_t,
            delta_bar=delta_bar,
            w=w,
            lambda_penalty=lambda_penalty,
            theta1=theta1,
            theta2=theta2,
            theta3=theta3,
            P1=P1,
            P2=P2,
            P3=P3,
            min_accept_reward=0.0,
            seed_candidate_limit=12,
            neighbor_candidate_limit=12,
            max_rounds=50,
        )

        summary = summarize_solution(
            X=X,
            groups=groups,
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

        if summary["total_reward"] > best_reward + 1e-12:
            best_reward = summary["total_reward"]
            best_groups = [sorted(g) for g in groups]
            used = set()
            for g in best_groups:
                used.update(g)
            best_leftover = [i for i in range(n) if i not in used]

    final_summary = summarize_solution(
        X=X,
        groups=best_groups,
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
        "method": "RRP_GRASP",
        "groups": best_groups,
        "leftover": best_leftover,
        "reward": final_summary["total_reward"],
        "n_packs": final_summary["n_packs"],
        "avg_delta": final_summary["avg_delta"],
        "avg_phi": final_summary["avg_phi"],
        "runtime": time.perf_counter() - start,
        "reward_per_pack": final_summary["reward_per_pack"],
        "positive_pack_ratio": final_summary["positive_pack_ratio"],
        "tier_counts": final_summary["tier_counts"],
    }