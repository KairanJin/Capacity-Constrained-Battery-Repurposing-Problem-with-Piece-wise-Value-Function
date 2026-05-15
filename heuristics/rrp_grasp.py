# heuristics/rrp_grasp.py
import time
import numpy as np

from utils import compute_group_reward, summarize_solution, piecewise_value
from heuristics.residual_packing import residual_pack_repair
from heuristics._grasp_stats import GroupStats, precompute_arrays, sq_dist_to_centroid

# =========================================================
# Basic evaluation
# =========================================================

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
    """Full evaluation for a completed pack."""
    if len(group) != K:
        return {
            "feasible": False,
            "reward": -np.inf,
            "phi": -np.inf,
            "delta": np.inf,
        }

    delta = float(np.mean(np.sum((X[group] - X[group].mean(axis=0)) ** 2, axis=1)))
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
# Helpers for incremental evaluation
# =========================================================

def _delta_after_swap(stats, remove_vec, add_vec, remove_sq, add_sq):
    """New delta if we remove one cell and add another. O(d)."""
    new_sum_sq = stats.sum_sq - remove_sq + add_sq
    new_sum_vec = stats.sum_vec - remove_vec + add_vec
    n = stats.count
    return new_sum_sq / n - (new_sum_vec @ new_sum_vec) / (n * n)


def _phi_after_swap(stats, remove_idx, add_idx, w_dot_X):
    """New phi after swap. O(1)."""
    return (stats.w_dot_sum - w_dot_X[remove_idx] + w_dot_X[add_idx]) / stats.count


def _reward_from_delta_phi(delta, phi, delta_bar, lambda_penalty,
                           theta1, theta2, theta3, P1, P2, P3):
    """Reward from pre-computed delta and phi. O(1)."""
    if delta > delta_bar:
        return -np.inf
    value = piecewise_value(phi, theta1, theta2, theta3, P1, P2, P3)
    return value - lambda_penalty * delta


# =========================================================
# Construction phase
# =========================================================

def _build_one_group_grasp(
    X: np.ndarray,
    available: list[int],
    K: int,
    w: np.ndarray,
    lambda_penalty: float,
    rng: np.random.Generator,
    rcl_size: int,
    X_sq_norms: np.ndarray,
    w_dot_X: np.ndarray,
) -> list[int]:
    """Build one group using GRASP construction with incremental stats."""
    if len(available) < K:
        return []

    qualities = [(w_dot_X[i], i) for i in available]
    qualities.sort(reverse=True)
    seed_pool = [i for _, i in qualities[:min(rcl_size, len(qualities))]]
    first = int(rng.choice(seed_pool))

    group = [first]
    stats = GroupStats()
    stats.add(first, X, X_sq_norms, w_dot_X)
    remaining = [i for i in available if i != first]

    while len(group) < K and len(remaining) > 0:
        gains = []
        for i in remaining:
            old_score = stats.partial_score(lambda_penalty)
            temp = stats.clone()
            temp.add(i, X, X_sq_norms, w_dot_X)
            mg = temp.partial_score(lambda_penalty) - old_score
            gains.append((mg, i))

        gains.sort(reverse=True, key=lambda x: x[0])
        rcl = [i for _, i in gains[:min(rcl_size, len(gains))]]
        chosen = int(rng.choice(rcl))
        group.append(chosen)
        stats.add(chosen, X, X_sq_norms, w_dot_X)
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
    X_sq_norms: np.ndarray,
    w_dot_X: np.ndarray,
):
    """Iteratively construct up to k_t feasible groups."""
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
            X_sq_norms=X_sq_norms,
            w_dot_X=w_dot_X,
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
            qualities = [(w_dot_X[i], i) for i in g]
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
    X_sq_norms: np.ndarray,
    w_dot_X: np.ndarray,
) -> bool:
    """1-1 swap between groups, first improvement, using incremental stats."""
    if len(groups) <= 1:
        return False

    all_stats = []
    all_rewards = []
    for g in groups:
        st = GroupStats()
        for c in g:
            st.add(c, X, X_sq_norms, w_dot_X)
        all_stats.append(st)
        all_rewards.append(_reward_from_delta_phi(
            st.delta, st.phi, delta_bar, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3
        ))

    scored = [(all_rewards[idx], idx) for idx in range(len(groups))]
    scored.sort()
    cand_group_ids = [idx for _, idx in scored[:min(group_candidate_limit, len(scored))]]

    for a in cand_group_ids:
        g1 = groups[a]
        st1 = all_stats[a]
        mu1 = st1.centroid

        d1 = [(sq_dist_to_centroid(g1[pos], mu1, X_sq_norms, X), pos) for pos in range(len(g1))]
        d1.sort(reverse=True)
        idxs1 = [pos for _, pos in d1[:min(cell_candidate_limit, len(d1))]]

        for b in range(len(groups)):
            if b == a:
                continue
            g2 = groups[b]
            st2 = all_stats[b]
            mu2 = st2.centroid

            d2 = [(sq_dist_to_centroid(g2[pos], mu2, X_sq_norms, X), pos) for pos in range(len(g2))]
            d2.sort(reverse=True)
            idxs2 = [pos for _, pos in d2[:min(cell_candidate_limit, len(d2))]]

            old_sum = all_rewards[a] + all_rewards[b]

            for i in idxs1:
                cell_i = g1[i]
                for j in idxs2:
                    cell_j = g2[j]

                    new_delta_a = _delta_after_swap(
                        st1, X[cell_i], X[cell_j],
                        X_sq_norms[cell_i], X_sq_norms[cell_j]
                    )
                    if new_delta_a > delta_bar:
                        continue
                    new_phi_a = _phi_after_swap(st1, cell_i, cell_j, w_dot_X)
                    new_reward_a = _reward_from_delta_phi(
                        new_delta_a, new_phi_a, delta_bar, lambda_penalty,
                        theta1, theta2, theta3, P1, P2, P3
                    )

                    new_delta_b = _delta_after_swap(
                        st2, X[cell_j], X[cell_i],
                        X_sq_norms[cell_j], X_sq_norms[cell_i]
                    )
                    if new_delta_b > delta_bar:
                        continue
                    new_phi_b = _phi_after_swap(st2, cell_j, cell_i, w_dot_X)
                    new_reward_b = _reward_from_delta_phi(
                        new_delta_b, new_phi_b, delta_bar, lambda_penalty,
                        theta1, theta2, theta3, P1, P2, P3
                    )

                    new_sum = new_reward_a + new_reward_b
                    if new_sum > old_sum + 1e-12:
                        new_g1 = g1[:]
                        new_g2 = g2[:]
                        new_g1[i], new_g2[j] = new_g2[j], new_g1[i]
                        groups[a] = sorted(new_g1)
                        groups[b] = sorted(new_g2)

                        st1.remove(cell_i, X, X_sq_norms, w_dot_X)
                        st1.add(cell_j, X, X_sq_norms, w_dot_X)
                        st2.remove(cell_j, X, X_sq_norms, w_dot_X)
                        st2.add(cell_i, X, X_sq_norms, w_dot_X)
                        all_rewards[a] = new_reward_a
                        all_rewards[b] = new_reward_b
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
    X_sq_norms: np.ndarray,
    w_dot_X: np.ndarray,
) -> bool:
    """Replace one outlier cell in a group with one promising leftover cell."""
    if len(leftover) == 0 or len(groups) == 0:
        return False

    all_stats = []
    all_rewards = []
    for g in groups:
        st = GroupStats()
        for c in g:
            st.add(c, X, X_sq_norms, w_dot_X)
        all_stats.append(st)
        all_rewards.append(_reward_from_delta_phi(
            st.delta, st.phi, delta_bar, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3
        ))

    scored = [(all_rewards[idx], idx) for idx in range(len(groups))]
    scored.sort()
    cand_group_ids = [idx for _, idx in scored[:min(group_candidate_limit, len(scored))]]

    for a in cand_group_ids:
        g = groups[a]
        st = all_stats[a]
        old_reward = all_rewards[a]
        mu = st.centroid

        d = [(sq_dist_to_centroid(g[pos], mu, X_sq_norms, X), pos) for pos in range(len(g))]
        d.sort(reverse=True)
        idxs = [pos for _, pos in d[:min(cell_candidate_limit, len(d))]]

        left_scores = []
        for c in leftover:
            score = w_dot_X[c] - 0.05 * sq_dist_to_centroid(c, mu, X_sq_norms, X)
            left_scores.append((score, c))
        left_scores.sort(reverse=True)
        cand_left = [c for _, c in left_scores[:min(leftover_candidate_limit, len(left_scores))]]

        for i in idxs:
            out_cell = g[i]
            for in_cell in cand_left:
                new_delta = _delta_after_swap(
                    st, X[out_cell], X[in_cell],
                    X_sq_norms[out_cell], X_sq_norms[in_cell]
                )
                if new_delta > delta_bar:
                    continue
                new_phi = _phi_after_swap(st, out_cell, in_cell, w_dot_X)
                new_reward = _reward_from_delta_phi(
                    new_delta, new_phi, delta_bar, lambda_penalty,
                    theta1, theta2, theta3, P1, P2, P3
                )

                if new_reward > old_reward + 1e-12:
                    new_g = g[:]
                    new_g[i] = in_cell
                    groups[a] = sorted(new_g)
                    leftover.remove(in_cell)
                    leftover.append(out_cell)
                    leftover.sort()

                    st.remove(out_cell, X, X_sq_norms, w_dot_X)
                    st.add(in_cell, X, X_sq_norms, w_dot_X)
                    all_rewards[a] = new_reward
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
    X_sq_norms: np.ndarray,
    w_dot_X: np.ndarray,
):
    for _ in range(max_local_iter):
        improved = False

        if _swap_first_improvement(
            X, groups, K, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3,
            group_candidate_limit, cell_candidate_limit,
            X_sq_norms, w_dot_X
        ):
            improved = True
            continue

        if _leftover_replace_first_improvement(
            X, groups, leftover, K, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3,
            group_candidate_limit, cell_candidate_limit, leftover_candidate_limit,
            X_sq_norms, w_dot_X
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

    X_sq_norms, w_dot_X = precompute_arrays(X, np.asarray(w))
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
            X_sq_norms=X_sq_norms,
            w_dot_X=w_dot_X,
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
            neighbor_candidate_limit=20,
            max_rounds=50,
            X_sq_norms=X_sq_norms,
            w_dot_X=w_dot_X,
        )

        groups, leftover = _local_search(
            X=X,
            groups=groups,
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
            max_local_iter=max_local_iter,
            group_candidate_limit=group_candidate_limit,
            cell_candidate_limit=cell_candidate_limit,
            leftover_candidate_limit=leftover_candidate_limit,
            X_sq_norms=X_sq_norms,
            w_dot_X=w_dot_X,
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