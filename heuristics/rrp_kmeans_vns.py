# heuristics/rrp_kmeans_vns.py
import time
import itertools
import numpy as np

from utils import (
    compute_centroid,
    compute_delta,
    compute_group_reward,
    summarize_solution,
)


# =========================================================
# Basic pack evaluation
# =========================================================

def _evaluate_group_info(
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
    """
    Evaluate one group only.
    Return a dict for local incremental evaluation.
    """
    if len(group) != K:
        return {
            "feasible": False,
            "reward": -np.inf,
            "phi": -np.inf,
            "delta": np.inf,
            "mu": None,
            "gap_to_next": np.inf,
        }

    mu = compute_centroid(X, group)
    delta = compute_delta(X, group)

    if delta > delta_bar:
        return {
            "feasible": False,
            "reward": -np.inf,
            "phi": -np.inf,
            "delta": delta,
            "mu": mu,
            "gap_to_next": np.inf,
        }

    reward, phi, delta = compute_group_reward(
        X, group, w, lambda_penalty, theta1, theta2, theta3, P1, P2, P3
    )

    # gap to next threshold: smaller means more worth improving
    if phi < theta3:
        gap = theta3 - phi
    elif phi < theta2:
        gap = theta2 - phi
    elif phi < theta1:
        gap = theta1 - phi
    else:
        gap = np.inf   # already highest tier

    return {
        "feasible": True,
        "reward": reward,
        "phi": phi,
        "delta": delta,
        "mu": mu,
        "gap_to_next": gap,
    }


def _recompute_all_infos(
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
    infos = []
    total_reward = 0.0
    for g in groups:
        info = _evaluate_group_info(
            X, g, K, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3
        )
        infos.append(info)
        if info["feasible"]:
            total_reward += info["reward"]
    return infos, total_reward


# =========================================================
# Prioritization: focus on the "most worth moving" packs/cells
# =========================================================

def _group_priority_score(info):
    """
    Larger score = more worth adjusting.
    Based on:
    1) low reward
    2) close to next pricing threshold
    """
    reward_term = -info["reward"] if np.isfinite(info["reward"]) else 1e6

    if np.isfinite(info["gap_to_next"]):
        gap_term = 1.0 / (info["gap_to_next"] + 1e-6)
    else:
        gap_term = 0.0

    return 0.7 * reward_term + 0.3 * gap_term


def _select_candidate_group_ids(infos, top_q, theta1):
    """
    仅从 P2 / P3 级别的组中选择候选组，P1 组不参与 VNS 搜索。
    """
    feasible_ids = [i for i, info in enumerate(infos) if info["feasible"] and info["phi"] < theta1]
    ranked = sorted(
        feasible_ids,
        key=lambda i: _group_priority_score(infos[i]),
        reverse=True,
    )
    return ranked[:min(top_q, len(ranked))]


def _nearest_partner_group_ids(infos, gid, top_q, theta1):
    mu = infos[gid]["mu"]
    candidates = []
    for j, info in enumerate(infos):
        if j == gid or not info["feasible"] or info["phi"] >= theta1:
            continue
        d = float(np.sum((mu - info["mu"]) ** 2))
        candidates.append((d, j))
    candidates.sort(key=lambda x: x[0])
    return [j for _, j in candidates[:min(top_q, len(candidates))]]


def _ordered_cells_in_group(X, group, mu, limit):
    """
    Prefer moving 'outlier' cells first.
    """
    scores = []
    for idx, cell in enumerate(group):
        dist = float(np.sum((X[cell] - mu) ** 2))
        scores.append((dist, idx))
    scores.sort(reverse=True)
    return [idx for _, idx in scores[:min(limit, len(scores))]]


def _ordered_leftover_for_group(X, leftover, mu, w, limit):
    """
    Prefer leftover cells that are:
    - close to the target group centroid
    - with better own quality contribution
    """
    scored = []
    for cell in leftover:
        dist = float(np.sum((X[cell] - mu) ** 2))
        contrib = float(np.dot(w, X[cell]))
        score = -dist + 0.05 * contrib
        scored.append((score, cell))
    scored.sort(reverse=True)
    return [cell for _, cell in scored[:min(limit, len(scored))]]


# =========================================================
# Initial K-means solution
# =========================================================

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
            full_groups.append(g[:])
        else:
            incomplete_cells.extend(g)

    if len(incomplete_cells) < K:
        return full_groups, incomplete_cells

    n_new = len(incomplete_cells) // K
    subX = X[incomplete_cells]

    rng = np.random.default_rng(seed)
    init_idx = rng.choice(len(incomplete_cells), size=n_new, replace=False)
    centers = subX[init_idx].copy()

    new_groups_local, rem_local = _assign_to_nearest_nonfull(subX, centers, K)

    new_groups = full_groups[:]
    for g in new_groups_local:
        mapped = [incomplete_cells[idx] for idx in g]
        new_groups.append(mapped)

    leftover = [incomplete_cells[idx] for idx in rem_local]
    return new_groups, leftover


def _initial_kmeans_solution(X, K, k_t, L1, tol, seed):
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    # Ensure enough centers so every cell can be assigned (len < k_t * K leaves orphans).
    n_centers = max(k_t, (n + K - 1) // K)

    init_idx = rng.choice(n, size=n_centers, replace=False)
    centers = X[init_idx].copy()

    groups = [[] for _ in range(n_centers)]

    for _ in range(L1):
        groups, _ = _assign_to_nearest_nonfull(X, centers, K)
        new_centers = _update_centers(X, groups, centers)
        shift = np.max(np.linalg.norm(new_centers - centers, axis=1))
        centers = new_centers
        if shift < tol:
            break

    groups, leftover = _recluster_incomplete_groups(X, groups, K, seed=seed)
    groups = [sorted(g) for g in groups if len(g) > 0]
    leftover = sorted(leftover)
    return groups, leftover


def _extract_feasible_groups_and_leftover(
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
):
    """
    Keep only feasible full groups, capped at k_t.
    Everything else goes to leftover.
    """
    feasible_groups = []
    used = set(leftover)

    for g in groups:
        if len(feasible_groups) >= k_t:
            used.update(g)
            continue
        info = _evaluate_group_info(
            X, g, K, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3
        )
        if info["feasible"]:
            feasible_groups.append(sorted(g))
            used.update(g)
        else:
            used.update(g)

    final_leftover = [i for i in range(X.shape[0]) if i not in {c for grp in feasible_groups for c in grp}]
    return feasible_groups, sorted(final_leftover)


# =========================================================
# Fast repair for large neighborhoods
# =========================================================

def _repair_from_pool_greedy(X, fixed_groups, pool, K, k_t, w):
    """
    Large-neighborhood greedy repair.
    Used only after local neighborhoods converge.
    """
    groups = [sorted(g) for g in fixed_groups]
    pool = sorted(pool)

    def quality(cell):
        return float(np.dot(w, X[cell]))

    while len(pool) >= K and len(groups) < k_t:
        seed = max(pool, key=quality)
        pool.remove(seed)

        dists = [(float(np.sum((X[c] - X[seed]) ** 2)), c) for c in pool]
        dists.sort(key=lambda x: x[0])

        chosen = [seed] + [c for _, c in dists[:K - 1]]
        chosen_set = set(chosen[1:])
        pool = [c for c in pool if c not in chosen_set]
        groups.append(sorted(chosen))

    return groups, pool


# =========================================================
# N1 / N2 / N3: first-improvement + local incremental evaluation
# =========================================================

def _n1_swap_first_improvement(
    X,
    groups,
    infos,
    total_reward,
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
    pack_candidate_limit,
    partner_limit,
    cell_candidate_limit,
):
    cand_ids = _select_candidate_group_ids(infos, pack_candidate_limit, theta1)

    for a in cand_ids:
        partner_ids = _nearest_partner_group_ids(infos, a, partner_limit, theta1)
        g1 = groups[a]
        mu1 = infos[a]["mu"]
        idxs1 = _ordered_cells_in_group(X, g1, mu1, cell_candidate_limit)

        for b in partner_ids:
            g2 = groups[b]
            mu2 = infos[b]["mu"]
            idxs2 = _ordered_cells_in_group(X, g2, mu2, cell_candidate_limit)

            old_sum = infos[a]["reward"] + infos[b]["reward"]

            for i in idxs1:
                for j in idxs2:
                    new_g1 = g1[:]
                    new_g2 = g2[:]
                    new_g1[i], new_g2[j] = new_g2[j], new_g1[i]

                    info1 = _evaluate_group_info(
                        X, new_g1, K, delta_bar, w, lambda_penalty,
                        theta1, theta2, theta3, P1, P2, P3
                    )
                    if not info1["feasible"]:
                        continue

                    info2 = _evaluate_group_info(
                        X, new_g2, K, delta_bar, w, lambda_penalty,
                        theta1, theta2, theta3, P1, P2, P3
                    )
                    if not info2["feasible"]:
                        continue

                    new_sum = info1["reward"] + info2["reward"]
                    if new_sum > old_sum + 1e-12:
                        groups[a] = sorted(new_g1)
                        groups[b] = sorted(new_g2)
                        return True

    return False


def _n2_leftover_swap_first_improvement(
    X,
    groups,
    infos,
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
    pack_candidate_limit,
    cell_candidate_limit,
    leftover_candidate_limit,
):
    if len(leftover) == 0:
        return False

    cand_ids = _select_candidate_group_ids(infos, pack_candidate_limit, theta1)

    for a in cand_ids:
        g = groups[a]
        mu = infos[a]["mu"]
        old_reward = infos[a]["reward"]

        idxs = _ordered_cells_in_group(X, g, mu, cell_candidate_limit)
        left_candidates = _ordered_leftover_for_group(X, leftover, mu, w, leftover_candidate_limit)

        for i in idxs:
            out_cell = g[i]
            for in_cell in left_candidates:
                new_g = g[:]
                new_g[i] = in_cell

                info_new = _evaluate_group_info(
                    X, new_g, K, delta_bar, w, lambda_penalty,
                    theta1, theta2, theta3, P1, P2, P3
                )
                if not info_new["feasible"]:
                    continue

                if info_new["reward"] > old_reward + 1e-12:
                    groups[a] = sorted(new_g)
                    leftover.remove(in_cell)
                    leftover.append(out_cell)
                    leftover.sort()
                    return True

    return False


def _n3_exchange22_first_improvement(
    X,
    groups,
    infos,
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
    pack_candidate_limit,
    partner_limit,
    cell_candidate_limit,
):
    if K < 2:
        return False

    cand_ids = _select_candidate_group_ids(infos, pack_candidate_limit, theta1)

    for a in cand_ids:
        partner_ids = _nearest_partner_group_ids(infos, a, partner_limit, theta1)
        g1 = groups[a]
        mu1 = infos[a]["mu"]
        idxs1 = _ordered_cells_in_group(X, g1, mu1, cell_candidate_limit)

        if len(idxs1) < 2:
            continue

        combs1 = list(itertools.combinations(idxs1, 2))

        for b in partner_ids:
            g2 = groups[b]
            mu2 = infos[b]["mu"]
            idxs2 = _ordered_cells_in_group(X, g2, mu2, cell_candidate_limit)

            if len(idxs2) < 2:
                continue

            combs2 = list(itertools.combinations(idxs2, 2))
            old_sum = infos[a]["reward"] + infos[b]["reward"]

            for c1 in combs1:
                for c2 in combs2:
                    new_g1 = g1[:]
                    new_g2 = g2[:]

                    cells1 = [g1[c1[0]], g1[c1[1]]]
                    cells2 = [g2[c2[0]], g2[c2[1]]]

                    new_g1[c1[0]], new_g1[c1[1]] = cells2[0], cells2[1]
                    new_g2[c2[0]], new_g2[c2[1]] = cells1[0], cells1[1]

                    info1 = _evaluate_group_info(
                        X, new_g1, K, delta_bar, w, lambda_penalty,
                        theta1, theta2, theta3, P1, P2, P3
                    )
                    if not info1["feasible"]:
                        continue

                    info2 = _evaluate_group_info(
                        X, new_g2, K, delta_bar, w, lambda_penalty,
                        theta1, theta2, theta3, P1, P2, P3
                    )
                    if not info2["feasible"]:
                        continue

                    new_sum = info1["reward"] + info2["reward"]
                    if new_sum > old_sum + 1e-12:
                        groups[a] = sorted(new_g1)
                        groups[b] = sorted(new_g2)
                        return True

    return False


# =========================================================
# N4 / N5: large neighborhoods
# triggered only after N1/N2/N3 converge
# =========================================================

def _n4_destroy_repair(
    X,
    groups,
    infos,
    leftover,
    total_reward,
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
    destroy_size,
    pack_candidate_limit,
):
    cand_ids = _select_candidate_group_ids(infos, max(pack_candidate_limit, destroy_size), theta1)
    if len(cand_ids) == 0:
        return False, groups, leftover

    destroy_ids = cand_ids[:min(destroy_size, len(cand_ids))]
    destroy_set = set(destroy_ids)

    fixed_groups = []
    pool = leftover[:]

    old_destroy_reward = 0.0
    for gid, g in enumerate(groups):
        if gid in destroy_set:
            pool.extend(g)
            old_destroy_reward += infos[gid]["reward"]
        else:
            fixed_groups.append(g[:])

    rebuilt_groups, new_leftover = _repair_from_pool_greedy(X, fixed_groups, pool, K, k_t, w)

    # full recompute is acceptable here because N4 is low-frequency
    new_infos, new_total = _recompute_all_infos(
        X, rebuilt_groups, K, delta_bar, w, lambda_penalty,
        theta1, theta2, theta3, P1, P2, P3
    )
    feasible_groups = [g for g, info in zip(rebuilt_groups, new_infos) if info["feasible"]]

    if new_total > total_reward + 1e-12:
        used = set()
        for g in feasible_groups:
            used.update(g)
        final_leftover = [i for i in range(X.shape[0]) if i not in used]
        return True, feasible_groups, final_leftover

    return False, groups, leftover


def _n5_merge_split(
    X,
    groups,
    infos,
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
    pack_candidate_limit,
    partner_limit,
):
    cand_ids = _select_candidate_group_ids(infos, pack_candidate_limit, theta1)

    for a in cand_ids:
        partner_ids = _nearest_partner_group_ids(infos, a, partner_limit, theta1)

        for b in partner_ids:
            if b <= a:
                continue

            g1 = groups[a]
            g2 = groups[b]
            pool = g1 + g2
            old_sum = infos[a]["reward"] + infos[b]["reward"]

            # exact balanced partition enumeration over 2K cells
            # for K=4, this is only C(8,4)=70
            for subset in itertools.combinations(pool, K):
                subset = set(subset)
                new_g1 = sorted(list(subset))
                new_g2 = sorted([c for c in pool if c not in subset])

                # avoid symmetric duplicate
                if new_g1[0] > new_g2[0]:
                    continue

                info1 = _evaluate_group_info(
                    X, new_g1, K, delta_bar, w, lambda_penalty,
                    theta1, theta2, theta3, P1, P2, P3
                )
                if not info1["feasible"]:
                    continue

                info2 = _evaluate_group_info(
                    X, new_g2, K, delta_bar, w, lambda_penalty,
                    theta1, theta2, theta3, P1, P2, P3
                )
                if not info2["feasible"]:
                    continue

                new_sum = info1["reward"] + info2["reward"]
                if new_sum > old_sum + 1e-12:
                    groups[a] = new_g1
                    groups[b] = new_g2
                    return True, groups, leftover

    return False, groups, leftover


# =========================================================
# Main solver
# =========================================================

def solve_rrp_kmeans_vns(
    X: np.ndarray,
    K: int,
    k_t: int,
    delta_bar: float,
    L1: int,
    tol: float,
    max_vns_iter: int,
    max_no_improve: int,
    w: np.ndarray,
    lambda_penalty: float,
    theta1: float,
    theta2: float,
    theta3: float,
    P1: float,
    P2: float,
    P3: float,
    seed: int | None = None,
    pack_candidate_limit: int = 8,
    partner_limit: int = 4,
    cell_candidate_limit: int = 3,
    leftover_candidate_limit: int = 12,
    destroy_size: int = 2,
    enable_n5: bool = False,   # 新增：默认关闭 N5 merge-split
):
    start = time.perf_counter()
    n = X.shape[0]

    if n < K or k_t <= 0:
        return {
            "method": "RRP_KMEANS_VNS",
            "groups": [],
            "leftover": list(range(n)),
            "reward": 0.0,
            "n_packs": 0,
            "avg_delta": 0.0,
            "avg_phi": 0.0,
            "runtime": time.perf_counter() - start,
        }

    seed_val = 0 if seed is None else seed

    # 1) Initial K-means solution
    raw_groups, raw_leftover = _initial_kmeans_solution(
        X=X,
        K=K,
        k_t=k_t,
        L1=L1,
        tol=tol,
        seed=seed_val,
    )

    groups, leftover = _extract_feasible_groups_and_leftover(
        X, raw_groups, raw_leftover, K, k_t, delta_bar, w, lambda_penalty,
        theta1, theta2, theta3, P1, P2, P3
    )

    infos, total_reward = _recompute_all_infos(
        X, groups, K, delta_bar, w, lambda_penalty,
        theta1, theta2, theta3, P1, P2, P3
    )

    no_improve_rounds = 0
    it = 0

    while it < max_vns_iter and no_improve_rounds < max_no_improve:
        improved = False

        # -------------------------------------------------
        # Phase A: light neighborhoods only (N1, N2, N3)
        # first-improvement + local incremental evaluation
        # -------------------------------------------------
        while True:
            light_improved = False

            if _n1_swap_first_improvement(
                X, groups, infos, total_reward, K, delta_bar, w, lambda_penalty,
                theta1, theta2, theta3, P1, P2, P3,
                pack_candidate_limit, partner_limit, cell_candidate_limit
            ):
                infos, total_reward = _recompute_all_infos(
                    X, groups, K, delta_bar, w, lambda_penalty,
                    theta1, theta2, theta3, P1, P2, P3
                )
                light_improved = True
                improved = True
                continue

            if _n2_leftover_swap_first_improvement(
                X, groups, infos, leftover, K, delta_bar, w, lambda_penalty,
                theta1, theta2, theta3, P1, P2, P3,
                pack_candidate_limit, cell_candidate_limit, leftover_candidate_limit
            ):
                infos, total_reward = _recompute_all_infos(
                    X, groups, K, delta_bar, w, lambda_penalty,
                    theta1, theta2, theta3, P1, P2, P3
                )
                light_improved = True
                improved = True
                continue

            if _n3_exchange22_first_improvement(
                X, groups, infos, K, delta_bar, w, lambda_penalty,
                theta1, theta2, theta3, P1, P2, P3,
                pack_candidate_limit, partner_limit, cell_candidate_limit
            ):
                infos, total_reward = _recompute_all_infos(
                    X, groups, K, delta_bar, w, lambda_penalty,
                    theta1, theta2, theta3, P1, P2, P3
                )
                light_improved = True
                improved = True
                continue

            if not light_improved:
                break

        # -------------------------------------------------
        # Phase B: heavy neighborhood N4
        # -------------------------------------------------
        if not improved:
            found_n4, new_groups, new_leftover = _n4_destroy_repair(
                X, groups, infos, leftover, total_reward,
                K, k_t, delta_bar, w, lambda_penalty,
                theta1, theta2, theta3, P1, P2, P3,
                destroy_size, pack_candidate_limit
            )
            if found_n4:
                groups, leftover = new_groups, new_leftover
                infos, total_reward = _recompute_all_infos(
                    X, groups, K, delta_bar, w, lambda_penalty,
                    theta1, theta2, theta3, P1, P2, P3
                )
                improved = True

        # -------------------------------------------------
        # Phase C: optional N5 merge-split
        # 默认关闭，因为 K=8 时 C(16,8)=12870，计算量很大
        # -------------------------------------------------
        if enable_n5 and not improved:
            found_n5, new_groups, new_leftover = _n5_merge_split(
                X, groups, infos, leftover, K, delta_bar, w, lambda_penalty,
                theta1, theta2, theta3, P1, P2, P3,
                pack_candidate_limit, partner_limit
            )
            if found_n5:
                groups, leftover = new_groups, new_leftover
                infos, total_reward = _recompute_all_infos(
                    X, groups, K, delta_bar, w, lambda_penalty,
                    theta1, theta2, theta3, P1, P2, P3
                )
                improved = True

        if improved:
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1

        it += 1

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

    used = set()
    for g in groups:
        used.update(g)
    leftover = [i for i in range(n) if i not in used]

    return {
        "method": "RRP_KMEANS_VNS",
        "groups": groups,
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