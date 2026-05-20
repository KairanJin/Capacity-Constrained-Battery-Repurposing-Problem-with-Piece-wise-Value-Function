# heuristics/rrp_ms_kmeans_vns.py
"""
Multi-Start KMeans + VNS (MS-KMeans-VNS)

Combines multiple diverse KMeans initializations with focused VNS local search.
The key insight: a single KMeans start often converges to a poor local optimum.
Running 3-4 different starts and applying VNS to each yields significantly better
solutions while staying within the 20s time budget.

Algorithm:
1. Generate diverse KMeans initial solutions using different seeds
2. Extract feasible groups from each, keep top candidates
3. Run focused VNS (N1/N2/N3) on each candidate in parallel fashion
4. On the best result, run residual packing + N4 destroy-repair
5. Return the best overall solution
"""
import time
import itertools
import numpy as np

from utils import (
    compute_centroid,
    compute_delta,
    compute_group_reward,
    summarize_solution,
    piecewise_value,
)
from heuristics.residual_packing import residual_pack_repair


# =========================================================
# KMeans initialization (copied from rrp_kmeans for self-containment)
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


def _kmeans_init(X, K, k_t, L1, tol, seed):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    # Ensure enough centers so every cell can be assigned.
    n_centers = max(k_t, (n + K - 1) // K)
    init_idx = rng.choice(n, size=n_centers, replace=False)
    centers = X[init_idx].copy()
    groups = [[] for _ in range(n_centers)]
    for _ in range(L1):
        groups, _ = _assign_to_nearest_nonfull(X, centers, K)
        new_centers = np.zeros_like(centers)
        for j, g in enumerate(groups):
            if len(g) > 0:
                new_centers[j] = X[g].mean(axis=0)
        shift = np.max(np.linalg.norm(new_centers - centers, axis=1))
        centers = new_centers
        if shift < tol:
            break
    groups, leftover = _recluster_incomplete_groups(X, groups, K, seed=seed)
    return groups, leftover


def _random_kmeans_init(X, K, k_t, L1, tol, seed, shuffle_frac=0.3):
    """KMeans with random shuffling of some assignments for diversity."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    groups, leftover = _kmeans_init(X, K, k_t, L1, tol, seed)

    # Shuffle a fraction of cells to different groups
    all_cells = []
    for g in groups:
        all_cells.extend(g)
    rng.shuffle(all_cells)
    n_shuffle = int(len(all_cells) * shuffle_frac)
    shuffled = all_cells[:n_shuffle]

    # Reassign shuffled cells randomly to groups of same size
    for cell in shuffled:
        old_gid = None
        for gid, g in enumerate(groups):
            if cell in g:
                old_gid = gid
                g.remove(cell)
                break
        if old_gid is not None:
            new_gid = int(rng.integers(0, len(groups)))
            groups[new_gid].append(cell)

    # Re-cluster incomplete
    groups, leftover = _recluster_incomplete_groups(X, groups, K, seed=seed + 1000)
    return groups, leftover


# =========================================================
# Group evaluation
# =========================================================

def _evaluate_group_info(X, group, K, delta_bar, w, lambda_penalty,
                         theta1, theta2, theta3, P1, P2, P3):
    if len(group) != K:
        return {"feasible": False, "reward": -np.inf, "phi": -np.inf,
                "delta": np.inf, "mu": None, "gap_to_next": np.inf}
    mu = X[group].mean(axis=0)
    delta = float(np.mean(np.sum((X[group] - mu) ** 2, axis=1)))
    if delta > delta_bar:
        return {"feasible": False, "reward": -np.inf, "phi": -np.inf,
                "delta": delta, "mu": mu, "gap_to_next": np.inf}
    phi = float(np.dot(w, mu))
    value = piecewise_value(phi, theta1, theta2, theta3, P1, P2, P3)
    reward = value - lambda_penalty * delta
    if phi < theta3:
        gap = theta3 - phi
    elif phi < theta2:
        gap = theta2 - phi
    elif phi < theta1:
        gap = theta1 - phi
    else:
        gap = np.inf
    return {"feasible": True, "reward": reward, "phi": phi,
            "delta": delta, "mu": mu, "gap_to_next": gap}


def _extract_feasible_and_leftover(X, groups, K, k_t, delta_bar, w, lambda_penalty,
                                    theta1, theta2, theta3, P1, P2, P3):
    feasible = []
    used = set()
    for g in groups:
        if len(feasible) >= k_t:
            used.update(g)
            continue
        info = _evaluate_group_info(X, g, K, delta_bar, w, lambda_penalty,
                                     theta1, theta2, theta3, P1, P2, P3)
        if info["feasible"]:
            feasible.append(sorted(g))
            used.update(g)
        else:
            used.update(g)
    leftover = [i for i in range(X.shape[0]) if i not in used]
    return feasible, sorted(leftover)


def _recompute_all_infos(X, groups, K, delta_bar, w, lambda_penalty,
                         theta1, theta2, theta3, P1, P2, P3):
    infos = []
    total = 0.0
    for g in groups:
        info = _evaluate_group_info(X, g, K, delta_bar, w, lambda_penalty,
                                     theta1, theta2, theta3, P1, P2, P3)
        infos.append(info)
        if info["feasible"]:
            total += info["reward"]
    return infos, total


# =========================================================
# Prioritization (same as VNS)
# =========================================================

def _group_priority_score(info):
    reward_term = -info["reward"] if np.isfinite(info["reward"]) else 1e6
    if np.isfinite(info["gap_to_next"]):
        gap_term = 1.0 / (info["gap_to_next"] + 1e-6)
    else:
        gap_term = 0.0
    return 0.7 * reward_term + 0.3 * gap_term


def _select_candidate_group_ids(infos, top_q):
    feasible = [i for i, info in enumerate(infos) if info["feasible"]]
    ranked = sorted(feasible, key=lambda i: _group_priority_score(infos[i]), reverse=True)
    return ranked[:min(top_q, len(ranked))]


def _nearest_partner_ids(infos, gid, top_q):
    mu = infos[gid]["mu"]
    cands = []
    for j, info in enumerate(infos):
        if j == gid or not info["feasible"]:
            continue
        d = float(np.sum((mu - info["mu"]) ** 2))
        cands.append((d, j))
    cands.sort(key=lambda x: x[0])
    return [j for _, j in cands[:min(top_q, len(cands))]]


def _ordered_outlier_indices(X, group, mu, limit):
    scores = [(float(np.sum((X[cell] - mu) ** 2)), idx) for idx, cell in enumerate(group)]
    scores.sort(reverse=True)
    return [idx for _, idx in scores[:min(limit, len(scores))]]


def _ordered_leftover(X, leftover, mu, w, limit):
    scored = []
    for cell in leftover:
        dist = float(np.sum((X[cell] - mu) ** 2))
        contrib = float(np.dot(w, X[cell]))
        scored.append((-dist + 0.05 * contrib, cell))
    scored.sort(reverse=True)
    return [cell for _, cell in scored[:min(limit, len(scored))]]


# =========================================================
# N1: 1-1 swap (first improvement)
# =========================================================

def _n1_swap(X, groups, infos, K, delta_bar, w, lambda_penalty,
             theta1, theta2, theta3, P1, P2, P3,
             pack_candidate_limit, partner_limit, cell_candidate_limit):
    cand_ids = _select_candidate_group_ids(infos, pack_candidate_limit)
    for a in cand_ids:
        partner_ids = _nearest_partner_ids(infos, a, partner_limit)
        g1 = groups[a]
        mu1 = infos[a]["mu"]
        idxs1 = _ordered_outlier_indices(X, g1, mu1, cell_candidate_limit)
        for b in partner_ids:
            g2 = groups[b]
            mu2 = infos[b]["mu"]
            idxs2 = _ordered_outlier_indices(X, g2, mu2, cell_candidate_limit)
            old_sum = infos[a]["reward"] + infos[b]["reward"]
            for i in idxs1:
                for j in idxs2:
                    new_g1 = g1[:]
                    new_g2 = g2[:]
                    new_g1[i], new_g2[j] = new_g2[j], new_g1[i]
                    info1 = _evaluate_group_info(X, new_g1, K, delta_bar, w, lambda_penalty,
                                                  theta1, theta2, theta3, P1, P2, P3)
                    if not info1["feasible"]:
                        continue
                    info2 = _evaluate_group_info(X, new_g2, K, delta_bar, w, lambda_penalty,
                                                  theta1, theta2, theta3, P1, P2, P3)
                    if not info2["feasible"]:
                        continue
                    if info1["reward"] + info2["reward"] > old_sum + 1e-12:
                        groups[a] = sorted(new_g1)
                        groups[b] = sorted(new_g2)
                        return True
    return False


# =========================================================
# N2: leftover swap (first improvement)
# =========================================================

def _n2_leftover(X, groups, infos, leftover, K, delta_bar, w, lambda_penalty,
                 theta1, theta2, theta3, P1, P2, P3,
                 pack_candidate_limit, cell_candidate_limit, leftover_candidate_limit):
    if len(leftover) == 0:
        return False
    cand_ids = _select_candidate_group_ids(infos, pack_candidate_limit)
    for a in cand_ids:
        g = groups[a]
        mu = infos[a]["mu"]
        old_reward = infos[a]["reward"]
        idxs = _ordered_outlier_indices(X, g, mu, cell_candidate_limit)
        left_cands = _ordered_leftover(X, leftover, mu, w, leftover_candidate_limit)
        for i in idxs:
            out_cell = g[i]
            for in_cell in left_cands:
                new_g = g[:]
                new_g[i] = in_cell
                info_new = _evaluate_group_info(X, new_g, K, delta_bar, w, lambda_penalty,
                                                 theta1, theta2, theta3, P1, P2, P3)
                if not info_new["feasible"]:
                    continue
                if info_new["reward"] > old_reward + 1e-12:
                    groups[a] = sorted(new_g)
                    leftover.remove(in_cell)
                    leftover.append(out_cell)
                    leftover.sort()
                    return True
    return False


# =========================================================
# N3: 2-2 exchange (first improvement)
# =========================================================

def _n3_exchange22(X, groups, infos, K, delta_bar, w, lambda_penalty,
                    theta1, theta2, theta3, P1, P2, P3,
                    pack_candidate_limit, partner_limit, cell_candidate_limit):
    if K < 2:
        return False
    cand_ids = _select_candidate_group_ids(infos, pack_candidate_limit)
    for a in cand_ids:
        partner_ids = _nearest_partner_ids(infos, a, partner_limit)
        g1 = groups[a]
        mu1 = infos[a]["mu"]
        idxs1 = _ordered_outlier_indices(X, g1, mu1, cell_candidate_limit)
        if len(idxs1) < 2:
            continue
        combs1 = list(itertools.combinations(idxs1, 2))
        for b in partner_ids:
            g2 = groups[b]
            mu2 = infos[b]["mu"]
            idxs2 = _ordered_outlier_indices(X, g2, mu2, cell_candidate_limit)
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
                    info1 = _evaluate_group_info(X, new_g1, K, delta_bar, w, lambda_penalty,
                                                  theta1, theta2, theta3, P1, P2, P3)
                    if not info1["feasible"]:
                        continue
                    info2 = _evaluate_group_info(X, new_g2, K, delta_bar, w, lambda_penalty,
                                                  theta1, theta2, theta3, P1, P2, P3)
                    if not info2["feasible"]:
                        continue
                    if info1["reward"] + info2["reward"] > old_sum + 1e-12:
                        groups[a] = sorted(new_g1)
                        groups[b] = sorted(new_g2)
                        return True
    return False


# =========================================================
# N4: destroy-repair (large neighborhood)
# =========================================================

def _repair_from_pool_greedy(X, fixed_groups, pool, K, k_t, w):
    groups = [sorted(g) for g in fixed_groups]
    pool = sorted(set(pool))

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


def _n4_destroy_repair(X, groups, infos, leftover, total_reward, K, k_t, delta_bar,
                       w, lambda_penalty, theta1, theta2, theta3, P1, P2, P3,
                       destroy_size, pack_candidate_limit):
    cand_ids = _select_candidate_group_ids(infos, max(pack_candidate_limit, destroy_size))
    if len(cand_ids) == 0:
        return False, groups, leftover
    destroy_ids = set(cand_ids[:min(destroy_size, len(cand_ids))])
    fixed_groups = []
    pool = leftover[:]
    for gid, g in enumerate(groups):
        if gid in destroy_ids:
            pool.extend(g)
        else:
            fixed_groups.append(g[:])
    rebuilt, new_leftover = _repair_from_pool_greedy(X, fixed_groups, pool, K, k_t, w)
    new_infos, new_total = _recompute_all_infos(
        X, rebuilt, K, delta_bar, w, lambda_penalty,
        theta1, theta2, theta3, P1, P2, P3)
    feasible = [g for g, info in zip(rebuilt, new_infos) if info["feasible"]]
    if new_total > total_reward + 1e-12:
        used = set()
        for g in feasible:
            used.update(g)
        final_leftover = [i for i in range(X.shape[0]) if i not in used]
        return True, feasible, final_leftover
    return False, groups, leftover


# =========================================================
# N5: merge-split (recombine 2 groups optimally)
# =========================================================

def _n5_merge_split(X, groups, infos, leftover, K, delta_bar, w, lambda_penalty,
                     theta1, theta2, theta3, P1, P2, P3, pack_candidate_limit, partner_limit):
    """Merge-split on the two lowest-reward groups. Only check top few partitions."""
    if len(groups) < 2:
        return False, groups, leftover

    # Only pick the two worst groups for efficiency
    scored = [(infos[i]["reward"] if infos[i]["feasible"] else -np.inf, i)
              for i in range(len(groups))]
    scored.sort()
    a, b = scored[0][1], scored[1][1]

    pool = groups[a] + groups[b]
    old_sum = infos[a]["reward"] + infos[b]["reward"]

    # Sample partitions instead of full enumeration
    # For K=8, C(16,8)=12870 is too slow. Use a subset.
    best_new_sum = old_sum
    best_split = None

    # Strategy: try partitions based on nearest-neighbor splits
    mu_a = infos[a]["mu"]
    mu_b = infos[b]["mu"]
    mid = (mu_a + mu_b) / 2.0

    # Split 1: assign each cell to whichever old centroid is closer
    g1_a, g1_b = [], []
    for cell in pool:
        if float(np.sum((X[cell] - mu_a) ** 2)) < float(np.sum((X[cell] - mu_b) ** 2)):
            g1_a.append(cell)
        else:
            g1_b.append(cell)

    # Balance to exactly K each
    while len(g1_a) > K and len(g1_b) < K:
        # move cell farthest from mu_a
        best_idx = max(range(len(g1_a)), key=lambda i: float(np.sum((X[g1_a[i]] - mu_a) ** 2)))
        g1_b.append(g1_a.pop(best_idx))
    while len(g1_b) > K and len(g1_a) < K:
        best_idx = max(range(len(g1_b)), key=lambda i: float(np.sum((X[g1_b[i]] - mu_b) ** 2)))
        g1_a.append(g1_b.pop(best_idx))

    if len(g1_a) == K and len(g1_b) == K:
        info1 = _evaluate_group_info(X, sorted(g1_a), K, delta_bar, w, lambda_penalty,
                                      theta1, theta2, theta3, P1, P2, P3)
        info2 = _evaluate_group_info(X, sorted(g1_b), K, delta_bar, w, lambda_penalty,
                                      theta1, theta2, theta3, P1, P2, P3)
        if info1["feasible"] and info2["feasible"]:
            s = info1["reward"] + info2["reward"]
            if s > best_new_sum + 1e-12:
                best_split = (sorted(g1_a), sorted(g1_b))
                best_new_sum = s

    # Split 2: random perturbations of the best split
    if best_split is None:
        rng = np.random.default_rng(42)
        for _ in range(20):
            rng.shuffle(pool)
            new_g1 = sorted(pool[:K])
            new_g2 = sorted(pool[K:])
            info1 = _evaluate_group_info(X, new_g1, K, delta_bar, w, lambda_penalty,
                                          theta1, theta2, theta3, P1, P2, P3)
            if not info1["feasible"]:
                continue
            info2 = _evaluate_group_info(X, new_g2, K, delta_bar, w, lambda_penalty,
                                          theta1, theta2, theta3, P1, P2, P3)
            if not info2["feasible"]:
                continue
            s = info1["reward"] + info2["reward"]
            if s > best_new_sum + 1e-12:
                best_split = (new_g1, new_g2)
                best_new_sum = s

    if best_split is not None:
        groups[a] = best_split[0]
        groups[b] = best_split[1]
        return True, groups, leftover

    return False, groups, leftover


# =========================================================
# Focused VNS on a single start
# =========================================================

def _run_vns(X, groups, leftover, K, k_t, delta_bar, w, lambda_penalty,
             theta1, theta2, theta3, P1, P2, P3,
             max_vns_iter, max_no_improve,
             pack_candidate_limit, partner_limit, cell_candidate_limit,
             leftover_candidate_limit, destroy_size):
    groups, leftover = _extract_feasible_and_leftover(
        X, groups, K, k_t, delta_bar, w, lambda_penalty,
        theta1, theta2, theta3, P1, P2, P3)
    infos, total_reward = _recompute_all_infos(
        X, groups, K, delta_bar, w, lambda_penalty,
        theta1, theta2, theta3, P1, P2, P3)

    no_improve = 0
    it = 0
    while it < max_vns_iter and no_improve < max_no_improve:
        improved = False
        # Phase A: N1, N2, N3
        while True:
            li = False
            if _n1_swap(X, groups, infos, K, delta_bar, w, lambda_penalty,
                         theta1, theta2, theta3, P1, P2, P3,
                         pack_candidate_limit, partner_limit, cell_candidate_limit):
                infos, total_reward = _recompute_all_infos(
                    X, groups, K, delta_bar, w, lambda_penalty,
                    theta1, theta2, theta3, P1, P2, P3)
                li = True
                improved = True
                continue
            if _n2_leftover(X, groups, infos, leftover, K, delta_bar, w, lambda_penalty,
                            theta1, theta2, theta3, P1, P2, P3,
                            pack_candidate_limit, cell_candidate_limit, leftover_candidate_limit):
                infos, total_reward = _recompute_all_infos(
                    X, groups, K, delta_bar, w, lambda_penalty,
                    theta1, theta2, theta3, P1, P2, P3)
                li = True
                improved = True
                continue
            if _n3_exchange22(X, groups, infos, K, delta_bar, w, lambda_penalty,
                               theta1, theta2, theta3, P1, P2, P3,
                               pack_candidate_limit, partner_limit, cell_candidate_limit):
                infos, total_reward = _recompute_all_infos(
                    X, groups, K, delta_bar, w, lambda_penalty,
                    theta1, theta2, theta3, P1, P2, P3)
                li = True
                improved = True
                continue
            if not li:
                break
        # Phase B: N4 destroy-repair
        if not improved:
            found, ng, nl = _n4_destroy_repair(
                X, groups, infos, leftover, total_reward, K, k_t, delta_bar,
                w, lambda_penalty, theta1, theta2, theta3, P1, P2, P3,
                destroy_size, pack_candidate_limit)
            if found:
                groups, leftover = ng, nl
                infos, total_reward = _recompute_all_infos(
                    X, groups, K, delta_bar, w, lambda_penalty,
                    theta1, theta2, theta3, P1, P2, P3)
                improved = True
        # Phase C: N5 merge-split (once, only when stuck)
        if not improved and no_improve >= max_no_improve - 2 and len(groups) >= 2:
            found, ng, nl = _n5_merge_split(
                X, groups, infos, leftover, K, delta_bar, w, lambda_penalty,
                theta1, theta2, theta3, P1, P2, P3,
                pack_candidate_limit, partner_limit)
            if found:
                groups, leftover = ng, nl
                infos, total_reward = _recompute_all_infos(
                    X, groups, K, delta_bar, w, lambda_penalty,
                    theta1, theta2, theta3, P1, P2, P3)
                improved = True
        if improved:
            no_improve = 0
        else:
            no_improve += 1
        it += 1
    return groups, leftover


# =========================================================
# Main solver
# =========================================================

def solve_rrp_ms_kmeans_vns(
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
    n_starts: int = 5,
    kmeans_L1: int = 15,
    kmeans_tol: float = 1e-4,
    max_vns_iter: int = 60,
    max_no_improve: int = 12,
    pack_candidate_limit: int = 8,
    partner_limit: int = 5,
    cell_candidate_limit: int = 3,
    leftover_candidate_limit: int = 12,
    destroy_size: int = 2,
):
    start = time.perf_counter()
    n = X.shape[0]

    if n < K or k_t <= 0:
        return {
            "method": "RRP_MS_KMEANS_VNS",
            "groups": [],
            "leftover": list(range(n)),
            "reward": 0.0,
            "n_packs": 0,
            "avg_delta": 0.0,
            "avg_phi": 0.0,
            "runtime": time.perf_counter() - start,
        }

    base_seed = 42 if seed is None else seed
    rng = np.random.default_rng(base_seed)

    # Step 1: Generate diverse initial solutions
    # - n_km seeds: pure KMeans with different seeds
    # - n_rk seeds: shuffled KMeans for more diversity
    n_km = max(1, n_starts // 2)
    n_rk = n_starts - n_km
    start_seeds = rng.integers(0, 2**31, size=n_starts).tolist()

    candidates = []
    for s in start_seeds[:n_km]:
        groups, leftover = _kmeans_init(X, K, k_t, kmeans_L1, kmeans_tol, seed=int(s))
        candidates.append((groups, leftover))
    for s in start_seeds[n_km:]:
        groups, leftover = _random_kmeans_init(
            X, K, k_t, kmeans_L1, kmeans_tol, seed=int(s), shuffle_frac=0.3)
        candidates.append((groups, leftover))

    # Step 2: Run focused VNS on each candidate
    best_groups = []
    best_leftover = list(range(n))
    best_reward = -np.inf

    for groups, leftover in candidates:
        g, l = _run_vns(
            X, groups, leftover, K, k_t, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3,
            max_vns_iter, max_no_improve,
            pack_candidate_limit, partner_limit, cell_candidate_limit,
            leftover_candidate_limit, destroy_size)

        summary = summarize_solution(
            X=X, groups=g, K=K, w=w, lambda_penalty=lambda_penalty,
            theta1=theta1, theta2=theta2, theta3=theta3,
            P1=P1, P2=P2, P3=P3)

        if summary["total_reward"] > best_reward + 1e-12:
            best_reward = summary["total_reward"]
            best_groups = [sorted(grp) for grp in g]
            best_leftover = l

    # Step 3: On the best solution, run residual packing + extra destroy-repair
    best_groups, best_leftover = residual_pack_repair(
        X=X, groups=best_groups, leftover=best_leftover,
        K=K, k_t=k_t, delta_bar=delta_bar, w=w,
        lambda_penalty=lambda_penalty,
        theta1=theta1, theta2=theta2, theta3=theta3,
        P1=P1, P2=P2, P3=P3,
        min_accept_reward=0.0,
        seed_candidate_limit=12,
        neighbor_candidate_limit=20,
        max_rounds=30,
    )

    # Step 4: One more round of N4 destroy-repair on the final solution
    groups_for_n4 = [sorted(g) for g in best_groups]
    leftover_for_n4 = sorted(best_leftover)
    infos, total = _recompute_all_infos(
        X, groups_for_n4, K, delta_bar, w, lambda_penalty,
        theta1, theta2, theta3, P1, P2, P3)

    # Run up to 3 N4 rounds
    for _ in range(3):
        found, ng, nl = _n4_destroy_repair(
            X, groups_for_n4, infos, leftover_for_n4, total, K, k_t, delta_bar,
            w, lambda_penalty, theta1, theta2, theta3, P1, P2, P3,
            destroy_size, pack_candidate_limit)
        if found:
            groups_for_n4, leftover_for_n4 = ng, nl
            infos, total = _recompute_all_infos(
                X, groups_for_n4, K, delta_bar, w, lambda_penalty,
                theta1, theta2, theta3, P1, P2, P3)
        else:
            break

    best_groups = groups_for_n4
    best_leftover = leftover_for_n4

    summary = summarize_solution(
        X=X, groups=best_groups, K=K, w=w, lambda_penalty=lambda_penalty,
        theta1=theta1, theta2=theta2, theta3=theta3,
        P1=P1, P2=P2, P3=P3)

    used = set()
    for g in best_groups:
        used.update(g)
    best_leftover = [i for i in range(n) if i not in used]

    return {
        "method": "RRP_MS_KMEANS_VNS",
        "groups": best_groups,
        "leftover": best_leftover,
        "reward": summary["total_reward"],
        "n_packs": summary["n_packs"],
        "avg_delta": summary["avg_delta"],
        "avg_phi": summary["avg_phi"],
        "runtime": time.perf_counter() - start,
        "reward_per_pack": summary["reward_per_pack"],
        "positive_pack_ratio": summary["positive_pack_ratio"],
        "tier_counts": summary["tier_counts"],
    }
