import time
import math
import itertools
import numpy as np

from heuristics.rrp_kmeans import solve_rrp_kmeans
from heuristics.rrp_kmeans_vns import solve_rrp_kmeans_vns
from heuristics.residual_packing import residual_pack_repair
from utils import compute_centroid, compute_delta, compute_group_reward, summarize_solution


# =========================================================
# Basic cached evaluation
# =========================================================

def _group_key(group):
    return tuple(sorted(group))


def _solution_key(groups):
    return tuple(sorted(tuple(sorted(g)) for g in groups))


def _evaluate_group_cached(
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
    reward_cache,
):
    key = _group_key(group)
    if key in reward_cache:
        return reward_cache[key]

    if len(key) != K:
        info = {
            "feasible": False,
            "reward": -np.inf,
            "phi": -np.inf,
            "delta": np.inf,
        }
        reward_cache[key] = info
        return info

    delta = compute_delta(X, list(key))
    if delta > delta_bar:
        info = {
            "feasible": False,
            "reward": -np.inf,
            "phi": -np.inf,
            "delta": float(delta),
        }
        reward_cache[key] = info
        return info

    reward, phi, delta = compute_group_reward(
        X, list(key), w, lambda_penalty, theta1, theta2, theta3, P1, P2, P3
    )
    info = {
        "feasible": True,
        "reward": float(reward),
        "phi": float(phi),
        "delta": float(delta),
    }
    reward_cache[key] = info
    return info


def _pack_reward(
    X,
    g,
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
    reward_cache,
):
    info = _evaluate_group_cached(
        X, g, K, delta_bar, w, lambda_penalty,
        theta1, theta2, theta3, P1, P2, P3,
        reward_cache
    )
    return info["reward"] if info["feasible"] else -np.inf


def _solution_reward(
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
    reward_cache,
):
    feasible_groups = []
    total_reward = 0.0

    for g in groups:
        info = _evaluate_group_cached(
            X, g, K, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3,
            reward_cache,
        )
        if info["feasible"]:
            feasible_groups.append(sorted(g))
            total_reward += info["reward"]

    used = set()
    for g in feasible_groups:
        used.update(g)
    leftover = [i for i in range(X.shape[0]) if i not in used]

    return total_reward, feasible_groups, leftover


# =========================================================
# Fast greedy repair
# =========================================================

def _sample_combinations(candidate_pool, choose_k, rng, sample_combination_limit):
    n_pool = len(candidate_pool)
    if n_pool < choose_k:
        return []

    total_combs = math.comb(n_pool, choose_k)
    if total_combs <= sample_combination_limit:
        return list(itertools.combinations(candidate_pool, choose_k))

    combos = set()
    max_trials = max(sample_combination_limit * 4, 30)
    trials = 0
    while len(combos) < sample_combination_limit and trials < max_trials:
        comb = tuple(sorted(rng.choice(candidate_pool, size=choose_k, replace=False).tolist()))
        combos.add(comb)
        trials += 1
    return list(combos)


def _greedy_repair(
    X,
    fixed_groups,
    pool,
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
    rng,
    reward_cache,
    neighbor_limit=5,
    sample_combination_limit=8,
):
    groups = [sorted(g) for g in fixed_groups]
    available = sorted(set(pool))
    cell_quality = X @ w

    while len(available) >= K and len(groups) < k_t:
        seed = max(available, key=lambda i: float(cell_quality[i]))
        available.remove(seed)

        x_seed = X[seed]
        scored = []
        for j in available:
            d = float(np.sum((X[j] - x_seed) ** 2))
            q = float(cell_quality[j])
            scored.append((-d + 0.05 * q, j))
        scored.sort(reverse=True)

        candidate_pool = [j for _, j in scored[:min(neighbor_limit, len(scored))]]
        if len(candidate_pool) < K - 1:
            available.append(seed)
            break

        best_group = None
        best_reward = -np.inf

        combos = _sample_combinations(
            candidate_pool=candidate_pool,
            choose_k=K - 1,
            rng=rng,
            sample_combination_limit=sample_combination_limit,
        )

        for comb in combos:
            g = sorted([seed] + list(comb))
            r = _pack_reward(
                X, g, K, delta_bar, w, lambda_penalty,
                theta1, theta2, theta3, P1, P2, P3,
                reward_cache,
            )
            if r > best_reward:
                best_reward = r
                best_group = g

        if best_group is None or not np.isfinite(best_reward):
            available.append(seed)
            available.sort(key=lambda i: float(cell_quality[i]))
            if len(available) > 0:
                available.pop(0)
            continue

        groups.append(best_group)
        used = set(best_group)
        available = [i for i in available if i not in used]

    return groups, sorted(available)


# =========================================================
# Lightweight local repair
# =========================================================

def _swap_local_search_once(
    groups,
    X,
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
    reward_cache,
    group_candidate_limit=3,
    cell_candidate_limit=1,
):
    if len(groups) <= 1:
        return groups, False

    scored = []
    for idx, g in enumerate(groups):
        r = _pack_reward(
            X, g, K, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3, reward_cache
        )
        scored.append((r, idx))
    scored.sort()

    cand_group_ids = [idx for _, idx in scored[:min(group_candidate_limit, len(scored))]]

    for a in cand_group_ids:
        g1 = groups[a]
        mu1 = compute_centroid(X, g1)
        idxs1 = sorted(
            range(len(g1)),
            key=lambda p: float(np.sum((X[g1[p]] - mu1) ** 2)),
            reverse=True
        )[:min(cell_candidate_limit, len(g1))]

        for b in range(len(groups)):
            if b == a:
                continue
            g2 = groups[b]
            mu2 = compute_centroid(X, g2)
            idxs2 = sorted(
                range(len(g2)),
                key=lambda p: float(np.sum((X[g2[p]] - mu2) ** 2)),
                reverse=True
            )[:min(cell_candidate_limit, len(g2))]

            old_sum = _pack_reward(
                X, g1, K, delta_bar, w, lambda_penalty,
                theta1, theta2, theta3, P1, P2, P3, reward_cache
            ) + _pack_reward(
                X, g2, K, delta_bar, w, lambda_penalty,
                theta1, theta2, theta3, P1, P2, P3, reward_cache
            )

            for i in idxs1:
                for j in idxs2:
                    new_g1 = g1[:]
                    new_g2 = g2[:]
                    new_g1[i], new_g2[j] = new_g2[j], new_g1[i]

                    new_sum = _pack_reward(
                        X, new_g1, K, delta_bar, w, lambda_penalty,
                        theta1, theta2, theta3, P1, P2, P3, reward_cache
                    ) + _pack_reward(
                        X, new_g2, K, delta_bar, w, lambda_penalty,
                        theta1, theta2, theta3, P1, P2, P3, reward_cache
                    )

                    if new_sum > old_sum + 1e-12:
                        groups[a] = sorted(new_g1)
                        groups[b] = sorted(new_g2)
                        return groups, True

    return groups, False


def _leftover_replace_once(
    groups,
    leftover,
    X,
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
    reward_cache,
    group_candidate_limit=3,
    cell_candidate_limit=1,
    leftover_candidate_limit=4,
):
    if len(groups) == 0 or len(leftover) == 0:
        return groups, leftover, False

    scored = []
    for idx, g in enumerate(groups):
        r = _pack_reward(
            X, g, K, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3, reward_cache
        )
        scored.append((r, idx))
    scored.sort()
    cand_group_ids = [idx for _, idx in scored[:min(group_candidate_limit, len(scored))]]

    for a in cand_group_ids:
        g = groups[a]
        mu = compute_centroid(X, g)
        old_reward = _pack_reward(
            X, g, K, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3, reward_cache
        )

        idxs = sorted(
            range(len(g)),
            key=lambda p: float(np.sum((X[g[p]] - mu) ** 2)),
            reverse=True
        )[:min(cell_candidate_limit, len(g))]

        cand_left = sorted(
            leftover,
            key=lambda c: float(np.dot(w, X[c])) - 0.05 * float(np.sum((X[c] - mu) ** 2)),
            reverse=True
        )[:min(leftover_candidate_limit, len(leftover))]

        for i in idxs:
            out_cell = g[i]
            for in_cell in cand_left:
                new_g = g[:]
                new_g[i] = in_cell

                new_reward = _pack_reward(
                    X, new_g, K, delta_bar, w, lambda_penalty,
                    theta1, theta2, theta3, P1, P2, P3, reward_cache
                )

                if new_reward > old_reward + 1e-12:
                    groups[a] = sorted(new_g)
                    leftover.remove(in_cell)
                    leftover.append(out_cell)
                    leftover.sort()
                    return groups, leftover, True

    return groups, leftover, False


def _light_repair(
    groups,
    leftover,
    X,
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
    reward_cache,
):
    groups = [g[:] for g in groups]
    leftover = leftover[:]

    groups, improved = _swap_local_search_once(
        groups, X, K, delta_bar, w, lambda_penalty,
        theta1, theta2, theta3, P1, P2, P3, reward_cache
    )
    if improved:
        return groups, leftover

    groups, leftover, _ = _leftover_replace_once(
        groups, leftover, X, K, delta_bar, w, lambda_penalty,
        theta1, theta2, theta3, P1, P2, P3, reward_cache
    )
    return groups, leftover


# =========================================================
# Seed solution generation
# =========================================================

def _make_seed_solutions(
    X,
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
    rng,
    n_random_seeds,
):
    seeds = []

    # 1) one light KMeans-VNS
    sol_vns = solve_rrp_kmeans_vns(
        X=X,
        K=K,
        k_t=k_t,
        delta_bar=delta_bar,
        L1=10,
        tol=1e-6,
        max_vns_iter=4,
        max_no_improve=2,
        w=w,
        lambda_penalty=lambda_penalty,
        theta1=theta1,
        theta2=theta2,
        theta3=theta3,
        P1=P1,
        P2=P2,
        P3=P3,
        seed=int(rng.integers(1, 10**9)),
        pack_candidate_limit=5,
        partner_limit=3,
        cell_candidate_limit=2,
        leftover_candidate_limit=6,
        destroy_size=1,
    )
    seeds.append((sol_vns["groups"], sol_vns["leftover"]))

    # 2) one plain KMeans
    sol_km = solve_rrp_kmeans(
        X=X,
        K=K,
        k_t=k_t,
        delta_bar=delta_bar,
        L1=10,
        L2=4,
        tol=1e-6,
        w=w,
        lambda_penalty=lambda_penalty,
        theta1=theta1,
        theta2=theta2,
        theta3=theta3,
        P1=P1,
        P2=P2,
        P3=P3,
        seed=int(rng.integers(1, 10**9)),
    )
    seeds.append((sol_km["groups"], sol_km["leftover"]))

    # 3) random greedy seeds
    reward_cache = {}
    for _ in range(n_random_seeds):
        pool = list(range(X.shape[0]))
        rng.shuffle(pool)
        groups, leftover = _greedy_repair(
            X=X,
            fixed_groups=[],
            pool=pool,
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
            reward_cache=reward_cache,
            neighbor_limit=5,
            sample_combination_limit=8,
        )
        seeds.append((groups, leftover))

    return seeds


# =========================================================
# Pack-pool recombination
# =========================================================

def _extract_pack_pool(
    seed_solutions,
    X,
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
    reward_cache,
):
    pool = {}
    for groups, _ in seed_solutions:
        for g in groups:
            key = _group_key(g)
            if key not in pool:
                info = _evaluate_group_cached(
                    X, g, K, delta_bar, w, lambda_penalty,
                    theta1, theta2, theta3, P1, P2, P3,
                    reward_cache,
                )
                if info["feasible"]:
                    pool[key] = {
                        "cells": list(key),
                        "reward": info["reward"],
                        "phi": info["phi"],
                        "delta": info["delta"],
                    }

    pack_list = list(pool.values())
    pack_list.sort(key=lambda c: c["reward"], reverse=True)
    return pack_list


def _recombine_from_pack_pool(
    X,
    pack_pool,
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
    rng,
    reward_cache,
    top_pool_ratio=0.7,
):
    if len(pack_pool) == 0:
        return [], list(range(X.shape[0]))

    top_cut = max(1, int(len(pack_pool) * top_pool_ratio))
    top_packs = pack_pool[:top_cut]
    rest_packs = pack_pool[top_cut:]

    selected = []
    used = set()

    # phase 1: greedily inherit from top pack pool
    for col in top_packs:
        g = col["cells"]
        if len(selected) >= k_t:
            break
        if all(c not in used for c in g):
            if rng.random() < 0.85:
                selected.append(sorted(g))
                used.update(g)

    # phase 2: fill with some lower ranked packs
    rng.shuffle(rest_packs)
    for col in rest_packs:
        g = col["cells"]
        if len(selected) >= k_t:
            break
        if all(c not in used for c in g):
            if rng.random() < 0.35:
                selected.append(sorted(g))
                used.update(g)

    pool = [i for i in range(X.shape[0]) if i not in used]

    groups, leftover = _greedy_repair(
        X=X,
        fixed_groups=selected,
        pool=pool,
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
        reward_cache=reward_cache,
        neighbor_limit=5,
        sample_combination_limit=8,
    )

    return groups, leftover


# =========================================================
# Main solver
# =========================================================

def solve_rrp_combine_repair(
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
    n_random_seeds: int = 3,
    n_recombine_rounds: int = 6,
    elite_keep: int = 4,
    use_final_residual_repair: bool = True,
):
    start = time.perf_counter()
    n = X.shape[0]

    if n < K or k_t <= 0:
        return {
            "method": "RRP_COMBINE_REPAIR",
            "groups": [],
            "leftover": list(range(n)),
            "reward": 0.0,
            "n_packs": 0,
            "avg_delta": 0.0,
            "avg_phi": 0.0,
            "runtime": time.perf_counter() - start,
        }

    rng = np.random.default_rng(seed)
    reward_cache = {}

    # -----------------------------------------------------
    # Step 1: seed solutions
    # -----------------------------------------------------
    seed_solutions = _make_seed_solutions(
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
        n_random_seeds=n_random_seeds,
    )

    # evaluate seed solutions
    elite = []
    seen = set()
    for groups, leftover in seed_solutions:
        reward, feasible_groups, leftover2 = _solution_reward(
            X, groups, K, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3, reward_cache
        )
        key = _solution_key(feasible_groups)
        if key not in seen:
            elite.append({
                "groups": feasible_groups,
                "leftover": leftover2,
                "fitness": reward,
            })
            seen.add(key)

    elite.sort(key=lambda s: s["fitness"], reverse=True)
    elite = elite[:elite_keep]

    # -----------------------------------------------------
    # Step 2: repeated recombination
    # -----------------------------------------------------
    best = elite[0]

    for _ in range(n_recombine_rounds):
        base_solutions = [(s["groups"], s["leftover"]) for s in elite]
        pack_pool = _extract_pack_pool(
            base_solutions,
            X, K, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3,
            reward_cache,
        )

        groups, leftover = _recombine_from_pack_pool(
            X=X,
            pack_pool=pack_pool,
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
            reward_cache=reward_cache,
        )

        groups, leftover = _light_repair(
            groups, leftover, X, K, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3,
            reward_cache,
        )

        reward, feasible_groups, leftover2 = _solution_reward(
            X, groups, K, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3, reward_cache
        )

        candidate = {
            "groups": feasible_groups,
            "leftover": leftover2,
            "fitness": reward,
        }

        key = _solution_key(candidate["groups"])
        if key not in seen:
            elite.append(candidate)
            seen.add(key)

        elite.sort(key=lambda s: s["fitness"], reverse=True)
        elite = elite[:elite_keep]
        best = elite[0]

    # -----------------------------------------------------
    # Step 3: optional final residual repair
    # -----------------------------------------------------
    best_groups = best["groups"]
    best_leftover = best["leftover"]

    if use_final_residual_repair:
        best_groups, best_leftover = residual_pack_repair(
            X=X,
            groups=best_groups,
            leftover=best_leftover,
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
            seed_candidate_limit=8,
            neighbor_candidate_limit=8,
            max_rounds=12,
        )

    summary = summarize_solution(
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
        "method": "RRP_COMBINE_REPAIR",
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