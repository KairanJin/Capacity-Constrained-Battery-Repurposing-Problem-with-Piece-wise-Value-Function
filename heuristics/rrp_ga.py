import time
import copy
import math
import itertools
import numpy as np

from heuristics.residual_packing import residual_pack_repair
from utils import compute_centroid, compute_delta, compute_group_reward, summarize_solution
from heuristics.rrp_kmeans import solve_rrp_kmeans
from heuristics.rrp_kmeans_vns import solve_rrp_kmeans_vns


# =========================================================
# Basic evaluation with cache
# =========================================================

def _group_key(group):
    return tuple(sorted(group))


def _solution_key(groups):
    """
    Canonical key for a whole solution, used for deduplication in population init.
    """
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
            "delta": delta,
        }
        reward_cache[key] = info
        return info

    reward, phi, delta = compute_group_reward(
        X, list(key), w, lambda_penalty, theta1, theta2, theta3, P1, P2, P3
    )
    info = {
        "feasible": True,
        "reward": reward,
        "phi": phi,
        "delta": delta,
    }
    reward_cache[key] = info
    return info


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
    total = 0.0
    feasible_groups = []

    for g in groups:
        info = _evaluate_group_cached(
            X, g, K, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3,
            reward_cache,
        )
        if info["feasible"]:
            g_sorted = sorted(g)
            feasible_groups.append(g_sorted)
            total += info["reward"]

    used = set()
    for g in feasible_groups:
        used.update(g)
    leftover = [i for i in range(X.shape[0]) if i not in used]

    return total, feasible_groups, leftover


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
        reward_cache,
    )
    return info["reward"] if info["feasible"] else -np.inf


# =========================================================
# Repair / construction helpers
# =========================================================

def _sample_combinations(candidate_pool, choose_k, rng, sample_combination_limit):
    """
    Restricted candidate search:
    - if total combinations small, enumerate all
    - otherwise randomly sample a limited number of unique combinations
    """
    n_pool = len(candidate_pool)
    if n_pool < choose_k:
        return []

    total_combs = math.comb(n_pool, choose_k)
    if total_combs <= sample_combination_limit:
        return list(itertools.combinations(candidate_pool, choose_k))

    combos = set()
    max_trials = max(sample_combination_limit * 5, 50)
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
    neighbor_limit=6,
    sample_combination_limit=15,
):
    groups = [sorted(g) for g in fixed_groups]
    available = sorted(set(pool))

    cell_quality = X @ w

    while len(available) >= K and len(groups) < k_t:
        seed = max(available, key=lambda i: float(cell_quality[i]))
        available.remove(seed)

        scored = []
        x_seed = X[seed]
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
# Individual / population
# =========================================================

def _make_individual(groups, leftover):
    return {
        "groups": [sorted(g) for g in groups],
        "leftover": sorted(leftover),
        "fitness": None,
    }


def _evaluate_individual(
    individual,
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
    reward, feasible_groups, leftover = _solution_reward(
        X,
        individual["groups"],
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
    )
    individual["groups"] = feasible_groups
    individual["leftover"] = leftover
    individual["fitness"] = reward
    return individual


def _random_greedy_individual(
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
    reward_cache,
    neighbor_limit,
    sample_combination_limit,
):
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
        neighbor_limit=neighbor_limit,
        sample_combination_limit=sample_combination_limit,
    )
    return _make_individual(groups, leftover)


def _try_add_individual(population, seen_keys, individual):
    key = _solution_key(individual["groups"])
    if key not in seen_keys:
        population.append(individual)
        seen_keys.add(key)
        return True
    return False


def _initialize_population(
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
    population_size,
    rng,
    reward_cache,
    neighbor_limit,
    sample_combination_limit,
    n_vns_seeds=None,
):
    """
    Improved initialization:
    1) Generate a few high-quality seeds using K-means-VNS (reduced for speed)
    2) Optional 1 plain K-means seed for diversity
    3) Fill the rest with random greedy individuals

    Optimized: reduced VNS seeds significantly for faster runtime.
    """
    population = []
    seen_keys = set()

    if n_vns_seeds is None:
        # Use fewer VNS seeds (only 2-3) for speed - GA will explore the space
        n_vns_seeds = max(2, min(3, population_size // 4))

    # -----------------------------------------------------
    # 1) Limited K-means-VNS seeds with reduced iterations
    # -----------------------------------------------------
    vns_trials = n_vns_seeds + 1

    for _ in range(vns_trials):
        if len(population) >= min(n_vns_seeds, population_size):
            break

        sol = solve_rrp_kmeans_vns(
            X=X,
            K=K,
            k_t=k_t,
            delta_bar=delta_bar,
            L1=12,
            tol=1e-6,
            max_vns_iter=8,
            max_no_improve=3,
            w=w,
            lambda_penalty=lambda_penalty,
            theta1=theta1,
            theta2=theta2,
            theta3=theta3,
            P1=P1,
            P2=P2,
            P3=P3,
            seed=int(rng.integers(1, 10 ** 9)),
        )
        ind = _make_individual(sol["groups"], sol["leftover"])
        _try_add_individual(population, seen_keys, ind)

    # -----------------------------------------------------
    # 2) One plain K-means seed (optional diversity seed)
    # -----------------------------------------------------
    if len(population) < population_size:
        km = solve_rrp_kmeans(
            X=X,
            K=K,
            k_t=k_t,
            delta_bar=delta_bar,
            L1=12,
            L2=6,
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
        ind = _make_individual(km["groups"], km["leftover"])
        _try_add_individual(population, seen_keys, ind)

    # -----------------------------------------------------
    # 3) Fill remaining slots with random greedy individuals
    # -----------------------------------------------------
    random_trials = 0
    max_random_trials = max(15, population_size * 5)

    while len(population) < population_size and random_trials < max_random_trials:
        ind = _random_greedy_individual(
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
            reward_cache=reward_cache,
            neighbor_limit=neighbor_limit,
            sample_combination_limit=sample_combination_limit,
        )
        _try_add_individual(population, seen_keys, ind)
        random_trials += 1

    # fallback: if deduplication makes population still too small, allow duplicates
    while len(population) < population_size:
        ind = _random_greedy_individual(
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
            reward_cache=reward_cache,
            neighbor_limit=neighbor_limit,
            sample_combination_limit=sample_combination_limit,
        )
        population.append(ind)

    return population


# =========================================================
# Selection
# =========================================================

def _tournament_selection(population, tournament_size, rng):
    idx = rng.choice(len(population), size=min(tournament_size, len(population)), replace=False)
    best_idx = max(idx, key=lambda i: population[i]["fitness"])
    return population[best_idx]


# =========================================================
# Reward-biased crossover
# =========================================================

def _reward_biased_crossover(
    parent1,
    parent2,
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
    reward_cache,
    neighbor_limit,
    sample_combination_limit,
    inherit_top_ratio=0.6,
):
    all_packs = []

    for g in parent1["groups"]:
        r = _pack_reward(
            X, g, K, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3,
            reward_cache,
        )
        if np.isfinite(r):
            all_packs.append(("p1", sorted(g), r))

    for g in parent2["groups"]:
        r = _pack_reward(
            X, g, K, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3,
            reward_cache,
        )
        if np.isfinite(r):
            all_packs.append(("p2", sorted(g), r))

    uniq = {}
    for _, g, r in all_packs:
        key = tuple(g)
        if key not in uniq or r > uniq[key][1]:
            uniq[key] = (g, r)
    all_packs = [(g, r) for g, r in uniq.values()]
    all_packs.sort(key=lambda x: x[1], reverse=True)

    child_groups = []
    used = set()

    top_cut = max(1, int(len(all_packs) * inherit_top_ratio))
    top_packs = all_packs[:top_cut]
    rest_packs = all_packs[top_cut:]

    for g, _ in top_packs:
        if len(child_groups) >= k_t:
            break
        if all(c not in used for c in g):
            if rng.random() < 0.85:
                child_groups.append(g)
                used.update(g)

    for g, _ in rest_packs:
        if len(child_groups) >= k_t:
            break
        if all(c not in used for c in g):
            if rng.random() < 0.4:
                child_groups.append(g)
                used.update(g)

    pool = [i for i in range(X.shape[0]) if i not in used]

    child_groups, leftover = _greedy_repair(
        X=X,
        fixed_groups=child_groups,
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
        neighbor_limit=neighbor_limit,
        sample_combination_limit=sample_combination_limit,
    )

    return _make_individual(child_groups, leftover)


# =========================================================
# Mutation: targeted destroy-repair
# =========================================================

def _targeted_destroy_repair_mutation(
    individual,
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
    destroy_size,
    reward_cache,
    neighbor_limit,
    sample_combination_limit,
):
    groups = [g[:] for g in individual["groups"]]
    leftover = individual["leftover"][:]

    if len(groups) == 0:
        return individual

    scored = []
    for idx, g in enumerate(groups):
        info = _evaluate_group_cached(
            X, g, K, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3,
            reward_cache,
        )
        gap = 0.0
        if info["feasible"]:
            phi = info["phi"]
            if phi < theta3:
                gap = theta3 - phi
            elif phi < theta2:
                gap = theta2 - phi
            elif phi < theta1:
                gap = theta1 - phi
            else:
                gap = 999.0
        score = info["reward"] - 0.05 * gap
        scored.append((score, idx))

    scored.sort()
    candidate_ids = [idx for _, idx in scored[:min(max(2, destroy_size + 1), len(scored))]]

    if rng.random() < 0.7:
        chosen = rng.choice(candidate_ids, size=min(destroy_size, len(candidate_ids)), replace=False)
    else:
        chosen = rng.choice(len(groups), size=min(destroy_size, len(groups)), replace=False)

    fixed_groups = []
    pool = leftover[:]

    chosen_set = set(int(x) for x in chosen)
    for idx, g in enumerate(groups):
        if idx in chosen_set:
            pool.extend(g)
        else:
            fixed_groups.append(g)

    new_groups, new_leftover = _greedy_repair(
        X=X,
        fixed_groups=fixed_groups,
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
        neighbor_limit=neighbor_limit,
        sample_combination_limit=sample_combination_limit,
    )

    return _make_individual(new_groups, new_leftover)


# =========================================================
# Selective local search (lightweight)
# =========================================================

def _swap_local_search_once(
    individual,
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
    group_candidate_limit,
    cell_candidate_limit,
    reward_cache,
):
    groups = individual["groups"]
    if len(groups) <= 1:
        return individual, False

    scored = []
    for idx, g in enumerate(groups):
        r = _pack_reward(
            X, g, K, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3,
            reward_cache,
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
                theta1, theta2, theta3, P1, P2, P3,
                reward_cache,
            ) + _pack_reward(
                X, g2, K, delta_bar, w, lambda_penalty,
                theta1, theta2, theta3, P1, P2, P3,
                reward_cache,
            )

            for i in idxs1:
                for j in idxs2:
                    new_g1 = g1[:]
                    new_g2 = g2[:]
                    new_g1[i], new_g2[j] = new_g2[j], new_g1[i]

                    new_sum = _pack_reward(
                        X, new_g1, K, delta_bar, w, lambda_penalty,
                        theta1, theta2, theta3, P1, P2, P3,
                        reward_cache,
                    ) + _pack_reward(
                        X, new_g2, K, delta_bar, w, lambda_penalty,
                        theta1, theta2, theta3, P1, P2, P3,
                        reward_cache,
                    )

                    if new_sum > old_sum + 1e-12:
                        groups[a] = sorted(new_g1)
                        groups[b] = sorted(new_g2)
                        return individual, True

    return individual, False


def _leftover_replace_local_search_once(
    individual,
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
    group_candidate_limit,
    cell_candidate_limit,
    leftover_candidate_limit,
    reward_cache,
):
    groups = individual["groups"]
    leftover = individual["leftover"]

    if len(groups) == 0 or len(leftover) == 0:
        return individual, False

    scored = []
    for idx, g in enumerate(groups):
        r = _pack_reward(
            X, g, K, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3,
            reward_cache,
        )
        scored.append((r, idx))
    scored.sort()
    cand_group_ids = [idx for _, idx in scored[:min(group_candidate_limit, len(scored))]]

    for a in cand_group_ids:
        g = groups[a]
        mu = compute_centroid(X, g)
        old_reward = _pack_reward(
            X, g, K, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3,
            reward_cache,
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
                    theta1, theta2, theta3, P1, P2, P3,
                    reward_cache,
                )

                if new_reward > old_reward + 1e-12:
                    groups[a] = sorted(new_g)
                    leftover.remove(in_cell)
                    leftover.append(out_cell)
                    leftover.sort()
                    return individual, True

    return individual, False


def _selective_local_search(
    individual,
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
    group_candidate_limit,
    cell_candidate_limit,
    leftover_candidate_limit,
    reward_cache,
):
    individual, improved = _swap_local_search_once(
        individual, X, K, delta_bar, w, lambda_penalty,
        theta1, theta2, theta3, P1, P2, P3,
        group_candidate_limit, cell_candidate_limit, reward_cache
    )
    if improved:
        return individual

    individual, _ = _leftover_replace_local_search_once(
        individual, X, K, delta_bar, w, lambda_penalty,
        theta1, theta2, theta3, P1, P2, P3,
        group_candidate_limit, cell_candidate_limit, leftover_candidate_limit, reward_cache
    )
    return individual


# =========================================================
# Main solver
# =========================================================

def solve_rrp_ga(
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
    population_size: int = 12,
    n_generations: int = 12,
    tournament_size: int = 3,
    crossover_prob: float = 0.85,
    mutation_prob: float = 0.35,
    destroy_size: int = 2,
    local_search_prob: float = 0.15,
    elitism_size: int = 2,
    group_candidate_limit: int = 4,
    cell_candidate_limit: int = 2,
    leftover_candidate_limit: int = 6,
    neighbor_limit: int = 6,
    sample_combination_limit: int = 15,
    n_vns_seeds: int | None = None,
    stall_limit: int = 3,
    min_improve: float = 1e-8,
):
    start = time.perf_counter()
    n = X.shape[0]

    if n < K or k_t <= 0:
        return {
            "method": "RRP_GA",
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

    population = _initialize_population(
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
        population_size=population_size,
        rng=rng,
        reward_cache=reward_cache,
        neighbor_limit=neighbor_limit,
        sample_combination_limit=sample_combination_limit,
        n_vns_seeds=n_vns_seeds,
    )

    population = [
        _evaluate_individual(
            ind, X, K, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3,
            reward_cache,
        )
        for ind in population
    ]

    population.sort(key=lambda ind: ind["fitness"], reverse=True)
    best_fitness_so_far = population[0]["fitness"]
    stall_count = 0

    for _ in range(n_generations):
        population.sort(key=lambda ind: ind["fitness"], reverse=True)
        new_population = [copy.deepcopy(ind) for ind in population[:min(elitism_size, len(population))]]

        offspring_pool = []

        while len(new_population) + len(offspring_pool) < population_size:
            parent1 = _tournament_selection(population, tournament_size, rng)
            parent2 = _tournament_selection(population, tournament_size, rng)

            if rng.random() < crossover_prob:
                child = _reward_biased_crossover(
                    parent1, parent2, X, K, k_t, delta_bar, w, lambda_penalty,
                    theta1, theta2, theta3, P1, P2, P3, rng,
                    reward_cache=reward_cache,
                    neighbor_limit=neighbor_limit,
                    sample_combination_limit=sample_combination_limit,
                )
            else:
                child = copy.deepcopy(parent1)

            if rng.random() < mutation_prob:
                child = _targeted_destroy_repair_mutation(
                    child, X, K, k_t, delta_bar, w, lambda_penalty,
                    theta1, theta2, theta3, P1, P2, P3, rng, destroy_size,
                    reward_cache=reward_cache,
                    neighbor_limit=neighbor_limit,
                    sample_combination_limit=sample_combination_limit,
                )

            child = _evaluate_individual(
                child, X, K, delta_bar, w, lambda_penalty,
                theta1, theta2, theta3, P1, P2, P3,
                reward_cache,
            )
            offspring_pool.append(child)

        offspring_pool.sort(key=lambda ind: ind["fitness"], reverse=True)

        if len(offspring_pool) > 0 and rng.random() < local_search_prob:
            offspring_pool[0] = _selective_local_search(
                offspring_pool[0], X, K, delta_bar, w, lambda_penalty,
                theta1, theta2, theta3, P1, P2, P3,
                group_candidate_limit, cell_candidate_limit, leftover_candidate_limit,
                reward_cache,
            )
            offspring_pool[0] = _evaluate_individual(
                offspring_pool[0], X, K, delta_bar, w, lambda_penalty,
                theta1, theta2, theta3, P1, P2, P3,
                reward_cache,
            )

        new_population.extend(offspring_pool)
        new_population.sort(key=lambda ind: ind["fitness"], reverse=True)
        population = new_population[:population_size]

        current_best = population[0]["fitness"]

        if current_best > best_fitness_so_far + min_improve:
            best_fitness_so_far = current_best
            stall_count = 0
        else:
            stall_count += 1

        if stall_count >= stall_limit:
            break

    population.sort(key=lambda ind: ind["fitness"], reverse=True)
    best = population[0]

    best_groups, best_leftover = residual_pack_repair(
        X=X,
        groups=best["groups"],
        leftover=best["leftover"],
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
        max_rounds=15,
        seed_candidate_limit=8,
        neighbor_candidate_limit=8,
    )

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
        "method": "RRP_GA",
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