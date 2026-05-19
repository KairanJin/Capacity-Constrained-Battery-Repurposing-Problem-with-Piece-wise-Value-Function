# heuristics/rrp_sa.py
"""
Adaptive Simulated Annealing (ASA) for RRP

Combines simulated annealing with VND intensification, tabu memory,
and adaptive temperature control to escape local optima that trap
greedy or VNS-based approaches.

Algorithm:
1. Multi-start initialization (K-means + greedy), pick best as starting point
2. Estimate initial temperature by sampling random moves
3. Main SA loop with geometric cooling:
   a) Pick random neighborhood move (swap / leftover-swap / 2-exchange)
   b) Accept if improving, or probabilistically if worse (exp(-dE/T))
   c) Track tabu moves to prevent cycling
4. Periodic VND intensification: systematic local descent
5. Reheating when stuck: boost temperature to escape deep local optima
6. Final residual packing to extract remaining value from leftovers
"""
import time
import numpy as np

from utils import compute_group_reward, summarize_solution, piecewise_value
from heuristics.residual_packing import residual_pack_repair
from heuristics._grasp_stats import GroupStats, precompute_arrays


# =========================================================
# Initialization helpers
# =========================================================

def _assign_to_nearest_nonfull(X, centers, K):
    n = X.shape[0]
    k = len(centers)
    groups = [[] for _ in range(k)]
    for i in range(n):
        dists = np.sum((centers - X[i]) ** 2, axis=1)
        order = np.argsort(dists)
        for j in order:
            if len(groups[j]) < K:
                groups[j].append(i)
                break
    leftover = []
    for g in groups:
        if len(g) < K:
            leftover.extend(g)
    full = [g for g in groups if len(g) == K]
    return full, leftover


def _kmeans_solution(X, K, k_t, L1, tol, seed):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    init_idx = rng.choice(n, size=k_t, replace=False)
    centers = X[init_idx].copy()
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
    full_groups, leftover = _assign_to_nearest_nonfull(X, centers, K)
    full_groups = [g for g in full_groups if len(g) == K]
    leftover = sorted(set(range(n)) - {c for g in full_groups for c in g})
    return full_groups, leftover


def _greedy_solution(X, K, k_t, w, rng):
    quality = X @ w
    available = list(range(X.shape[0]))
    groups = []
    while len(available) >= K and len(groups) < k_t:
        seed = max(available, key=lambda i: float(quality[i]))
        available.remove(seed)
        scored = []
        x_seed = X[seed]
        for j in available:
            d = float(np.sum((X[j] - x_seed) ** 2))
            q = float(quality[j])
            scored.append((-d + 0.05 * q, j))
        scored.sort(reverse=True)
        chosen = [seed] + [j for _, j in scored[:K - 1]]
        if len(chosen) < K:
            available.append(seed)
            break
        groups.append(sorted(chosen))
        used = set(chosen)
        available = [i for i in available if i not in used]
    return groups, sorted(available)


# =========================================================
# Group evaluation
# =========================================================

def _evaluate_group_simple(X, group, K, delta_bar, w, lambda_penalty,
                            theta1, theta2, theta3, P1, P2, P3):
    """Full evaluation returning (reward, feasible)."""
    if len(group) != K:
        return -np.inf, False
    mu = X[group].mean(axis=0)
    delta = float(np.mean(np.sum((X[group] - mu) ** 2, axis=1)))
    if delta > delta_bar:
        return -np.inf, False
    phi = float(np.dot(w, mu))
    value = piecewise_value(phi, theta1, theta2, theta3, P1, P2, P3)
    return value - lambda_penalty * delta, True


def _build_stats_list(groups, X, X_sq_norms, w_dot_X):
    stats_list = []
    for g in groups:
        st = GroupStats()
        for c in g:
            st.add(c, X, X_sq_norms, w_dot_X)
        stats_list.append(st)
    return stats_list


def _build_single_stats(group, X, X_sq_norms, w_dot_X):
    st = GroupStats()
    for c in group:
        st.add(c, X, X_sq_norms, w_dot_X)
    return st


def _recompute_all_rewards(X, groups, K, delta_bar, w, lambda_penalty,
                            theta1, theta2, theta3, P1, P2, P3):
    total = 0.0
    infos = []
    for g in groups:
        r, feas = _evaluate_group_simple(X, g, K, delta_bar, w, lambda_penalty,
                                          theta1, theta2, theta3, P1, P2, P3)
        info = {"feasible": feas, "reward": r if feas else -np.inf}
        infos.append(info)
        if feas:
            total += r
    return total, infos


# =========================================================
# Incremental delta/phi after swap (O(d))
# =========================================================

def _swap_rewards(stats_a, stats_b, cell_a, cell_b, X, X_sq_norms, w_dot_X,
                  lambda_penalty, delta_bar, theta1, theta2, theta3, P1, P2, P3):
    """
    Compute new rewards for groups a and b after swapping cell_a <-> cell_b.
    Returns (new_reward_a, new_reward_b, feasible_a, feasible_b).
    """
    n = stats_a.count

    # Group a: remove cell_a, add cell_b
    new_sum_sq_a = stats_a.sum_sq - X_sq_norms[cell_a] + X_sq_norms[cell_b]
    new_sum_vec_a = stats_a.sum_vec - X[cell_a] + X[cell_b]
    new_delta_a = new_sum_sq_a / n - (new_sum_vec_a @ new_sum_vec_a) / (n * n)
    if new_delta_a > delta_bar:
        return -np.inf, -np.inf, False, False
    new_phi_a = (stats_a.w_dot_sum - w_dot_X[cell_a] + w_dot_X[cell_b]) / n
    new_reward_a = piecewise_value(new_phi_a, theta1, theta2, theta3, P1, P2, P3) - lambda_penalty * new_delta_a

    # Group b: remove cell_b, add cell_a
    new_sum_sq_b = stats_b.sum_sq - X_sq_norms[cell_b] + X_sq_norms[cell_a]
    new_sum_vec_b = stats_b.sum_vec - X[cell_b] + X[cell_a]
    new_delta_b = new_sum_sq_b / n - (new_sum_vec_b @ new_sum_vec_b) / (n * n)
    if new_delta_b > delta_bar:
        return -np.inf, -np.inf, False, False
    new_phi_b = (stats_b.w_dot_sum - w_dot_X[cell_b] + w_dot_X[cell_a]) / n
    new_reward_b = piecewise_value(new_phi_b, theta1, theta2, theta3, P1, P2, P3) - lambda_penalty * new_delta_b

    return new_reward_a, new_reward_b, True, True


def _leftover_swap_reward(stats, out_cell, in_cell, X, X_sq_norms, w_dot_X,
                           lambda_penalty, delta_bar, theta1, theta2, theta3, P1, P2, P3):
    """Compute new group reward after swapping out_cell with in_cell from leftover."""
    n = stats.count
    new_sum_sq = stats.sum_sq - X_sq_norms[out_cell] + X_sq_norms[in_cell]
    new_sum_vec = stats.sum_vec - X[out_cell] + X[in_cell]
    new_delta = new_sum_sq / n - (new_sum_vec @ new_sum_vec) / (n * n)
    if new_delta > delta_bar:
        return -np.inf, False
    new_phi = (stats.w_dot_sum - w_dot_X[out_cell] + w_dot_X[in_cell]) / n
    new_reward = piecewise_value(new_phi, theta1, theta2, theta3, P1, P2, P3) - lambda_penalty * new_delta
    return new_reward, True


# =========================================================
# Neighborhood move generators (return candidate + dE, don't modify in-place)
# =========================================================

def _generate_swap_candidate(groups, stats_list, infos, tabu_set, rng,
                              X, X_sq_norms, w_dot_X, K,
                              delta_bar, lambda_penalty, theta1, theta2, theta3, P1, P2, P3):
    """
    Generate a random 1-1 swap candidate.
    Returns (a, b, pos_a, pos_b, cell_a, cell_b, dE) or None if no feasible move found.
    """
    if len(groups) < 2:
        return None

    max_tries = 20
    for _ in range(max_tries):
        a = int(rng.integers(0, len(groups)))
        b = int(rng.integers(0, len(groups)))
        if a == b:
            b = (b + 1) % len(groups)
        pa = int(rng.integers(0, len(groups[a])))
        pb = int(rng.integers(0, len(groups[b])))

        cell_a = groups[a][pa]
        cell_b = groups[b][pb]

        tabu_key = (min(cell_a, cell_b), max(cell_a, cell_b))
        if tabu_key in tabu_set:
            continue

        nr_a, nr_b, fa, fb = _swap_rewards(
            stats_list[a], stats_list[b], cell_a, cell_b, X, X_sq_norms, w_dot_X,
            lambda_penalty, delta_bar, theta1, theta2, theta3, P1, P2, P3,
        )
        if not fa or not fb:
            continue

        dE = (nr_a + nr_b) - (infos[a]["reward"] + infos[b]["reward"])
        return a, b, pa, pb, cell_a, cell_b, dE

    return None


def _generate_leftover_candidate(groups, stats_list, infos, leftover, rng,
                                  X, X_sq_norms, w_dot_X, K,
                                  delta_bar, lambda_penalty, theta1, theta2, theta3, P1, P2, P3):
    """Generate a random leftover swap candidate."""
    if len(groups) == 0 or len(leftover) == 0:
        return None

    max_tries = 20
    for _ in range(max_tries):
        a = int(rng.integers(0, len(groups)))
        pa = int(rng.integers(0, len(groups[a])))
        lc = int(rng.integers(0, len(leftover)))

        out_cell = groups[a][pa]
        in_cell = leftover[lc]

        nr, feas = _leftover_swap_reward(
            stats_list[a], out_cell, in_cell, X, X_sq_norms, w_dot_X,
            lambda_penalty, delta_bar, theta1, theta2, theta3, P1, P2, P3,
        )
        if not feas:
            continue

        dE = nr - infos[a]["reward"]
        return a, pa, lc, out_cell, in_cell, dE

    return None


def _generate_2exchange_candidate(groups, stats_list, infos, rng, X, w,
                                   K, delta_bar, lambda_penalty, theta1, theta2, theta3, P1, P2, P3):
    """Generate a random 2-2 exchange candidate (full evaluation)."""
    if len(groups) < 2 or K < 2:
        return None

    max_tries = 15
    for _ in range(max_tries):
        a = int(rng.integers(0, len(groups)))
        b = int(rng.integers(0, len(groups)))
        if a == b:
            b = (b + 1) % len(groups)
        if len(groups[a]) < 2 or len(groups[b]) < 2:
            continue

        pos_a1, pos_a2 = sorted(rng.choice(len(groups[a]), size=2, replace=False).tolist())
        pos_b1, pos_b2 = sorted(rng.choice(len(groups[b]), size=2, replace=False).tolist())

        new_ga = groups[a][:]
        new_gb = groups[b][:]
        new_ga[pos_a1], new_ga[pos_a2] = new_gb[pos_b1], new_gb[pos_b2]
        new_gb[pos_b1], new_gb[pos_b2] = groups[a][pos_a1], groups[a][pos_a2]
        new_ga.sort()
        new_gb.sort()

        r_a, fa = _evaluate_group_simple(X, new_ga, K, delta_bar, w, lambda_penalty,
                                          theta1, theta2, theta3, P1, P2, P3)
        r_b, fb = _evaluate_group_simple(X, new_gb, K, delta_bar, w, lambda_penalty,
                                          theta1, theta2, theta3, P1, P2, P3)
        if not fa or not fb:
            continue

        dE = (r_a + r_b) - (infos[a]["reward"] + infos[b]["reward"])
        return a, b, pos_a1, pos_a2, pos_b1, pos_b2, dE, new_ga, new_gb

    return None


# =========================================================
# Apply moves in-place
# =========================================================

def _apply_swap(groups, stats_list, a, b, pa, pb, cell_a, cell_b, X, X_sq_norms, w_dot_X):
    groups[a][pa], groups[b][pb] = cell_b, cell_a
    groups[a].sort()
    groups[b].sort()
    stats_list[a] = _build_single_stats(groups[a], X, X_sq_norms, w_dot_X)
    stats_list[b] = _build_single_stats(groups[b], X, X_sq_norms, w_dot_X)


def _apply_leftover_swap(groups, stats_list, leftover, a, pa, lc, out_cell, in_cell,
                          X, X_sq_norms, w_dot_X):
    groups[a][pa] = in_cell
    groups[a].sort()
    stats_list[a] = _build_single_stats(groups[a], X, X_sq_norms, w_dot_X)
    leftover[lc] = out_cell
    leftover.sort()


def _apply_2exchange(groups, stats_list, a, b, pos_a1, pos_a2, pos_b1, pos_b2,
                      new_ga, new_gb, X, X_sq_norms, w_dot_X):
    groups[a] = new_ga
    groups[b] = new_gb
    stats_list[a] = _build_single_stats(new_ga, X, X_sq_norms, w_dot_X)
    stats_list[b] = _build_single_stats(new_gb, X, X_sq_norms, w_dot_X)


# =========================================================
# VND intensification (best-improvement, exhaustive within neighborhoods)
# =========================================================

def _vnd_intensification(groups, leftover, X, K, k_t, delta_bar, w, lambda_penalty,
                          theta1, theta2, theta3, P1, P2, P3,
                          X_sq_norms, w_dot_X, max_vnd_rounds=10):
    """
    Variable Neighborhood Descent: systematically search N1 then N2 with
    best-improvement acceptance until no improvement.
    """
    if len(groups) == 0:
        return groups, leftover

    stats_list = _build_stats_list(groups, X, X_sq_norms, w_dot_X)
    total_reward, infos = _recompute_all_rewards(
        X, groups, K, delta_bar, w, lambda_penalty,
        theta1, theta2, theta3, P1, P2, P3,
    )

    for _ in range(max_vnd_rounds):
        # N1: best 1-1 swap
        best_dE = 0.0
        best_move = None

        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                for pa in range(len(groups[a])):
                    for pb in range(len(groups[b])):
                        cell_a = groups[a][pa]
                        cell_b = groups[b][pb]
                        nr_a, nr_b, fa, fb = _swap_rewards(
                            stats_list[a], stats_list[b], cell_a, cell_b, X, X_sq_norms, w_dot_X,
                            lambda_penalty, delta_bar, theta1, theta2, theta3, P1, P2, P3,
                        )
                        if not fa or not fb:
                            continue
                        dE = (nr_a + nr_b) - (infos[a]["reward"] + infos[b]["reward"])
                        if dE > best_dE + 1e-12:
                            best_dE = dE
                            best_move = ('swap', a, b, pa, pb, cell_a, cell_b)

        if best_move is not None:
            _, a, b, pa, pb, ca, cb = best_move
            _apply_swap(groups, stats_list, a, b, pa, pb, ca, cb, X, X_sq_norms, w_dot_X)
            total_reward, infos = _recompute_all_rewards(
                X, groups, K, delta_bar, w, lambda_penalty,
                theta1, theta2, theta3, P1, P2, P3,
            )
            continue

        # N2: best leftover swap
        if len(leftover) > 0:
            best_dE2 = 0.0
            best_move2 = None

            for a in range(len(groups)):
                for pa in range(len(groups[a])):
                    for lc in range(len(leftover)):
                        nr, feas = _leftover_swap_reward(
                            stats_list[a], groups[a][pa], leftover[lc], X, X_sq_norms, w_dot_X,
                            lambda_penalty, delta_bar, theta1, theta2, theta3, P1, P2, P3,
                        )
                        if not feas:
                            continue
                        dE = nr - infos[a]["reward"]
                        if dE > best_dE2 + 1e-12:
                            best_dE2 = dE
                            best_move2 = ('leftover', a, pa, lc, groups[a][pa], leftover[lc])

            if best_move2 is not None:
                _, a, pa, lc, out_c, in_c = best_move2
                _apply_leftover_swap(groups, stats_list, leftover, a, pa, lc, out_c, in_c,
                                      X, X_sq_norms, w_dot_X)
                total_reward, infos = _recompute_all_rewards(
                    X, groups, K, delta_bar, w, lambda_penalty,
                    theta1, theta2, theta3, P1, P2, P3,
                )
                continue

        break  # no improvement in any neighborhood

    return groups, leftover


# =========================================================
# Adaptive temperature estimation
# =========================================================

def _estimate_initial_temperature(groups, leftover, X, K, delta_bar, w, lambda_penalty,
                                   theta1, theta2, theta3, P1, P2, P3,
                                   X_sq_norms, w_dot_X, rng, n_samples=50):
    """
    Sample random moves, compute worsening |dE| values, set T = mean(|dE|) / ln(2)
    so that exp(-mean_de / T) ~ 0.5.
    """
    if len(groups) < 2:
        return 1.0

    _, infos = _recompute_all_rewards(
        X, groups, K, delta_bar, w, lambda_penalty,
        theta1, theta2, theta3, P1, P2, P3,
    )
    stats_list = _build_stats_list(groups, X, X_sq_norms, w_dot_X)
    worsening = []

    for _ in range(n_samples):
        a = int(rng.integers(0, len(groups)))
        b = int(rng.integers(0, len(groups)))
        if a == b:
            b = (b + 1) % len(groups)
        pa = int(rng.integers(0, len(groups[a])))
        pb = int(rng.integers(0, len(groups[b])))
        cell_a = groups[a][pa]
        cell_b = groups[b][pb]

        nr_a, nr_b, fa, fb = _swap_rewards(
            stats_list[a], stats_list[b], cell_a, cell_b, X, X_sq_norms, w_dot_X,
            lambda_penalty, delta_bar, theta1, theta2, theta3, P1, P2, P3,
        )
        if fa and fb:
            dE = (nr_a + nr_b) - (infos[a]["reward"] + infos[b]["reward"])
            if dE < 0:
                worsening.append(abs(dE))

    if len(leftover) > 0:
        for _ in range(n_samples // 2):
            a = int(rng.integers(0, len(groups)))
            pa = int(rng.integers(0, len(groups[a])))
            lc = int(rng.integers(0, len(leftover)))
            nr, feas = _leftover_swap_reward(
                stats_list[a], groups[a][pa], leftover[lc], X, X_sq_norms, w_dot_X,
                lambda_penalty, delta_bar, theta1, theta2, theta3, P1, P2, P3,
            )
            if feas and nr < infos[a]["reward"]:
                worsening.append(infos[a]["reward"] - nr)

    if not worsening:
        return 1.0
    mean_de = float(np.mean(worsening))
    return mean_de / np.log(2) if mean_de > 1e-10 else 1.0


# =========================================================
# Main SA solver
# =========================================================

def solve_rrp_sa(
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
    # SA parameters
    initial_temperature: float | None = None,
    cooling_rate: float = 0.995,
    min_temperature: float = 1e-4,
    max_sa_iterations: int = 5000,
    # VND parameters
    vnd_interval: int = 200,
    max_vnd_rounds: int = 5,
    # Reheating
    reheating_ratio: float = 3.0,
    reheating_stall: int = 500,
    max_reheats: int = 3,
    # Tabu
    tabu_tenure: int = 25,
    # Initialization
    n_init_starts: int = 5,
    kmeans_L1: int = 15,
    kmeans_tol: float = 1e-4,
    # Residual packing
    residual_rounds: int = 20,
):
    start = time.perf_counter()
    n = X.shape[0]

    if n < K or k_t <= 0:
        return {
            "method": "RRP_SA",
            "groups": [],
            "leftover": list(range(n)),
            "reward": 0.0,
            "n_packs": 0,
            "avg_delta": 0.0,
            "avg_phi": 0.0,
            "runtime": time.perf_counter() - start,
        }

    rng = np.random.default_rng(seed)
    X_sq_norms, w_dot_X = precompute_arrays(X, np.asarray(w))
    w_arr = np.asarray(w)

    # ---- Step 1: Multi-start initialization ----
    candidates = []

    # K-means starts
    km_seeds = rng.integers(0, 2**31, size=n_init_starts).tolist()
    for s in km_seeds:
        groups, leftover = _kmeans_solution(X, K, k_t, kmeans_L1, kmeans_tol, seed=int(s))
        feasible = []
        used = set()
        for g in groups:
            r, feas = _evaluate_group_simple(X, g, K, delta_bar, w_arr, lambda_penalty,
                                              theta1, theta2, theta3, P1, P2, P3)
            if feas:
                feasible.append(sorted(g))
                used.update(g)
        candidates.append((feasible, sorted(set(range(n)) - used)))

    # Greedy start
    groups, leftover = _greedy_solution(X, K, k_t, w_arr, rng)
    feasible = []
    used = set()
    for g in groups:
        r, feas = _evaluate_group_simple(X, g, K, delta_bar, w_arr, lambda_penalty,
                                          theta1, theta2, theta3, P1, P2, P3)
        if feas:
            feasible.append(sorted(g))
            used.update(g)
    candidates.append((feasible, sorted(set(range(n)) - used)))

    # Pick best initial solution
    best_groups, best_leftover = [], list(range(n))
    best_reward = -np.inf
    for groups, leftover in candidates:
        total, _ = _recompute_all_rewards(
            X, groups, K, delta_bar, w_arr, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3,
        )
        if total > best_reward:
            best_reward = total
            best_groups = [g[:] for g in groups]
            best_leftover = leftover[:]

    # ---- Step 2: Set up current solution ----
    current_groups = [g[:] for g in best_groups]
    current_leftover = best_leftover[:]
    current_reward, current_infos = _recompute_all_rewards(
        X, current_groups, K, delta_bar, w_arr, lambda_penalty,
        theta1, theta2, theta3, P1, P2, P3,
    )
    current_stats = _build_stats_list(current_groups, X, X_sq_norms, w_dot_X)

    best_groups = [g[:] for g in current_groups]
    best_leftover = current_leftover[:]
    best_reward = current_reward

    # ---- Step 3: Estimate initial temperature ----
    T = initial_temperature if initial_temperature is not None else _estimate_initial_temperature(
        current_groups, current_leftover, X, K, delta_bar, w_arr, lambda_penalty,
        theta1, theta2, theta3, P1, P2, P3, X_sq_norms, w_dot_X, rng, n_samples=50,
    )
    T0 = T

    # ---- Step 4: SA main loop ----
    tabu_set = set()
    tabu_list = []  # FIFO list for managing tabu tenure

    stall_count = 0
    reheat_count = 0
    n_accepted = 0

    for sa_iter in range(max_sa_iterations):
        # Clean expired tabu entries
        while tabu_list and tabu_list[0][1] <= sa_iter:
            expired_key = tabu_list.pop(0)[0]
            tabu_set.discard(expired_key)

        # Choose neighborhood type
        r = rng.random()
        candidate = None  # Will hold move info
        move_type = None

        if r < 0.5 and len(current_groups) >= 2:
            candidate = _generate_swap_candidate(
                current_groups, current_stats, current_infos, tabu_set, rng,
                X, X_sq_norms, w_dot_X, K,
                delta_bar, lambda_penalty, theta1, theta2, theta3, P1, P2, P3,
            )
            if candidate is not None:
                move_type = 'swap'

        elif r < 0.8 and len(current_leftover) > 0:
            candidate = _generate_leftover_candidate(
                current_groups, current_stats, current_infos, current_leftover, rng,
                X, X_sq_norms, w_dot_X, K,
                delta_bar, lambda_penalty, theta1, theta2, theta3, P1, P2, P3,
            )
            if candidate is not None:
                move_type = 'leftover'

        elif len(current_groups) >= 2 and K >= 2:
            candidate = _generate_2exchange_candidate(
                current_groups, current_stats, current_infos, rng, X, w_arr,
                K, delta_bar, lambda_penalty, theta1, theta2, theta3, P1, P2, P3,
            )
            if candidate is not None:
                move_type = '2exchange'

        if candidate is None:
            continue

        # Extract dE based on move type
        if move_type == 'swap':
            dE = candidate[6]
        elif move_type == 'leftover':
            dE = candidate[5]
        else:  # 2exchange
            dE = candidate[6]

        # SA acceptance
        accept = dE >= 0 or (T > 1e-10 and rng.random() < np.exp(dE / T))

        if accept:
            if move_type == 'swap':
                a, b, pa, pb, ca, cb, _ = candidate
                _apply_swap(current_groups, current_stats, a, b, pa, pb, ca, cb,
                            X, X_sq_norms, w_dot_X)
                tabu_key = (min(ca, cb), max(ca, cb))
                tabu_set.add(tabu_key)
                tabu_list.append((tabu_key, sa_iter + tabu_tenure))

            elif move_type == 'leftover':
                a, pa, lc, out_c, in_c, _ = candidate
                _apply_leftover_swap(current_groups, current_stats, current_leftover,
                                      a, pa, lc, out_c, in_c, X, X_sq_norms, w_dot_X)

            elif move_type == '2exchange':
                a, b, pa1, pa2, pb1, pb2, _, new_ga, new_gb = candidate
                _apply_2exchange(current_groups, current_stats, a, b, pa1, pa2, pb1, pb2,
                                  new_ga, new_gb, X, X_sq_norms, w_dot_X)

            # Recompute exact reward and infos
            current_reward, current_infos = _recompute_all_rewards(
                X, current_groups, K, delta_bar, w_arr, lambda_penalty,
                theta1, theta2, theta3, P1, P2, P3,
            )
            current_stats = _build_stats_list(current_groups, X, X_sq_norms, w_dot_X)
            n_accepted += 1

            if current_reward > best_reward + 1e-12:
                best_reward = current_reward
                best_groups = [g[:] for g in current_groups]
                best_leftover = current_leftover[:]
                stall_count = 0
            else:
                stall_count += 1
        else:
            stall_count += 1

        # Reheating
        if stall_count >= reheating_stall and reheat_count < max_reheats:
            T = T0 * reheating_ratio
            reheat_count += 1
            stall_count = 0

        # VND intensification
        if (sa_iter + 1) % vnd_interval == 0 and len(current_groups) > 0:
            current_groups, current_leftover = _vnd_intensification(
                current_groups, current_leftover, X, K, k_t, delta_bar, w_arr, lambda_penalty,
                theta1, theta2, theta3, P1, P2, P3, X_sq_norms, w_dot_X, max_vnd_rounds,
            )
            current_reward, current_infos = _recompute_all_rewards(
                X, current_groups, K, delta_bar, w_arr, lambda_penalty,
                theta1, theta2, theta3, P1, P2, P3,
            )
            current_stats = _build_stats_list(current_groups, X, X_sq_norms, w_dot_X)

            if current_reward > best_reward + 1e-12:
                best_reward = current_reward
                best_groups = [g[:] for g in current_groups]
                best_leftover = current_leftover[:]
                stall_count = 0

        # Cooling
        T *= cooling_rate
        if T < min_temperature:
            T = min_temperature

    # ---- Step 5: Residual packing ----
    best_groups, best_leftover = residual_pack_repair(
        X=X, groups=best_groups, leftover=best_leftover,
        K=K, k_t=k_t, delta_bar=delta_bar, w=w_arr,
        lambda_penalty=lambda_penalty,
        theta1=theta1, theta2=theta2, theta3=theta3,
        P1=P1, P2=P2, P3=P3,
        min_accept_reward=0.0,
        seed_candidate_limit=12, neighbor_candidate_limit=20,
        max_rounds=residual_rounds,
        X_sq_norms=X_sq_norms, w_dot_X=w_dot_X,
    )

    summary = summarize_solution(
        X=X, groups=best_groups, K=K, w=w_arr, lambda_penalty=lambda_penalty,
        theta1=theta1, theta2=theta2, theta3=theta3, P1=P1, P2=P2, P3=P3,
    )

    used = set()
    for g in best_groups:
        used.update(g)
    best_leftover = [i for i in range(n) if i not in used]

    return {
        "method": "RRP_SA",
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
