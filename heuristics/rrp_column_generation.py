# heuristics/rrp_column_generation.py
import time
import itertools
import numpy as np
import pulp

from utils import compute_delta, compute_group_reward

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


def _group_key(group):
    return tuple(sorted(group))


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
        return None

    delta = compute_delta(X, list(group))
    if delta > delta_bar:
        return None

    reward, phi, delta = compute_group_reward(
        X, list(group), w, lambda_penalty, theta1, theta2, theta3, P1, P2, P3
    )

    return {
        "cells": tuple(sorted(group)),
        "reward": reward,
        "phi": phi,
        "delta": delta,
    }


def _build_initial_columns(
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
    seed=0,
    n_starts=30,
    neighbor_size=8,
):
    """
    初始列集：围绕 seed cell 构造若干局部邻域组合
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    starts = rng.choice(n, size=min(n, n_starts), replace=False)

    columns = {}
    dmat = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)

    for i in starts:
        neigh = np.argsort(dmat[i])[1:1 + min(neighbor_size, n - 1)]
        if len(neigh) < K - 1:
            continue

        for comb in itertools.combinations(neigh, K - 1):
            group = tuple(sorted((i,) + comb))
            col = _evaluate_group(
                X, group, K, delta_bar, w, lambda_penalty,
                theta1, theta2, theta3, P1, P2, P3
            )
            if col is not None:
                columns[_group_key(group)] = col

    return list(columns.values())


def _solve_rmp_lp(columns, n_cells, k_t):
    """
    Solve LP relaxation of restricted master problem.
    max sum r_g x_g
    s.t. sum_{g: i in g} x_g <= 1      for each cell i
         sum_g x_g <= k_t
         x_g >= 0
    """
    prob = pulp.LpProblem("RMP_LP", pulp.LpMaximize)

    x_vars = {}
    for idx, col in enumerate(columns):
        x_vars[idx] = pulp.LpVariable(f"x_{idx}", lowBound=0, cat="Continuous")

    prob += pulp.lpSum(columns[idx]["reward"] * x_vars[idx] for idx in x_vars)

    cover_cons = {}
    for i in range(n_cells):
        involved = [idx for idx, col in enumerate(columns) if i in col["cells"]]
        cons = pulp.lpSum(x_vars[idx] for idx in involved) <= 1
        cname = f"cover_{i}"
        prob += cons, cname
        cover_cons[i] = prob.constraints[cname]

    prob += pulp.lpSum(x_vars[idx] for idx in x_vars) <= k_t, "pack_limit"
    pack_limit_cons = prob.constraints["pack_limit"]

    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    pi = np.array([cover_cons[i].pi for i in range(n_cells)], dtype=float)
    sigma = float(pack_limit_cons.pi)

    x_val = np.array([pulp.value(x_vars[idx]) for idx in x_vars], dtype=float)
    obj_val = float(pulp.value(prob.objective)) if prob.objective is not None else 0.0

    return {
        "objective": obj_val,
        "x_val": x_val,
        "pi": pi,
        "sigma": sigma,
    }


def _pricing_by_local_enumeration(
    X,
    pi,
    sigma,
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
    existing_keys,
    max_new_cols=20,
    n_seeds=40,
    neighbor_size=10,
    seed=0,
):
    """
    子问题：基于对偶变量做 reduced-cost 搜索
    reduced cost = reward(g) - sum_{i in g} pi_i - sigma
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    dmat = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)

    seed_nodes = rng.choice(n, size=min(n, n_seeds), replace=False)

    candidates = []
    seen = set()

    for i in seed_nodes:
        neigh = np.argsort(dmat[i])[1:1 + min(neighbor_size, n - 1)]
        if len(neigh) < K - 1:
            continue

        for comb in itertools.combinations(neigh, K - 1):
            group = tuple(sorted((i,) + comb))
            gkey = _group_key(group)
            if gkey in existing_keys or gkey in seen:
                continue
            seen.add(gkey)

            col = _evaluate_group(
                X, group, K, delta_bar, w, lambda_penalty,
                theta1, theta2, theta3, P1, P2, P3
            )
            if col is None:
                continue

            reduced_cost = col["reward"] - np.sum(pi[list(group)]) - sigma
            if reduced_cost > 1e-8:
                col["reduced_cost"] = float(reduced_cost)
                candidates.append(col)

    candidates.sort(key=lambda c: c["reduced_cost"], reverse=True)
    return candidates[:max_new_cols]


def _pricing_by_gurobi(
    X,
    pi,
    sigma,
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
    existing_keys,
    max_new_cols=20,
    n_clusters=8,
    neighbor_size=15,
    seed=0,
):
    """
    子问题：使用 Gurobi 求解
    由于 reward 函数是非线性的（基于 tier 阈值），
    我们使用 KMeans 聚类 + Gurobi 来寻找高质量的 group。

    策略：
    1. 对每个高质量 cell 作为中心，构建候选集合
    2. 使用 Gurobi 求解：从候选集合中选择 K 个 cells，最大化 reduced cost
    """
    if not GUROBI_AVAILABLE:
        # Fallback to local enumeration if Gurobi is not available
        return _pricing_by_local_enumeration(
            X, pi, sigma, K, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3, existing_keys,
            max_new_cols=max_new_cols, n_seeds=40, neighbor_size=10, seed=seed
        )

    n = X.shape[0]
    rng = np.random.default_rng(seed)
    dmat = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)

    # Calculate adjusted quality (pi acts as a cost, so higher pi = lower quality)
    adjusted_quality = -pi  # We want to minimize sum of pi, so negative pi is "quality"

    # Select seed cells with high adjusted quality
    seed_scores = adjusted_quality + 0.1 * (X @ w)
    top_seeds = np.argsort(seed_scores)[-min(n_clusters * 3, n):][::-1]

    candidates = []
    seen = set()

    for center_idx in top_seeds:
        # Build candidate pool based on distance and quality
        neigh = np.argsort(dmat[center_idx])[1:1 + min(neighbor_size, n - 1)]
        candidate_pool = list(neigh) + [center_idx]
        candidate_pool = list(set(candidate_pool))

        if len(candidate_pool) < K:
            continue

        # Solve subproblem using Gurobi
        model = gp.Model("Pricing")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', 5)  # 5 second limit per subproblem

        # y[i] = 1 if cell i is selected
        y = model.addVars(len(candidate_pool), vtype=GRB.BINARY, name="y")

        # Constraint: exactly K cells selected
        model.addConstr(gp.quicksum(y[i] for i in range(len(candidate_pool))) == K, "cardinality")

        # Pre-compute delta for all candidate pairs
        pool_indices = candidate_pool
        X_pool = X[pool_indices]

        # Auxiliary variables for delta calculation
        # delta = max pairwise distance in the group
        M = 2 * delta_bar  # Big-M

        # d_max >= distance between any pair of selected cells
        d_max = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="d_max")

        # Add constraints for delta using big-M formulation
        for i_idx, i in enumerate(pool_indices):
            for j_idx, j in enumerate(pool_indices):
                if i_idx < j_idx:
                    dist = float(np.sqrt(np.sum((X[i] - X[j]) ** 2)))
                    # d_max >= dist if both i and j are selected
                    model.addConstr(
                        d_max >= dist - M * (2 - y[i_idx] - y[j_idx]),
                        f"dist_{i_idx}_{j_idx}"
                    )

        # Constraint: delta <= delta_bar
        model.addConstr(d_max <= delta_bar, "delta_limit")

        # Pre-compute rewards for all possible subsets (this is still complex)
        # For efficiency, we use a linear approximation of the reward
        # reward_approx = sum(quality[i] * y[i]) - penalty * variance_approx

        # Simplified objective: maximize sum of adjusted qualities
        objective = gp.quicksum(adjusted_quality[pool_indices[i]] * y[i] for i in range(len(candidate_pool)))
        model.setObjective(objective, GRB.MAXIMIZE)

        # Solve
        model.optimize()

        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            # Extract solution
            selected = [pool_indices[i] for i in range(len(candidate_pool)) if y[i].X > 0.5]

            if len(selected) == K:
                group = tuple(sorted(selected))
                gkey = _group_key(group)

                if gkey not in existing_keys and gkey not in seen:
                    seen.add(gkey)

                    col = _evaluate_group(
                        X, group, K, delta_bar, w, lambda_penalty,
                        theta1, theta2, theta3, P1, P2, P3
                    )

                    if col is not None:
                        reduced_cost = col["reward"] - np.sum(pi[list(group)]) - sigma
                        if reduced_cost > 1e-8:
                            col["reduced_cost"] = float(reduced_cost)
                            candidates.append(col)

        if len(candidates) >= max_new_cols:
            break

    # If we didn't find enough columns, fall back to local enumeration
    if len(candidates) < max_new_cols:
        fallback = _pricing_by_local_enumeration(
            X, pi, sigma, K, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3,
            existing_keys=existing_keys | seen,
            max_new_cols=max_new_cols - len(candidates),
            n_seeds=20,
            neighbor_size=8,
            seed=seed + 1000,
        )
        candidates.extend(fallback)

    candidates.sort(key=lambda c: c["reduced_cost"], reverse=True)
    return candidates[:max_new_cols]


def _solve_final_ip(columns, n_cells, k_t):
    prob = pulp.LpProblem("RMP_IP", pulp.LpMaximize)

    x_vars = {}
    for idx, col in enumerate(columns):
        x_vars[idx] = pulp.LpVariable(f"x_{idx}", lowBound=0, upBound=1, cat="Binary")

    prob += pulp.lpSum(columns[idx]["reward"] * x_vars[idx] for idx in x_vars)

    for i in range(n_cells):
        involved = [idx for idx, col in enumerate(columns) if i in col["cells"]]
        prob += pulp.lpSum(x_vars[idx] for idx in involved) <= 1

    prob += pulp.lpSum(x_vars[idx] for idx in x_vars) <= k_t

    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    chosen = []
    for idx in x_vars:
        val = pulp.value(x_vars[idx])
        if val is not None and val > 0.5:
            chosen.append(columns[idx])

    obj_val = float(pulp.value(prob.objective)) if prob.objective is not None else 0.0
    return chosen, obj_val


def solve_rrp_column_generation(
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
    max_cg_iter: int = 30,
    init_n_starts: int = 30,
    init_neighbor_size: int = 8,
    pricing_n_seeds: int = 40,
    pricing_neighbor_size: int = 10,
    max_new_cols: int = 20,
    seed: int | None = None,
    use_gurobi_pricing: bool = True,
):
    start = time.perf_counter()
    n = X.shape[0]

    if n < K or k_t <= 0:
        return {
            "method": "RRP_COLUMN_GENERATION",
            "groups": [],
            "leftover": list(range(n)),
            "reward": 0.0,
            "n_packs": 0,
            "avg_delta": 0.0,
            "avg_phi": 0.0,
            "runtime": time.perf_counter() - start,
            "n_columns": 0,
        }

    seed_val = 0 if seed is None else seed

    columns = _build_initial_columns(
        X, K, k_t, delta_bar, w, lambda_penalty,
        theta1, theta2, theta3, P1, P2, P3,
        seed=seed_val,
        n_starts=init_n_starts,
        neighbor_size=init_neighbor_size,
    )

    # 若初始列过少，尝试扩大初始列
    if len(columns) == 0:
        columns = _build_initial_columns(
            X, K, k_t, delta_bar, w, lambda_penalty,
            theta1, theta2, theta3, P1, P2, P3,
            seed=seed_val + 1,
            n_starts=min(n, 60),
            neighbor_size=min(n - 1, 12),
        )

    if len(columns) == 0:
        return {
            "method": "RRP_COLUMN_GENERATION",
            "groups": [],
            "leftover": list(range(n)),
            "reward": 0.0,
            "n_packs": 0,
            "avg_delta": 0.0,
            "avg_phi": 0.0,
            "runtime": time.perf_counter() - start,
            "n_columns": 0,
        }

    existing_keys = {_group_key(col["cells"]) for col in columns}

    for it in range(max_cg_iter):
        lp_res = _solve_rmp_lp(columns, n, k_t)

        if use_gurobi_pricing and GUROBI_AVAILABLE:
            new_cols = _pricing_by_gurobi(
                X=X,
                pi=lp_res["pi"],
                sigma=lp_res["sigma"],
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
                existing_keys=existing_keys,
                max_new_cols=max_new_cols,
                n_clusters=pricing_n_seeds // 5,
                neighbor_size=pricing_neighbor_size,
                seed=seed_val + 100 + it,
            )
        else:
            new_cols = _pricing_by_local_enumeration(
                X=X,
                pi=lp_res["pi"],
                sigma=lp_res["sigma"],
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
                existing_keys=existing_keys,
                max_new_cols=max_new_cols,
                n_seeds=pricing_n_seeds,
                neighbor_size=pricing_neighbor_size,
                seed=seed_val + 100 + it,
            )

        if len(new_cols) == 0:
            break

        for col in new_cols:
            gkey = _group_key(col["cells"])
            if gkey not in existing_keys:
                columns.append(col)
                existing_keys.add(gkey)

    chosen_cols, obj_val = _solve_final_ip(columns, n, k_t)

    used = set()
    groups = []
    rewards = []
    phis = []
    deltas = []

    for col in chosen_cols:
        g = list(col["cells"])
        groups.append(g)
        used.update(g)
        rewards.append(col["reward"])
        phis.append(col["phi"])
        deltas.append(col["delta"])

    leftover = [i for i in range(n) if i not in used]

    return {
        "method": "RRP_COLUMN_GENERATION",
        "groups": groups,
        "leftover": leftover,
        "reward": float(np.sum(rewards)) if rewards else 0.0,
        "n_packs": len(groups),
        "avg_delta": float(np.mean(deltas)) if deltas else 0.0,
        "avg_phi": float(np.mean(phis)) if phis else 0.0,
        "runtime": time.perf_counter() - start,
        "n_columns": len(columns),
    }