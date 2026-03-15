# heuristics/rrp_column_generation.py
import time
import itertools
import numpy as np
import pulp

from utils import compute_delta, compute_group_reward


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