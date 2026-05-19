# heuristics/rrp_gurobi_exact.py
"""
Gurobi-based exact solver for the Battery Cell Reorganization Problem (RRP).

Two strategies:
1. Full enumeration + set-partitioning IP (pure exact method)
   - Enumerate all C(n,K) combinations, filter by delta_bar
   - Solve set-partitioning IP via Gurobi (pure binary)
   - For N <= ~45, K <= 8 this is tractable

2. Direct MIP formulation via Gurobi
   - Binary assignment x[i,k] + quadratic delta constraint
   - NonConvex=2 for nonconvex quadratic
   - For small instances (N <= 60, k_t <= 10)
"""
from __future__ import annotations

import time
import itertools
import numpy as np

from utils import compute_delta, compute_group_reward, summarize_solution

from config import setup_gurobi_license
setup_gurobi_license()

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = True  # will be set below


# =========================================================
# Method 1: Full enumeration + set-partitioning IP
# =========================================================

def solve_rrp_gurobi_enumeration(
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
    time_limit: float = 300.0,
    max_groups: int = 5000000,
    seed: int | None = None,
):
    """
    Solve RRP by enumerating all feasible groups, then solving
    a set-partitioning integer program via Gurobi.

    Best for small instances (N <= ~45, K <= 8).
    """
    start = time.perf_counter()
    n = X.shape[0]
    w = np.asarray(w, dtype=float)

    if not GUROBI_AVAILABLE:
        raise RuntimeError("gurobipy is required for the exact solver")

    if n < K or k_t <= 0:
        return _empty_result(n, start)

    print(f"  Enumerating all C({n},{K}) combinations...")

    feasible_groups = []
    feasible_rewards = []

    count = 0
    for combo in itertools.combinations(range(n), K):
        count += 1
        if count % 1000000 == 0:
            print(f"  Progress: {count/1e6:.1f}M combos checked, {len(feasible_groups)} feasible so far...")
        if count > max_groups:
            print(f"  Reached max_groups limit ({max_groups}), stopping enumeration.")
            break

        group = list(combo)
        delta_val = float(np.mean(np.sum((X[group] - X[group].mean(axis=0)) ** 2, axis=1)))
        if delta_val > delta_bar:
            continue

        mu = X[group].mean(axis=0)
        phi_val = float(np.dot(w, mu))
        reward = _piecewise_value(phi_val, theta1, theta2, theta3, P1, P2, P3) - lambda_penalty * delta_val

        feasible_groups.append(group)
        feasible_rewards.append(reward)

    n_feasible = len(feasible_groups)
    print(f"  Found {n_feasible} feasible groups out of {count} evaluated.")

    if n_feasible == 0:
        return _empty_result(n, start)

    return _solve_set_partitioning(
        X=X, n=n, K=K, k_t=k_t, w=w, lambda_penalty=lambda_penalty,
        theta1=theta1, theta2=theta2, theta3=theta3, P1=P1, P2=P2, P3=P3,
        feasible_groups=feasible_groups, feasible_rewards=feasible_rewards,
        time_limit=time_limit, seed=seed, start=start,
        method_name="RRP_GUROBI_ENUM",
    )


# =========================================================
# Method 2: Direct MIP formulation via Gurobi
# =========================================================

def solve_rrp_gurobi_mip(
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
    time_limit: float = 60.0,
    mip_gap: float = 0.001,
    threads: int = 4,
    presolve: int = 2,
    method: int = -1,
):
    """
    Solve RRP directly via Gurobi MIP with nonconvex quadratic constraints.

    For small instances only (N <= ~60, k_t <= 10).
    """
    start = time.perf_counter()
    n = X.shape[0]
    d = X.shape[1]
    w = np.asarray(w, dtype=float)

    if not GUROBI_AVAILABLE:
        raise RuntimeError("gurobipy is required for the exact MIP solver")

    if n < K or k_t <= 0:
        return _empty_result(n, start)

    # Precompute squared norms
    X_sq = np.sum(X ** 2, axis=1)

    # Precompute phi bounds for big-M
    all_phi = X @ w
    phi_min = float(np.min(all_phi))
    phi_max = float(np.max(all_phi))
    M_tier = max(phi_max - phi_min, 1e-6)

    eps = 1e-6

    model = gp.Model("RRP_Exact_MIP")
    model.setParam("OutputFlag", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("MIPGap", mip_gap)
    model.setParam("Threads", threads)
    model.setParam("Presolve", presolve)
    model.setParam("Method", method)
    model.setParam("NonConvex", 2)
    if seed is not None:
        model.setParam("Seed", seed)

    # x[i,k] = 1 if cell i assigned to group k
    x = model.addVars(n, k_t, vtype=GRB.BINARY, name="x")

    # tier[k,j] = 1 if group k is in tier j
    tier = model.addVars(k_t, 4, vtype=GRB.BINARY, name="tier")

    # mu[k,d] = centroid coordinate for group k, dimension d
    mu = model.addVars(k_t, d, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="mu")

    # phi[k] = w @ mu[k] for group k
    phi = model.addVars(k_t, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="phi")

    # --- Constraints ---

    # 1. Each cell assigned to at most one group
    model.addConstrs(
        (gp.quicksum(x[i, k] for k in range(k_t)) <= 1 for i in range(n)),
        name="assign_once"
    )

    # 2. Each group has exactly K cells
    model.addConstrs(
        (gp.quicksum(x[i, k] for i in range(n)) == K for k in range(k_t)),
        name="group_size"
    )

    # 3. Centroid definition: mu[k] = (1/K) * sum_i x[i,k] * X[i]
    for k in range(k_t):
        for dd in range(d):
            model.addConstr(
                mu[k, dd] == gp.quicksum(x[i, k] * X[i, dd] for i in range(n)) / K,
                name=f"centroid_{k}_{dd}"
            )

    # 4. Phi definition
    for k in range(k_t):
        model.addConstr(
            phi[k] == gp.quicksum(w[dd] * mu[k, dd] for dd in range(d)),
            name=f"phi_def_{k}"
        )

    # 5. Delta bound: sum_i x[i,k]*||X[i]||^2 - K*sum_d mu[k,d]^2 <= K*delta_bar
    for k in range(k_t):
        linear_part = gp.quicksum(x[i, k] * X_sq[i] for i in range(n))
        quad_part = gp.quicksum(mu[k, dd] * mu[k, dd] for dd in range(d))
        model.addQConstr(
            linear_part - K * quad_part <= K * delta_bar,
            name=f"delta_bound_{k}"
        )

    # 6. Tier selection constraints
    for k in range(k_t):
        model.addConstr(
            gp.quicksum(tier[k, j] for j in range(4)) == 1,
            name=f"tier_one_{k}"
        )
        model.addConstr(phi[k] >= theta1 - M_tier * (1 - tier[k, 1]), name=f"tier1_lower_{k}")
        model.addConstr(phi[k] >= theta2 - M_tier * (1 - tier[k, 2]), name=f"tier2_lower_{k}")
        model.addConstr(phi[k] <= theta1 - eps + M_tier * (1 - tier[k, 2]), name=f"tier2_upper_{k}")
        model.addConstr(phi[k] >= theta3 - M_tier * (1 - tier[k, 3]), name=f"tier3_lower_{k}")
        model.addConstr(phi[k] <= theta2 - eps + M_tier * (1 - tier[k, 3]), name=f"tier3_upper_{k}")
        model.addConstr(phi[k] <= theta3 - eps + M_tier * (1 - tier[k, 0]), name=f"tier0_upper_{k}")

    # --- Objective ---
    obj = gp.LinExpr()
    for k in range(k_t):
        obj += P1 * tier[k, 1] + P2 * tier[k, 2] + P3 * tier[k, 3]

    for i in range(n):
        for k in range(k_t):
            obj += (-lambda_penalty / K) * X_sq[i] * x[i, k]

    for k in range(k_t):
        for dd in range(d):
            obj += lambda_penalty * mu[k, dd] * mu[k, dd]

    model.setObjective(obj, GRB.MAXIMIZE)

    # --- Solve ---
    model.optimize()

    # --- Extract solution ---
    status = model.Status
    if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        return _empty_result(n, start, "RRP_GUROBI_MIP")

    groups = []
    used = set()

    for k in range(k_t):
        group = []
        for i in range(n):
            xval = x[i, k].getAttr(GRB.Attr.X) if model.SolCount > 0 else 0.0
            if xval > 0.5:
                group.append(i)
                used.add(i)
        if len(group) == K:
            delta_val = compute_delta(X, group)
            if delta_val <= delta_bar + 1e-6:
                groups.append(sorted(group))

    leftover = [i for i in range(n) if i not in used]

    summary = summarize_solution(
        X=X, groups=groups, K=K, w=w, lambda_penalty=lambda_penalty,
        theta1=theta1, theta2=theta2, theta3=theta3,
        P1=P1, P2=P2, P3=P3,
    )

    return {
        "method": "RRP_GUROBI_MIP",
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
        "gurobi_status": status,
        "gurobi_obj_val": model.ObjVal if hasattr(model, "ObjVal") else None,
        "gurobi_best_bound": model.ObjBound if hasattr(model, "ObjBound") else None,
        "gurobi_gap": model.MIPGap if hasattr(model, "MIPGap") else None,
        "n_vars": model.NumVars,
        "n_constrs": model.NumConstrs,
    }


# =========================================================
# Shared: Solve set-partitioning IP
# =========================================================

def _solve_set_partitioning(
    X, n, K, k_t, w, lambda_penalty, theta1, theta2, theta3, P1, P2, P3,
    feasible_groups, feasible_rewards, time_limit, seed, start,
    method_name="RRP_GUROBI_ENUM",
):
    """Solve the set-partitioning IP given a list of candidate groups."""
    n_feasible = len(feasible_groups)

    model = gp.Model("RRP_SetPartitioning")
    model.setParam("OutputFlag", 1)
    model.setParam("TimeLimit", time_limit)
    if seed is not None:
        model.setParam("Seed", seed)

    y = model.addVars(n_feasible, vtype=GRB.BINARY, name="y")

    model.setObjective(
        gp.quicksum(feasible_rewards[g] * y[g] for g in range(n_feasible)),
        GRB.MAXIMIZE
    )

    # Each cell used at most once
    for i in range(n):
        cells_in_group = [g for g in range(n_feasible) if i in feasible_groups[g]]
        if cells_in_group:
            model.addConstr(
                gp.quicksum(y[g] for g in cells_in_group) <= 1,
                name=f"cell_{i}"
            )

    # At most k_t groups
    model.addConstr(
        gp.quicksum(y[g] for g in range(n_feasible)) <= k_t,
        name="pack_limit"
    )

    model.optimize()

    status = model.Status
    if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        return _empty_result(n, start, method_name)

    groups = []
    used = set()
    for g in range(n_feasible):
        yval = y[g].getAttr(GRB.Attr.X) if model.SolCount > 0 else 0.0
        if yval > 0.5:
            groups.append(feasible_groups[g])
            used.update(feasible_groups[g])

    leftover = [i for i in range(n) if i not in used]

    summary = summarize_solution(
        X=X, groups=groups, K=K, w=w, lambda_penalty=lambda_penalty,
        theta1=theta1, theta2=theta2, theta3=theta3,
        P1=P1, P2=P2, P3=P3,
    )

    return {
        "method": method_name,
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
        "gurobi_status": status,
        "gurobi_obj_val": model.ObjVal if hasattr(model, "ObjVal") else None,
        "gurobi_best_bound": model.ObjBound if hasattr(model, "ObjBound") else None,
        "gurobi_gap": model.MIPGap if hasattr(model, "MIPGap") else None,
        "n_feasible_groups": n_feasible,
    }


# =========================================================
# Helpers
# =========================================================

def _piecewise_value(phi, theta1, theta2, theta3, P1, P2, P3):
    if phi >= theta1:
        return P1
    elif phi >= theta2:
        return P2
    elif phi >= theta3:
        return P3
    return 0.0


def _empty_result(n, start, method="RRP_GUROBI"):
    return {
        "method": method,
        "groups": [],
        "leftover": list(range(n)),
        "reward": 0.0,
        "n_packs": 0,
        "avg_delta": 0.0,
        "avg_phi": 0.0,
        "runtime": time.perf_counter() - start,
        "reward_per_pack": 0.0,
        "positive_pack_ratio": 0.0,
        "tier_counts": {"P1": 0, "P2": 0, "P3": 0, "P0": 0},
    }
