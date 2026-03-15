# run_two_stage_experiment.py
from __future__ import annotations

import os
import math
import copy
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from config import Config
from data_generator import generate_cells

from heuristics.rrp_kmeans import solve_rrp_kmeans
from heuristics.rrp_kmeans_vns import solve_rrp_kmeans_vns
from heuristics.rrp_grasp import solve_rrp_grasp
from heuristics.rrp_ga import solve_rrp_ga

from outer.tsrah import tsrah_scrapping_decision, default_quality_score
from outer.arrival import gaussian_arrival_generator


# =========================================================
# Utility
# =========================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def stack_inventory_and_arrival(I_t: np.ndarray, A_t: np.ndarray) -> np.ndarray:
    if len(I_t) == 0:
        return A_t.copy()
    if len(A_t) == 0:
        return I_t.copy()
    return np.vstack([I_t, A_t])


# =========================================================
# Inner solver wrapper
# =========================================================

def solve_inner_rrp(X: np.ndarray, cfg: Config, seed: int | None = None, method: str = "VNS") -> dict:
    k_t = min(cfg.problem.k_max, X.shape[0] // cfg.problem.K)

    if method == "KMEANS":
        return solve_rrp_kmeans(
            X=X,
            K=cfg.problem.K,
            k_t=k_t,
            delta_bar=cfg.problem.delta_bar,
            L1=cfg.kmeans.L1,
            L2=cfg.kmeans.L2,
            tol=cfg.kmeans.tol,
            w=cfg.problem.w,
            lambda_penalty=cfg.problem.lambda_penalty,
            theta1=cfg.problem.theta1,
            theta2=cfg.problem.theta2,
            theta3=cfg.problem.theta3,
            P1=cfg.problem.P1,
            P2=cfg.problem.P2,
            P3=cfg.problem.P3,
            seed=seed,
        )

    if method == "GRASP":
        return solve_rrp_grasp(
            X=X,
            K=cfg.problem.K,
            k_t=k_t,
            delta_bar=cfg.problem.delta_bar,
            w=cfg.problem.w,
            lambda_penalty=cfg.problem.lambda_penalty,
            theta1=cfg.problem.theta1,
            theta2=cfg.problem.theta2,
            theta3=cfg.problem.theta3,
            P1=cfg.problem.P1,
            P2=cfg.problem.P2,
            P3=cfg.problem.P3,
            seed=seed,
            n_starts=cfg.grasp.n_starts,
            rcl_size=cfg.grasp.rcl_size,
            max_group_attempts=cfg.grasp.max_group_attempts,
            max_local_iter=cfg.grasp.max_local_iter,
            group_candidate_limit=cfg.grasp.group_candidate_limit,
            cell_candidate_limit=cfg.grasp.cell_candidate_limit,
            leftover_candidate_limit=cfg.grasp.leftover_candidate_limit,
        )

    if method == "GA":
        return solve_rrp_ga(
            X=X,
            K=cfg.problem.K,
            k_t=k_t,
            delta_bar=cfg.problem.delta_bar,
            w=cfg.problem.w,
            lambda_penalty=cfg.problem.lambda_penalty,
            theta1=cfg.problem.theta1,
            theta2=cfg.problem.theta2,
            theta3=cfg.problem.theta3,
            P1=cfg.problem.P1,
            P2=cfg.problem.P2,
            P3=cfg.problem.P3,
            seed=seed,
            population_size=cfg.ga.population_size,
            n_generations=cfg.ga.n_generations,
            tournament_size=cfg.ga.tournament_size,
            crossover_prob=cfg.ga.crossover_prob,
            mutation_prob=cfg.ga.mutation_prob,
            destroy_size=cfg.ga.destroy_size,
            local_search_prob=cfg.ga.local_search_prob,
            elitism_size=cfg.ga.elitism_size,
            group_candidate_limit=cfg.ga.group_candidate_limit,
            cell_candidate_limit=cfg.ga.cell_candidate_limit,
            leftover_candidate_limit=cfg.ga.leftover_candidate_limit,
        )

    # default
    return solve_rrp_kmeans_vns(
        X=X,
        K=cfg.problem.K,
        k_t=k_t,
        delta_bar=cfg.problem.delta_bar,
        L1=cfg.vns.L1,
        tol=cfg.vns.tol,
        max_vns_iter=cfg.vns.max_vns_iter,
        max_no_improve=cfg.vns.max_no_improve,
        w=cfg.problem.w,
        lambda_penalty=cfg.problem.lambda_penalty,
        theta1=cfg.problem.theta1,
        theta2=cfg.problem.theta2,
        theta3=cfg.problem.theta3,
        P1=cfg.problem.P1,
        P2=cfg.problem.P2,
        P3=cfg.problem.P3,
        seed=seed,
        pack_candidate_limit=cfg.vns.pack_candidate_limit,
        partner_limit=cfg.vns.partner_limit,
        cell_candidate_limit=cfg.vns.cell_candidate_limit,
        leftover_candidate_limit=cfg.vns.leftover_candidate_limit,
        destroy_size=cfg.vns.destroy_size,
    )


# =========================================================
# Realized arrival path generation
# =========================================================

def generate_initial_inventory(cfg: Config, seed: int, n_init: int = 80) -> np.ndarray:
    return generate_cells(
        n_cells=n_init,
        mu_C=cfg.data.mu_C,
        sigma_C=cfg.data.sigma_C,
        mu_R=cfg.data.mu_R,
        sigma_R=cfg.data.sigma_R,
        seed=seed,
    )


def generate_arrival_sequence(
    cfg: Config,
    n_periods: int,
    seed: int,
    n_arrivals_each_period: int = 130,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    seq = []
    for _ in range(n_periods):
        A_t = gaussian_arrival_generator(
            rng=rng,
            n_arrivals=n_arrivals_each_period,
            mu_C=cfg.data.mu_C,
            sigma_C=cfg.data.sigma_C,
            mu_R=cfg.data.mu_R,
            sigma_R=cfg.data.sigma_R,
        )
        seq.append(A_t)
    return seq


# =========================================================
# Scrap action helpers
# =========================================================

def get_quality_scores(X: np.ndarray, wq: np.ndarray) -> np.ndarray:
    return default_quality_score(X, wq=wq)


def apply_threshold_scrap(I_t_plus: np.ndarray, eta: float, wq: np.ndarray) -> Tuple[List[int], np.ndarray]:
    q_vals = get_quality_scores(I_t_plus, wq)
    D_idx = np.where(q_vals < eta)[0].tolist()

    if len(D_idx) == 0:
        return D_idx, I_t_plus.copy()

    mask = np.ones(len(I_t_plus), dtype=bool)
    mask[np.array(D_idx, dtype=int)] = False
    I_t_pp = I_t_plus[mask].copy()
    return D_idx, I_t_pp


# =========================================================
# Core simulation for one policy on one realized path
# =========================================================

def simulate_policy_on_path(
    cfg: Config,
    initial_inventory: np.ndarray,
    arrivals_seq: List[np.ndarray],
    n_periods: int,
    inner_method: str,
    H_scrap: int,
    policy_name: str,
    threshold_grid: List[float],
    fixed_eta: float | None,
    m_list: List[int],
    rho: float,
    gamma: float,
    s0: float,
    wq: np.ndarray,
    master_seed: int,
    return_period_df: bool = False,
) -> Tuple[float, pd.DataFrame]:
    """
    policy_name in:
      - "NO_SCRAP"
      - "FIXED_THRESHOLD"
      - "TSRH"
    """
    rng = np.random.default_rng(master_seed)
    I_t = initial_inventory.copy()
    records = []

    for t in range(1, n_periods + 1):
        A_t = arrivals_seq[t - 1]
        U_t = stack_inventory_and_arrival(I_t, A_t)

        inner_seed = int(rng.integers(1, 10**9))
        inner_res = solve_inner_rrp(U_t, cfg, seed=inner_seed, method=inner_method)

        groups_t = inner_res.get("groups", [])
        R_t_grp = float(inner_res.get("reward", 0.0))

        used = set()
        for g in groups_t:
            used.update(g)
        leftover_idx = [i for i in range(len(U_t)) if i not in used]
        I_t_plus = U_t[leftover_idx].copy()

        eta_star = np.nan
        D_t = []
        I_t_pp = I_t_plus.copy()
        R_t_scr = 0.0

        if t % H_scrap == 0:
            if policy_name == "NO_SCRAP":
                pass

            elif policy_name == "FIXED_THRESHOLD":
                eta_star = float(fixed_eta)
                D_t, I_t_pp = apply_threshold_scrap(I_t_plus, eta_star, wq)
                R_t_scr = s0 * len(D_t)

            elif policy_name == "TSRH":
                def inner_solver_for_rollout(X, **kwargs):
                    rollout_seed = int(rng.integers(1, 10**9))
                    return solve_inner_rrp(X, cfg, seed=rollout_seed, method=inner_method)

                def arrival_fn_for_rollout(rng, **kwargs):
                    return gaussian_arrival_generator(
                        rng=rng,
                        n_arrivals=len(arrivals_seq[0]),
                        mu_C=cfg.data.mu_C,
                        sigma_C=cfg.data.sigma_C,
                        mu_R=cfg.data.mu_R,
                        sigma_R=cfg.data.sigma_R,
                    )

                tsrah_res = tsrah_scrapping_decision(
                    t=t,
                    I_t_plus=I_t_plus,
                    E=threshold_grid,
                    H=H_scrap,
                    m_list=m_list,
                    rho=rho,
                    gamma=gamma,
                    s0=s0,
                    quality_score_fn=default_quality_score,
                    inner_solver_fn=inner_solver_for_rollout,
                    arrival_generator_fn=arrival_fn_for_rollout,
                    quality_score_kwargs={"wq": wq},
                    inner_solver_kwargs={},
                    arrival_generator_kwargs={},
                    seed=int(rng.integers(1, 10**9)),
                    verbose=False,
                )

                eta_star = tsrah_res.eta_star
                D_t = tsrah_res.D_t
                I_t_pp = tsrah_res.I_t_pp
                R_t_scr = tsrah_res.R_t_scr

            else:
                raise ValueError(f"Unknown policy_name={policy_name}")

        R_t_total = R_t_grp + R_t_scr
        I_t = I_t_pp.copy()

        records.append({
            "t": t,
            "policy": policy_name if policy_name != "FIXED_THRESHOLD" else f"FIXED_{fixed_eta}",
            "arrivals": len(A_t),
            "available_before_rrp": len(U_t),
            "n_packs": len(groups_t),
            "inventory_after_rrp": len(I_t_plus),
            "scrap_count": len(D_t),
            "inventory_after_scrap": len(I_t_pp),
            "R_t_grp": R_t_grp,
            "R_t_scr": R_t_scr,
            "R_t_total": R_t_total,
            "eta_star": eta_star,
        })

    period_df = pd.DataFrame(records)
    total_reward = float(period_df["R_t_total"].sum())
    return total_reward, period_df if return_period_df else pd.DataFrame()


# =========================================================
# Clairvoyant upper bound within threshold-policy class
# =========================================================

def simulate_tail_deterministic(
    cfg: Config,
    I_start: np.ndarray,
    arrivals_seq: List[np.ndarray],
    t_start: int,
    n_periods: int,
    inner_method: str,
    H_scrap: int,
    eta_plan: Dict[int, float | None],
    s0: float,
    wq: np.ndarray,
    master_seed: int,
) -> float:
    """
    Simulate from period t_start to n_periods under a fixed threshold plan,
    with full future arrivals known.
    eta_plan[t] is only used if t is a review epoch.
    """
    rng = np.random.default_rng(master_seed + 777)
    I_t = I_start.copy()
    total = 0.0

    for t in range(t_start, n_periods + 1):
        A_t = arrivals_seq[t - 1]
        U_t = stack_inventory_and_arrival(I_t, A_t)

        inner_seed = int(rng.integers(1, 10**9))
        inner_res = solve_inner_rrp(U_t, cfg, seed=inner_seed, method=inner_method)
        groups_t = inner_res.get("groups", [])
        R_t_grp = float(inner_res.get("reward", 0.0))

        used = set()
        for g in groups_t:
            used.update(g)
        leftover_idx = [i for i in range(len(U_t)) if i not in used]
        I_t_plus = U_t[leftover_idx].copy()

        R_t_scr = 0.0
        I_t_pp = I_t_plus.copy()

        if t % H_scrap == 0:
            eta = eta_plan.get(t, None)
            if eta is not None:
                D_t, I_t_pp = apply_threshold_scrap(I_t_plus, float(eta), wq)
                R_t_scr = s0 * len(D_t)

        total += R_t_grp + R_t_scr
        I_t = I_t_pp.copy()

    return float(total)


def clairvoyant_threshold_upper_bound(
    cfg: Config,
    initial_inventory: np.ndarray,
    arrivals_seq: List[np.ndarray],
    n_periods: int,
    inner_method: str,
    H_scrap: int,
    threshold_grid: List[float],
    s0: float,
    wq: np.ndarray,
    master_seed: int,
    include_no_scrap_action: bool = True,
) -> Tuple[float, Dict[int, float | None]]:
    """
    Perfect-information benchmark within the threshold policy class.

    It knows all future arrivals exactly and searches over threshold choices
    at all review epochs.

    IMPORTANT:
    This is an upper bound *within the threshold-policy class and fixed inner solver*,
    not necessarily a global upper bound of the original MDP.
    """
    review_epochs = [t for t in range(1, n_periods + 1) if t % H_scrap == 0]
    action_set = [None] + list(threshold_grid) if include_no_scrap_action else list(threshold_grid)

    best_total = -np.inf
    best_plan = None

    # brute-force threshold plans across review epochs
    def dfs_build_plan(idx: int, current_plan: Dict[int, float | None]):
        nonlocal best_total, best_plan
        if idx == len(review_epochs):
            total = simulate_tail_deterministic(
                cfg=cfg,
                I_start=initial_inventory,
                arrivals_seq=arrivals_seq,
                t_start=1,
                n_periods=n_periods,
                inner_method=inner_method,
                H_scrap=H_scrap,
                eta_plan=current_plan,
                s0=s0,
                wq=wq,
                master_seed=master_seed,
            )
            if total > best_total:
                best_total = total
                best_plan = copy.deepcopy(current_plan)
            return

        t_review = review_epochs[idx]
        for a in action_set:
            current_plan[t_review] = a
            dfs_build_plan(idx + 1, current_plan)

    dfs_build_plan(0, {})
    return float(best_total), best_plan if best_plan is not None else {}


# =========================================================
# Batch experiment
# =========================================================

def run_batch_two_stage_experiments():
    cfg = Config()
    ensure_dir("results_two_stage")

    # -----------------------------
    # User-controlled experiment setup
    # -----------------------------
    n_periods = 20
    inner_method = "VNS"   # choose among: KMEANS / VNS / GRASP / GA
    s0 = 5.0
    gamma = 0.95
    rho = 0.5
    m_list = [2, 4, 8]

    seeds = [42, 43, 44]
    H_list = [4, 5, 6]

    threshold_grids = {
        "grid_coarse": [110, 120, 130, 140],
        "grid_fine":   [110, 115, 120, 125, 130, 135, 140],
    }

    wq = np.array([1.0, -1.0], dtype=float)

    raw_rows = []

    for seed in seeds:
        initial_inventory = generate_initial_inventory(cfg, seed=seed, n_init=80)
        arrivals_seq = generate_arrival_sequence(
            cfg=cfg,
            n_periods=n_periods,
            seed=seed,
            n_arrivals_each_period=130,
        )

        for H_scrap in H_list:
            for grid_name, grid_vals in threshold_grids.items():
                print(f"\n[seed={seed}] [H={H_scrap}] [grid={grid_name}]")

                # -----------------------------
                # Upper bound within threshold policy class
                # -----------------------------
                ub_total, ub_plan = clairvoyant_threshold_upper_bound(
                    cfg=cfg,
                    initial_inventory=initial_inventory,
                    arrivals_seq=arrivals_seq,
                    n_periods=n_periods,
                    inner_method=inner_method,
                    H_scrap=H_scrap,
                    threshold_grid=grid_vals,
                    s0=s0,
                    wq=wq,
                    master_seed=seed,
                    include_no_scrap_action=True,
                )

                # -----------------------------
                # Policy 1: No-scrap
                # -----------------------------
                total_no_scrap, _ = simulate_policy_on_path(
                    cfg=cfg,
                    initial_inventory=initial_inventory,
                    arrivals_seq=arrivals_seq,
                    n_periods=n_periods,
                    inner_method=inner_method,
                    H_scrap=H_scrap,
                    policy_name="NO_SCRAP",
                    threshold_grid=grid_vals,
                    fixed_eta=None,
                    m_list=m_list,
                    rho=rho,
                    gamma=gamma,
                    s0=s0,
                    wq=wq,
                    master_seed=seed,
                    return_period_df=False,
                )
                gap_no_scrap = 100.0 * max(0.0, ub_total - total_no_scrap) / ub_total if ub_total > 0 else np.nan

                raw_rows.append({
                    "seed": seed,
                    "inner_method": inner_method,
                    "H_scrap": H_scrap,
                    "grid_name": grid_name,
                    "policy": "NO_SCRAP",
                    "fixed_eta": np.nan,
                    "total_reward": total_no_scrap,
                    "upper_bound_reward": ub_total,
                    "gap_pct": gap_no_scrap,
                    "upper_bound_plan": str(ub_plan),
                })

                # -----------------------------
                # Policy 2: Fixed-threshold (one row per eta)
                # -----------------------------
                for eta in grid_vals:
                    total_fixed, _ = simulate_policy_on_path(
                        cfg=cfg,
                        initial_inventory=initial_inventory,
                        arrivals_seq=arrivals_seq,
                        n_periods=n_periods,
                        inner_method=inner_method,
                        H_scrap=H_scrap,
                        policy_name="FIXED_THRESHOLD",
                        threshold_grid=grid_vals,
                        fixed_eta=float(eta),
                        m_list=m_list,
                        rho=rho,
                        gamma=gamma,
                        s0=s0,
                        wq=wq,
                        master_seed=seed,
                        return_period_df=False,
                    )
                    gap_fixed = 100.0 * max(0.0, ub_total - total_fixed) / ub_total if ub_total > 0 else np.nan

                    raw_rows.append({
                        "seed": seed,
                        "inner_method": inner_method,
                        "H_scrap": H_scrap,
                        "grid_name": grid_name,
                        "policy": "FIXED_THRESHOLD",
                        "fixed_eta": float(eta),
                        "total_reward": total_fixed,
                        "upper_bound_reward": ub_total,
                        "gap_pct": gap_fixed,
                        "upper_bound_plan": str(ub_plan),
                    })

                # -----------------------------
                # Policy 3: TSRH
                # -----------------------------
                total_tsrah, _ = simulate_policy_on_path(
                    cfg=cfg,
                    initial_inventory=initial_inventory,
                    arrivals_seq=arrivals_seq,
                    n_periods=n_periods,
                    inner_method=inner_method,
                    H_scrap=H_scrap,
                    policy_name="TSRH",
                    threshold_grid=grid_vals,
                    fixed_eta=None,
                    m_list=m_list,
                    rho=rho,
                    gamma=gamma,
                    s0=s0,
                    wq=wq,
                    master_seed=seed,
                    return_period_df=False,
                )
                gap_tsrah = 100.0 * max(0.0, ub_total - total_tsrah) / ub_total if ub_total > 0 else np.nan

                raw_rows.append({
                    "seed": seed,
                    "inner_method": inner_method,
                    "H_scrap": H_scrap,
                    "grid_name": grid_name,
                    "policy": "TSRH",
                    "fixed_eta": np.nan,
                    "total_reward": total_tsrah,
                    "upper_bound_reward": ub_total,
                    "gap_pct": gap_tsrah,
                    "upper_bound_plan": str(ub_plan),
                })

    raw_df = pd.DataFrame(raw_rows)

    # summary
    summary_df = (
        raw_df
        .groupby(["inner_method", "H_scrap", "grid_name", "policy", "fixed_eta"], dropna=False)
        .agg(
            total_reward_mean=("total_reward", "mean"),
            total_reward_std=("total_reward", "std"),
            upper_bound_mean=("upper_bound_reward", "mean"),
            gap_pct_mean=("gap_pct", "mean"),
            gap_pct_std=("gap_pct", "std"),
        )
        .reset_index()
    )

    raw_path = os.path.join("results_two_stage", "two_stage_raw_results.csv")
    summary_path = os.path.join("results_two_stage", "two_stage_summary_results.csv")

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nSaved:")
    print(raw_path)
    print(summary_path)

    print("\n=== Summary Preview ===")
    with pd.option_context("display.max_columns", None, "display.width", 260):
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    run_batch_two_stage_experiments()