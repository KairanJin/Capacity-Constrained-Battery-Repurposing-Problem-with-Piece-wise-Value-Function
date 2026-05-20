# main_two_stage.py
from __future__ import annotations

import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import Config
from data_generator import generate_cells
from heuristics.rrp_kmeans import solve_rrp_kmeans
from heuristics.rrp_kmeans_vns import solve_rrp_kmeans_vns
from heuristics.rrp_grasp import solve_rrp_grasp
from heuristics.rrp_ga import solve_rrp_ga
from heuristics.rrp_sa import solve_rrp_sa
from heuristics.rrp_ms_kmeans_vns import solve_rrp_ms_kmeans_vns
from outer.tsrah import tsrah_scrapping_decision, default_quality_score
from outer.arrival import gaussian_arrival_generator


# =========================================================
# 可在此处灵活控制参与实验的启发式算法种类
# 注释/取消注释即可开关对应算法
# =========================================================

METHODS = [
    "KMEANS",
    "VNS",
    # "GRASP",
    # "GA",
    # "SA",
    # "MS_VNS",
]

METHOD_LABELS = {
    "KMEANS": "KMeans",
    "VNS": "KMeans_VNS",
    "GRASP": "GRASP",
    "GA": "GA",
    "SA": "SA",
    "MS_VNS": "MS_KMeans_VNS",
}


# =========================================================
# Inner solver wrapper
# =========================================================

def solve_inner_rrp(
    X: np.ndarray,
    cfg: Config,
    method: str = "VNS",
    seed: int | None = None
) -> dict:
    """
    Unified wrapper for the inner RRP solver.
    method options: "KMEANS", "VNS", "GRASP", "GA"
    """
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

    if method == "VNS":
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

    if method == "SA":
        return solve_rrp_sa(
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
            initial_temperature=cfg.sa.initial_temperature,
            cooling_rate=cfg.sa.cooling_rate,
            min_temperature=cfg.sa.min_temperature,
            max_sa_iterations=cfg.sa.max_sa_iterations,
            vnd_interval=cfg.sa.vnd_interval,
            max_vnd_rounds=cfg.sa.max_vnd_rounds,
            reheating_ratio=cfg.sa.reheating_ratio,
            reheating_stall=cfg.sa.reheating_stall,
            max_reheats=cfg.sa.max_reheats,
            tabu_tenure=cfg.sa.tabu_tenure,
            n_init_starts=cfg.sa.n_init_starts,
            kmeans_L1=cfg.sa.kmeans_L1,
            kmeans_tol=cfg.sa.kmeans_tol,
            residual_rounds=cfg.sa.residual_rounds,
        )

    if method == "MS_VNS":
        return solve_rrp_ms_kmeans_vns(
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
        )

    raise ValueError(f"Unknown method: {method}")


# =========================================================
# Arrival path generation
# =========================================================

def generate_arrival_batch(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    return gaussian_arrival_generator(
        rng=rng,
        n_arrivals=130,
        mu_C=cfg.data.mu_C,
        sigma_C=cfg.data.sigma_C,
        mu_R=cfg.data.mu_R,
        sigma_R=cfg.data.sigma_R,
    )


def generate_arrival_sequence(cfg: Config, n_periods: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [generate_arrival_batch(cfg, rng) for _ in range(n_periods)]


# =========================================================
# Scrap helpers
# =========================================================

def apply_threshold_scrap(I_t_plus: np.ndarray, eta: float, wq: np.ndarray):
    """
    根据质量阈值废弃电芯。
    eta 现在解释为分位数比例（0~1），而非绝对分数。
    例如 eta=0.25 表示废弃质量得分最低的 25% 电芯。
    """
    if len(I_t_plus) == 0:
        return [], I_t_plus.copy()

    q_vals = default_quality_score(I_t_plus, wq=wq)
    # eta 为分位数比例，计算对应的绝对分数阈值
    abs_threshold = float(np.quantile(q_vals, eta))
    D_t = np.where(q_vals < abs_threshold)[0].tolist()

    if len(D_t) == 0:
        return D_t, I_t_plus.copy()

    mask = np.ones(len(I_t_plus), dtype=bool)
    mask[np.array(D_t, dtype=int)] = False
    I_t_pp = I_t_plus[mask].copy()
    return D_t, I_t_pp


def evaluate_eta_with_realized_horizon(
    cfg: Config,
    I_t_plus: np.ndarray,
    eta: float,
    arrivals_seq: list[np.ndarray],
    t: int,
    H_scrap: int,
    gamma: float,
    s0: float,
    wq: np.ndarray,
    inner_method: str = "VNS",
    seed: int = 42,
) -> float:
    rng = np.random.default_rng(seed)

    D_t, I_curr = apply_threshold_scrap(I_t_plus, eta, wq)
    immediate_scrap_reward = s0 * len(D_t)

    total_future_reward = 0.0
    n_periods_total = len(arrivals_seq)

    for tau in range(1, H_scrap + 1):
        future_t = t + tau
        if future_t > n_periods_total:
            break

        A_tau = arrivals_seq[future_t - 1]

        if len(I_curr) == 0:
            U_tau = A_tau.copy()
        elif len(A_tau) == 0:
            U_tau = I_curr.copy()
        else:
            U_tau = np.vstack([I_curr, A_tau])

        inner_seed = int(rng.integers(1, 10**9))
        inner_res = solve_inner_rrp(U_tau, cfg, method=inner_method, seed=inner_seed)

        R_grp = float(inner_res.get("reward", 0.0))
        total_future_reward += (gamma ** tau) * R_grp

        used = set()
        for g in inner_res.get("groups", []):
            used.update(g)

        leftover_idx = [i for i in range(len(U_tau)) if i not in used]
        I_curr = U_tau[leftover_idx].copy()

    return immediate_scrap_reward + total_future_reward


def choose_clairvoyant_eta_for_current_stage(
    cfg: Config,
    I_t_plus: np.ndarray,
    E_thresholds: list[float],
    arrivals_seq: list[np.ndarray],
    t: int,
    H_scrap: int,
    gamma: float,
    s0: float,
    wq: np.ndarray,
    inner_method: str = "VNS",
    seed: int = 42,
):
    best_eta = None
    best_value = -np.inf
    eta_values = {}

    for eta in E_thresholds:
        val = evaluate_eta_with_realized_horizon(
            cfg=cfg,
            I_t_plus=I_t_plus,
            eta=float(eta),
            arrivals_seq=arrivals_seq,
            t=t,
            H_scrap=H_scrap,
            gamma=gamma,
            s0=s0,
            wq=wq,
            inner_method=inner_method,
            seed=seed + int(eta * 10),
        )
        eta_values[float(eta)] = val

        if val > best_value:
            best_value = val
            best_eta = float(eta)

    return best_eta, best_value, eta_values


def simulate_stagewise_clairvoyant_upper_bound(
    cfg: Config,
    arrivals_seq: list[np.ndarray],
    n_periods: int,
    H_scrap: int,
    E_thresholds: list[float],
    gamma: float,
    s0: float,
    inner_method: str = "VNS",
    seed: int = 42,
    verbose: bool = True,
):
    rng = np.random.default_rng(seed)

    I_t = generate_cells(
        n_cells=0,
        mu_C=cfg.data.mu_C,
        sigma_C=cfg.data.sigma_C,
        mu_R=cfg.data.mu_R,
        sigma_R=cfg.data.sigma_R,
        seed=seed,
    )

    records = []
    wq = np.array([1.0, -1.0], dtype=float)

    for t in range(1, n_periods + 1):
        A_t = arrivals_seq[t - 1]

        if len(I_t) == 0:
            U_t = A_t.copy()
        elif len(A_t) == 0:
            U_t = I_t.copy()
        else:
            U_t = np.vstack([I_t, A_t])

        inner_seed = int(rng.integers(1, 10**9))
        inner_res = solve_inner_rrp(U_t, cfg, method=inner_method, seed=inner_seed)

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
        clairvoyant_stage_value = np.nan

        if t % H_scrap == 0:
            eta_star, clairvoyant_stage_value, eta_values = choose_clairvoyant_eta_for_current_stage(
                cfg=cfg,
                I_t_plus=I_t_plus,
                E_thresholds=E_thresholds,
                arrivals_seq=arrivals_seq,
                t=t,
                H_scrap=H_scrap,
                gamma=gamma,
                s0=s0,
                wq=wq,
                inner_method=inner_method,
                seed=seed,
            )

            D_t, I_t_pp = apply_threshold_scrap(I_t_plus, eta_star, wq)
            R_t_scr = s0 * len(D_t)
        else:
            eta_values = {}

        R_t = R_t_grp + R_t_scr
        I_t = I_t_pp.copy()

        rec = {
            "method": inner_method,
            "t": t,
            "arrivals": len(A_t),
            "available_before_rrp": len(U_t),
            "n_packs": len(groups_t),
            "used_cells": sum(len(g) for g in groups_t),
            "inventory_after_rrp": len(I_t_plus),
            "scrap_count": len(D_t),
            "inventory_after_scrap": len(I_t_pp),
            "R_t_grp": R_t_grp,
            "R_t_scr": R_t_scr,
            "R_t_total": R_t,
            "eta_star": eta_star,
            "clairvoyant_stage_value": clairvoyant_stage_value,
            "eta_values": str(eta_values),
        }
        records.append(rec)

        if verbose:
            print(
                f"[UB-{inner_method}] t={t:2d} | arrivals={len(A_t):3d} | "
                f"packs={len(groups_t):2d} | I_t^+={len(I_t_plus):3d} | "
                f"scrap={len(D_t):3d} | I_t++={len(I_t_pp):3d} | "
                f"R_grp={R_t_grp:8.2f} | R_scr={R_t_scr:6.2f} | eta={eta_star}"
            )

    return pd.DataFrame(records)


# =========================================================
# One pass simulation under a fixed threshold plan
# =========================================================

def simulate_with_fixed_threshold_plan(
    cfg: Config,
    arrivals_seq: list[np.ndarray],
    n_periods: int,
    H_scrap: int,
    s0: float,
    threshold_plan: dict[int, float | None],
    inner_method: str = "VNS",
    seed: int = 42,
    verbose: bool = False,
):
    rng = np.random.default_rng(seed)

    I_t = generate_cells(
        n_cells=0,
        mu_C=cfg.data.mu_C,
        sigma_C=cfg.data.sigma_C,
        mu_R=cfg.data.mu_R,
        sigma_R=cfg.data.sigma_R,
        seed=seed,
    )

    records = []
    wq = np.array([1.0, -1.0], dtype=float)

    for t in range(1, n_periods + 1):
        A_t = arrivals_seq[t - 1]

        if len(I_t) == 0:
            U_t = A_t.copy()
        elif len(A_t) == 0:
            U_t = I_t.copy()
        else:
            U_t = np.vstack([I_t, A_t])

        inner_seed = int(rng.integers(1, 10**9))
        inner_res = solve_inner_rrp(U_t, cfg, method=inner_method, seed=inner_seed)

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
            eta_t = threshold_plan.get(t, None)
            if eta_t is not None:
                eta_star = float(eta_t)
                D_t, I_t_pp = apply_threshold_scrap(I_t_plus, eta_star, wq)
                R_t_scr = s0 * len(D_t)

        R_t = R_t_grp + R_t_scr
        I_t = I_t_pp.copy()

        rec = {
            "method": inner_method,
            "t": t,
            "arrivals": len(A_t),
            "available_before_rrp": len(U_t),
            "n_packs": len(groups_t),
            "used_cells": sum(len(g) for g in groups_t),
            "inventory_after_rrp": len(I_t_plus),
            "scrap_count": len(D_t),
            "inventory_after_scrap": len(I_t_pp),
            "R_t_grp": R_t_grp,
            "R_t_scr": R_t_scr,
            "R_t_total": R_t,
            "eta_star": eta_star,
        }
        records.append(rec)

    return pd.DataFrame(records)


# =========================================================
# Main TSRH simulation on a fixed arrival path
# =========================================================

def simulate_two_stage_system(
    cfg: Config,
    n_periods: int,
    H_scrap: int,
    E_thresholds: list[float],
    m_list: list[int],
    rho: float,
    gamma: float,
    s0: float,
    arrivals_seq: list[np.ndarray] | None = None,
    inner_method: str = "VNS",
    seed: int = 42,
    verbose: bool = True,
):
    rng = np.random.default_rng(seed)

    if arrivals_seq is None:
        arrivals_seq = generate_arrival_sequence(cfg, n_periods, seed)

    I_t = generate_cells(
        n_cells=0,
        mu_C=cfg.data.mu_C,
        sigma_C=cfg.data.sigma_C,
        mu_R=cfg.data.mu_R,
        sigma_R=cfg.data.sigma_R,
        seed=seed,
    )

    records = []
    wq = np.array([1.0, -1.0], dtype=float)

    for t in range(1, n_periods + 1):
        A_t = arrivals_seq[t - 1]

        if len(I_t) == 0:
            U_t = A_t.copy()
        elif len(A_t) == 0:
            U_t = I_t.copy()
        else:
            U_t = np.vstack([I_t, A_t])

        inner_seed = int(rng.integers(1, 10**9))
        inner_res = solve_inner_rrp(U_t, cfg, method=inner_method, seed=inner_seed)

        groups_t = inner_res.get("groups", [])
        R_t_grp = float(inner_res.get("reward", 0.0))

        used = set()
        for g in groups_t:
            used.update(g)

        leftover_idx = [i for i in range(len(U_t)) if i not in used]
        I_t_plus = U_t[leftover_idx].copy()

        if t % H_scrap == 0:
            def inner_solver_for_rollout(X, **kwargs):
                rollout_seed = int(rng.integers(1, 10**9))
                return solve_inner_rrp(X, cfg, method=inner_method, seed=rollout_seed)

            def arrival_fn_for_rollout(rng, **kwargs):
                return gaussian_arrival_generator(
                    rng=rng,
                    n_arrivals=30,
                    mu_C=cfg.data.mu_C,
                    sigma_C=cfg.data.sigma_C,
                    mu_R=cfg.data.mu_R,
                    sigma_R=cfg.data.sigma_R,
                )

            tsrah_res = tsrah_scrapping_decision(
                t=t,
                I_t_plus=I_t_plus,
                E=E_thresholds,
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
            final_candidates = tsrah_res.final_candidates

        else:
            eta_star = np.nan
            D_t = []
            I_t_pp = I_t_plus.copy()
            R_t_scr = 0.0
            final_candidates = []

        R_t = R_t_grp + R_t_scr
        I_t = I_t_pp.copy()

        rec = {
            "method": inner_method,
            "t": t,
            "arrivals": len(A_t),
            "available_before_rrp": len(U_t),
            "n_packs": len(groups_t),
            "used_cells": sum(len(g) for g in groups_t),
            "inventory_after_rrp": len(I_t_plus),
            "scrap_count": len(D_t),
            "inventory_after_scrap": len(I_t_pp),
            "R_t_grp": R_t_grp,
            "R_t_scr": R_t_scr,
            "R_t_total": R_t,
            "eta_star": eta_star,
            "final_candidates": str(final_candidates),
        }
        records.append(rec)

        if verbose:
            print(
                f"[ONLINE-{inner_method}] t={t:2d} | arrivals={len(A_t):3d} | "
                f"packs={len(groups_t):2d} | "
                f"I_t^+={len(I_t_plus):3d} | "
                f"scrap={len(D_t):3d} | "
                f"I_t++={len(I_t_pp):3d} | "
                f"R_grp={R_t_grp:8.2f} | "
                f"R_scr={R_t_scr:6.2f} | "
                f"eta={eta_star}"
            )

    return pd.DataFrame(records)


# =========================================================
# Comparison utilities
# =========================================================

def summarize_run(df_online: pd.DataFrame, df_ub: pd.DataFrame, method: str) -> dict:
    online_total = float(df_online["R_t_total"].sum())
    ub_total = float(df_ub["R_t_total"].sum())
    gap_pct = 100.0 * (ub_total - online_total) / ub_total if ub_total > 0 else np.nan

    return {
        "method": method,
        "online_group_reward": float(df_online["R_t_grp"].sum()),
        "online_scrap_reward": float(df_online["R_t_scr"].sum()),
        "online_total_reward": online_total,
        "online_total_scrap": int(df_online["scrap_count"].sum()),
        "online_avg_packs": float(df_online["n_packs"].mean()),
        "ub_total_reward": ub_total,
        "ub_total_scrap": int(df_ub["scrap_count"].sum()),
        "ub_avg_packs": float(df_ub["n_packs"].mean()),
        "gap_pct": gap_pct,
    }


def compare_inner_methods(
    cfg: Config,
    methods: list[str],
    n_periods: int,
    H_scrap: int,
    E_thresholds: list[float],
    m_list: list[int],
    rho: float,
    gamma: float,
    s0: float,
    seed: int = 42,
):
    arrivals_seq = generate_arrival_sequence(cfg, n_periods, seed)

    all_online = {}
    all_ub = {}
    summary_records = []

    for method in methods:
        print(f"\n{'=' * 25}")
        print(f"Running method: {method}")
        print(f"{'=' * 25}")

        df_online = simulate_two_stage_system(
            cfg=cfg,
            n_periods=n_periods,
            H_scrap=H_scrap,
            E_thresholds=E_thresholds,
            m_list=m_list,
            rho=rho,
            gamma=gamma,
            s0=s0,
            arrivals_seq=arrivals_seq,
            inner_method=method,
            seed=seed,
            verbose=True,
        )

        df_ub = simulate_stagewise_clairvoyant_upper_bound(
            cfg=cfg,
            arrivals_seq=arrivals_seq,
            n_periods=n_periods,
            H_scrap=H_scrap,
            E_thresholds=E_thresholds,
            gamma=gamma,
            s0=s0,
            inner_method=method,
            seed=seed,
            verbose=True,
        )

        all_online[method] = df_online
        all_ub[method] = df_ub
        summary_records.append(summarize_run(df_online, df_ub, method))

    summary_df = pd.DataFrame(summary_records)
    return summary_df, all_online, all_ub


def plot_method_comparison(summary_df: pd.DataFrame):
    methods = summary_df["method"].values
    online_total = summary_df["online_total_reward"].values
    ub_total = summary_df["ub_total_reward"].values
    gap_pct = summary_df["gap_pct"].values

    fig = plt.figure(figsize=(14, 4))

    ax1 = plt.subplot(1, 3, 1)
    ax1.bar(methods, online_total)
    ax1.set_title("Online Total Reward")
    ax1.set_ylabel("Reward")
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(1, 3, 2)
    ax2.bar(methods, ub_total)
    ax2.set_title("Stage-wise UB Total Reward")
    ax2.set_ylabel("Reward")
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(1, 3, 3)
    ax3.bar(methods, gap_pct)
    ax3.set_title("Gap to UB (%)")
    ax3.set_ylabel("Gap %")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def run_one_experiment_with_seed(
    seed: int,
    n_periods: int = 20,
    H_scrap: int = 5,
    methods: list[str] | None = None,
) -> pd.DataFrame:
    """
    使用指定种子跑一次两阶段仿真实验。
    methods 默认为顶部 METHODS 列表中启用的算法。
    """
    cfg = Config()
    # 废弃分位数比例：0 表示不弃置，0.25 表示废弃最差 25%，以此类推
    E_thresholds = [0.0, 0.25, 0.5, 0.75, 1.0]
    m_list = [2, 4, 8]
    rho = 0.5
    gamma = 0.95
    s0 = 5.0

    if methods is None:
        methods = list(METHODS)

    summary_df, all_online, all_ub = compare_inner_methods(
        cfg=cfg,
        methods=methods,
        n_periods=n_periods,
        H_scrap=H_scrap,
        E_thresholds=E_thresholds,
        m_list=m_list,
        rho=rho,
        gamma=gamma,
        s0=s0,
        seed=seed,
    )

    return summary_df


def main():
    cfg = Config()

    n_periods = 20
    H_scrap = 5
    # 废弃分位数比例：0 表示不弃置，0.25 表示废弃最差 25%，以此类推
    E_thresholds = [0.0, 0.25, 0.5, 0.75, 1.0]
    m_list = [2, 4, 8]
    rho = 0.5
    gamma = 0.95
    s0 = 5.0
    seed = 43

    # 参与实验的算法列表，修改顶部 METHODS 即可控制
    methods = list(METHODS)

    summary_df, all_online, all_ub = compare_inner_methods(
        cfg=cfg,
        methods=methods,
        n_periods=n_periods,
        H_scrap=H_scrap,
        E_thresholds=E_thresholds,
        m_list=m_list,
        rho=rho,
        gamma=gamma,
        s0=s0,
        seed=seed,
    )

    print("\n=== Summary Comparison Across Inner Methods ===")
    with pd.option_context("display.max_columns", None, "display.width", 260):
        print(summary_df.to_string(index=False))

    plot_method_comparison(summary_df)

    # 打印所有已运行方法的详细结果
    for method in methods:
        print(f"\n=== Detailed Online Results: {method} ===")
        with pd.option_context("display.max_columns", None, "display.width", 260):
            print(all_online[method].to_string(index=False))

        print(f"\n=== Detailed UB Results: {method} ===")
        with pd.option_context("display.max_columns", None, "display.width", 260):
            print(all_ub[method].to_string(index=False))


if __name__ == "__main__":
    main()