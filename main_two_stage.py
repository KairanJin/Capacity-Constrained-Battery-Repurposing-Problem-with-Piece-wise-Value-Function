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
from outer.tsrah import tsrah_scrapping_decision, default_quality_score
from outer.arrival import gaussian_arrival_generator


# =========================================================
# Inner solver wrapper
# =========================================================

def solve_inner_rrp(X: np.ndarray, cfg: Config, seed: int | None = None) -> dict:
    """
    Unified wrapper for the inner RRP solver.
    You can switch the method here.
    """
    k_t = min(cfg.problem.k_max, X.shape[0] // cfg.problem.K)

    method = "VNS"   # options: "KMEANS", "VNS", "GRASP", "GA"

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

    # default: VNS
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
    """
    Pre-generate the entire realized future arrival path.
    This is needed for fair comparison with the clairvoyant upper bound.
    """
    rng = np.random.default_rng(seed)
    return [generate_arrival_batch(cfg, rng) for _ in range(n_periods)]


# =========================================================
# Scrap helpers
# =========================================================

def apply_threshold_scrap(I_t_plus: np.ndarray, eta: float, wq: np.ndarray):
    """
    Threshold scrapping:
        D_t(eta) = { z in I_t^+ : q(z) < eta }
    """
    if len(I_t_plus) == 0:
        return [], I_t_plus.copy()

    q_vals = default_quality_score(I_t_plus, wq=wq)
    D_t = np.where(q_vals < eta)[0].tolist()

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
    seed: int = 42,
) -> float:
    """
    Evaluate one threshold eta at current review epoch using the REALIZED future
    arrivals over the next H_scrap periods.

    This is a stage-wise clairvoyant evaluation:
    - immediate scrapping now
    - then simulate next H_scrap periods with known arrivals
    - no further scrapping during this horizon
    """
    rng = np.random.default_rng(seed)

    # immediate scrap
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
        inner_res = solve_inner_rrp(U_tau, cfg, seed=inner_seed)

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
    seed: int = 42,
):
    """
    At the current review epoch, choose the best threshold using the REALIZED
    future arrivals over the next H_scrap periods.
    """
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
    seed: int = 42,
    verbose: bool = True,
):
    """
    Stage-wise clairvoyant benchmark:
    At each review epoch t, choose the best eta using the realized arrivals
    from t+1 to t+H_scrap, but only for the current scrapping decision.
    """
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
        inner_res = solve_inner_rrp(U_t, cfg, seed=inner_seed)

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
                seed=seed,
            )

            D_t, I_t_pp = apply_threshold_scrap(I_t_plus, eta_star, wq)
            R_t_scr = s0 * len(D_t)
        else:
            eta_values = {}

        R_t = R_t_grp + R_t_scr
        I_t = I_t_pp.copy()

        rec = {
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
                f"[UB-stage] t={t:2d} | arrivals={len(A_t):3d} | "
                f"packs={len(groups_t):2d} | I_t^+={len(I_t_plus):3d} | "
                f"scrap={len(D_t):3d} | I_t++={len(I_t_pp):3d} | "
                f"R_grp={R_t_grp:8.2f} | R_scr={R_t_scr:6.2f} | eta={eta_star}"
            )

    df = pd.DataFrame(records)
    return df

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
    seed: int = 42,
    verbose: bool = False,
):
    """
    Simulate the whole system on a fixed realized arrival path, with a fixed threshold
    chosen at each review epoch.

    threshold_plan[t] is used only when t % H_scrap == 0.
    If threshold_plan[t] is None, that review epoch does no scrapping.
    """
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
        inner_res = solve_inner_rrp(U_t, cfg, seed=inner_seed)

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

        if verbose:
            print(
                f"[FIXED/UB] t={t:2d} | arrivals={len(A_t):3d} | "
                f"packs={len(groups_t):2d} | I_t^+={len(I_t_plus):3d} | "
                f"scrap={len(D_t):3d} | I_t++={len(I_t_pp):3d} | "
                f"R_grp={R_t_grp:8.2f} | R_scr={R_t_scr:6.2f} | eta={eta_star}"
            )

    df = pd.DataFrame(records)
    return df


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
    seed: int = 42,
    verbose: bool = True,
):
    """
    Main two-stage control loop with TSRH.
    If arrivals_seq is provided, it uses that fixed realized future path.
    """
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
        inner_res = solve_inner_rrp(U_t, cfg, seed=inner_seed)

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
                return solve_inner_rrp(X, cfg, seed=rollout_seed)

            # rollout阶段仍然采用随机到达，用于实际在线策略评估
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
                f"t={t:2d} | arrivals={len(A_t):3d} | "
                f"packs={len(groups_t):2d} | "
                f"I_t^+={len(I_t_plus):3d} | "
                f"scrap={len(D_t):3d} | "
                f"I_t++={len(I_t_pp):3d} | "
                f"R_grp={R_t_grp:8.2f} | "
                f"R_scr={R_t_scr:6.2f} | "
                f"eta={eta_star}"
            )

    df = pd.DataFrame(records)
    return df


# =========================================================
# Clairvoyant upper bound within threshold-policy class
# =========================================================

def compute_clairvoyant_threshold_upper_bound(
    cfg: Config,
    arrivals_seq: list[np.ndarray],
    n_periods: int,
    H_scrap: int,
    E_thresholds: list[float],
    s0: float,
    seed: int = 42,
    include_no_scrap_action: bool = True,
    verbose: bool = True,
):
    """
    Perfect-information benchmark within the threshold-policy class.

    It knows the full realized future arrival path and chooses the best threshold
    action at each review epoch over the whole horizon.

    IMPORTANT:
    This is an upper bound within the threshold-policy class,
    not a global upper bound of the original MDP.
    """
    review_epochs = [t for t in range(1, n_periods + 1) if t % H_scrap == 0]

    if include_no_scrap_action:
        action_set = [None] + [float(x) for x in E_thresholds]
    else:
        action_set = [float(x) for x in E_thresholds]

    best_total_reward = -np.inf
    best_plan = None
    best_df = None

    plan_count = 0
    for action_tuple in itertools.product(action_set, repeat=len(review_epochs)):
        threshold_plan = {t_review: eta for t_review, eta in zip(review_epochs, action_tuple)}
        plan_count += 1

        df_plan = simulate_with_fixed_threshold_plan(
            cfg=cfg,
            arrivals_seq=arrivals_seq,
            n_periods=n_periods,
            H_scrap=H_scrap,
            s0=s0,
            threshold_plan=threshold_plan,
            seed=seed,
            verbose=False,
        )

        total_reward = float(df_plan["R_t_total"].sum())

        if total_reward > best_total_reward:
            best_total_reward = total_reward
            best_plan = threshold_plan
            best_df = df_plan.copy()

    if verbose:
        print("\n=== Clairvoyant Upper Bound ===")
        print(f"Number of threshold plans evaluated: {plan_count}")
        print(f"Best total reward: {best_total_reward:.2f}")
        print(f"Best threshold plan: {best_plan}")

    return best_total_reward, best_plan, best_df

def plot_online_vs_benchmark(df_online: pd.DataFrame, df_benchmark: pd.DataFrame):
    """
    Plot period-by-period and cumulative comparison between:
    - Online TSRH
    - Stage-wise clairvoyant benchmark
    """
    t = df_online["t"].values

    # per-period rewards
    online_grp = df_online["R_t_grp"].values
    online_scr = df_online["R_t_scr"].values
    online_total = df_online["R_t_total"].values

    bench_grp = df_benchmark["R_t_grp"].values
    bench_scr = df_benchmark["R_t_scr"].values
    bench_total = df_benchmark["R_t_total"].values

    # cumulative rewards
    online_cum = np.cumsum(online_total)
    bench_cum = np.cumsum(bench_total)

    # per-period difference
    diff_total = online_total - bench_total
    diff_cum = online_cum - bench_cum

    fig = plt.figure(figsize=(14, 10))

    # -------------------------------------------------
    # 1) Per-period total reward comparison
    # -------------------------------------------------
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(t, online_total, marker="o", label="Online TSRH total")
    ax1.plot(t, bench_total, marker="s", label="Clairvoyant benchmark total")
    ax1.set_title("Per-period Total Reward")
    ax1.set_xlabel("Period t")
    ax1.set_ylabel("Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # -------------------------------------------------
    # 2) Cumulative total reward comparison
    # -------------------------------------------------
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(t, online_cum, marker="o", label="Online TSRH cumulative")
    ax2.plot(t, bench_cum, marker="s", label="Clairvoyant benchmark cumulative")
    ax2.set_title("Cumulative Total Reward")
    ax2.set_xlabel("Period t")
    ax2.set_ylabel("Cumulative reward")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # -------------------------------------------------
    # 3) Per-period reward decomposition
    # -------------------------------------------------
    ax3 = plt.subplot(2, 2, 3)
    width = 0.18
    ax3.bar(t - 1.5 * width, online_grp, width=width, label="Online group")
    ax3.bar(t - 0.5 * width, online_scr, width=width, label="Online scrap")
    ax3.bar(t + 0.5 * width, bench_grp, width=width, label="Benchmark group")
    ax3.bar(t + 1.5 * width, bench_scr, width=width, label="Benchmark scrap")
    ax3.set_title("Per-period Reward Decomposition")
    ax3.set_xlabel("Period t")
    ax3.set_ylabel("Reward")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # -------------------------------------------------
    # 4) Difference plot
    # -------------------------------------------------
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(t, diff_total, marker="o", label="Per-period total diff")
    ax4.plot(t, diff_cum, marker="s", label="Cumulative total diff")
    ax4.axhline(0, linestyle="--", linewidth=1)
    ax4.set_title("Online TSRH - Benchmark")
    ax4.set_xlabel("Period t")
    ax4.set_ylabel("Difference")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def main():
    cfg = Config()

    n_periods = 20
    H_scrap = 5
    E_thresholds = [70, 80, 90, 100, 110, 115, 120, 125, 130, 135, 140]
    m_list = [2, 4, 8]
    rho = 0.5
    gamma = 0.95
    s0 = 5.0
    seed = 43

    # fixed realized arrival path for fair comparison
    arrivals_seq = generate_arrival_sequence(cfg, n_periods, seed)

    # -------------------------------------------------
    # Online TSRH policy
    # -------------------------------------------------
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
        seed=seed,
        verbose=True,
    )

    # -------------------------------------------------
    # Stage-wise clairvoyant upper bound
    # -------------------------------------------------
    df_ub = simulate_stagewise_clairvoyant_upper_bound(
        cfg=cfg,
        arrivals_seq=arrivals_seq,
        n_periods=n_periods,
        H_scrap=H_scrap,
        E_thresholds=E_thresholds,
        gamma=gamma,
        s0=s0,
        seed=seed,
        verbose=True,
    )

    online_total = float(df_online["R_t_total"].sum())
    ub_total = float(df_ub["R_t_total"].sum())
    gap_pct = 100.0 * (ub_total - online_total) / ub_total if ub_total > 0 else np.nan

    print("\n=== Period-by-period Results: Online TSRH ===")
    with pd.option_context("display.max_columns", None, "display.width", 260):
        print(df_online.to_string(index=False))

    print("\n=== Summary: Online TSRH ===")
    print(f"Total group reward : {df_online['R_t_grp'].sum():.2f}")
    print(f"Total scrap reward : {df_online['R_t_scr'].sum():.2f}")
    print(f"Total reward       : {df_online['R_t_total'].sum():.2f}")
    print(f"Total scrap count  : {df_online['scrap_count'].sum()}")
    print(f"Average packs/period: {df_online['n_packs'].mean():.2f}")

    print("\n=== Period-by-period Results: Stage-wise Clairvoyant UB ===")
    with pd.option_context("display.max_columns", None, "display.width", 260):
        print(df_ub.to_string(index=False))

    print("\n=== Summary: Stage-wise Clairvoyant UB ===")
    print(f"Total group reward : {df_ub['R_t_grp'].sum():.2f}")
    print(f"Total scrap reward : {df_ub['R_t_scr'].sum():.2f}")
    print(f"Total reward       : {df_ub['R_t_total'].sum():.2f}")
    print(f"Total scrap count  : {df_ub['scrap_count'].sum()}")
    print(f"Average packs/period: {df_ub['n_packs'].mean():.2f}")

    print("\n=== GAP ===")
    print(f"Online TSRH total reward = {online_total:.2f}")
    print(f"Stage-wise clairvoyant UB = {ub_total:.2f}")
    print(f"GAP (%) = {gap_pct:.4f}")

    plot_online_vs_benchmark(df_online, df_ub)

if __name__ == "__main__":
    main()