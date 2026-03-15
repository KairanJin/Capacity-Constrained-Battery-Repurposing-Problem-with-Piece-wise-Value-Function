# upper_bound.py
from __future__ import annotations

import pandas as pd
import numpy as np

from config import Config
from data_generator import generate_cells
from heuristics.rrp_column_generation import solve_rrp_column_generation
from outer.tsrah import tsrah_scrapping_decision, default_quality_score
from outer.arrival import gaussian_arrival_generator


# =========================================================
# Inner solver: Column Generation
# =========================================================

def solve_inner_rrp_cg(X: np.ndarray, cfg: Config, seed: int | None = None) -> dict:
    """
    Unified inner solver wrapper using Column Generation only.
    """
    k_t = min(cfg.problem.k_max, X.shape[0] // cfg.problem.K)

    return solve_rrp_column_generation(
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
        max_cg_iter=cfg.cg.max_cg_iter,
        init_n_starts=cfg.cg.init_n_starts,
        init_neighbor_size=cfg.cg.init_neighbor_size,
        pricing_n_seeds=cfg.cg.pricing_n_seeds,
        pricing_neighbor_size=cfg.cg.pricing_neighbor_size,
        max_new_cols=cfg.cg.max_new_cols,
        seed=seed,
    )


# =========================================================
# Arrival path generation
# =========================================================

def generate_arrival_batch(cfg: Config, rng: np.random.Generator, n_arrivals: int = 130) -> np.ndarray:
    return gaussian_arrival_generator(
        rng=rng,
        n_arrivals=n_arrivals,
        mu_C=cfg.data.mu_C,
        sigma_C=cfg.data.sigma_C,
        mu_R=cfg.data.mu_R,
        sigma_R=cfg.data.sigma_R,
    )


def generate_arrival_sequence(
    cfg: Config,
    n_periods: int,
    seed: int,
    n_arrivals: int = 130,
) -> list[np.ndarray]:
    """
    Pre-generate one realized future arrival path.
    """
    rng = np.random.default_rng(seed)
    return [generate_arrival_batch(cfg, rng, n_arrivals=n_arrivals) for _ in range(n_periods)]


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


# =========================================================
# Stage-wise perfect-information benchmark
# =========================================================

def evaluate_eta_with_realized_horizon_cg(
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
    Evaluate one threshold eta at the current review epoch using the REALIZED
    future arrivals over the next H_scrap periods.

    This is stage-wise perfect-information evaluation:
    - immediate scrapping now
    - then simulate the next H_scrap periods with known arrivals
    - no further scrapping during this horizon
    """
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
        inner_res = solve_inner_rrp_cg(U_tau, cfg, seed=inner_seed)

        R_grp = float(inner_res.get("reward", 0.0))
        total_future_reward += (gamma ** tau) * R_grp

        used = set()
        for g in inner_res.get("groups", []):
            used.update(g)

        leftover_idx = [i for i in range(len(U_tau)) if i not in used]
        I_curr = U_tau[leftover_idx].copy()

    return immediate_scrap_reward + total_future_reward


def choose_clairvoyant_eta_for_current_stage_cg(
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
    At the current review epoch, choose the best threshold using REALIZED
    future arrivals over the next H_scrap periods.
    """
    best_eta = None
    best_value = -np.inf
    eta_values = {}

    for eta in E_thresholds:
        val = evaluate_eta_with_realized_horizon_cg(
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


# =========================================================
# Online TSRH simulation with Column Generation
# =========================================================

def simulate_online_tsrah_cg(
    cfg: Config,
    arrivals_seq: list[np.ndarray],
    n_periods: int,
    H_scrap: int,
    E_thresholds: list[float],
    m_list: list[int],
    rho: float,
    gamma: float,
    s0: float,
    n_rollout_arrivals: int = 30,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Online TSRH:
    - inner layer uses Column Generation
    - outer layer uses TSRH with Monte Carlo rollout
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
        inner_res = solve_inner_rrp_cg(U_t, cfg, seed=inner_seed)

        groups_t = inner_res.get("groups", [])
        R_t_grp = float(inner_res.get("reward", 0.0))
        n_columns = int(inner_res.get("n_columns", 0))

        used = set()
        for g in groups_t:
            used.update(g)
        leftover_idx = [i for i in range(len(U_t)) if i not in used]
        I_t_plus = U_t[leftover_idx].copy()

        eta_star = np.nan
        D_t = []
        I_t_pp = I_t_plus.copy()
        R_t_scr = 0.0
        final_candidates = []

        if t % H_scrap == 0:
            def inner_solver_for_rollout(X, **kwargs):
                rollout_seed = int(rng.integers(1, 10**9))
                return solve_inner_rrp_cg(X, cfg, seed=rollout_seed)

            def arrival_fn_for_rollout(rng, **kwargs):
                return gaussian_arrival_generator(
                    rng=rng,
                    n_arrivals=n_rollout_arrivals,
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

        R_t = R_t_grp + R_t_scr
        I_t = I_t_pp.copy()

        rec = {
            "t": t,
            "policy": "ONLINE_TSRH_CG",
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
            "n_columns": n_columns,
            "final_candidates": str(final_candidates),
        }
        records.append(rec)

        if verbose:
            print(
                f"[ONLINE] t={t:2d} | arrivals={len(A_t):3d} | "
                f"packs={len(groups_t):2d} | cols={n_columns:4d} | "
                f"I_t^+={len(I_t_plus):3d} | scrap={len(D_t):3d} | "
                f"I_t++={len(I_t_pp):3d} | R_grp={R_t_grp:8.2f} | "
                f"R_scr={R_t_scr:6.2f} | eta={eta_star}"
            )

    return pd.DataFrame(records)


# =========================================================
# Stage-wise clairvoyant benchmark with Column Generation
# =========================================================

def simulate_stagewise_clairvoyant_benchmark_cg(
    cfg: Config,
    arrivals_seq: list[np.ndarray],
    n_periods: int,
    H_scrap: int,
    E_thresholds: list[float],
    gamma: float,
    s0: float,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Stage-wise perfect-information benchmark:
    - inner layer uses Column Generation
    - at each review epoch, choose the best current threshold using REALIZED
      arrivals over the next H_scrap periods
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
        inner_res = solve_inner_rrp_cg(U_t, cfg, seed=inner_seed)

        groups_t = inner_res.get("groups", [])
        R_t_grp = float(inner_res.get("reward", 0.0))
        n_columns = int(inner_res.get("n_columns", 0))

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
        eta_values = {}

        if t % H_scrap == 0:
            eta_star, clairvoyant_stage_value, eta_values = choose_clairvoyant_eta_for_current_stage_cg(
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

        R_t = R_t_grp + R_t_scr
        I_t = I_t_pp.copy()

        rec = {
            "t": t,
            "policy": "STAGEWISE_CLAIRVOYANT_BENCHMARK_CG",
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
            "n_columns": n_columns,
            "clairvoyant_stage_value": clairvoyant_stage_value,
            "eta_values": str(eta_values),
        }
        records.append(rec)

        if verbose:
            print(
                f"[BENCH]  t={t:2d} | arrivals={len(A_t):3d} | "
                f"packs={len(groups_t):2d} | cols={n_columns:4d} | "
                f"I_t^+={len(I_t_plus):3d} | scrap={len(D_t):3d} | "
                f"I_t++={len(I_t_pp):3d} | R_grp={R_t_grp:8.2f} | "
                f"R_scr={R_t_scr:6.2f} | eta={eta_star}"
            )

    return pd.DataFrame(records)


# =========================================================
# Excel export
# =========================================================

def build_summary_df(df_online: pd.DataFrame, df_bench: pd.DataFrame) -> pd.DataFrame:
    online_total = float(df_online["R_t_total"].sum())
    bench_total = float(df_bench["R_t_total"].sum())

    online_grp_total = float(df_online["R_t_grp"].sum())
    bench_grp_total = float(df_bench["R_t_grp"].sum())

    online_scr_total = float(df_online["R_t_scr"].sum())
    bench_scr_total = float(df_bench["R_t_scr"].sum())

    diff_total = online_total - bench_total
    diff_pct = 100.0 * diff_total / bench_total if bench_total > 0 else np.nan

    return pd.DataFrame([
        {"metric": "online_total_reward", "value": online_total},
        {"metric": "benchmark_total_reward", "value": bench_total},
        {"metric": "online_group_reward", "value": online_grp_total},
        {"metric": "benchmark_group_reward", "value": bench_grp_total},
        {"metric": "online_scrap_reward", "value": online_scr_total},
        {"metric": "benchmark_scrap_reward", "value": bench_scr_total},
        {"metric": "difference_online_minus_benchmark", "value": diff_total},
        {"metric": "relative_difference_pct", "value": diff_pct},
        {"metric": "online_total_columns", "value": float(df_online["n_columns"].sum())},
        {"metric": "benchmark_total_columns", "value": float(df_bench["n_columns"].sum())},
    ])


def build_period_compare_df(df_online: pd.DataFrame, df_bench: pd.DataFrame) -> pd.DataFrame:
    compare = pd.DataFrame({
        "t": df_online["t"],
        "online_R_t_grp": df_online["R_t_grp"],
        "bench_R_t_grp": df_bench["R_t_grp"],
        "diff_R_t_grp": df_online["R_t_grp"] - df_bench["R_t_grp"],

        "online_R_t_scr": df_online["R_t_scr"],
        "bench_R_t_scr": df_bench["R_t_scr"],
        "diff_R_t_scr": df_online["R_t_scr"] - df_bench["R_t_scr"],

        "online_R_t_total": df_online["R_t_total"],
        "bench_R_t_total": df_bench["R_t_total"],
        "diff_R_t_total": df_online["R_t_total"] - df_bench["R_t_total"],

        "online_cum_total": df_online["R_t_total"].cumsum(),
        "bench_cum_total": df_bench["R_t_total"].cumsum(),
        "diff_cum_total": df_online["R_t_total"].cumsum() - df_bench["R_t_total"].cumsum(),

        "online_eta": df_online["eta_star"],
        "bench_eta": df_bench["eta_star"],
        "online_columns": df_online["n_columns"],
        "bench_columns": df_bench["n_columns"],
    })
    return compare


def export_results_to_excel(
    df_online: pd.DataFrame,
    df_bench: pd.DataFrame,
    filepath: str,
):
    summary_df = build_summary_df(df_online, df_bench)
    compare_df = build_period_compare_df(df_online, df_bench)

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        df_online.to_excel(writer, sheet_name="Online_TSRH_CG", index=False)
        df_bench.to_excel(writer, sheet_name="Stagewise_Benchmark_CG", index=False)
        compare_df.to_excel(writer, sheet_name="Period_Comparison", index=False)


# =========================================================
# Main
# =========================================================

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
    # Online TSRH with Column Generation
    # -------------------------------------------------
    df_online = simulate_online_tsrah_cg(
        cfg=cfg,
        arrivals_seq=arrivals_seq,
        n_periods=n_periods,
        H_scrap=H_scrap,
        E_thresholds=E_thresholds,
        m_list=m_list,
        rho=rho,
        gamma=gamma,
        s0=s0,
        seed=seed,
        verbose=True,
    )

    # -------------------------------------------------
    # Stage-wise clairvoyant benchmark with Column Generation
    # -------------------------------------------------
    df_bench = simulate_stagewise_clairvoyant_benchmark_cg(
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

    # -------------------------------------------------
    # Export to Excel
    # -------------------------------------------------
    output_path = "upper_bound_results.xlsx"
    export_results_to_excel(df_online, df_bench, output_path)

    # -------------------------------------------------
    # Print summary
    # -------------------------------------------------
    summary_df = build_summary_df(df_online, df_bench)
    compare_df = build_period_compare_df(df_online, df_bench)

    print("\n=== Summary ===")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(summary_df.to_string(index=False))

    print("\n=== Period Comparison ===")
    with pd.option_context("display.max_columns", None, "display.width", 260):
        print(compare_df.to_string(index=False))

    print(f"\nSaved Excel file to: {output_path}")


if __name__ == "__main__":
    main()
