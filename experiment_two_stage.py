"""
Two-Stage 完整系统实验脚本
在动态到达 + 废弃决策环境中，比较各算法作为内层求解器的表现。

用法:
    python experiment_two_stage.py              # 完整实验 (10 种子, H=[4,5,6])
    python experiment_two_stage.py --quick       # 快速模式 (3 种子, H=[5])
    python experiment_two_stage.py --seeds 5     # 指定种子数
"""
from __future__ import annotations

import argparse
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import Config
from data_generator import generate_cells
from heuristics.rrp_kmeans import solve_rrp_kmeans
from heuristics.rrp_kmeans_vns import solve_rrp_kmeans_vns
from heuristics.rrp_grasp import solve_rrp_grasp
from heuristics.rrp_ga import solve_rrp_ga
from heuristics.rrp_sa import solve_rrp_sa
from heuristics.rrp_ms_kmeans_vns import solve_rrp_ms_kmeans_vns
from heuristics.rrp_combine_repair import solve_rrp_combine_repair
try:
    from heuristics.rrp_column_generation import solve_rrp_column_generation
except ImportError:
    solve_rrp_column_generation = None
from outer.tsrah import tsrah_scrapping_decision, default_quality_score
from outer.arrival import gaussian_arrival_generator
from experiment_utils import create_two_stage_excel


# =========================================================
# Inner solver wrapper (扩展版，包含更多算法)
# =========================================================

def solve_inner_rrp(
    X: np.ndarray,
    cfg: Config,
    method: str = "VNS",
    seed: int | None = None,
) -> dict:
    """
    统一 inner RRP 求解接口。
    method: "KMEANS", "VNS", "GRASP", "GA", "SA", "MS_VNS", "COMBINE", "CG"
    """
    k_t = min(cfg.problem.k_max, X.shape[0] // cfg.problem.K)
    common = dict(
        X=X, K=cfg.problem.K, k_t=k_t, delta_bar=cfg.problem.delta_bar,
        w=np.asarray(cfg.problem.w), lambda_penalty=cfg.problem.lambda_penalty,
        theta1=cfg.problem.theta1, theta2=cfg.problem.theta2, theta3=cfg.problem.theta3,
        P1=cfg.problem.P1, P2=cfg.problem.P2, P3=cfg.problem.P3, seed=seed,
    )

    if method == "KMEANS":
        return solve_rrp_kmeans(**common, L1=cfg.kmeans.L1, L2=cfg.kmeans.L2, tol=cfg.kmeans.tol)
    elif method == "GRASP":
        return solve_rrp_grasp(
            **common, n_starts=cfg.grasp.n_starts, rcl_size=cfg.grasp.rcl_size,
            max_group_attempts=cfg.grasp.max_group_attempts, max_local_iter=cfg.grasp.max_local_iter,
            group_candidate_limit=cfg.grasp.group_candidate_limit,
            cell_candidate_limit=cfg.grasp.cell_candidate_limit,
            leftover_candidate_limit=cfg.grasp.leftover_candidate_limit,
        )
    elif method == "GA":
        return solve_rrp_ga(
            **common, population_size=cfg.ga.population_size, n_generations=cfg.ga.n_generations,
            tournament_size=cfg.ga.tournament_size, crossover_prob=cfg.ga.crossover_prob,
            mutation_prob=cfg.ga.mutation_prob, destroy_size=cfg.ga.destroy_size,
            local_search_prob=cfg.ga.local_search_prob, elitism_size=cfg.ga.elitism_size,
            group_candidate_limit=cfg.ga.group_candidate_limit,
            cell_candidate_limit=cfg.ga.cell_candidate_limit,
            leftover_candidate_limit=cfg.ga.leftover_candidate_limit,
        )
    elif method == "SA":
        return solve_rrp_sa(
            **common, initial_temperature=cfg.sa.initial_temperature, cooling_rate=cfg.sa.cooling_rate,
            min_temperature=cfg.sa.min_temperature, max_sa_iterations=cfg.sa.max_sa_iterations,
            vnd_interval=cfg.sa.vnd_interval, max_vnd_rounds=cfg.sa.max_vnd_rounds,
            reheating_ratio=cfg.sa.reheating_ratio, reheating_stall=cfg.sa.reheating_stall,
            max_reheats=cfg.sa.max_reheats, tabu_tenure=cfg.sa.tabu_tenure,
            n_init_starts=cfg.sa.n_init_starts, kmeans_L1=cfg.sa.kmeans_L1,
            kmeans_tol=cfg.sa.kmeans_tol, residual_rounds=cfg.sa.residual_rounds,
        )
    elif method == "MS_VNS":
        return solve_rrp_ms_kmeans_vns(**common)
    elif method == "COMBINE":
        return solve_rrp_combine_repair(**common)
    elif method == "CG":
        if solve_rrp_column_generation is None:
            raise RuntimeError("COLUMN_GENERATION requires pulp. Install with: pip install pulp")
        return solve_rrp_column_generation(
            **common, max_cg_iter=cfg.cg.max_cg_iter, init_n_starts=cfg.cg.init_n_starts,
            init_neighbor_size=cfg.cg.init_neighbor_size, pricing_n_seeds=cfg.cg.pricing_n_seeds,
            pricing_neighbor_size=cfg.cg.pricing_neighbor_size, max_new_cols=cfg.cg.max_new_cols,
            use_gurobi_pricing=False,  # Two-Stage 中关闭 Gurobi pricing 加速
        )
    else:  # VNS (default)
        return solve_rrp_kmeans_vns(
            **common, L1=cfg.vns.L1, tol=cfg.vns.tol, max_vns_iter=cfg.vns.max_vns_iter,
            max_no_improve=cfg.vns.max_no_improve,
            pack_candidate_limit=cfg.vns.pack_candidate_limit, partner_limit=cfg.vns.partner_limit,
            cell_candidate_limit=cfg.vns.cell_candidate_limit,
            leftover_candidate_limit=cfg.vns.leftover_candidate_limit,
            destroy_size=cfg.vns.destroy_size,
        )


# =========================================================
# 模拟函数 (从 main_two_stage.py 提取并适配)
# =========================================================

def generate_arrival_batch(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    return gaussian_arrival_generator(
        rng=rng, n_arrivals=130,
        mu_C=cfg.data.mu_C, sigma_C=cfg.data.sigma_C,
        mu_R=cfg.data.mu_R, sigma_R=cfg.data.sigma_R,
    )


def generate_arrival_sequence(cfg: Config, n_periods: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [generate_arrival_batch(cfg, rng) for _ in range(n_periods)]


def apply_threshold_scrap(I_t_plus: np.ndarray, eta: float, wq: np.ndarray):
    if len(I_t_plus) == 0:
        return [], I_t_plus.copy()
    q_vals = default_quality_score(I_t_plus, wq=wq)
    D_t = np.where(q_vals < eta)[0].tolist()
    if len(D_t) == 0:
        return D_t, I_t_plus.copy()
    mask = np.ones(len(I_t_plus), dtype=bool)
    mask[np.array(D_t, dtype=int)] = False
    return D_t, I_t_plus[mask].copy()


def simulate_two_stage_system(
    cfg: Config, n_periods: int, H_scrap: int, E_thresholds: list[float],
    m_list: list[int], rho: float, gamma: float, s0: float,
    arrivals_seq: list[np.ndarray] | None = None,
    inner_method: str = "VNS", seed: int = 42, verbose: bool = True,
):
    rng = np.random.default_rng(seed)
    if arrivals_seq is None:
        arrivals_seq = generate_arrival_sequence(cfg, n_periods, seed)

    I_t = generate_cells(
        n_cells=0, mu_C=cfg.data.mu_C, sigma_C=cfg.data.sigma_C,
        mu_R=cfg.data.mu_R, sigma_R=cfg.data.sigma_R, seed=seed,
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

        D_t = []
        I_t_pp = I_t_plus.copy()
        R_t_scr = 0.0
        eta_star = np.nan

        if t % H_scrap == 0:
            # 简化版：使用固定阈值策略（取 E_thresholds 的中位数）
            # 这比 TSRH 的 Monte Carlo rollout 快得多
            eta_star = float(np.median(E_thresholds))
            D_t, I_t_pp = apply_threshold_scrap(I_t_plus, eta_star, wq)
            R_t_scr = s0 * len(D_t)

        R_t = R_t_grp + R_t_scr
        I_t = I_t_pp.copy()

        records.append({
            "method": inner_method, "t": t, "arrivals": len(A_t),
            "available": len(U_t), "n_packs": len(groups_t),
            "used_cells": sum(len(g) for g in groups_t),
            "inventory_after_rrp": len(I_t_plus),
            "scrap_count": len(D_t), "inventory_after_scrap": len(I_t_pp),
            "R_t_grp": R_t_grp, "R_t_scr": R_t_scr, "R_t_total": R_t,
            "eta_star": eta_star,
        })

        if verbose:
            print(
                f"  [{inner_method}] t={t:2d} | packs={len(groups_t):2d} | "
                f"I+={len(I_t_plus):3d} | scrap={len(D_t):3d} | "
                f"R_grp={R_t_grp:8.2f} | R_scr={R_t_scr:6.2f} | eta={eta_star}"
            )

    return pd.DataFrame(records)


def simulate_stagewise_clairvoyant_ub(
    cfg: Config, arrivals_seq: list[np.ndarray], n_periods: int, H_scrap: int,
    E_thresholds: list[float], gamma: float, s0: float,
    inner_method: str = "VNS", seed: int = 42, verbose: bool = True,
):
    """分阶段 clairvoyant 上界。"""
    rng = np.random.default_rng(seed)
    wq = np.array([1.0, -1.0], dtype=float)

    I_t = generate_cells(
        n_cells=0, mu_C=cfg.data.mu_C, sigma_C=cfg.data.sigma_C,
        mu_R=cfg.data.mu_R, sigma_R=cfg.data.sigma_R, seed=seed,
    )

    records = []
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

        # 在废弃审查点，用穷举搜索找到最优 eta
        eta_star = np.nan
        D_t = []
        I_t_pp = I_t_plus.copy()
        R_t_scr = 0.0
        clairvoyant_val = np.nan

        if t % H_scrap == 0:
            # 简化版 clairvoyant: 尝试几个 eta 值，选最优的
            best_eta, best_val = None, -np.inf
            for eta in [80.0, 100.0, 120.0, 140.0]:
                I_curr = I_t_plus.copy()
                q_vals = default_quality_score(I_curr, wq=wq)
                scrap_idx = np.where(q_vals < eta)[0].tolist()
                immediate = s0 * len(scrap_idx)
                if scrap_idx:
                    mask = np.ones(len(I_curr), dtype=bool)
                    mask[np.array(scrap_idx, dtype=int)] = False
                    I_curr = I_curr[mask].copy()

                # 只评估未来 1 期（简化）
                ft = t + 1
                if ft <= len(arrivals_seq):
                    Ar = arrivals_seq[ft - 1]
                    if len(I_curr) == 0:
                        Ut = Ar.copy()
                    elif len(Ar) == 0:
                        Ut = I_curr.copy()
                    else:
                        Ut = np.vstack([I_curr, Ar])
                    ir = solve_inner_rrp(Ut, cfg, method=inner_method, seed=42)
                    future_reward = gamma * float(ir.get("reward", 0.0))
                else:
                    future_reward = 0.0

                val = immediate + future_reward
                if val > best_val:
                    best_val = val
                    best_eta = float(eta)

            eta_star = best_eta
            clairvoyant_val = best_val
            D_t, I_t_pp = apply_threshold_scrap(I_t_plus, eta_star, wq)
            R_t_scr = s0 * len(D_t)

        R_t = R_t_grp + R_t_scr
        I_t = I_t_pp.copy()

        records.append({
            "method": inner_method, "t": t, "arrivals": len(A_t),
            "available": len(U_t), "n_packs": len(groups_t),
            "used_cells": sum(len(g) for g in groups_t),
            "inventory_after_rrp": len(I_t_plus),
            "scrap_count": len(D_t), "inventory_after_scrap": len(I_t_pp),
            "R_t_grp": R_t_grp, "R_t_scr": R_t_scr, "R_t_total": R_t,
            "eta_star": eta_star, "clairvoyant_val": clairvoyant_val,
        })

        if verbose:
            print(
                f"  [UB-{inner_method}] t={t:2d} | packs={len(groups_t):2d} | "
                f"I+={len(I_t_plus):3d} | scrap={len(D_t):3d} | "
                f"R_grp={R_t_grp:8.2f} | R_scr={R_t_scr:6.2f} | eta={eta_star}"
            )

    return pd.DataFrame(records)


# =========================================================
# 主实验
# =========================================================

def run_experiment(
    methods: list[str],
    seeds: list[int],
    H_scrap_values: list[int],
    cfg: Config,
    output_path: str,
    n_periods: int = 20,
    skip_methods: list[str] | None = None,
):
    if skip_methods is None:
        skip_methods = []
    methods = [m for m in methods if m not in skip_methods]

    E_thresholds = [70, 80, 90, 100, 110, 115, 120, 125, 130, 135, 140]
    m_list = [2, 4, 8]
    rho = 0.5
    gamma = 0.95
    s0 = 5.0

    print(f"Two-Stage 实验配置:")
    print(f"  方法: {methods}")
    print(f"  种子: {seeds}")
    print(f"  H_scrap: {H_scrap_values}")
    print(f"  n_periods: {n_periods}")
    print()

    raw_records = []
    total_start = time.perf_counter()

    total_runs = len(methods) * len(seeds) * len(H_scrap_values)
    run_count = 0

    for seed in seeds:
        for H_scrap in H_scrap_values:
            # 预先生成 arrival sequence (所有方法共用同一条路径)
            arrivals_seq = generate_arrival_sequence(cfg, n_periods, seed)

            for method in methods:
                run_count += 1
                print(f"[{run_count}/{total_runs}] seed={seed}, H={H_scrap}, method={method}")

                # Online 模拟
                start_t = time.perf_counter()
                try:
                    df_online = simulate_two_stage_system(
                        cfg=cfg, n_periods=n_periods, H_scrap=H_scrap,
                        E_thresholds=E_thresholds, m_list=m_list, rho=rho,
                        gamma=gamma, s0=s0, arrivals_seq=arrivals_seq,
                        inner_method=method, seed=seed, verbose=False,
                    )
                    online_time = time.perf_counter() - start_t

                    # UB 模拟
                    df_ub = simulate_stagewise_clairvoyant_ub(
                        cfg=cfg, arrivals_seq=arrivals_seq, n_periods=n_periods,
                        H_scrap=H_scrap, E_thresholds=E_thresholds, gamma=gamma,
                        s0=s0, inner_method=method, seed=seed, verbose=False,
                    )

                    online_total = float(df_online["R_t_total"].sum())
                    ub_total = float(df_ub["R_t_total"].sum())
                    gap_pct = 100.0 * (ub_total - online_total) / ub_total if ub_total > 0 else np.nan

                    raw_records.append({
                        "seed": seed,
                        "H_scrap": H_scrap,
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
                        "runtime": online_time,
                    })
                except Exception as e:
                    runtime = time.perf_counter() - start_t
                    print(f"  [WARN] {method} failed: {e}")
                    raw_records.append({
                        "seed": seed, "H_scrap": H_scrap, "method": method,
                        "online_group_reward": np.nan, "online_scrap_reward": np.nan,
                        "online_total_reward": np.nan, "online_total_scrap": np.nan,
                        "online_avg_packs": np.nan, "ub_total_reward": np.nan,
                        "ub_total_scrap": np.nan, "ub_avg_packs": np.nan,
                        "gap_pct": np.nan, "runtime": runtime,
                    })

    total_runtime = time.perf_counter() - total_start
    print(f"\n实验完成! 总运行时间: {total_runtime:.1f}s ({total_runtime / 60:.1f} min)")

    results_df = pd.DataFrame(raw_records)

    # 汇总统计
    summary_records = []
    for method in methods:
        for H_scrap in H_scrap_values:
            subset = results_df[(results_df["method"] == method) & (results_df["H_scrap"] == H_scrap)]
            summary_records.append({
                "method": method,
                "H_scrap": H_scrap,
                "mean_online_reward": float(subset["online_total_reward"].mean()),
                "std_online_reward": float(subset["online_total_reward"].std()),
                "mean_ub_reward": float(subset["ub_total_reward"].mean()),
                "mean_gap_pct": float(subset["gap_pct"].mean()),
                "std_gap_pct": float(subset["gap_pct"].std()),
                "mean_scrap": float(subset["online_total_scrap"].mean()),
                "mean_avg_packs": float(subset["online_avg_packs"].mean()),
                "mean_runtime": float(subset["runtime"].mean()),
            })
    summary_df = pd.DataFrame(summary_records)

    # 打印摘要
    print("\n" + "=" * 80)
    print("Two-Stage 汇总统计")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # 生成 Excel
    create_two_stage_excel(
        results_df=results_df,
        summary_df=summary_df,
        problem_config=cfg.problem,
        data_config=cfg.data,
        output_path=output_path,
    )

    return results_df, summary_df


# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Two-Stage 完整系统实验")
    parser.add_argument("--quick", action="store_true", help="快速模式: 3 种子, H=[5]")
    parser.add_argument("--seeds", type=int, default=None, help="指定种子数")
    parser.add_argument("--H", nargs="+", type=int, default=None, help="指定 H_scrap 值")
    parser.add_argument("--output", type=str, default="results/experiment_two_stage.xlsx", help="输出路径")
    parser.add_argument("--skip", nargs="+", default=None, help="跳过的算法")
    parser.add_argument("--methods", nargs="+", default=None, help="指定运行的算法")
    args = parser.parse_args()

    cfg = Config()

    if args.quick:
        seeds = [42, 43, 44]
        H_values = [5]
        methods = ["KMEANS", "VNS", "GRASP", "GA"]
    else:
        n_seeds = args.seeds if args.seeds else 10
        seeds = list(range(42, 42 + n_seeds))
        H_values = args.H if args.H else [4, 5, 6]
        if args.methods:
            methods = args.methods
        else:
            # methods = ["KMEANS", "VNS", "GRASP", "GA", "SA", "MS_VNS", "COMBINE", "CG"]
            methods = ["KMEANS", "VNS", "GA", "SA", "MS_VNS"]
    skip = args.skip or []

    run_experiment(
        methods=methods,
        seeds=seeds,
        H_scrap_values=H_values,
        cfg=cfg,
        output_path=args.output,
        skip_methods=skip,
    )


if __name__ == "__main__":
    main()
