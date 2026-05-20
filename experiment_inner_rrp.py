"""
Inner RRP 基准实验脚本
比较多种启发式算法在固定电芯实例上的表现。

用法:
    python experiment_inner_rrp.py              # 完整实验
    python experiment_inner_rrp.py --quick       # 快速模式 (N=30,40, 3 种子)
    python experiment_inner_rrp.py --sizes 100 200  # 指定实例规模
    python experiment_inner_rrp.py --seeds 5      # 指定种子数
"""
from __future__ import annotations

import argparse
import sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import Config, ProblemConfig, DataConfig
from data_generator import generate_cells
from experiment_utils import (
    run_all_statistical_tests,
    create_inner_rrp_excel,
)


# =========================================================
# 算法调度器
# =========================================================

# 定义所有可用的算法及其求解函数
AVAILABLE_METHODS = {
    "KMEANS",
    "KMEANS_VNS",
    "GRASP",
    "GA",
    "SA",
    "MS_KMEANS_VNS",
    "COMBINE_REPAIR",
    "COLUMN_GENERATION",
    "GUROBI_ENUM",
}


def solve_method(
    method: str,
    X: np.ndarray,
    cfg: Config,
    k_t: int,
    seed: int | None = None,
) -> dict:
    """统一求解接口，返回标准格式的结果 dict。"""
    common = dict(
        X=X,
        K=cfg.problem.K,
        k_t=k_t,
        delta_bar=cfg.problem.delta_bar,
        w=np.asarray(cfg.problem.w),
        lambda_penalty=cfg.problem.lambda_penalty,
        theta1=cfg.problem.theta1,
        theta2=cfg.problem.theta2,
        theta3=cfg.problem.theta3,
        P1=cfg.problem.P1,
        P2=cfg.problem.P2,
        P3=cfg.problem.P3,
        seed=seed,
    )

    if method == "KMEANS":
        from heuristics.rrp_kmeans import solve_rrp_kmeans
        return solve_rrp_kmeans(
            **common,
            L1=cfg.kmeans.L1,
            L2=cfg.kmeans.L2,
            tol=cfg.kmeans.tol,
        )

    elif method == "KMEANS_VNS":
        from heuristics.rrp_kmeans_vns import solve_rrp_kmeans_vns
        return solve_rrp_kmeans_vns(
            **common,
            L1=cfg.vns.L1,
            tol=cfg.vns.tol,
            max_vns_iter=cfg.vns.max_vns_iter,
            max_no_improve=cfg.vns.max_no_improve,
            pack_candidate_limit=cfg.vns.pack_candidate_limit,
            partner_limit=cfg.vns.partner_limit,
            cell_candidate_limit=cfg.vns.cell_candidate_limit,
            leftover_candidate_limit=cfg.vns.leftover_candidate_limit,
            destroy_size=cfg.vns.destroy_size,
        )

    elif method == "GRASP":
        from heuristics.rrp_grasp import solve_rrp_grasp
        return solve_rrp_grasp(
            **common,
            n_starts=cfg.grasp.n_starts,
            rcl_size=cfg.grasp.rcl_size,
            max_group_attempts=cfg.grasp.max_group_attempts,
            max_local_iter=cfg.grasp.max_local_iter,
            group_candidate_limit=cfg.grasp.group_candidate_limit,
            cell_candidate_limit=cfg.grasp.cell_candidate_limit,
            leftover_candidate_limit=cfg.grasp.leftover_candidate_limit,
        )

    elif method == "GA":
        from heuristics.rrp_ga import solve_rrp_ga
        return solve_rrp_ga(
            **common,
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

    elif method == "SA":
        from heuristics.rrp_sa import solve_rrp_sa
        return solve_rrp_sa(
            **common,
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

    elif method == "MS_KMEANS_VNS":
        from heuristics.rrp_ms_kmeans_vns import solve_rrp_ms_kmeans_vns
        return solve_rrp_ms_kmeans_vns(
            **common,
        )

    elif method == "COMBINE_REPAIR":
        from heuristics.rrp_combine_repair import solve_rrp_combine_repair
        return solve_rrp_combine_repair(
            **common,
        )

    elif method == "COLUMN_GENERATION":
        from heuristics.rrp_column_generation import solve_rrp_column_generation
        return solve_rrp_column_generation(
            **common,
            max_cg_iter=cfg.cg.max_cg_iter,
            init_n_starts=cfg.cg.init_n_starts,
            init_neighbor_size=cfg.cg.init_neighbor_size,
            pricing_n_seeds=cfg.cg.pricing_n_seeds,
            pricing_neighbor_size=cfg.cg.pricing_neighbor_size,
            max_new_cols=cfg.cg.max_new_cols,
            use_gurobi_pricing=cfg.cg.max_cg_iter > 0,  # 默认开启
        )

    elif method == "GUROBI_ENUM":
        from heuristics.rrp_gurobi_exact import solve_rrp_gurobi_enumeration
        return solve_rrp_gurobi_enumeration(
            **common,
            time_limit=300.0,
            max_groups=5000000,
        )

    else:
        raise ValueError(f"Unknown method: {method}")


# =========================================================
# 实例生成
# =========================================================

def generate_instances(
    n_cells_list: list[int],
    n_seeds: int,
    data_config: DataConfig,
    base_seed: int = 42,
) -> list[tuple[int, int, np.ndarray]]:
    """
    生成实验实例列表。
    返回 [(instance_id, n_cells, X), ...]
    """
    instances = []
    inst_id = 0
    for n_cells in n_cells_list:
        for s in range(n_seeds):
            seed = base_seed + s
            X = generate_cells(
                n_cells=n_cells,
                mu_C=data_config.mu_C,
                sigma_C=data_config.sigma_C,
                mu_R=data_config.mu_R,
                sigma_R=data_config.sigma_R,
                seed=seed,
            )
            instances.append((inst_id, n_cells, X))
            inst_id += 1
    return instances


# =========================================================
# 标准化函数
# =========================================================

def standardize_cells(X: np.ndarray) -> np.ndarray:
    """
    将原始 (C, R) 转换为 z-score。
    使用全局均值/方差（而非实例内），与 multi_inner_opt.py 一致。
    """
    # 对于实验，使用实例内的标准化
    mu_C, sigma_C = X[:, 0].mean(), X[:, 0].std()
    mu_R, sigma_R = X[:, 1].mean(), X[:, 1].std()
    if sigma_C < 1e-10:
        sigma_C = 1.0
    if sigma_R < 1e-10:
        sigma_R = 1.0
    C_std = (X[:, 0] - mu_C) / sigma_C
    R_std = (X[:, 1] - mu_R) / sigma_R
    return np.column_stack([C_std, R_std])


# =========================================================
# 主实验流程
# =========================================================

def run_experiment(
    methods: list[str],
    n_cells_list: list[int],
    n_seeds: int,
    cfg: Config,
    output_path: str,
    skip_methods: list[str] | None = None,
):
    if skip_methods is None:
        skip_methods = []

    methods = [m for m in methods if m not in skip_methods]
    print(f"实验方法: {methods}")
    print(f"实例规模: {n_cells_list}")
    print(f"种子数: {n_seeds}")
    print(f"问题参数: K={cfg.problem.K}, delta_bar={cfg.problem.delta_bar}, k_max={cfg.problem.k_max}")
    print()

    # 生成实例
    instances = generate_instances(
        n_cells_list=n_cells_list,
        n_seeds=n_seeds,
        data_config=cfg.data,
        base_seed=42,
    )
    print(f"总实例数: {len(instances)}")
    print(f"总实验数: {len(instances)} x {len(methods)} = {len(instances) * len(methods)}")
    print()

    # 结果收集
    raw_records = []
    optimal_records = []  # 用于小规模最优解对比

    total_start = time.perf_counter()

    for inst_id, n_cells, X_raw in tqdm(instances, desc="Instances"):
        # 标准化
        X = standardize_cells(X_raw)
        k_t = min(cfg.problem.k_max, n_cells // cfg.problem.K)

        for method in methods:
            method_start = time.perf_counter()

            try:
                seed = 42  # 固定种子确保可复现
                result = solve_method(method, X, cfg, k_t, seed=seed)

                runtime = time.perf_counter() - method_start

                rec = {
                    "instance_id": inst_id,
                    "n_cells": n_cells,
                    "seed": seed,
                    "method": method,
                    "reward": result.get("reward", 0.0),
                    "n_packs": result.get("n_packs", 0),
                    "avg_delta": result.get("avg_delta", 0.0),
                    "avg_phi": result.get("avg_phi", 0.0),
                    "runtime": runtime,
                    "reward_per_pack": result.get("reward_per_pack", 0.0),
                    "positive_pack_ratio": result.get("positive_pack_ratio", 0.0),
                    "leftover_count": len(result.get("leftover", [])),
                    "P1": result.get("tier_counts", {}).get("P1", 0),
                    "P2": result.get("tier_counts", {}).get("P2", 0),
                    "P3": result.get("tier_counts", {}).get("P3", 0),
                    "P0": result.get("tier_counts", {}).get("P0", 0),
                    "gap_to_optimal": np.nan,  # 后续填充
                }
                raw_records.append(rec)

                # GUROBI_ENUM 的结果用于计算最优性间隔
                if method == "GUROBI_ENUM":
                    opt_reward = result.get("reward", 0.0)
                    for r in raw_records:
                        if r["instance_id"] == inst_id and r["method"] != "GUROBI_ENUM":
                            if opt_reward > 0:
                                r["gap_to_optimal"] = 100.0 * (opt_reward - r["reward"]) / opt_reward
                            else:
                                r["gap_to_optimal"] = 0.0
                    # 记录最优解
                    optimal_records.append({
                        "n_cells": n_cells,
                        "seed": seed,
                        "method": "GUROBI_ENUM",
                        "reward": opt_reward,
                        "optimal_reward": opt_reward,
                        "gap_pct": 0.0,
                    })

            except Exception as e:
                runtime = time.perf_counter() - method_start
                print(f"\n  [WARN] {method} failed on instance {inst_id} (N={n_cells}): {e}")
                raw_records.append({
                    "instance_id": inst_id,
                    "n_cells": n_cells,
                    "seed": seed,
                    "method": method,
                    "reward": np.nan,
                    "n_packs": np.nan,
                    "avg_delta": np.nan,
                    "avg_phi": np.nan,
                    "runtime": runtime,
                    "reward_per_pack": np.nan,
                    "positive_pack_ratio": np.nan,
                    "leftover_count": np.nan,
                    "P1": np.nan, "P2": np.nan, "P3": np.nan, "P0": np.nan,
                    "gap_to_optimal": np.nan,
                })

    total_runtime = time.perf_counter() - total_start
    print(f"\n实验完成! 总运行时间: {total_runtime:.1f}s ({total_runtime / 60:.1f} min)")

    # 构建 DataFrame
    results_df = pd.DataFrame(raw_records)

    # 汇总统计
    summary_records = []
    for method in methods:
        for n_cells in n_cells_list:
            subset = results_df[(results_df["method"] == method) & (results_df["n_cells"] == n_cells)]
            rewards = subset["reward"].dropna()
            runtimes = subset["runtime"].dropna()
            n_packs = subset["n_packs"].dropna()
            rpp = subset["reward_per_pack"].dropna()
            ppr = subset["positive_pack_ratio"].dropna()
            leftovers = subset["leftover_count"].dropna()

            summary_records.append({
                "method": method,
                "n_cells": n_cells,
                "mean_reward": float(rewards.mean()) if len(rewards) > 0 else 0.0,
                "std_reward": float(rewards.std()) if len(rewards) > 1 else 0.0,
                "median_reward": float(rewards.median()) if len(rewards) > 0 else 0.0,
                "min_reward": float(rewards.min()) if len(rewards) > 0 else 0.0,
                "max_reward": float(rewards.max()) if len(rewards) > 0 else 0.0,
                "mean_runtime": float(runtimes.mean()) if len(runtimes) > 0 else 0.0,
                "mean_n_packs": float(n_packs.mean()) if len(n_packs) > 0 else 0.0,
                "mean_reward_per_pack": float(rpp.mean()) if len(rpp) > 0 else 0.0,
                "mean_positive_ratio": float(ppr.mean()) if len(ppr) > 0 else 0.0,
                "mean_leftover": float(leftovers.mean()) if len(leftovers) > 0 else 0.0,
                "total_P1": int(subset["P1"].sum()),
                "total_P2": int(subset["P2"].sum()),
                "total_P3": int(subset["P3"].sum()),
                "total_P0": int(subset["P0"].sum()),
            })

    summary_df = pd.DataFrame(summary_records)

    # 最优性间隔
    optimality_gap_df = pd.DataFrame(optimal_records) if optimal_records else None

    # 统计检验
    print("\n运行统计检验...")
    stat_results = run_all_statistical_tests(
        results_df=results_df,
        methods=methods,
        ref_method="KMEANS_VNS",
        metric="reward",
    )

    # 打印摘要
    print("\n" + "=" * 80)
    print("汇总统计 (按 n_cells 分组)")
    print("=" * 80)
    for n_cells in n_cells_list:
        print(f"\n--- N = {n_cells} ---")
        sub = summary_df[summary_df["n_cells"] == n_cells]
        print(sub[["method", "mean_reward", "std_reward", "mean_runtime", "mean_n_packs"]].to_string(index=False))

    # 打印统计检验结果
    print("\n" + "=" * 80)
    print("配对 t 检验 (KMEANS_VNS vs 其他, 指标: reward)")
    print("=" * 80)
    for tt in stat_results["paired_ttests"]:
        sig = "***" if tt["p_value"] < 0.001 else "**" if tt["p_value"] < 0.01 else "*" if tt["p_value"] < 0.05 else "n.s."
        print(f"  KMEANS_VNS vs {tt['method_b']}: mean_diff={tt['mean_diff']:+.4f}, p={tt['p_value']:.4f} {sig}")

    print("\nCohen's d 效应量:")
    for m, d in stat_results["cohens_ds"].items():
        print(f"  KMEANS_VNS vs {m}: d={d:.4f}")

    if stat_results["anova"]:
        anova = stat_results["anova"]
        sig = "***" if anova["p_value"] < 0.001 else "**" if anova["p_value"] < 0.01 else "*" if anova["p_value"] < 0.05 else "n.s."
        print(f"\nANOVA: F={anova['f_statistic']:.4f}, p={anova['p_value']:.2e} {sig}")

    # 生成 Excel
    print("\n生成 Excel 报告...")
    create_inner_rrp_excel(
        results_df=results_df,
        stat_results=stat_results,
        optimality_gap_df=optimality_gap_df,
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
    parser = argparse.ArgumentParser(description="Inner RRP 基准实验")
    parser.add_argument("--quick", action="store_true", help="快速模式: N=[30,40], 3 种子")
    parser.add_argument("--sizes", nargs="+", type=int, default=None, help="指定实例规模 (空格分隔)")
    parser.add_argument("--seeds", type=int, default=None, help="指定种子数")
    parser.add_argument("--output", type=str, default="results/experiment_inner_rrp.xlsx", help="输出路径")
    parser.add_argument("--skip", nargs="+", default=None, help="跳过的算法 (空格分隔)")
    args = parser.parse_args()

    cfg = Config()

    if args.quick:
        n_cells_list = [30, 40]
        n_seeds = 3
        methods = ["KMEANS", "KMEANS_VNS", "GRASP", "GA", "SA", "MS_KMEANS_VNS", "COMBINE_REPAIR"]
        # 快速模式跳过 CG 和 GUROBI (需要 Gurobi 许可证)
    elif args.sizes:
        n_cells_list = args.sizes
        n_seeds = args.seeds if args.seeds else 20
        methods = list(AVAILABLE_METHODS)
    else:
        # 默认完整实验
        n_cells_list = [30, 40, 50, 100, 200, 300, 500]
        n_seeds = args.seeds if args.seeds else 20
        methods = ["KMEANS", "KMEANS_VNS", "GRASP", "GA", "SA", "MS_KMEANS_VNS", "COMBINE_REPAIR", "COLUMN_GENERATION"]
        # GUROBI_ENUM 仅对 N<=50 运行，需要单独处理

    skip = args.skip or []

    run_experiment(
        methods=methods,
        n_cells_list=n_cells_list,
        n_seeds=n_seeds,
        cfg=cfg,
        output_path=args.output,
        skip_methods=skip,
    )


if __name__ == "__main__":
    main()
