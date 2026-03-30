# run_experiment.py
from __future__ import annotations

import os
import time
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from config import Config
from data_generator import generate_cells
from heuristics.rrp_kmeans import solve_rrp_kmeans
from heuristics.rrp_kmeans_vns import solve_rrp_kmeans_vns
from heuristics.rrp_grasp import solve_rrp_grasp
from heuristics.rrp_ga import solve_rrp_ga
from heuristics.rrp_column_generation import solve_rrp_column_generation
from utils import summarize_solution, safe_div


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def format_tier_distribution(tier_counts: dict) -> str:
    return (
        f"P1:{tier_counts.get('P1', 0)}, "
        f"P2:{tier_counts.get('P2', 0)}, "
        f"P3:{tier_counts.get('P3', 0)}, "
        f"P0:{tier_counts.get('P0', 0)}"
    )


def enrich_result(result: Dict[str, Any], X: np.ndarray, cfg: Config) -> Dict[str, Any]:
    summary = summarize_solution(
        X=X,
        groups=result["groups"],
        K=cfg.problem.K,
        w=cfg.problem.w,
        lambda_penalty=cfg.problem.lambda_penalty,
        theta1=cfg.problem.theta1,
        theta2=cfg.problem.theta2,
        theta3=cfg.problem.theta3,
        P1=cfg.problem.P1,
        P2=cfg.problem.P2,
        P3=cfg.problem.P3,
    )

    total_cells = X.shape[0]
    used_cells = cfg.problem.K * summary["n_packs"]
    utilization_rate = safe_div(used_cells, total_cells)

    result["reward"] = summary["total_reward"]
    result["n_packs"] = summary["n_packs"]
    result["avg_delta"] = summary["avg_delta"]
    result["avg_phi"] = summary["avg_phi"]
    result["reward_per_pack"] = summary["reward_per_pack"]
    result["positive_pack_ratio"] = summary["positive_pack_ratio"]
    result["tier_counts"] = summary["tier_counts"]
    result["tier_distribution"] = format_tier_distribution(summary["tier_counts"])
    result["utilization_rate"] = utilization_rate

    return result


def build_row(
    result: Dict[str, Any],
    instance_name: str,
    seed: int,
    cfg: Config,
    extra_tags: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    row = {
        "instance_name": instance_name,
        "seed": seed,
        "method": result["method"],
        "n_cells": cfg.problem.n_cells,
        "K": cfg.problem.K,
        "k_max": cfg.problem.k_max,
        "delta_bar": cfg.problem.delta_bar,
        "lambda_penalty": cfg.problem.lambda_penalty,
        "theta1": cfg.problem.theta1,
        "theta2": cfg.problem.theta2,
        "theta3": cfg.problem.theta3,
        "P1": cfg.problem.P1,
        "P2": cfg.problem.P2,
        "P3": cfg.problem.P3,
        "reward": result["reward"],
        "n_packs": result["n_packs"],
        "avg_delta": result["avg_delta"],
        "avg_phi": result["avg_phi"],
        "reward_per_pack": result["reward_per_pack"],
        "utilization_rate": result["utilization_rate"],
        "positive_pack_ratio": result["positive_pack_ratio"],
        "runtime": result["runtime"],
        "leftover": len(result["leftover"]),
        "tier_P1": result["tier_counts"].get("P1", 0),
        "tier_P2": result["tier_counts"].get("P2", 0),
        "tier_P3": result["tier_counts"].get("P3", 0),
        "tier_P0": result["tier_counts"].get("P0", 0),
        "tier_distribution": result["tier_distribution"],
    }

    if "n_columns" in result:
        row["n_columns"] = result["n_columns"]
    else:
        row["n_columns"] = np.nan

    if extra_tags is not None:
        row.update(extra_tags)

    return row


def aggregate_results(raw_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    numeric_cols = [
        "reward",
        "n_packs",
        "avg_delta",
        "avg_phi",
        "reward_per_pack",
        "utilization_rate",
        "positive_pack_ratio",
        "runtime",
        "leftover",
        "tier_P1",
        "tier_P2",
        "tier_P3",
        "tier_P0",
        "n_columns",
    ]

    agg_dict = {}
    for c in numeric_cols:
        if c in raw_df.columns:
            agg_dict[c] = ["mean", "std"]

    summary = raw_df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary


def run_methods_on_instance(X: np.ndarray, cfg: Config, seed: int) -> List[Dict[str, Any]]:
    results = []
    k_t = min(cfg.problem.k_max, X.shape[0] // cfg.problem.K)

    if cfg.experiment.run_kmeans:
        res_kmeans = solve_rrp_kmeans(
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
        results.append(enrich_result(res_kmeans, X, cfg))

    if cfg.experiment.run_kmeans_vns:
        res_vns = solve_rrp_kmeans_vns(
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
        results.append(enrich_result(res_vns, X, cfg))

    if cfg.experiment.run_grasp:
        res_grasp = solve_rrp_grasp(
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
        results.append(enrich_result(res_grasp, X, cfg))

    if cfg.experiment.run_ga:
        res_ga = solve_rrp_ga(
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
        results.append(enrich_result(res_ga, X, cfg))

    run_cg = cfg.experiment.run_column_generation
    if cfg.experiment.skip_cg_for_large_instances and cfg.problem.n_cells > cfg.experiment.cg_size_threshold:
        run_cg = False

    if run_cg:
        res_cg = solve_rrp_column_generation(
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
        results.append(enrich_result(res_cg, X, cfg))

    return results


def generate_instance(cfg: Config, seed: int) -> np.ndarray:
    return generate_cells(
        n_cells=cfg.problem.n_cells,
        mu_C=cfg.data.mu_C,
        sigma_C=cfg.data.sigma_C,
        mu_R=cfg.data.mu_R,
        sigma_R=cfg.data.sigma_R,
        seed=seed,
    )


def run_base_experiment(cfg: Config) -> pd.DataFrame:
    rows = []
    seeds = cfg.get_seed_list()

    if cfg.experiment.verbose:
        print("\n[Base Experiment]")

    for rep_id, seed in enumerate(seeds, start=1):
        X = generate_instance(cfg, seed)
        results = run_methods_on_instance(X, cfg, seed)

        for res in results:
            rows.append(
                build_row(
                    result=res,
                    instance_name="base",
                    seed=seed,
                    cfg=cfg,
                    extra_tags={
                        "experiment_type": "base",
                        "replication_id": rep_id,
                    },
                )
            )

        if cfg.experiment.verbose:
            print(f"  finished replication {rep_id}/{len(seeds)} (seed={seed})")

    return pd.DataFrame(rows)


def run_size_experiment(cfg: Config) -> pd.DataFrame:
    rows = []
    seeds = cfg.get_seed_list()
    original_n = cfg.problem.n_cells

    if cfg.experiment.verbose:
        print("\n[Size Experiment]")

    for n_cells in cfg.experiment.instance_sizes:
        cfg.problem.n_cells = n_cells

        if cfg.experiment.verbose:
            print(f"  size = {n_cells}")

        for rep_id, seed in enumerate(seeds, start=1):
            X = generate_instance(cfg, seed)
            results = run_methods_on_instance(X, cfg, seed)

            for res in results:
                rows.append(
                    build_row(
                        result=res,
                        instance_name=f"size_{n_cells}",
                        seed=seed,
                        cfg=cfg,
                        extra_tags={
                            "experiment_type": "size",
                            "replication_id": rep_id,
                            "size_level": n_cells,
                        },
                    )
                )

    cfg.problem.n_cells = original_n
    return pd.DataFrame(rows)


def run_delta_sensitivity(cfg: Config) -> pd.DataFrame:
    rows = []
    seeds = cfg.get_seed_list()
    original_delta = cfg.problem.delta_bar

    if cfg.experiment.verbose:
        print("\n[Delta-bar Sensitivity]")

    for delta_bar in cfg.experiment.delta_bar_grid:
        cfg.problem.delta_bar = delta_bar

        if cfg.experiment.verbose:
            print(f"  delta_bar = {delta_bar}")

        for rep_id, seed in enumerate(seeds, start=1):
            X = generate_instance(cfg, seed)
            results = run_methods_on_instance(X, cfg, seed)

            for res in results:
                rows.append(
                    build_row(
                        result=res,
                        instance_name=f"delta_{delta_bar}",
                        seed=seed,
                        cfg=cfg,
                        extra_tags={
                            "experiment_type": "delta_sensitivity",
                            "replication_id": rep_id,
                            "delta_level": delta_bar,
                        },
                    )
                )

    cfg.problem.delta_bar = original_delta
    return pd.DataFrame(rows)


def run_lambda_sensitivity(cfg: Config) -> pd.DataFrame:
    rows = []
    seeds = cfg.get_seed_list()
    original_lambda = cfg.problem.lambda_penalty

    if cfg.experiment.verbose:
        print("\n[Lambda Sensitivity]")

    for lam in cfg.experiment.lambda_grid:
        cfg.problem.lambda_penalty = lam

        if cfg.experiment.verbose:
            print(f"  lambda_penalty = {lam}")

        for rep_id, seed in enumerate(seeds, start=1):
            X = generate_instance(cfg, seed)
            results = run_methods_on_instance(X, cfg, seed)

            for res in results:
                rows.append(
                    build_row(
                        result=res,
                        instance_name=f"lambda_{lam}",
                        seed=seed,
                        cfg=cfg,
                        extra_tags={
                            "experiment_type": "lambda_sensitivity",
                            "replication_id": rep_id,
                            "lambda_level": lam,
                        },
                    )
                )

    cfg.problem.lambda_penalty = original_lambda
    return pd.DataFrame(rows)


def run_kmax_sensitivity(cfg: Config) -> pd.DataFrame:
    rows = []
    seeds = cfg.get_seed_list()
    original_kmax = cfg.problem.k_max

    if cfg.experiment.verbose:
        print("\n[k_max Sensitivity]")

    for kmax in cfg.experiment.kmax_grid:
        cfg.problem.k_max = kmax

        if cfg.experiment.verbose:
            print(f"  k_max = {kmax}")

        for rep_id, seed in enumerate(seeds, start=1):
            X = generate_instance(cfg, seed)
            results = run_methods_on_instance(X, cfg, seed)

            for res in results:
                rows.append(
                    build_row(
                        result=res,
                        instance_name=f"kmax_{kmax}",
                        seed=seed,
                        cfg=cfg,
                        extra_tags={
                            "experiment_type": "kmax_sensitivity",
                            "replication_id": rep_id,
                            "kmax_level": kmax,
                        },
                    )
                )

    cfg.problem.k_max = original_kmax
    return pd.DataFrame(rows)


def main():
    cfg = Config()

    t0 = time.time()
    ensure_dir(cfg.experiment.results_dir)

    all_raw_parts = [run_base_experiment(cfg)]

    if cfg.experiment.run_size_experiment:
        all_raw_parts.append(run_size_experiment(cfg))

    if cfg.experiment.run_delta_sensitivity:
        all_raw_parts.append(run_delta_sensitivity(cfg))

    if cfg.experiment.run_lambda_sensitivity:
        all_raw_parts.append(run_lambda_sensitivity(cfg))

    if cfg.experiment.run_kmax_sensitivity:
        all_raw_parts.append(run_kmax_sensitivity(cfg))

    raw_df = pd.concat(all_raw_parts, ignore_index=True)

    summary_group_cols = [
        "experiment_type",
        "instance_name",
        "method",
        "n_cells",
        "K",
        "k_max",
        "delta_bar",
        "lambda_penalty",
    ]
    summary_df = aggregate_results(raw_df, summary_group_cols)

    if cfg.experiment.save_results:
        raw_path = os.path.join(cfg.experiment.results_dir, cfg.experiment.raw_csv)
        summary_path = os.path.join(cfg.experiment.results_dir, cfg.experiment.summary_csv)

        raw_df.to_csv(raw_path, index=False)
        summary_df.to_csv(summary_path, index=False)

        if cfg.experiment.verbose:
            print(f"\nRaw results saved to: {raw_path}")
            print(f"Summary results saved to: {summary_path}")

    elapsed = time.time() - t0

    if cfg.experiment.verbose:
        print(f"\nTotal experiment time: {elapsed:.2f}s")
        print("\n=== Summary Preview ===")
        with pd.option_context("display.max_columns", None, "display.width", 260):
            print(summary_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()