# main.py
from __future__ import annotations

import pandas as pd

from heuristics.residual_packing import residual_pack_repair
from config import Config
from data_generator import generate_cells
from heuristics.rrp_kmeans import solve_rrp_kmeans
from heuristics.rrp_kmeans_vns import solve_rrp_kmeans_vns
from heuristics.rrp_grasp import solve_rrp_grasp
from heuristics.rrp_ga import solve_rrp_ga
from heuristics.rrp_column_generation import solve_rrp_column_generation
from heuristics.solve_rrp_lns import solve_rrp_lns
from heuristics.rrp_gurobi_exact import solve_rrp_gurobi, GUROBI_AVAILABLE
from utils import summarize_solution, safe_div


def format_tier_distribution(tier_counts: dict) -> str:
    return (
        f"P1:{tier_counts.get('P1', 0)}, "
        f"P2:{tier_counts.get('P2', 0)}, "
        f"P3:{tier_counts.get('P3', 0)}, "
        f"P0:{tier_counts.get('P0', 0)}"
    )


def enrich_result(result, X, cfg):
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


def main():
    cfg = Config()
    seed = cfg.experiment.base_seed

    X = generate_cells(
        n_cells=cfg.problem.n_cells,
        mu_C=cfg.data.mu_C,
        sigma_C=cfg.data.sigma_C,
        mu_R=cfg.data.mu_R,
        sigma_R=cfg.data.sigma_R,
        seed=seed,
    )

    k_t = min(cfg.problem.k_max, X.shape[0] // cfg.problem.K)

    results = []

    if cfg.experiment.run_kmeans:
        res = solve_rrp_kmeans(
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
        results.append(enrich_result(res, X, cfg))

    if cfg.experiment.run_kmeans_vns:
        res = solve_rrp_kmeans_vns(
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
        results.append(enrich_result(res, X, cfg))

    if cfg.experiment.run_grasp:
        res = solve_rrp_grasp(
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
        results.append(enrich_result(res, X, cfg))

    if cfg.experiment.run_ga:
        res = solve_rrp_ga(
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
        results.append(enrich_result(res, X, cfg))

    # Large Neighborhood Search (LNS)
    if getattr(cfg.experiment, 'run_lns', True):
        res = solve_rrp_lns(
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
        results.append(enrich_result(res, X, cfg))

    run_cg = cfg.experiment.run_column_generation
    if cfg.experiment.skip_cg_for_large_instances and cfg.problem.n_cells > cfg.experiment.cg_size_threshold:
        run_cg = False

    if cfg.experiment.run_gurobi_exact:
        if not GUROBI_AVAILABLE:
            print("WARNING: gurobipy not installed, skipping RRP_GUROBI")
        else:
            res = solve_rrp_gurobi(
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
                method=cfg.gurobi.method,
                time_limit=cfg.gurobi.time_limit,
                mip_gap=cfg.gurobi.mip_gap,
                threads=cfg.gurobi.threads,
                max_enumeration=cfg.gurobi.max_enumeration,
            )
            results.append(enrich_result(res, X, cfg))

    if run_cg:
        res = solve_rrp_column_generation(
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
        results.append(enrich_result(res, X, cfg))

    df = pd.DataFrame([
        {
            "method": r["method"],
            "reward": r["reward"],
            "n_packs": r["n_packs"],
            "avg_delta": r["avg_delta"],
            "avg_phi": r["avg_phi"],
            "reward_per_pack": r["reward_per_pack"],
            "utilization_rate": r["utilization_rate"],
            "positive_pack_ratio": r["positive_pack_ratio"],
            "runtime": r["runtime"],
            "leftover": len(r["leftover"]),
            "tier_distribution": r["tier_distribution"],
            "n_columns": r.get("n_columns", float("nan")),
            "gurobi_gap": r.get("gurobi_gap", float("nan")),
            "n_feasible_groups": r.get("n_feasible_groups", float("nan")),
        }
        for r in results
    ])

    print("\n=== Single Instance Results ===")
    with pd.option_context("display.max_columns", None, "display.width", 260):
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()
