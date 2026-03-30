# test_outer.py
from functools import partial
import numpy as np

from config import Config
from data_generator import generate_cells
from heuristics.rrp_kmeans_vns import solve_rrp_kmeans_vns
from outer.tsrah import tsrah_scrapping_decision, default_quality_score
from outer.arrival import gaussian_arrival_generator


def main():
    cfg = Config()

    # 假设当前期重组后库存 I_t^+
    I_t_plus = generate_cells(
        n_cells=120,
        mu_C=cfg.data.mu_C,
        sigma_C=cfg.data.sigma_C,
        mu_R=cfg.data.mu_R,
        sigma_R=cfg.data.sigma_R,
        seed=cfg.experiment.base_seed,
    )

    # 用 VNS 作为 rollout 里的内层求解器
    def inner_solver_fn(X, **kwargs):
        k_t = min(cfg.problem.k_max, X.shape[0] // cfg.problem.K)
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
            seed=None,
            pack_candidate_limit=cfg.vns.pack_candidate_limit,
            partner_limit=cfg.vns.partner_limit,
            cell_candidate_limit=cfg.vns.cell_candidate_limit,
            leftover_candidate_limit=cfg.vns.leftover_candidate_limit,
            destroy_size=cfg.vns.destroy_size,
        )

    # 到达生成器
    def arrival_fn(rng, **kwargs):
        return gaussian_arrival_generator(
            rng=rng,
            n_arrivals=30,
            mu_C=cfg.data.mu_C,
            sigma_C=cfg.data.sigma_C,
            mu_R=cfg.data.mu_R,
            sigma_R=cfg.data.sigma_R,
        )

    # 质量评分：示例 q(z)=wq^T z
    # 你可以自己改成更符合报废逻辑的分数
    wq = np.array([1.0, -1.0])

    res = tsrah_scrapping_decision(
        t=10,                     # 当前期
        I_t_plus=I_t_plus,
        E=[100,110,120, 130, 140, 150, 160],
        H=5,
        m_list=[2, 4, 8],
        rho=0.5,
        gamma=0.95,
        s0=5.0,
        quality_score_fn=default_quality_score,
        inner_solver_fn=inner_solver_fn,
        arrival_generator_fn=arrival_fn,
        quality_score_kwargs={"wq": wq},
        inner_solver_kwargs={},
        arrival_generator_kwargs={},
        seed=42,
        verbose=True,
    )

    print("\nDetailed stats:")
    for eta, st in sorted(res.stats.items()):
        print(
            f"eta={eta}, "
            f"N={st['N_eta']}, "
            f"mean={st['Jhat_eta']:.4f}, "
            f"std={st['sample_std']:.4f}, "
            f"var={st['sample_var']:.4f}"
        )

    print("\nLayer logs:")
    for layer in res.layer_logs:
        print(f"Layer {layer['layer']}: entering={layer['entering_candidates']}, next={layer['next_candidates']}")
        for item in sorted(layer["summary"], key=lambda x: x["Jhat_eta"], reverse=True):
            print(
                f"  eta={item['eta']}, "
                f"N={item['N_eta']}, "
                f"mean={item['Jhat_eta']:.4f}, "
                f"std={item['sample_std']:.4f}, "
                f"survives={item['survives']}"
            )


if __name__ == "__main__":
    main()