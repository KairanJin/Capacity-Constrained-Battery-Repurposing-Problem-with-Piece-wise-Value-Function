"""
测试 RRP_GUROBI_MIP 在 3600s 时限下能求解的最大单轮 N 值。
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["GRB_LICENSE_FILE"] = r"C:\gurobi1001\win64\bin\gurobi.lic"

import numpy as np
from heuristics.rrp_gurobi_exact import solve_rrp_gurobi_mip
from data_generator import generate_cells
from config import Config

cfg = Config()

np.random.seed(cfg.experiment.base_seed)
cells_raw = generate_cells(
    n_cells=500,
    mu_C=cfg.data.mu_C, sigma_C=cfg.data.sigma_C,
    mu_R=cfg.data.mu_R, sigma_R=cfg.data.sigma_R,
)
X_all = np.column_stack([
    (cells_raw[:, 0] - cfg.data.mu_C) / cfg.data.sigma_C,
    (cells_raw[:, 1] - cfg.data.mu_R) / cfg.data.sigma_R,
])

K = cfg.problem.K

# 从 N=40 开始逐步增加
for n in [40, 50, 60, 70, 80, 100, 120, 150, 200]:
    X = X_all[:n]
    k_t = min(cfg.problem.k_max, n // K)
    print(f"\n{'='*60}")
    print(f"Testing N={n}, K={K}, k_t={k_t} (time_limit=3600s)")
    print(f"{'='*60}", flush=True)

    try:
        res = solve_rrp_gurobi_mip(
            X=X, K=K, k_t=k_t,
            delta_bar=cfg.gurobi.delta_bar,
            w=cfg.problem.w,
            lambda_penalty=cfg.problem.lambda_penalty,
            theta1=cfg.problem.theta1,
            theta2=cfg.problem.theta2,
            theta3=cfg.problem.theta3,
            P1=cfg.problem.P1, P2=cfg.problem.P2, P3=cfg.problem.P3,
            time_limit=3600.0, seed=42,
        )

        gap = res.get("gurobi_gap", "N/A")
        status = res.get("gurobi_status", "N/A")
        obj = res.get("gurobi_obj_val", "N/A")
        bound = res.get("gurobi_best_bound", "N/A")
        n_vars = res.get("n_vars", "N/A")
        n_constrs = res.get("n_constrs", "N/A")

        print(f"\n  packs={res['n_packs']}, reward={res['reward']:.2f}, "
              f"time={res['runtime']:.1f}s, "
              f"vars={n_vars}, constrs={n_constrs}, "
              f"status={status}, obj={obj}, bound={bound}, gap={gap}", flush=True)

        if res['n_packs'] == 0:
            print(f"  *** NO FEASIBLE SOLUTION at N={n}, stopping ***", flush=True)
            break

    except Exception as e:
        print(f"  *** ERROR at N={n}: {e} ***", flush=True)
        print(f"  *** STOPPING TEST ***", flush=True)
        break

print(f"\n{'='*60}")
print("Test complete.", flush=True)
