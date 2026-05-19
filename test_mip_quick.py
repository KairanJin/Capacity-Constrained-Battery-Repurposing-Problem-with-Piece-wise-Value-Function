"""
快速测试 RRP_GUROBI_MIP 求解能力，每个 N 用 300s 时限。
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
time_limit = 300.0  # 5 minutes per instance

results = []

for n in [20, 30, 40, 50, 60, 80, 100]:
    X = X_all[:n]
    k_t = min(cfg.problem.k_max, n // K)
    print(f"\n{'='*60}")
    print(f"Testing N={n}, K={K}, k_t={k_t} (time_limit={time_limit:.0f}s)")
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
            time_limit=time_limit, seed=42,
        )

        gap = res.get("gurobi_gap")
        status = res.get("gurobi_status")
        obj = res.get("gurobi_obj_val")
        bound = res.get("gurobi_best_bound")
        n_vars = res.get("n_vars")
        n_constrs = res.get("n_constrs")

        gap_str = f"{gap:.2f}%" if gap is not None else "N/A"
        status_map = {2: "OPTIMAL", 9: "TIME_LIMIT", 3: "INFEASIBLE"}
        status_str = status_map.get(status, str(status))

        print(f"\n  Result: packs={res['n_packs']}, reward={res['reward']:.2f}, "
              f"time={res['runtime']:.1f}s, "
              f"vars={n_vars}, constrs={n_constrs}, "
              f"status={status_str}, obj={obj}, bound={bound}, gap={gap_str}", flush=True)

        results.append({
            'n': n, 'k_t': k_t, 'packs': res['n_packs'],
            'reward': res['reward'], 'time': res['runtime'],
            'vars': n_vars, 'constrs': n_constrs,
            'status': status_str, 'gap': gap_str,
        })

        # 如果无可行解，停止
        if res['n_packs'] == 0:
            print(f"  *** NO FEASIBLE SOLUTION at N={n} ***", flush=True)
            break

    except Exception as e:
        print(f"  *** ERROR at N={n}: {e} ***", flush=True)
        results.append({'n': n, 'error': str(e)})
        break

print(f"\n{'='*60}")
print("SUMMARY:")
print(f"{'='*60}")
for r in results:
    if 'error' in r:
        print(f"  N={r['n']}: ERROR - {r['error']}")
    else:
        print(f"  N={r['n']:3d} | k_t={r['k_t']:2d} | packs={r['packs']} | "
              f"reward={r['reward']:8.2f} | time={r['time']:8.1f}s | "
              f"vars={r['vars']:5d} | status={r['status']:12s} | gap={r['gap']}")

print(f"\n{'='*60}")
print("Test complete.", flush=True)
