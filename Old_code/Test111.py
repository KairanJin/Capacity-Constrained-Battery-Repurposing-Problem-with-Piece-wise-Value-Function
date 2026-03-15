from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Config
# =========================

@dataclass
class Config:
    # Simulation
    seed: int = 42
    n_periods: int = 20
    T_scrap: int = 5           # scrap every T_scrap periods
    H_rh: int = 5              # rolling horizon length (no scrapping during rollout)
    M: int = 5                 # Monte Carlo replications per eta (FAST)

    # Arrivals
    N_arrivals: int = 220
    mu_C: float = 200.0
    sigma_C: float = 20.0
    mu_R: float = 50.0
    sigma_R: float = 5.0
    trunc_eps: float = 1e-6

    # Inner reassembly
    K: int = 8
    k_max: int = 30
    L1_init: int = 10          # init iterations
    L2_swap: int = 4           # swap passes (outer iterations)
    barDelta: float = 1
    lam: float = 0.2

    # Repair step (your new Step 3)
    repair_max_rounds: int = 20  # safety cap to avoid infinite loops

    # Reward / pricing
    gamma: float = 0.99
    w: Tuple[float, float] = (0.5, 0.5)
    theta1: float = 1.0
    theta2: float = 0.3
    theta3: float = -0.3
    P1: float = 10.0
    P2: float = 6.0
    P3: float = 3.0

    # Scrapping
    etas: Tuple[float, ...] = tuple(np.round(np.arange(0, 1.0, 0.2), 1).tolist())
    s0: float = 4.0

    # Output
    out_dir: str = "outputs_fast"
    save_png: bool = True
    save_csv: bool = True


# =========================
# Data structures
# =========================

@dataclass
class Cell:
    cid: int
    C: float
    R: float
    Ct: float
    Rt: float

    def z_tilde(self) -> np.ndarray:
        return np.array([self.Ct, self.Rt], dtype=float)


# =========================
# Utilities
# =========================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def truncated_normal(rng: np.random.Generator, mu: float, sigma: float, size: int, eps: float) -> np.ndarray:
    """Generate N(mu, sigma^2) truncated to [eps, +inf) by clipping."""
    x = rng.normal(loc=mu, scale=sigma, size=size)
    return np.maximum(x, eps)


def standardize_C_R(C: np.ndarray, R: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    # global standardization
    Ct = (C - cfg.mu_C) / cfg.sigma_C
    Rt = (R - cfg.mu_R) / cfg.sigma_R
    return Ct, Rt


def quality_score(cell: Cell) -> float:
    # q(z) = C̃ - R̃
    return cell.Ct - cell.Rt


def pack_center(pack_cells: List[Cell]) -> np.ndarray:
    return np.mean(np.stack([c.z_tilde() for c in pack_cells], axis=0), axis=0)


def delta_pack(pack_cells: List[Cell]) -> float:
    # Δ(G) = (1/|G|) * sum ||z - mu||^2  in standardized space
    if len(pack_cells) == 0:
        return float("inf")
    mu = pack_center(pack_cells)
    X = np.stack([c.z_tilde() for c in pack_cells], axis=0)
    sse = float(np.sum(np.sum((X - mu) ** 2, axis=1)))
    return sse / len(pack_cells)


def perf_score(pack_cells: List[Cell], cfg: Config) -> float:
    zbar = np.mean(np.stack([c.z_tilde() for c in pack_cells], axis=0), axis=0)
    w = np.array(cfg.w, dtype=float)
    return float(w @ zbar)


def tier_price(perf: float, cfg: Config) -> float:
    if perf >= cfg.theta1:
        return cfg.P1
    if cfg.theta2 <= perf < cfg.theta1:
        return cfg.P2
    if cfg.theta3 <= perf < cfg.theta2:
        return cfg.P3
    return 0.0


def total_sse(packs: List[List[Cell]]) -> float:
    sse = 0.0
    for G in packs:
        if len(G) == 0:
            continue
        mu = pack_center(G)
        X = np.stack([c.z_tilde() for c in G], axis=0)
        sse += float(np.sum(np.sum((X - mu) ** 2, axis=1)))
    return sse


# =========================
# Arrivals
# =========================

def generate_arrivals_batch(rng: np.random.Generator, cfg: Config, start_cid: int) -> Tuple[List[Cell], int]:
    C = truncated_normal(rng, cfg.mu_C, cfg.sigma_C, cfg.N_arrivals, cfg.trunc_eps)
    R = truncated_normal(rng, cfg.mu_R, cfg.sigma_R, cfg.N_arrivals, cfg.trunc_eps)
    Ct, Rt = standardize_C_R(C, R, cfg)

    cells = []
    cid = start_cid
    for i in range(cfg.N_arrivals):
        cells.append(Cell(cid=cid, C=float(C[i]), R=float(R[i]), Ct=float(Ct[i]), Rt=float(Rt[i])))
        cid += 1
    return cells, cid


# =========================
# Inner heuristic: Step 2 (init) + Step 3 (repair underfilled) + Step 4 (swap) + Step 5 (filter)
# =========================

def capacity_constrained_init(U: List[Cell], k_t: int, cfg: Config, rng: np.random.Generator) -> Tuple[List[List[Cell]], List[Cell]]:
    """
    Step 2: capacity-feasible initialization
    - choose k_t distinct centers
    - greedy assign with capacity K
    - leftover if all full
    - repeat update/assign for L1_init iterations
    """
    if k_t <= 0 or len(U) < cfg.K:
        return [], U.copy()

    n = len(U)
    center_indices = rng.choice(n, size=k_t, replace=False)
    centers = np.stack([U[i].z_tilde() for i in center_indices], axis=0)

    def assign(centers_now: np.ndarray) -> Tuple[List[List[Cell]], List[Cell]]:
        packs = [[] for _ in range(k_t)]
        caps = [cfg.K] * k_t
        leftover: List[Cell] = []

        for cell in U:
            z = cell.z_tilde()
            d2 = np.sum((centers_now - z) ** 2, axis=1)
            order = np.argsort(d2)
            placed = False
            for j in order:
                if caps[j] > 0:
                    packs[j].append(cell)
                    caps[j] -= 1
                    placed = True
                    break
            if not placed:
                leftover.append(cell)
        return packs, leftover

    packs, leftover = assign(centers)

    for _ in range(cfg.L1_init):
        new_centers = centers.copy()
        for j in range(k_t):
            if len(packs[j]) > 0:
                new_centers[j] = pack_center(packs[j])

        if np.allclose(new_centers, centers, atol=1e-6, rtol=0.0):
            centers = new_centers
            break

        centers = new_centers
        packs, leftover = assign(centers)

    return packs, leftover


def repair_underfilled_packs(
    packs: List[List[Cell]],
    leftover: List[Cell],
    cfg: Config,
    rng: np.random.Generator
) -> Tuple[List[List[Cell]], List[Cell]]:
    """
    Step 3:
    - collect all cells from underfilled packs (|G| < K) and leftover
    - rerun Step 2 on that pool to form more full packs
    - repeat until remaining pool size < K or max rounds hit
    """
    full_packs: List[List[Cell]] = []
    pool: List[Cell] = leftover.copy()

    for G in packs:
        if len(G) == cfg.K:
            full_packs.append(G)
        else:
            pool.extend(G)

    rounds = 0
    while len(pool) >= cfg.K and rounds < cfg.repair_max_rounds:
        rounds += 1
        k_t = min(cfg.k_max, len(pool) // cfg.K)
        if k_t <= 0:
            break

        new_packs, new_leftover = capacity_constrained_init(pool, k_t, cfg, rng)

        pool = new_leftover
        for G in new_packs:
            if len(G) == cfg.K:
                full_packs.append(G)
            else:
                pool.extend(G)

    return full_packs, pool


def swap_local_improvement_fullpacks_fast(packs: List[List[Cell]], cfg: Config) -> List[List[Cell]]:
    if len(packs) <= 1:
        return packs

    P = len(packs)
    K = cfg.K

    # precompute z for each cell (avoid repeated allocations)
    zpacks = [[c.z_tilde() for c in G] for G in packs]

    # per-pack stats
    sum_z = np.zeros((P, 2), dtype=float)
    sum_norm2 = np.zeros(P, dtype=float)
    for p in range(P):
        Z = np.stack(zpacks[p], axis=0)  # Kx2
        sum_z[p] = Z.sum(axis=0)
        sum_norm2[p] = np.sum(Z[:, 0] ** 2 + Z[:, 1] ** 2)

    def pack_sse(p: int) -> float:
        # SSE = sum||z||^2 - n||mean||^2, n=K fixed
        mz = sum_z[p] / K
        return float(sum_norm2[p] - K * (mz[0] ** 2 + mz[1] ** 2))

    base_sse_total = float(np.sum([pack_sse(p) for p in range(P)]))

    for _pass in range(cfg.L2_swap):
        improved = False

        # centers from stats (no recompute)
        centers = sum_z / K  # Px2

        for a in range(P):
            for i in range(K):
                za = zpacks[a][i]

                # find candidate packs in increasing distance to current centers
                d2 = np.sum((centers - za) ** 2, axis=1)
                order = np.argsort(d2)

                for b in order:
                    if b == a:
                        continue

                    for j in range(K):
                        zb = zpacks[b][j]

                        # current SSE of affected packs
                        sse_a_old = pack_sse(a)
                        sse_b_old = pack_sse(b)

                        # stats update if swap i<->j
                        # remove za add zb in pack a; remove zb add za in pack b
                        sum_z_a_new = sum_z[a] - za + zb
                        sum_z_b_new = sum_z[b] - zb + za

                        norm_za = float(za[0]**2 + za[1]**2)
                        norm_zb = float(zb[0]**2 + zb[1]**2)
                        sum_norm2_a_new = sum_norm2[a] - norm_za + norm_zb
                        sum_norm2_b_new = sum_norm2[b] - norm_zb + norm_za

                        # new SSE
                        mz_a = sum_z_a_new / K
                        mz_b = sum_z_b_new / K
                        sse_a_new = float(sum_norm2_a_new - K * (mz_a[0]**2 + mz_a[1]**2))
                        sse_b_new = float(sum_norm2_b_new - K * (mz_b[0]**2 + mz_b[1]**2))

                        trial_total = base_sse_total - sse_a_old - sse_b_old + sse_a_new + sse_b_new

                        if trial_total + 1e-12 < base_sse_total:
                            # accept swap in packs + zpacks
                            packs[a][i], packs[b][j] = packs[b][j], packs[a][i]
                            zpacks[a][i], zpacks[b][j] = zpacks[b][j], zpacks[a][i]

                            # commit stats
                            sum_z[a] = sum_z_a_new
                            sum_z[b] = sum_z_b_new
                            sum_norm2[a] = sum_norm2_a_new
                            sum_norm2[b] = sum_norm2_b_new
                            centers[a] = sum_z[a] / K
                            centers[b] = sum_z[b] / K

                            base_sse_total = trial_total
                            improved = True
                            break

                    if improved:
                        break
                # 继续搜索其它 swap（也可以加“first-improvement”策略：一旦找到 improvement 就 break）
        if not improved:
            break

    return packs


def feasibility_filter(packs: List[List[Cell]], leftover: List[Cell], cfg: Config) -> Tuple[List[List[Cell]], List[Cell]]:
    """
    Step 5: Feasibility filtering
    dissolve if |G|<K or Δ(G)>barDelta; dissolved cells go to leftover
    """
    feasible: List[List[Cell]] = []
    W = leftover.copy()
    for G in packs:
        if len(G) != cfg.K:
            W.extend(G)
            continue
        if delta_pack(G) > cfg.barDelta:
            W.extend(G)
            continue
        feasible.append(G)
    return feasible, W


def inner_reassembly(U: List[Cell], cfg: Config, rng: np.random.Generator) -> Tuple[List[List[Cell]], List[Cell]]:
    k_t = min(cfg.k_max, len(U) // cfg.K)

    packs_init, leftover = capacity_constrained_init(U, k_t, cfg, rng)
    packs_full, pool_leftover = repair_underfilled_packs(packs_init, leftover, cfg, rng)
    packs_improved = swap_local_improvement_fullpacks_fast(packs_full, cfg)
    packs_feasible, W = feasibility_filter(packs_improved, pool_leftover, cfg)
    return packs_feasible, W


def reward_reassembly(packs: List[List[Cell]], cfg: Config) -> Tuple[float, float, float]:
    if len(packs) == 0:
        return 0.0, float("nan"), float("nan")

    deltas = []
    perfs = []
    total = 0.0
    for G in packs:
        perf = perf_score(G, cfg)
        price = tier_price(perf, cfg)
        d = delta_pack(G)
        total += (price - cfg.lam * d)
        deltas.append(d)
        perfs.append(perf)
    return float(total), float(np.mean(deltas)), float(np.mean(perfs))


# =========================
# Visualization / Tables
# =========================

def plot_reassembly(U: List[Cell], packs: List[List[Cell]], W: List[Cell], t: int, cfg: Config) -> None:
    if not cfg.save_png:
        return
    fig = plt.figure()
    ax = plt.gca()
    ax.set_title(f"Period {t}: Reassembly (packs colored, leftover gray)")
    ax.set_xlabel("Capacitance C")
    ax.set_ylabel("Resistance R")

    if len(W) > 0:
        ax.scatter([c.C for c in W], [c.R for c in W], alpha=0.35, label="Leftover")

    for j, G in enumerate(packs):
        ax.scatter([c.C for c in G], [c.R for c in G], alpha=0.85, label=f"Pack {j}")

    # Avoid legend warning if nothing has labels
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) > 0:
        ax.legend(loc="best", fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(cfg.out_dir, f"reassembly_t{t:02d}.png"), dpi=150)
    plt.close(fig)


def plot_scrap(I_old: List[Cell], D: List[Cell], t: int, cfg: Config) -> None:
    """
    Plot scrapping on beginning-of-period inventory (old cells only).
    """
    if not cfg.save_png:
        return
    D_ids = set(c.cid for c in D)
    kept = [c for c in I_old if c.cid not in D_ids]

    fig = plt.figure()
    ax = plt.gca()
    ax.set_title(f"Period {t}: Scrapping (old inventory only)")
    ax.set_xlabel("Capacitance C")
    ax.set_ylabel("Resistance R")

    if len(kept) > 0:
        ax.scatter([c.C for c in kept], [c.R for c in kept], alpha=0.35, label="Kept")
    if len(D) > 0:
        ax.scatter([c.C for c in D], [c.R for c in D], alpha=0.85, label="Scrapped")

    handles, labels = ax.get_legend_handles_labels()
    if len(labels) > 0:
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(cfg.out_dir, f"scrap_t{t:02d}.png"), dpi=150)
    plt.close(fig)


def save_reassembly_table(U: List[Cell], packs: List[List[Cell]], W: List[Cell], t: int, cfg: Config) -> None:
    if not cfg.save_csv:
        return
    rows = []
    for j, G in enumerate(packs):
        for c in G:
            rows.append({
                "t": t, "cid": c.cid, "C": c.C, "R": c.R, "Ct": c.Ct, "Rt": c.Rt,
                "group_id": j, "status": "packed"
            })
    for c in W:
        rows.append({
            "t": t, "cid": c.cid, "C": c.C, "R": c.R, "Ct": c.Ct, "Rt": c.Rt,
            "group_id": -1, "status": "leftover"
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.out_dir, f"reassembly_table_t{t:02d}.csv"), index=False)


# =========================
# Rolling-horizon scrapping (CRN)
# =========================

def select_scrap_set(I: List[Cell], eta: float) -> List[Cell]:
    return [c for c in I if quality_score(c) < eta]

def _eval_one_eta_worker(args) -> Tuple[float, float]:
    """
    Top-level worker for multiprocessing (must be picklable).
    Returns (eta, J).
    """
    I_old, eta, cfg, master_seed = args
    rng_master = np.random.default_rng(int(master_seed))  # CRN across eta
    J = rh_evaluate_eta(I_old, float(eta), cfg, rng_master)
    return float(eta), float(J)

def rh_evaluate_eta(I_old: List[Cell], eta: float, cfg: Config, rng_master: np.random.Generator) -> float:
    """
    Evaluate eta on OLD inventory only (no arrivals included at decision moment).
    Then rollout H_rh periods, no scrapping during rollout.
    """
    D = select_scrap_set(I_old, eta)
    D_ids = set(c.cid for c in D)
    I0 = [c for c in I_old if c.cid not in D_ids]
    immediate = cfg.s0 * len(D)

    seeds = rng_master.integers(low=0, high=2**31 - 1, size=cfg.M, dtype=np.int64)

    vals = []
    for m in range(cfg.M):
        rng_m = np.random.default_rng(int(seeds[m]))
        I_sim = I0.copy()
        cid_next = 10_000_000 + m * 1_000_000

        total = immediate
        for h in range(1, cfg.H_rh + 1):
            A_sim, cid_next = generate_arrivals_batch(rng_m, cfg, cid_next)
            U_sim = I_sim + A_sim
            packs, W = inner_reassembly(U_sim, cfg, rng_m)
            R_grp, _, _ = reward_reassembly(packs, cfg)
            total += (cfg.gamma ** h) * R_grp
            I_sim = W  # assembled removed
        vals.append(total)

    return float(np.mean(vals))


def rh_choose_eta_parallel(
    I_old: List[Cell],
    cfg: Config,
    rng: np.random.Generator,
    max_workers: int | None = None
) -> Tuple[float, Dict[float, float]]:
    """
    Parallel evaluation over etas using ProcessPoolExecutor.
    Keeps CRN by using the same master_seed for all etas.
    """
    master_seed = int(rng.integers(low=0, high=2**31 - 1, size=1)[0])

    # prepare tasks
    tasks = [(I_old, eta, cfg, master_seed) for eta in cfg.etas]

    # run in parallel
    eta_to_J: Dict[float, float] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for eta, J in ex.map(_eval_one_eta_worker, tasks, chunksize=1):
            eta_to_J[eta] = J

    eta_star = max(eta_to_J.keys(), key=lambda e: eta_to_J[e])
    return float(eta_star), eta_to_J


def do_scrap_on_inventory(
    inventory_old: List[Cell],
    t: int,
    cfg: Config,
    rng: np.random.Generator
) -> Tuple[List[Cell], float | None, int, float]:
    """
    In scrap periods, scrap happens BEFORE arrivals & reassembly.
    Only scrap from beginning-of-period inventory (old cells).
    """
    if len(inventory_old) == 0:
        print(f"[t={t:02d}] Scrap skipped (no initial inventory)")
        return inventory_old, None, 0, 0.0

    eta_star, _ = rh_choose_eta_parallel(inventory_old, cfg, rng)

    D = select_scrap_set(inventory_old, eta_star)
    n_scrap = len(D)
    R_scr = cfg.s0 * n_scrap

    plot_scrap(inventory_old, D, t, cfg)

    D_ids = set(c.cid for c in D)
    inventory_after = [c for c in inventory_old if c.cid not in D_ids]
    return inventory_after, eta_star, n_scrap, R_scr


# =========================
# Main simulation
# =========================

def run_sim(cfg: Config) -> pd.DataFrame:
    ensure_dir(cfg.out_dir)
    rng = np.random.default_rng(cfg.seed)

    inventory: List[Cell] = []  # this is I_t at beginning of each period
    next_cid = 1
    logs = []

    for t in range(1, cfg.n_periods + 1):

        # --------- NEW TIMING ---------
        # (1) If scrap period: scrap on beginning-of-period inventory only
        inv_before = len(inventory)
        R_scr = 0.0
        eta_star = None
        n_scrap = 0

        if (t % cfg.T_scrap) == 0:
            inventory, eta_star, n_scrap, R_scr = do_scrap_on_inventory(inventory, t, cfg, rng)

        inv_after_scrap = len(inventory)

        # (2) Observe arrivals after scrapping
        A_t, next_cid = generate_arrivals_batch(rng, cfg, next_cid)

        # (3) Reassembly on (post-scrap inventory + new arrivals)
        U_t = inventory + A_t
        packs, W = inner_reassembly(U_t, cfg, rng)
        R_grp, avg_delta, avg_perf = reward_reassembly(packs, cfg)

        # Save per-period outputs
        plot_reassembly(U_t, packs, W, t, cfg)
        save_reassembly_table(U_t, packs, W, t, cfg)

        # (4) End-of-period inventory becomes leftover after reassembly
        inventory = W
        R_total = R_grp + R_scr

        logs.append({
            "t": t,
            "arrivals": len(A_t),
            "U_size": len(U_t),
            "packs_formed": len(packs),
            "cells_packed": len(packs) * cfg.K,
            "leftover": len(W),
            "avg_delta": avg_delta,
            "avg_perf": avg_perf,
            "R_grp": R_grp,
            "R_scr": R_scr,
            "R_total": R_total,
            "inventory_before": inv_before,
            "inventory_after_scrap": inv_after_scrap,
            "inventory_next": len(inventory),
            "scrap_period": int((t % cfg.T_scrap) == 0),
            "eta_star": eta_star if eta_star is not None else np.nan,
            "n_scrap": n_scrap,
        })

        print(f"[t={t:02d}] packs={len(packs):3d}, leftover={len(W):4d}, "
              f"R_grp={R_grp:8.3f}, R_scr={R_scr:6.3f}, inv_next={len(inventory):4d}, eta*={eta_star}")

    df = pd.DataFrame(logs)
    df.to_csv(os.path.join(cfg.out_dir, "summary.csv"), index=False)

    # inventory trajectory
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(df["t"], df["inventory_next"], marker="o")
    ax.set_title("Inventory size trajectory")
    ax.set_xlabel("Period t")
    ax.set_ylabel("Inventory size |I_{t+1}|")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.out_dir, "inventory_trajectory.png"), dpi=150)
    plt.close(fig)

    return df


if __name__ == "__main__":
    cfg = Config(
        seed=42,
        M=20,
        k_max=30,
        K=8,
        mu_C=200, sigma_C=20,
        mu_R=50, sigma_R=5,
        barDelta=0.3,
        lam=0.2,
        out_dir="outputs_fast",
    )
    df = run_sim(cfg)
    print("\nSaved outputs to:", cfg.out_dir)
    print(df.head())