# outer/tsrah.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, Sequence, List
import math
import numpy as np


@dataclass
class TSRHResult:
    eta_star: float
    D_t: list[int]
    I_t_pp: np.ndarray
    R_t_scr: float
    stats: Dict[float, Dict[str, float]]
    final_candidates: list[float]
    layer_logs: list[dict]


# =========================================================
# Quality score / threshold-induced scrapping
# =========================================================

def default_quality_score(X: np.ndarray, wq: np.ndarray) -> np.ndarray:
    """
    q(z) = wq^T z
    X: shape (n, d)
    return: shape (n,)
    """
    return X @ wq


def get_scrap_indices_by_threshold(
    I_t_plus: np.ndarray,
    eta: float,
    quality_score_fn: Callable[..., np.ndarray],
    quality_score_kwargs: Dict[str, Any] | None = None,
) -> list[int]:
    quality_score_kwargs = quality_score_kwargs or {}
    q_vals = quality_score_fn(I_t_plus, **quality_score_kwargs)
    return np.where(q_vals < eta)[0].tolist()


def apply_scrapping(
    I_t_plus: np.ndarray,
    scrap_indices: list[int],
) -> np.ndarray:
    if len(scrap_indices) == 0:
        return I_t_plus.copy()

    mask = np.ones(len(I_t_plus), dtype=bool)
    mask[np.array(scrap_indices, dtype=int)] = False
    return I_t_plus[mask].copy()


# =========================================================
# Rollout
# =========================================================

def one_rollout_for_eta(
    I_t_plus: np.ndarray,
    eta: float,
    H: int,
    gamma: float,
    s0: float,
    quality_score_fn: Callable[..., np.ndarray],
    inner_solver_fn: Callable[..., Dict[str, Any]],
    arrival_generator_fn: Callable[..., np.ndarray],
    quality_score_kwargs: Dict[str, Any] | None = None,
    inner_solver_kwargs: Dict[str, Any] | None = None,
    arrival_generator_kwargs: Dict[str, Any] | None = None,
    rng_seed: int | None = None,
) -> float:
    quality_score_kwargs = quality_score_kwargs or {}
    inner_solver_kwargs = inner_solver_kwargs or {}
    arrival_generator_kwargs = arrival_generator_kwargs or {}

    rng = np.random.default_rng(rng_seed)

    scrap_indices = get_scrap_indices_by_threshold(
        I_t_plus=I_t_plus,
        eta=eta,
        quality_score_fn=quality_score_fn,
        quality_score_kwargs=quality_score_kwargs,
    )
    I_curr = apply_scrapping(I_t_plus, scrap_indices)
    immediate_scrap_reward = s0 * len(scrap_indices)

    total_future_reward = 0.0

    for tau in range(1, H + 1):
        A_tau = arrival_generator_fn(rng=rng, **arrival_generator_kwargs)

        if len(I_curr) == 0:
            U_tau = A_tau.copy()
        elif len(A_tau) == 0:
            U_tau = I_curr.copy()
        else:
            U_tau = np.vstack([I_curr, A_tau])

        inner_res = inner_solver_fn(X=U_tau, **inner_solver_kwargs)
        R_grp = float(inner_res.get("reward", 0.0))
        total_future_reward += (gamma ** tau) * R_grp

        used = set()
        for g in inner_res.get("groups", []):
            used.update(g)

        leftover_idx = [i for i in range(len(U_tau)) if i not in used]
        I_curr = U_tau[leftover_idx].copy()

    return immediate_scrap_reward + total_future_reward


# =========================================================
# Helpers
# =========================================================

def top_k_candidates(
    candidates: Sequence[float],
    mean_dict: Dict[float, float],
    k: int,
) -> list[float]:
    ranked = sorted(candidates, key=lambda eta: mean_dict[eta], reverse=True)
    return ranked[:max(1, min(k, len(ranked)))]


def compute_sample_variance(sum_x: float, sum_x2: float, n: int) -> float:
    """
    Unbiased sample variance.
    """
    if n <= 1:
        return 0.0
    mean = sum_x / n
    var = (sum_x2 - n * mean * mean) / (n - 1)
    return max(0.0, float(var))


# =========================================================
# TSRH
# =========================================================

def tsrah_scrapping_decision(
    t: int,
    I_t_plus: np.ndarray,
    E: Sequence[float],
    H: int,
    m_list: Sequence[int],
    rho: float,
    gamma: float,
    s0: float,
    quality_score_fn: Callable[..., np.ndarray],
    inner_solver_fn: Callable[..., Dict[str, Any]],
    arrival_generator_fn: Callable[..., np.ndarray],
    quality_score_kwargs: Dict[str, Any] | None = None,
    inner_solver_kwargs: Dict[str, Any] | None = None,
    arrival_generator_kwargs: Dict[str, Any] | None = None,
    seed: int | None = None,
    verbose: bool = False,
) -> TSRHResult:
    quality_score_kwargs = quality_score_kwargs or {}
    inner_solver_kwargs = inner_solver_kwargs or {}
    arrival_generator_kwargs = arrival_generator_kwargs or {}

    if t % H != 0:
        return TSRHResult(
            eta_star=float("nan"),
            D_t=[],
            I_t_pp=I_t_plus.copy(),
            R_t_scr=0.0,
            stats={},
            final_candidates=[],
            layer_logs=[],
        )

    rng = np.random.default_rng(seed)

    current_candidates = [float(x) for x in E]

    S_eta = {float(eta): 0.0 for eta in E}
    SS_eta = {float(eta): 0.0 for eta in E}   # sum of squares
    N_eta = {float(eta): 0 for eta in E}
    Jhat_eta = {float(eta): 0.0 for eta in E}
    Var_eta = {float(eta): 0.0 for eta in E}
    Std_eta = {float(eta): 0.0 for eta in E}

    layer_logs: list[dict] = []
    L = len(m_list)

    for ell in range(L):
        if len(current_candidates) <= 1:
            break

        m_ell = int(m_list[ell])
        layer_candidate_snapshot = list(current_candidates)
        layer_rollout_values: Dict[float, list[float]] = {}

        if verbose:
            print(f"\n[TSRH] Layer {ell}")
            print(f"  entering candidates: {layer_candidate_snapshot}")
            print(f"  additional rollouts per candidate: {m_ell}")

        for eta in current_candidates:
            eta = float(eta)

            rollout_vals = []
            for _ in range(m_ell):
                rollout_seed = int(rng.integers(1, 10**9))
                val = one_rollout_for_eta(
                    I_t_plus=I_t_plus,
                    eta=eta,
                    H=H,
                    gamma=gamma,
                    s0=s0,
                    quality_score_fn=quality_score_fn,
                    inner_solver_fn=inner_solver_fn,
                    arrival_generator_fn=arrival_generator_fn,
                    quality_score_kwargs=quality_score_kwargs,
                    inner_solver_kwargs=inner_solver_kwargs,
                    arrival_generator_kwargs=arrival_generator_kwargs,
                    rng_seed=rollout_seed,
                )
                rollout_vals.append(val)

            layer_rollout_values[eta] = rollout_vals

            S_eta[eta] += float(np.sum(rollout_vals))
            SS_eta[eta] += float(np.sum(np.square(rollout_vals)))
            N_eta[eta] += m_ell
            Jhat_eta[eta] = S_eta[eta] / N_eta[eta]
            Var_eta[eta] = compute_sample_variance(S_eta[eta], SS_eta[eta], N_eta[eta])
            Std_eta[eta] = math.sqrt(Var_eta[eta])

        keep_k = math.ceil(rho * len(current_candidates))
        next_candidates = top_k_candidates(current_candidates, Jhat_eta, keep_k)

        layer_summary = []
        for eta in layer_candidate_snapshot:
            layer_summary.append({
                "eta": float(eta),
                "new_rollouts": layer_rollout_values.get(float(eta), []),
                "S_eta": S_eta[float(eta)],
                "SS_eta": SS_eta[float(eta)],
                "N_eta": N_eta[float(eta)],
                "Jhat_eta": Jhat_eta[float(eta)],
                "sample_var": Var_eta[float(eta)],
                "sample_std": Std_eta[float(eta)],
                "survives": float(eta) in next_candidates,
            })

        layer_logs.append({
            "layer": ell,
            "m_ell": m_ell,
            "entering_candidates": layer_candidate_snapshot,
            "keep_k": keep_k,
            "summary": layer_summary,
            "next_candidates": list(next_candidates),
        })

        if verbose:
            print("  layer summary:")
            for item in sorted(layer_summary, key=lambda x: x["Jhat_eta"], reverse=True):
                print(
                    f"    eta={item['eta']:.4f}, "
                    f"N={item['N_eta']}, "
                    f"mean={item['Jhat_eta']:.4f}, "
                    f"std={item['sample_std']:.4f}, "
                    f"survives={item['survives']}"
                )
            print(f"  next candidates: {next_candidates}")

        current_candidates = next_candidates

    if len(current_candidates) == 0:
        eta_star = float(E[0])
    else:
        eta_star = max(current_candidates, key=lambda eta: Jhat_eta[float(eta)])

    D_t = get_scrap_indices_by_threshold(
        I_t_plus=I_t_plus,
        eta=eta_star,
        quality_score_fn=quality_score_fn,
        quality_score_kwargs=quality_score_kwargs,
    )
    I_t_pp = apply_scrapping(I_t_plus, D_t)
    R_t_scr = s0 * len(D_t)

    stats = {
        float(eta): {
            "S_eta": S_eta[float(eta)],
            "SS_eta": SS_eta[float(eta)],
            "N_eta": N_eta[float(eta)],
            "Jhat_eta": Jhat_eta[float(eta)],
            "sample_var": Var_eta[float(eta)],
            "sample_std": Std_eta[float(eta)],
        }
        for eta in E
    }

    if verbose:
        print("\n[TSRH] Final selection")
        print(f"  eta_star = {eta_star}")
        print(f"  scrap count = {len(D_t)}")
        print(f"  R_t_scr = {R_t_scr}")

    return TSRHResult(
        eta_star=float(eta_star),
        D_t=D_t,
        I_t_pp=I_t_pp,
        R_t_scr=float(R_t_scr),
        stats=stats,
        final_candidates=[float(x) for x in current_candidates],
        layer_logs=layer_logs,
    )