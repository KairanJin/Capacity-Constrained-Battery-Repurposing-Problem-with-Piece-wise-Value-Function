# utils.py
import numpy as np


def compute_centroid(X: np.ndarray, group: list[int]) -> np.ndarray:
    return X[group].mean(axis=0)


def compute_delta(X: np.ndarray, group: list[int]) -> float:
    mu = compute_centroid(X, group)
    return float(np.mean(np.sum((X[group] - mu) ** 2, axis=1)))


def compute_phi(X: np.ndarray, group: list[int], w: np.ndarray) -> float:
    mu = compute_centroid(X, group)
    return float(np.dot(w, mu))


def piecewise_value(
    phi: float,
    theta1: float,
    theta2: float,
    theta3: float,
    P1: float,
    P2: float,
    P3: float,
) -> float:
    if phi >= theta1:
        return P1
    elif phi >= theta2:
        return P2
    elif phi >= theta3:
        return P3
    else:
        return 0.0


def get_tier(
    phi: float,
    theta1: float,
    theta2: float,
    theta3: float,
) -> str:
    if phi >= theta1:
        return "P1"
    elif phi >= theta2:
        return "P2"
    elif phi >= theta3:
        return "P3"
    else:
        return "P0"


def compute_group_reward(
    X: np.ndarray,
    group: list[int],
    w: np.ndarray,
    lambda_penalty: float,
    theta1: float,
    theta2: float,
    theta3: float,
    P1: float,
    P2: float,
    P3: float,
) -> tuple[float, float, float]:
    phi = compute_phi(X, group, w)
    delta = compute_delta(X, group)
    value = piecewise_value(phi, theta1, theta2, theta3, P1, P2, P3)
    reward = value - lambda_penalty * delta
    return reward, phi, delta


def total_sse(X: np.ndarray, groups: list[list[int]]) -> float:
    sse = 0.0
    for g in groups:
        if len(g) == 0:
            continue
        mu = compute_centroid(X, g)
        sse += float(np.sum(np.sum((X[g] - mu) ** 2, axis=1)))
    return sse


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def summarize_solution(
    X: np.ndarray,
    groups: list[list[int]],
    K: int,
    w: np.ndarray,
    lambda_penalty: float,
    theta1: float,
    theta2: float,
    theta3: float,
    P1: float,
    P2: float,
    P3: float,
):
    """
    对一个解做统一统计。
    groups 默认应为最终可行 pack 集合。
    """
    pack_rewards = []
    phis = []
    deltas = []
    tier_counts = {"P1": 0, "P2": 0, "P3": 0, "P0": 0}

    for g in groups:
        if len(g) != K:
            continue
        reward, phi, delta = compute_group_reward(
            X, g, w, lambda_penalty, theta1, theta2, theta3, P1, P2, P3
        )
        pack_rewards.append(reward)
        phis.append(phi)
        deltas.append(delta)
        tier_counts[get_tier(phi, theta1, theta2, theta3)] += 1

    n_packs = len(pack_rewards)
    total_reward = float(np.sum(pack_rewards)) if pack_rewards else 0.0
    avg_phi = float(np.mean(phis)) if phis else 0.0
    avg_delta = float(np.mean(deltas)) if deltas else 0.0
    reward_per_pack = safe_div(total_reward, n_packs)
    positive_pack_ratio = safe_div(sum(r > 0 for r in pack_rewards), n_packs)

    return {
        "total_reward": total_reward,
        "n_packs": n_packs,
        "avg_phi": avg_phi,
        "avg_delta": avg_delta,
        "reward_per_pack": reward_per_pack,
        "positive_pack_ratio": positive_pack_ratio,
        "tier_counts": tier_counts,
        "pack_rewards": pack_rewards,
        "phis": phis,
        "deltas": deltas,
    }