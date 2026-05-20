# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project solves the **Capacity-Constrained Battery Repurposing Problem (RRP)** with a piece-wise value function. It studies how to re-group retired EV battery cells (characterized by capacity C and internal resistance R) into new battery packs to maximize total value.

The project uses a **two-stage optimization framework**:
- **Inner problem (RRP)**: Given a batch of cells, how to group/pack them to maximize reward
- **Outer problem**: In a dynamic arrival environment, decide when and how to scrap low-quality inventory

## Quick Start

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run two-stage simulation (compares KMEANS, VNS, GRASP, GA)
python main_two_stage.py

# Run batch experiment across seeds/scrap-periods/threshold grids
python run_two_stage_experiment.py

# Run multi-round re-packing experiment (outputs Excel)
python multi_inner_opt.py

# Run upper bound analysis with Column Generation
python upper_bound.py
```

## Code Architecture

```
two-stage/
├── config.py                  # All dataclass configs (Experiment, Problem, Data, algorithm-specific)
├── data_generator.py          # generate_cells(): produces (C, R) arrays from normal distributions
├── utils.py                   # compute_centroid, compute_delta, compute_phi, piecewise_value, compute_group_reward, summarize_solution
├── main_two_stage.py          # Two-stage system: compare inner methods + TSRH outer + clairvoyant UB
├── run_two_stage_experiment.py # Batch experiment runner across seeds/H_scrap/threshold grids
├── multi_inner_opt.py         # Multi-round re-packing experiment with Excel export
├── upper_bound.py             # Upper bound analysis using Column Generation inner solver
│
├── heuristics/                # Inner RRP solvers (all return dict with "groups", "reward", "leftover", "runtime")
│   ├── rrp_kmeans.py          # K-means clustering (fast approximation)
│   ├── rrp_kmeans_vns.py      # K-means + Variable Neighborhood Search (default solver)
│   ├── rrp_ms_kmeans_vns.py   # Multi-start K-means + VNS
│   ├── rrp_grasp.py           # GRASP (Greedy Randomized Adaptive Search)
│   ├── rrp_ga.py              # Genetic Algorithm
│   ├── rrp_sa.py              # Simulated Annealing with Tabu-VND
│   ├── rrp_column_generation.py # Column Generation (CG) with PuLP/Gurobi pricing
│   ├── rrp_gurobi_exact.py    # Gurobi exact solver (MIP + enumeration)
│   ├── rrp_combine_repair.py  # Combine + repair utility
│   ├── residual_packing.py    # Residual packing helper
│   └── _grasp_stats.py        # GRASP statistics helper
│
├── outer/                     # Outer optimization (dynamic scrapping decisions)
│   ├── tsrah.py               # TSRH: Two-Stage Robust Heuristic with Monte Carlo rollout
│   └── arrival.py             # gaussian_arrival_generator(): generates new cell batches
│
├── results/                   # Result storage directory
├── Old_code/                  # Deprecated code archive
└── results_two_stage/         # Batch experiment output (CSV)
```

## Inner Solver Interface

All inner solvers follow the same interface: they accept `X` (n x 2 array of standardized cells) and return a `dict`:

```python
result = {
    "groups": [[cell_indices], ...],  # list of lists, each list has K cell indices
    "leftover": [unassigned_cell_indices],
    "reward": float,                   # total reward = sum(V(phi) - lambda*delta)
    "runtime": float,
    ...
}
```

The unified wrapper `solve_inner_rrp()` in `main_two_stage.py` and `run_two_stage_experiment.py` dispatches to the appropriate solver based on `method` string: `"KMEANS"`, `"VNS"`, `"GRASP"`, `"GA"`.

## Key Data Flow

1. **Cell generation**: `generate_cells()` returns raw (C, R) arrays from N(mu, sigma)
2. **Standardization**: `standardize_cells()` in `multi_inner_opt.py` converts to z-scores: `(C - mu_C)/sigma_C`, `(R - mu_R)/sigma_R`
3. **Inner solve**: Standardized cells are passed to a heuristic solver
4. **Reward computation**: `compute_group_reward()` = `piecewise_value(phi)` - `lambda * delta`
5. **Leftover carry-forward**: Unassigned cells accumulate into the next round's pool

## Two-Stage Simulation (`main_two_stage.py`)

The `simulate_two_stage_system()` function runs the core loop:
- Each period t: merge leftover inventory + new arrivals -> solve inner RRP -> leftover becomes I_t^+
- Every H_scrap periods: run TSRH to decide which cells to scrap
- TSRH uses Monte Carlo rollout to evaluate threshold candidates across multiple filtering layers

## Configuration

All parameters are in `config.py` as dataclasses. The top-level `Config` class composes all sub-configs. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `problem.n_cells` | 500 | Initial cell count |
| `problem.K` | 8 | Cells per pack |
| `problem.k_max` | 30 | Maximum number of packs |
| `problem.delta_bar` | 99 | Variance constraint (heuristic penalty) |
| `problem.theta1/2/3` | 0.5/0.0/-0.5 | Tier thresholds |
| `problem.P1/P2/P3` | 10.0/6.0/3.0 | Tier values |
| `data.mu_C/sigma_C` | 200/20 | Capacity distribution |
| `data.mu_R/sigma_R` | 50/5 | Resistance distribution |

## Gurobi

Gurobi is available for exact solving (`rrp_gurobi_exact.py`) and CG pricing (`rrp_column_generation.py`). License file path is configured in `config.py`. The `GUROBI_AVAILABLE` flag is checked before running Gurobi-dependent methods. C(n,K) enumeration is auto-skipped when >50M combinations.
