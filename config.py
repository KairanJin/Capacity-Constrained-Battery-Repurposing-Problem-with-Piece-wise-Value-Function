from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class ExperimentConfig:
    run_kmeans: bool = True
    run_kmeans_vns: bool = True
    run_grasp: bool = False
    run_ga: bool = True
    run_column_generation: bool = False
    skip_cg_for_large_instances: bool = True
    cg_size_threshold: int = 1000
    base_seed: int = 43
    verbose: bool = True
    run_lns: bool = True # Added for LNS
    run_gurobi_exact: bool = False

@dataclass
class ProblemConfig:
    n_cells: int = 500
    K: int = 8
    k_max: int = 150
    delta_bar: float = 0.8
    w: Tuple[float, float] = (0.5, 0.5)
    lambda_penalty: float = 0.05
    theta1: float = 0.5  # Adjusted to align with data distribution
    theta2: float = 0.0
    theta3: float = -0.5
    P1: float = 10.0
    P2: float = 6.0
    P3: float = 3.0

@dataclass
class DataConfig:
    mu_C: float = 200.0
    sigma_C: float = 20.0
    mu_R: float = 50.0
    sigma_R: float = 5.0

@dataclass
class KMeansConfig:
    L1: int = 10
    L2: int = 4
    tol: float = 1e-4

@dataclass
class VNSConfig:
    L1: int = 15
    tol: float = 1e-4
    max_vns_iter: int = 80
    max_no_improve: int = 15
    pack_candidate_limit: int = 8
    partner_limit: int = 6
    cell_candidate_limit: int = 3
    leftover_candidate_limit: int = 12
    destroy_size: int = 3
    n_starts: int = 3

@dataclass
class GRASPConfig:
    n_starts: int = 30
    rcl_size: int = 4
    max_group_attempts: int = 200
    max_local_iter: int = 30
    group_candidate_limit: int = 6
    cell_candidate_limit: int = 2
    leftover_candidate_limit: int = 10

@dataclass
class GAConfig:
    population_size: int = 50
    n_generations: int = 100
    tournament_size: int = 5
    crossover_prob: float = 0.8
    mutation_prob: float = 0.1
    destroy_size: int = 2
    local_search_prob: float = 0.2
    elitism_size: int = 2
    group_candidate_limit: int = 6
    cell_candidate_limit: int = 2
    leftover_candidate_limit: int = 10

@dataclass
class CGConfig:
    max_cg_iter: int = 30
    init_n_starts: int = 30
    init_neighbor_size: int = 8
    pricing_n_seeds: int = 40
    pricing_neighbor_size: int = 10
    max_new_cols: int = 20

@dataclass
class SAConfig:
    initial_temperature: float | None = None
    cooling_rate: float = 0.995
    min_temperature: float = 1e-4
    max_sa_iterations: int = 2000
    vnd_interval: int = 150
    max_vnd_rounds: int = 3
    reheating_ratio: float = 3.0
    reheating_stall: int = 300
    max_reheats: int = 3
    tabu_tenure: int = 15
    n_init_starts: int = 3
    kmeans_L1: int = 15
    kmeans_tol: float = 1e-4
    residual_rounds: int = 20

import os

# Gurobi license file path for the full (unrestricted) license
# Located at Gurobi 10.0.1 installation, used with gurobipy 10.x in conda env `gurobi10`
GUROBI_LICENSE_FILE = r"C:\gurobi1001\win64\bin\gurobi.lic"


def setup_gurobi_license():
    """Set the Gurobi license file environment variable before importing gurobipy."""
    if os.path.isfile(GUROBI_LICENSE_FILE):
        os.environ["GRB_LICENSE_FILE"] = GUROBI_LICENSE_FILE


@dataclass
class GurobiConfig:
    time_limit: float = 3600.0
    license_file: str = GUROBI_LICENSE_FILE
    # delta_bar for Gurobi feasibility constraint (differs from problem.delta_bar
    # which is used as penalty in heuristic objectives)
    delta_bar: float = 0.8
    # Smart sampling parameters
    max_candidates: int = 200000
    n_sampling_rounds: int = 500

@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    problem: ProblemConfig = field(default_factory=ProblemConfig)
    data: DataConfig = field(default_factory=DataConfig)
    kmeans: KMeansConfig = field(default_factory=KMeansConfig)
    vns: VNSConfig = field(default_factory=VNSConfig)
    grasp: GRASPConfig = field(default_factory=GRASPConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    cg: CGConfig = field(default_factory=CGConfig)
    sa: SAConfig = field(default_factory=SAConfig)
    gurobi: GurobiConfig = field(default_factory=GurobiConfig)
