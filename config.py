# config.py
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class ProblemConfig:
    """
    Problem-related parameters.
    """
    n_cells: int = 200
    dim: int = 2
    K: int = 4
    k_max: int = 40

    delta_bar: float = 25.0
    lambda_penalty: float = 0.5

    # phi(G) = w^T mu_G
    w: Optional[np.ndarray] = None

    # tier thresholds
    theta1: float = 180.0
    theta2: float = 165.0
    theta3: float = 150.0

    # tier prices
    P1: float = 100.0
    P2: float = 70.0
    P3: float = 40.0

    def __post_init__(self):
        if self.w is None:
            self.w = np.array([1.0, -0.5], dtype=float)

    @property
    def tier_prices(self):
        return {
            "P1": self.P1,
            "P2": self.P2,
            "P3": self.P3,
            "P0": 0.0,
        }

    @property
    def tier_thresholds(self):
        return {
            "theta1": self.theta1,
            "theta2": self.theta2,
            "theta3": self.theta3,
        }


@dataclass
class DataConfig:
    """
    Synthetic data generation parameters.
    """
    mu_C: float = 180.0
    sigma_C: float = 15.0
    mu_R: float = 45.0
    sigma_R: float = 5.0

    trunc_lb_C: float = 1e-6
    trunc_lb_R: float = 1e-6

    # reserved for future extension
    use_correlated_sampling: bool = False
    rho_CR: float = -0.3

    use_mixture: bool = False
    mixture_weights: List[float] = field(default_factory=lambda: [0.5, 0.5])

    mu_C_1: float = 190.0
    sigma_C_1: float = 10.0
    mu_R_1: float = 40.0
    sigma_R_1: float = 4.0

    mu_C_2: float = 165.0
    sigma_C_2: float = 12.0
    mu_R_2: float = 50.0
    sigma_R_2: float = 5.0


@dataclass
class KMeansConfig:
    """
    Parameters for the basic improved K-means heuristic.
    """
    L1: int = 20
    L2: int = 20
    tol: float = 1e-6

    use_multiple_starts: bool = False
    n_starts: int = 5


@dataclass
class VNSConfig:
    """
    Parameters for the lightweight K-means + VNS heuristic.
    """
    L1: int = 20
    tol: float = 1e-6

    max_vns_iter: int = 12
    max_no_improve: int = 4

    pack_candidate_limit: int = 6
    partner_limit: int = 3
    cell_candidate_limit: int = 2
    leftover_candidate_limit: int = 10

    destroy_size: int = 2

    use_multiple_starts: bool = False
    n_starts: int = 3


@dataclass
class GRASPConfig:
    """
    Parameters for the GRASP heuristic.
    """
    n_starts: int = 15
    rcl_size: int = 4
    max_group_attempts: int = 150
    max_local_iter: int = 20

    group_candidate_limit: int = 5
    cell_candidate_limit: int = 2
    leftover_candidate_limit: int = 8


@dataclass
class ColumnGenerationConfig:
    """
    Parameters for the column generation benchmark.
    """
    max_cg_iter: int = 30

    init_n_starts: int = 30
    init_neighbor_size: int = 8

    pricing_n_seeds: int = 40
    pricing_neighbor_size: int = 10
    max_new_cols: int = 20

    max_total_columns_soft: int = 5000
    solver_msg: bool = False

@dataclass
class GAConfig:
    population_size: int = 16
    n_generations: int = 20
    tournament_size: int = 3
    crossover_prob: float = 0.9
    mutation_prob: float = 0.3
    destroy_size: int = 2
    local_search_prob: float = 0.5
    elitism_size: int = 2
    group_candidate_limit: int = 5
    cell_candidate_limit: int = 2
    leftover_candidate_limit: int = 8

@dataclass
class ExperimentConfig:
    """
    Experiment-level settings for batch runs.
    """
    base_seed: int = 52
    n_replications: int = 10

    run_kmeans: bool = True
    run_kmeans_vns: bool = True
    run_grasp: bool = False
    run_column_generation: bool = False
    run_ga: bool = True

    save_results: bool = True
    results_dir: str = "results"
    summary_csv: str = "summary_results.csv"
    raw_csv: str = "raw_results.csv"

    verbose: bool = True

    skip_cg_for_large_instances: bool = True
    cg_size_threshold: int = 400

    instance_sizes: List[int] = field(default_factory=lambda: [120, 180, 240, 300])

    delta_bar_grid: List[float] = field(default_factory=lambda: [15.0, 20.0, 25.0, 30.0])
    lambda_grid: List[float] = field(default_factory=lambda: [0.2, 0.5, 1.0])
    kmax_grid: List[int] = field(default_factory=lambda: [20, 30, 40, 50])

    run_size_experiment: bool = False
    run_delta_sensitivity: bool = False
    run_lambda_sensitivity: bool = False
    run_kmax_sensitivity: bool = False


@dataclass
class Config:
    """
    Master config.
    """
    problem: ProblemConfig = field(default_factory=ProblemConfig)
    data: DataConfig = field(default_factory=DataConfig)
    kmeans: KMeansConfig = field(default_factory=KMeansConfig)
    vns: VNSConfig = field(default_factory=VNSConfig)
    grasp: GRASPConfig = field(default_factory=GRASPConfig)
    cg: ColumnGenerationConfig = field(default_factory=ColumnGenerationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    ga: GAConfig = field(default_factory=GAConfig)

    def get_seed_list(self) -> List[int]:
        return [
            self.experiment.base_seed + i
            for i in range(self.experiment.n_replications)
        ]