from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, replace
from typing import Sequence

#: Repo root, so the default network and dataset paths resolve from anywhere.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Dataset row layout: [idx | t | params | species].
PARAM_NAMES = ["nH", "T", "Tgrain", "Av", "uv_flux"]
N_LEADING = 2                # idx and t, ahead of the parameters
N_PARAMS = len(PARAM_NAMES)  # width of the parameter block
SUBSTEPS = 5                 # chemistry snapshots per hydro segment
NSEC = 3600 * 24 * 365.25    # seconds per Julian year

# Must match how the dataset's full-order solves were run.
SELF_SHIELDING = True
DUST_ATTENUATION = False     # uv_flux is already attenuated; would double-count

QOI_FLOOR = 1e-30            # clip before log10; a zero abundance breaks k-means

# Log10-transformed when forming the training/testing partition, independent of Config.log_cols.
SPLIT_LOG_COLS = ("nH", "Av", "uv_flux")


@dataclass
class Config:
    """Everything the build and evaluation need."""

    # ---- data / paths ----
    out_dir: str                      # holds the feature matrices and artifacts
    # The hydro parameters and the chemistry; read only by --build-features.
    chemical_hydro_file: str = os.path.join(
        REPO_ROOT, "datasets/grav_collapse/nelson/R0.1_M6_trace_cells.npy")
    # Hydro parameters at the knots
    params_file: str = os.path.join(
        REPO_ROOT, "datasets/grav_collapse/R0.1_M6_trace_cells_params.npz")
    network: str = os.path.join(REPO_ROOT, "networks/nelson/gas_reactions.in")
    abundances: str = os.path.join(REPO_ROOT, "networks/nelson/abundances.in")
    build_features: bool = False      # rebuild the feature matrices from the raw

    # ---- clustering feature space ----
    qoi: Sequence[str] = ("CO", "e-")             # error is measured on these
    log_cols: Sequence[str] = ("nH", "Av", "uv_flux")
    drop_cols: Sequence[str] = ("Tgrain",)        # excluded from the features

    # ---- training / testing partition ----
    n_train: int = 4096               # tracers fitted on
    n_test: int = 2048                # tracers held out
    rng_seed: int = 42                # the training/testing partition and the k-means cluster splits
    base_seed: int = 0                # the base k-means

    # ---- local bases + adaptive refinement ----
    eta: float = 1e-4                 # SVD energy a basis may discard; must be > 0
    tau_split: float = 0.10           # a cluster over this error is split in two
    n_clusters: int = 8               # clusters the base k-means starts from
    max_clusters: int = 1024          # cap on the refined partition
    min_cluster_size: int = 50        # smallest child a split may produce
    increase_rank: bool = True        # raise rank when a split is not possible
    rank_proxy: bool = True           # seed the rank walk from the projection proxy; forced off when error_norm is absolute

    # ---- system conventions ----
    # Chemistry abundances are positive, span decades and are stiff
    ode_method: str = "BDF"           # "BDF" stiff, "RK45" explicit
    scale_method: str = "pareto"      # "pareto" = 1/sqrt(std), or "none"
    qoi_log: bool = True              # log10 the QoI clustering features
    error_norm: str = "relative"      # "relative" to ||FOM||, or "absolute"

    # ---- inference ----
    positivity: bool = True           # project each step's state nonnegative
    dt_hydro_yr: float = 250.0        # Hydrodynamic time step
    atol: float = 1e-6                # solve_ivp
    rtol: float = 1e-6                # solve_ivp
    n_eval: int = 2048                # number of test parameter trajectories
    n_workers: int = 24               # cpu cores

    def __post_init__(self):
        if not 0.0 < self.eta < 1.0:
            raise ValueError(f"eta must be in (0, 1); got {self.eta!r}")
        if self.tau_split <= 0:
            raise ValueError(f"tau_split must be > 0; got {self.tau_split!r}")
        if self.max_clusters < self.n_clusters:
            raise ValueError("max_clusters < n_clusters")
        if self.scale_method not in ("pareto", "none"):
            raise ValueError(f"scale_method must be 'pareto' or 'none'; "
                             f"got {self.scale_method!r}")
        if self.error_norm not in ("relative", "absolute"):
            raise ValueError(f"error_norm must be 'relative' or 'absolute'; "
                             f"got {self.error_norm!r}")
        if self.error_norm == "absolute" and self.rank_proxy:
            self.rank_proxy = False
            warnings.warn(
                "error_norm='absolute': rank proxy disabled, so the rank walk "
                "re-scores every rank from r0+1 and the build is slower.",
                RuntimeWarning, stacklevel=2)
        self.qoi = list(self.qoi)
        self.log_cols = list(self.log_cols)
        self.drop_cols = list(self.drop_cols)

    @property
    def var_threshold(self) -> float:
        """Retained cumulative variance: the complement of eta."""
        return 1.0 - self.eta

    @property
    def dt_hydro(self) -> float:
        return self.dt_hydro_yr * NSEC

    def replace(self, **over) -> "Config":
        return replace(self, **over)


# ---------------------------------------------------------------------------
# Artifact naming
# ---------------------------------------------------------------------------
# Kept here rather than in plots.py so the compute path never imports matplotlib.

def eta_tag(eta):
    """``eta1e-4`` for an exact power of ten, else ``eta0.0003``."""
    import math
    exp = math.log10(eta)
    if abs(exp - round(exp)) < 1e-12:
        return f"eta1e{int(round(exp))}"
    return f"eta{eta:g}"


def combo_stem(tau, eta):
    """Artifact stem for one (tau_split, eta) combination."""
    return f"tol{tau:g}_{eta_tag(eta)}"


def split_suffix(eval_split):
    """Output suffix: empty for test, so the train pass cannot overwrite it."""
    return "" if eval_split == "test" else f"_{eval_split}"
