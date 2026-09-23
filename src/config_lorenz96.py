"""Run settings for the Lorenz-96 forcing ensemble.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, replace
from typing import Sequence

from .config import REPO_ROOT

PARAM_NAMES = ["F"]          # the forcing is the only parameter
N_LEADING = 2                # idx and t, ahead of the parameters
N_PARAMS = len(PARAM_NAMES)  # width of the parameter block


@dataclass
class LorenzConfig:
    """Everything the Lorenz-96 build and evaluation need."""

    # ---- data / paths ----
    out_dir: str = os.path.join(REPO_ROOT, "experiments/lorenz96/N100_F0-2.5")
    gen_dir: str = os.path.join(REPO_ROOT, "datasets/lorenz96/N100_F0-2.5")  # generator output
    build_features: bool = False      # rebuild the shifted source and matrices

    # ---- clustering feature space ----
    qoi: Sequence[int] = (0, 25, 50, 75)   # state components scored on
    log_cols: Sequence[str] = ()           
    drop_cols: Sequence[str] = ()

    # ---- training / testing partition ----
    n_train: int = 150                # of 200 forcings, drawn at random
    n_test: int = 50                  # the rest, held out
    rng_seed: int = 42                # the training/testing partition and the k-means cluster splits
    base_seed: int = 0                # the base k-means

    # ---- local bases + adaptive refinement ----
    eta: float = 1e-4                 # SVD energy a basis may discard; must be > 0
    tau_split: float = 0.02           # state units, since error_norm is absolute
    n_clusters: int = 16              # clusters the base k-means starts from
    max_clusters: int = 2048          # cap on the refined partition
    min_cluster_size: int = 50        # smallest child a split may produce
    increase_rank: bool = True        # raise rank when a split is not possible
    rank_proxy: bool = True           # seed the rank walk from the projection proxy; forced off when error_norm is absolute

    # ---- system conventions ----
    ode_method: str = "RK45"          # not stiff 
    scale_method: str = "none"        # no scaling
    qoi_log: bool = False             # the shifted state is signed
    error_norm: str = "absolute"      # a relative denominator collapses near zero

    # ---- inference ----
    positivity: bool = False          # not needed for a signed state
    atol: float = 1e-8                # solve_ivp
    rtol: float = 1e-8                # solve_ivp
    n_eval: int = 50                  # number of test forcings to evaluate
    n_workers: int = 24               # cpu cores

    def __post_init__(self):
        if not 0.0 < self.eta < 1.0:
            raise ValueError(f"eta must be in (0, 1); got {self.eta!r}")
        if self.tau_split <= 0:
            raise ValueError(f"tau_split must be > 0; got {self.tau_split!r}")
        if self.max_clusters < self.n_clusters:
            raise ValueError("max_clusters < n_clusters")
        if self.positivity:
            raise ValueError("positivity is undefined for a signed state")
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

    def replace(self, **over) -> "LorenzConfig":
        return replace(self, **over)
