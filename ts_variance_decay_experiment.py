"""Approximate Thompson sampling with a parameter-space kernel regularizer.

The policy is deliberately small and explicit: one freshly initialized MLP is
fit on D_t, then its finite-grid maximizer is queried.  Diagnostics never make
decisions.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    """All scientific and reproducibility choices exposed by the CLI."""

    output: str = "runs/smoke"
    log_level: str = "INFO"
    plot: bool = False
    plot_out: str | None = None
    plot_max_iterations: int | None = None
    plot_gp_only: bool = False
    plot_font_size: float = 10.0
    plot_figsize_width: float = 7.0
    plot_figsize_height: float = 4.0
    plot_no_titles: bool = False
    seed: int = 0
    num_runs: int = 1
    seeds: list[int] | None = None
    teacher_seed: int | None = None
    initial_seed: int | None = None
    noise_seed: int | None = None
    input_dim: int = 1
    grid_size: int = 41
    lower: float = -1.0
    upper: float = 1.0
    hidden: int = 16
    layers: int = 2
    initialization: str = "lecun_normal"
    initialization_scale: float = 1.0
    teacher_initialization_scale: float = 1.0
    teacher_noise: float = 0.04
    initial_size: int = 3
    rounds: int = 20
    steps: int = 250
    learning_rate: float = 0.03
    optimizer: str = "adam"
    schedule: str = "log"
    lambda_base: float = 0.5
    schedule_q: float = 1.5
    theta_lengthscale_c: float = 1.0
    regularizer_kernel: str = "rbf"
    regularizer_matern_nu: float = 2.5
    regularizer_rq_alpha: float = 1.0
    gp_lengthscale: float = 0.35
    gp_nu: float = 2.5
    gp_jitter: float = 1e-6
    ensemble_members: int = 5
    ensemble_frequency: int = 1
    snapshot_frequency: int = 5
    # Reserved for the optional Lemma 4.3 reference plot; no rate curve is
    # fitted by this compact implementation yet.
    d_eff: float | None = None
    rate_window: str = "2:20"
    device: str = "cpu"


class MLP(nn.Module):
    """Small smooth learner/teacher network used by both policy and diagnostics."""

    def __init__(self, input_dim: int, hidden: int, layers: int):
        super().__init__()
        mods: list[nn.Module] = []
        d = input_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        mods.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def make_grid(c: Config) -> np.ndarray:
    if c.input_dim == 1:
        return np.linspace(c.lower, c.upper, c.grid_size)[:, None]
    # A tensor grid makes the 2D extension useful without changing the policy.
    side = round(c.grid_size ** (1 / c.input_dim))
    axes = [np.linspace(c.lower, c.upper, side) for _ in range(c.input_dim)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, c.input_dim)


def seed_model(
    model: nn.Module,
    seed: int,
    initialization: str = "lecun_normal",
    scale: float = 1.0,
) -> None:
    """Deterministically initialize ``model`` using PyTorch's global RNG.

    The initialization functions consume PyTorch's default RNG. Resetting it
    immediately before initializing each fresh model makes a policy fit
    reproducible from its explicit seed; the training loop itself does not
    draw additional random numbers. For Gaussian initializers, ``scale``
    multiplies the sampled weight standard deviation; for Xavier uniform it
    is the initialization gain.
    """
    if scale <= 0:
        raise ValueError("initialization scale must be positive")
    torch.manual_seed(seed)
    for p in model.parameters():
        if p.ndim > 1:
            if initialization == "lecun_normal":
                nn.init.normal_(
                    p, mean=0.0, std=scale / np.sqrt(p.shape[1])
                )
            elif initialization == "kaiming_normal":
                nn.init.kaiming_normal_(p, mode="fan_in", nonlinearity="relu")
                with torch.no_grad():
                    p.mul_(scale)
            elif initialization == "xavier_uniform":
                nn.init.xavier_uniform_(p, gain=scale)
            else:
                raise ValueError(
                    "initialization must be lecun_normal, kaiming_normal, or "
                    "xavier_uniform"
                )
        else:
            nn.init.zeros_(p)


def make_parameter_kernel(
    kernel: str,
    lengthscale: float,
    matern_nu: float,
    rq_alpha: float,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Construct the fixed parameter-space regularizer term.

    For the stationary unit-diagonal kernels, this returns ``-k`` because the
    additive self-kernel constants have no effect on optimization. For the
    linear kernel, it returns the full squared-distance term used for L2
    regularization.
    """
    if kernel == "linear":
        def evaluate(theta: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
            return 0.5 * ((theta - anchor) ** 2).sum()

        return evaluate
    if kernel == "rbf":
        def evaluate(theta: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
            squared_distance = ((theta - anchor) ** 2).sum()
            return -torch.exp(-0.5 * squared_distance / lengthscale**2)

        return evaluate
    if kernel == "matern":
        distance_scale = np.sqrt(2.0 * matern_nu) / lengthscale
        if np.isclose(matern_nu, 0.5):
            def evaluate(theta: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
                distance = torch.sqrt(((theta - anchor) ** 2).sum().clamp_min(0.0))
                return -torch.exp(-distance_scale * distance)

            return evaluate
        if np.isclose(matern_nu, 1.5):
            def evaluate(theta: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
                distance = torch.sqrt(((theta - anchor) ** 2).sum().clamp_min(0.0))
                scaled_distance = distance_scale * distance
                return -(1.0 + scaled_distance) * torch.exp(-scaled_distance)

            return evaluate
        if np.isclose(matern_nu, 2.5):
            def evaluate(theta: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
                distance = torch.sqrt(((theta - anchor) ** 2).sum().clamp_min(0.0))
                scaled_distance = distance_scale * distance
                return (
                    -(
                        1.0 + scaled_distance + scaled_distance**2 / 3.0
                    )
                    * torch.exp(-scaled_distance)
                )

            return evaluate
        raise ValueError("regularizer_matern_nu must be 0.5, 1.5, or 2.5")
    if kernel == "rational_quadratic":
        if rq_alpha <= 0:
            raise ValueError("regularizer_rq_alpha must be positive")

        def evaluate(theta: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
            squared_distance = ((theta - anchor) ** 2).sum()
            return -(
                1.0 + squared_distance / (2.0 * rq_alpha * lengthscale**2)
            ) ** (-rq_alpha)

        return evaluate
    raise ValueError(
        "regularizer_kernel must be linear, rbf, matern, or rational_quadratic"
    )


def parameter_count(c: Config) -> int:
    """Return the number of scalar parameters in the configured MLP."""
    if c.layers == 0:
        return c.input_dim + 1
    return (
        c.input_dim * c.hidden
        + c.hidden
        + (c.layers - 1) * (c.hidden**2 + c.hidden)
        + c.hidden
        + 1
    )


def make_run_parameter_kernel(
    c: Config,
) -> tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], float, int]:
    """Construct the fixed parameter kernel once for a worker process."""
    count = parameter_count(c)
    lengthscale = c.theta_lengthscale_c * np.sqrt(count)
    kernel = make_parameter_kernel(
        c.regularizer_kernel,
        lengthscale,
        c.regularizer_matern_nu,
        c.regularizer_rq_alpha,
    )
    return kernel, lengthscale, count


def lambda_at(c: Config, t: int) -> float:
    """Return the regularization strength for a dataset of size ``t``."""
    if c.schedule == "constant":
        return c.lambda_base
    if c.schedule != "log":
        raise ValueError(f"unknown schedule {c.schedule}")
    return c.lambda_base * (1.0 + np.log1p(t)) ** c.schedule_q


def fit_network(
    c: Config,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    parameter_kernel: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> tuple[MLP, dict]:
    """Fit one fresh policy network to ``D_t`` and return diagnostics.

    The loss is a sum of half-squared residuals plus the parameter-space
    penalty. An empty dataset is supported so that a zero-observation initial
    design remains a valid configuration.
    """
    model = MLP(c.input_dim, c.hidden, c.layers).to(c.device)
    seed_model(model, seed, c.initialization, c.initialization_scale)
    anchor = parameters_to_vector(model.parameters()).detach().clone()
    m = anchor.numel()
    ell = c.theta_lengthscale_c * np.sqrt(m)
    lambda_t = lambda_at(c, len(x))
    if c.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=c.learning_rate)
    elif c.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=c.learning_rate)
    else:
        raise ValueError("optimizer must be adam or sgd")
    LOGGER.debug(
        "fit start: seed=%d observations=%d optimizer=%s steps=%d",
        seed,
        len(x),
        c.optimizer,
        c.steps,
    )
    xt = torch.as_tensor(x, dtype=torch.float32, device=c.device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=c.device)
    start = time.perf_counter()
    for _ in range(c.steps):
        opt.zero_grad(set_to_none=True)
        theta = parameters_to_vector(model.parameters())
        # Keep an autograd path through theta even when there are no data terms.
        pred = model(xt) if len(x) else theta[:1].sum() * xt.new_zeros((0,))
        data_loss = 0.5 * ((pred - yt) ** 2).sum()
        kernel_value = parameter_kernel(theta, anchor)
        # For stationary kernels, the factory returns the equivalent -k term
        # after dropping constants. For the linear kernel, it returns the full
        # half squared-distance L2 term.
        penalty = lambda_t * kernel_value
        loss = data_loss + penalty
        loss.backward()
        opt.step()
    diagnostics = {
        "train_seconds": time.perf_counter() - start,
        "parameter_count": m,
        "theta_lengthscale": ell,
    }
    LOGGER.debug(
        "fit complete: seed=%d time=%.3fs",
        seed,
        diagnostics["train_seconds"],
    )
    return model, diagnostics


def make_matern_kernel(
    lengthscale: float, nu: float
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Construct a fixed isotropic Matérn covariance evaluator."""
    distance_scale = np.sqrt(2.0 * nu) / lengthscale
    if np.isclose(nu, 0.5):
        def evaluate(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
            distance = torch.cdist(x1, x2)
            return torch.exp(-distance_scale * distance)

        return evaluate
    if np.isclose(nu, 1.5):
        def evaluate(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
            scaled_distance = distance_scale * torch.cdist(x1, x2)
            return (1.0 + scaled_distance) * torch.exp(-scaled_distance)

        return evaluate
    if np.isclose(nu, 2.5):
        def evaluate(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
            scaled_distance = distance_scale * torch.cdist(x1, x2)
            return (
                1.0 + scaled_distance + scaled_distance**2 / 3.0
            ) * torch.exp(-scaled_distance)

        return evaluate
    raise ValueError("gp_nu must be 0.5, 1.5, or 2.5")


def matern_variance(
    c: Config,
    x_obs: np.ndarray,
    x_star: np.ndarray,
    matern_kernel: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> np.ndarray:
    """Compute fixed-ridge latent Matérn variance at one or more query points."""
    matern_kernel = matern_kernel or make_matern_kernel(c.gp_lengthscale, c.gp_nu)
    q = torch.as_tensor(x_star, dtype=torch.float32)
    if len(x_obs) == 0:
        return matern_kernel(q, q).diag().detach().numpy()
    xo = torch.as_tensor(x_obs, dtype=torch.float32)
    K = matern_kernel(xo, xo)
    ks = matern_kernel(xo, q)
    kss = matern_kernel(q, q).diag()
    # The ridge is lambda_0 from Corollary 4.2, not observation noise or
    # lambda_t. Jitter is added separately only for numerical stability.
    ridge = c.lambda_base + c.gp_jitter
    sol = torch.linalg.solve(K + ridge * torch.eye(len(x_obs)), ks)
    return torch.clamp(kss - (ks * sol).sum(dim=0), min=0).detach().numpy()


def matern_posterior(
    c: Config,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    query: np.ndarray,
    matern_kernel: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed-hyperparameter latent GP mean and variance.

    Observation rows are never deduplicated, so repeated candidate visits are
    conditioned on with the same sufficient matrix expression as the paper's
    diagnostic quantity.
    """
    matern_kernel = matern_kernel or make_matern_kernel(c.gp_lengthscale, c.gp_nu)
    q = torch.as_tensor(query, dtype=torch.float32)
    if len(x_obs) == 0:
        return np.zeros(len(query)), matern_kernel(q, q).diag().detach().numpy()
    xo = torch.as_tensor(x_obs, dtype=torch.float32)
    K = matern_kernel(xo, xo)
    ks = matern_kernel(xo, q)
    kss = matern_kernel(q, q).diag()
    A = K + (c.lambda_base + c.gp_jitter) * torch.eye(len(x_obs))
    mean = (
        ks * torch.linalg.solve(A, torch.as_tensor(y_obs, dtype=torch.float32)[:, None])
    ).sum(dim=0)
    var = torch.clamp(kss - (ks * torch.linalg.solve(A, ks)).sum(dim=0), min=0)
    return mean.detach().numpy(), var.detach().numpy()


def ensemble_predict(
    c: Config,
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    t: int,
    seed: int,
    parameter_kernel: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> tuple[np.ndarray, np.ndarray]:
    """Train independent diagnostic networks; these predictions never select actions."""
    if c.ensemble_members < 2:
        raise ValueError("ensemble_members must be at least two when enabled")
    vals = []
    for j in range(c.ensemble_members):
        model, _ = fit_network(
            c,
            x,
            y,
            seed + 100003 * (j + 1) + 7919 * t,
            parameter_kernel,
        )
        with torch.no_grad():
            vals.append(
                model(torch.as_tensor(grid, dtype=torch.float32, device=c.device))
                .cpu()
                .numpy()
            )
    a = np.stack(vals)
    return a.mean(0), a.var(0, ddof=1)


def direct_ridge_variance(
    kernel: np.ndarray, obs: np.ndarray, star: int, ridge: float
) -> float:
    """Reference implementation used by tests; duplicates are intentionally retained."""
    if len(obs) == 0:
        return float(kernel[star, star])
    k = kernel[np.ix_(obs, obs)]
    ks = kernel[obs, star]
    return float(
        kernel[star, star] - ks @ np.linalg.solve(k + ridge * np.eye(len(obs)), ks)
    )


def run_one(
    c: Config,
    policy_seed: int,
    grid: np.ndarray,
    teacher_values: np.ndarray,
    initial_idx: np.ndarray,
    noise: np.ndarray,
    root: Path,
    progress_position: int = 0,
) -> None:
    """Run one policy seed and save data aligned by observation count.

    Record ``t`` is produced from ``D_t``: it contains the fitted network,
    diagnostics, and the grid maximizer that will become ``x_(t+1)``.
    """
    out = root / f"seed_{policy_seed}"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run: {out}")
    out.mkdir(parents=True, exist_ok=True)
    LOGGER.info("starting policy seed %d; output=%s", policy_seed, out)
    parameter_kernel, lengthscale, count = make_run_parameter_kernel(c)
    matern_kernel = make_matern_kernel(c.gp_lengthscale, c.gp_nu)
    LOGGER.debug(
        "configured parameter kernel=%s lengthscale=%.6g parameters=%d",
        c.regularizer_kernel,
        lengthscale,
        count,
    )
    x = grid[initial_idx].copy()
    y = teacher_values[initial_idx] + noise[: len(initial_idx)]
    x_queries, y_queries = [], []
    records, snapshots = [], []
    star = int(np.argmax(teacher_values))
    progress = tqdm(
        total=c.rounds + 1,
        desc=f"seed {policy_seed}",
        unit="iter",
        position=progress_position,
        dynamic_ncols=True,
        leave=True,
    )
    for t in range(c.rounds + 1):
        model, fit = fit_network(
            c, x, y, policy_seed + 1009 * t, parameter_kernel
        )
        with torch.no_grad():
            pred = (
                model(torch.as_tensor(grid, dtype=torch.float32, device=c.device))
                .cpu()
                .numpy()
            )
        qidx = int(np.argmax(pred))  # reproducible first-index tie break
        gp_var = matern_variance(c, x, grid[star : star + 1], matern_kernel)[0]
        rec = dict(
            observation_count=len(x),
            lambda_t=lambda_at(c, len(x)),
            gp_latent_variance_at_xstar=float(gp_var),
            exact_optimum_visits=int(np.sum(np.all(np.isclose(x, grid[star]), axis=1))),
            best_true_queried_value=(
                float(
                    np.max(
                        teacher_values[
                            np.array(
                                [np.argmin(np.sum((grid - z) ** 2, axis=1)) for z in x]
                            )
                        ]
                    )
                )
                if len(x)
                else float("nan")
            ),
            simple_regret=(
                float(
                    teacher_values[star]
                    - np.max(
                        teacher_values[
                            np.array(
                                [np.argmin(np.sum((grid - z) ** 2, axis=1)) for z in x]
                            )
                        ]
                    )
                )
                if len(x)
                else float("nan")
            ),
            cumulative_regret=float(
                sum(
                    teacher_values[star]
                    - teacher_values[np.argmin(np.sum((grid - z) ** 2, axis=1))]
                    for z in x_queries
                )
            ),
            next_query_index=qidx,
            **fit,
        )
        if c.ensemble_members and (t % c.ensemble_frequency == 0):
            em, ev = ensemble_predict(
                c,
                x,
                y,
                grid,
                len(x),
                policy_seed + 500000,
                parameter_kernel,
            )
            rec["ensemble_variance_at_xstar"] = float(ev[star])
            rec["ensemble_available"] = True
        else:
            rec["ensemble_variance_at_xstar"] = float("nan")
            rec["ensemble_available"] = False
        records.append(rec)
        LOGGER.debug(
            "seed=%d observations=%d lambda=%.6g next_query=%d gp_var=%.6g ensemble_var=%s",
            policy_seed,
            len(x),
            rec["lambda_t"],
            qidx,
            rec["gp_latent_variance_at_xstar"],
            (f"{rec['ensemble_variance_at_xstar']:.6g}"
             if rec["ensemble_available"] else "unavailable"),
        )
        progress.set_postfix(n=len(x), query=qidx, gp=f"{gp_var:.2g}")
        progress.update(1)
        if c.snapshot_frequency and t % c.snapshot_frequency == 0:
            gp_mean, gp_grid_var = matern_posterior(c, x, y, grid, matern_kernel)
            snap = {
                "observation_count": len(x),
                "teacher": teacher_values,
                "policy": pred,
                "gp_mean": gp_mean,
                "gp_variance": gp_grid_var,
                "observed_x": x.copy(),
                "observed_y": y.copy(),
                "x_star": grid[star],
                "next_query": grid[qidx],
            }
            if rec["ensemble_available"]:
                snap.update(ensemble_mean=em, ensemble_variance=ev)
            snapshots.append(snap)
        if t == c.rounds:
            break
        xq, yq = grid[qidx], teacher_values[qidx] + noise[len(initial_idx) + t]
        x_queries.append(xq.copy())
        y_queries.append(yq)
        x = np.vstack([x, xq])
        y = np.append(y, yq)
    progress.close()
    np.savez_compressed(
        out / "arrays.npz",
        grid=grid,
        teacher_values=teacher_values,
        initial_indices=initial_idx,
        x_obs=x,
        y_obs=y,
        x_queries=np.array(x_queries),
        y_queries=np.array(y_queries),
    )
    (out / "records.json").write_text(json.dumps(records, indent=2, allow_nan=True))
    np.savez_compressed(
        out / "snapshots.npz",
        **{
            f"s{i}_{k}": v
            for i, s in enumerate(snapshots)
            for k, v in s.items()
            if isinstance(v, (np.ndarray, int, float))
        },
    )
    (out / "config.json").write_text(
        json.dumps(
            {
                "config": asdict(c),
                "policy_seed": policy_seed,
                "teacher_seed": c.teacher_seed,
                "platform": platform.platform(),
                "python": sys.version,
            },
            indent=2,
        )
    )
    LOGGER.info(
        "completed policy seed %d; observations=%d records=%d saved=%s",
        policy_seed,
        len(x),
        len(records),
        out,
    )


ARGUMENT_HELP = {
    "output": "Directory for the run and all saved artifacts.",
    "log_level": "Logging verbosity: DEBUG, INFO, WARNING, ERROR, or CRITICAL.",
    "plot": "Run plot.py after all policy subprocesses finish.",
    "plot_out": "Plot output directory when --plot is enabled; defaults to OUTPUT/plots.",
    "plot_max_iterations": "Maximum observation-count iterations to plot.",
    "plot_gp_only": "Show only GP latent variance in the uncertainty plot.",
    "plot_font_size": "Master plot font size in points; other plot font sizes are derived from it.",
    "plot_figsize_width": "Plot figure width in inches when --plot is enabled.",
    "plot_figsize_height": "Plot figure height in inches when --plot is enabled.",
    "plot_no_titles": "Omit titles from generated plots.",
    "seed": "Master seed used to derive all unspecified experiment seeds.",
    "num_runs": "Number of policy runs when --seeds is not provided.",
    "seeds": "Directly specified policy seeds; overrides --seed and --num-runs for policy runs.",
    "teacher_seed": "Seed for the frozen teacher; derived from --seed when omitted.",
    "initial_seed": "Seed for initial-design sampling; derived from --seed when omitted.",
    "noise_seed": "Seed for observation noise; derived from --seed when omitted.",
    "input_dim": "Input dimension; one-dimensional runs use a line grid.",
    "grid_size": "Number of grid points in 1D, or approximate total points in higher dimensions.",
    "lower": "Lower bound of every input coordinate.",
    "upper": "Upper bound of every input coordinate.",
    "hidden": "Width of each hidden tanh layer.",
    "layers": "Number of hidden tanh layers.",
    "initialization": "Policy weight initialization: lecun_normal, kaiming_normal, or xavier_uniform.",
    "initialization_scale": "Policy Gaussian standard-deviation multiplier, or Xavier gain.",
    "teacher_initialization_scale": "Scale of the frozen teacher's Gaussian LeCun-normal initialization.",
    "teacher_noise": "Standard deviation of independent Gaussian observation noise.",
    "initial_size": "Number of randomly selected initial-design observations.",
    "rounds": "Number of sequential query rounds after the initial design.",
    "steps": "Optimizer steps used for each fresh policy fit.",
    "learning_rate": "Learning rate for the selected optimizer.",
    "optimizer": "Optimizer used for each fit (adam or sgd).",
    "schedule": "Regularization schedule (log or constant).",
    "lambda_base": "Shared lower bound lambda_0 and fixed GP diagnostic ridge.",
    "schedule_q": "Exponent q in lambda_t=lambda_base*[1+log(1+t)]^q.",
    "theta_lengthscale_c": "Multiplier c in the parameter-kernel lengthscale c*sqrt(M).",
    "regularizer_kernel": "Parameter-space kernel: linear, rbf, matern, or rational_quadratic.",
    "regularizer_matern_nu": "Matérn smoothness for the regularizer; must be 0.5, 1.5, or 2.5.",
    "regularizer_rq_alpha": "Rational-quadratic shape parameter; must be positive.",
    "gp_lengthscale": "Fixed input-space Matérn lengthscale for the diagnostic GP.",
    "gp_nu": "Diagnostic Matérn smoothness; must be 0.5, 1.5, or 2.5.",
    "gp_jitter": "Numerical jitter added to the GP conditioning matrix.",
    "ensemble_members": "Number of diagnostic ensemble members; zero disables the ensemble.",
    "ensemble_frequency": "Run the ensemble every N diagnostic observation counts.",
    "snapshot_frequency": "Save a full-grid snapshot every N rounds; zero disables snapshots.",
    "d_eff": "Explicit effective dimension for optional rate references; unused by training.",
    "rate_window": "Observation-count window reserved for optional rate fitting.",
    "device": "Torch device for training and evaluation.",
}


def resolve_seeds(c: Config) -> None:
    """Fill unspecified streams with deterministic draws from the master seed."""
    if c.num_runs < 1:
        raise ValueError("num_runs must be positive")
    rng = np.random.default_rng(c.seed)

    def next_seed() -> int:
        return int(rng.integers(0, 2**63 - 1))

    if c.teacher_seed is None:
        c.teacher_seed = next_seed()
    if c.initial_seed is None:
        c.initial_seed = next_seed()
    if c.noise_seed is None:
        c.noise_seed = next_seed()
    if c.seeds is None:
        c.seeds = [next_seed() for _ in range(c.num_runs)]


def run_plots(c: Config) -> None:
    """Run the plotting script after the experiment has completed."""
    command = [sys.executable, str(Path(__file__).with_name("plot.py")), c.output]
    if c.plot_out is not None:
        command.extend(["--out", c.plot_out])
    if c.plot_max_iterations is not None:
        command.extend(["--max-iterations", str(c.plot_max_iterations)])
    if c.plot_gp_only:
        command.append("--gp-only")
    command.extend(["--font-size", str(c.plot_font_size)])
    command.extend(["--figsize-width", str(c.plot_figsize_width)])
    command.extend(["--figsize-height", str(c.plot_figsize_height)])
    if c.plot_no_titles:
        command.append("--no-titles")
    command.extend(["--log-level", c.log_level])
    LOGGER.info("plotting results: %s", " ".join(command))
    subprocess.run(command, check=True)


def parse() -> tuple[Config, int | None, int]:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # These private options are used by the parent process to start one worker
    # subprocess per policy seed. They intentionally stay out of public help.
    p.add_argument("--_config-file", default=None, help=argparse.SUPPRESS)
    p.add_argument("--_worker-seed", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_worker-position", type=int, default=0, help=argparse.SUPPRESS)
    for name, field in Config.__dataclass_fields__.items():
        if name == "seeds":
            p.add_argument(
                "--seeds",
                nargs="+",
                type=int,
                default=None,
                help=ARGUMENT_HELP[name],
            )
            continue
        typ = field.type
        if typ in (bool, "bool"):
            flag = f"--{name}"
            alias = f"--{name.replace('_', '-')}"
            flags = [flag] if alias == flag else [flag, alias]
            p.add_argument(
                *flags,
                action="store_true",
                default=field.default,
                help=ARGUMENT_HELP[name],
            )
        else:
            flag = f"--{name}"
            alias = f"--{name.replace('_', '-')}"
            flags = [flag] if alias == flag else [flag, alias]
            p.add_argument(
                *flags,
                type=(
                    int
                    if name
                    in {
                        "seed",
                        "num_runs",
                        "teacher_seed",
                        "initial_seed",
                        "noise_seed",
                        "plot_max_iterations",
                    }
                    else (
                        str
                        if typ in (str, "str") or field.default is None
                        else type(field.default)
                    )
                ),
                default=field.default,
                help=ARGUMENT_HELP[name],
            )
    args = vars(p.parse_args())
    config_file = args.pop("_config_file")
    worker_seed = args.pop("_worker_seed")
    worker_position = args.pop("_worker_position")
    if config_file is not None:
        c = Config(**json.loads(Path(config_file).read_text()))
        if worker_seed is None:
            raise ValueError("a worker seed is required with --_config-file")
        c.seeds = [worker_seed]
        return c, worker_seed, worker_position
    return Config(**args), None, 0


def main() -> None:
    c, worker_seed, worker_position = parse()
    resolve_seeds(c)
    level = getattr(logging, c.log_level.upper(), None)
    if not isinstance(level, int):
        raise TypeError("log_level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
    logging.basicConfig(
        level=level,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(processName)s[%(process)d] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    LOGGER.info(
        "starting experiment: output=%s seeds=%s device=%s log_level=%s",
        c.output,
        c.seeds,
        c.device,
        c.log_level.upper(),
    )
    root = Path(c.output)
    root.mkdir(parents=True, exist_ok=True)
    if worker_seed is not None:
        setup_path = root / "policy_setup.npz"
        with np.load(setup_path) as setup:
            run_one(
                c,
                worker_seed,
                setup["grid"],
                setup["teacher_values"],
                setup["initial_indices"],
                setup["noise"],
                root,
                progress_position=worker_position,
            )
        return
    grid = make_grid(c)
    teacher = MLP(c.input_dim, c.hidden, c.layers)
    seed_model(
        teacher,
        c.teacher_seed,
        "lecun_normal",
        c.teacher_initialization_scale,
    )
    teacher.eval()
    with torch.no_grad():
        teacher_values = teacher(torch.as_tensor(grid, dtype=torch.float32)).numpy()
    star = int(np.argmax(teacher_values))
    sorted_v = np.sort(teacher_values)[::-1]
    if len(sorted_v) > 1 and np.isclose(sorted_v[0], sorted_v[1]):
        raise ValueError("teacher has a tied finite-grid optimum")
    LOGGER.info(
        "teacher ready: grid_points=%d x_star_index=%d x_star=%s f_star=%.6g gap=%.6g",
        len(grid),
        star,
        grid[star].tolist(),
        teacher_values[star],
        sorted_v[0] - sorted_v[1],
    )
    initial_idx = (
        np.random.default_rng(c.initial_seed).choice(
            len(grid), size=c.initial_size, replace=False
        )
        if c.initial_size
        else np.array([], dtype=int)
    )
    noise = np.random.default_rng(c.noise_seed).normal(
        0, c.teacher_noise, size=c.initial_size + c.rounds
    )
    torch.save(teacher.state_dict(), root / "teacher.pt")
    (root / "teacher_metadata.json").write_text(
        json.dumps(
            {
                "teacher_seed": c.teacher_seed,
                "grid": grid.tolist(),
                "x_star_index": star,
                "x_star": grid[star].tolist(),
                "f_star": float(teacher_values[star]),
                "gap_to_second": float(sorted_v[0] - sorted_v[1]),
            },
            indent=2,
        )
    )
    np.savez_compressed(
        root / "policy_setup.npz",
        grid=grid,
        teacher_values=teacher_values,
        initial_indices=initial_idx,
        noise=noise,
    )
    config_path = root / "run_config.json"
    config_path.write_text(json.dumps(asdict(c), indent=2))
    LOGGER.info("launching %d independent policy subprocesses", len(c.seeds))
    processes = {
        subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--_config-file",
                str(config_path),
                "--_worker-seed",
                str(seed),
                "--_worker-position",
                str(position),
            ]
        ): (seed, position)
        for position, seed in enumerate(c.seeds)
    }
    failures = []
    while processes:
        for process, (seed, _position) in list(processes.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            del processes[process]
            if return_code == 0:
                LOGGER.info("policy subprocess finished: seed=%d", seed)
            else:
                failures.append((seed, return_code))
                LOGGER.error(
                    "policy subprocess failed: seed=%d return_code=%d",
                    seed,
                    return_code,
                )
        if processes:
            time.sleep(0.1)
    if failures:
        raise RuntimeError(f"policy subprocess failures: {failures}")
    if c.plot:
        run_plots(c)
    LOGGER.info(
        "experiment complete: output=%s x_star_index=%d x_star=%s gap=%.4g",
        root,
        star,
        grid[star].tolist(),
        sorted_v[0] - sorted_v[1],
    )


if __name__ == "__main__":
    main()
