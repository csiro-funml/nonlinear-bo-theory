"""Compare aggregate metrics across saved experiment directories.

Metrics: ``gp_variance`` is diagnostic GP latent variance at the optimum;
``ensemble_variance`` is diagnostic ensemble sample variance there;
``simple_regret`` is the current optimum gap; ``cumulative_regret`` is its
running sum; and ``optimum_visits`` counts exact visits to the grid optimum.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot import (
    DEFAULT_FIGSIZE,
    configure_fonts,
    load_runs,
    metric_matrix,
    validate_figsize,
)

LOGGER = logging.getLogger(__name__)

# Okabe-Ito palette, designed to remain distinguishable for common forms of
# colour-vision deficiency. The final black entry is useful for an additional
# trace but is intentionally not used for axes or text styling.
COLORBLIND_PALETTE = (
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#000000",  # black
)

METRICS = {
    "gp_variance": (
        "gp_latent_variance_at_xstar",
        "GP latent variance at x-star",
        "variance (log scale)",
        True,
    ),
    "ensemble_variance": (
        "ensemble_variance_at_xstar",
        "ensemble variance at x-star",
        "variance (log scale)",
        True,
    ),
    "simple_regret": ("simple_regret", "simple regret", "regret", False),
    "cumulative_regret": (
        "cumulative_regret",
        "cumulative regret",
        "regret",
        False,
    ),
    "optimum_visits": (
        "exact_optimum_visits",
        "exact optimum visits",
        "visits",
        False,
    ),
}


def plot_comparison(
    roots: list[Path],
    out: Path,
    metric: str,
    max_iterations: int | None,
    show_titles: bool = True,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    log_y: bool = False,
) -> None:
    """Plot one aggregate trace for each experiment directory."""
    figsize = validate_figsize(figsize)
    key, title, ylabel, log_scale = METRICS[metric]
    fig, ax = plt.subplots(figsize=figsize)
    for index, root in enumerate(roots):
        runs = load_runs(root)
        if max_iterations is not None:
            if max_iterations < 1:
                raise ValueError("max_iterations must be positive")
            runs = [(name, records[:max_iterations]) for name, records in runs]
        counts = np.array([record["observation_count"] for record in runs[0][1]])
        values = metric_matrix(runs, key)
        mean = np.nanmean(values, axis=0)
        standard_error = (
            np.nanstd(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
            if values.shape[0] > 1
            else np.zeros_like(mean)
        )
        color = COLORBLIND_PALETTE[index % len(COLORBLIND_PALETTE)]
        use_log_scale = log_scale or log_y
        plot_mean = np.maximum(mean, np.finfo(float).tiny) if use_log_scale else mean
        ax.plot(
            counts,
            plot_mean,
            color=color,
            linewidth=2,
            label=root.name,
        )
        lower = mean - standard_error
        upper = mean + standard_error
        if use_log_scale:
            lower = np.maximum(lower, np.finfo(float).tiny)
            upper = np.maximum(upper, np.finfo(float).tiny)
        ax.fill_between(
            counts,
            lower,
            upper,
            color=color,
            alpha=0.18,
            linewidth=0,
        )

    if log_scale or log_y:
        ax.set_yscale("log")
    ax.set(xlabel="observation count", ylabel=ylabel)
    if show_titles:
        ax.set_title(title)
    ax.legend(title="run")
    fig.tight_layout()
    fig.savefig(out / f"{metric}.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "roots", nargs="+", type=Path, help="Saved experiment directories to compare."
    )
    parser.add_argument(
        "--out", type=Path, default=Path("comparisons"), help="Output directory."
    )
    parser.add_argument(
        "--metric",
        choices=sorted(METRICS),
        default="gp_variance",
        help=(
            "Metric to compare: gp_variance (GP latent variance), "
            "ensemble_variance (ensemble sample variance), "
            "simple_regret (current optimum gap), "
            "cumulative_regret (running regret sum), or "
            "optimum_visits (exact optimum visits)."
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum number of observation-count iterations to plot.",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=10.0,
        help="Master plot font size in points; other plot font sizes are derived from it.",
    )
    parser.add_argument(
        "--figsize-width",
        type=float,
        default=DEFAULT_FIGSIZE[0],
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--figsize-height",
        type=float,
        default=DEFAULT_FIGSIZE[1],
        help="Figure height in inches.",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use a logarithmic y-axis, including for metrics that default to linear scale.",
    )
    parser.add_argument(
        "--no-titles",
        action="store_true",
        help="Omit the title from the comparison plot.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    args = parser.parse_args()
    configure_fonts(args.font_size)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s.%(msecs)03d %(levelname)s %(process)d %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args.out.mkdir(parents=True, exist_ok=True)
    figsize = (args.figsize_width, args.figsize_height)
    plot_comparison(
        args.roots,
        args.out,
        args.metric,
        args.max_iterations,
        not args.no_titles,
        figsize,
        args.log_y,
    )
    LOGGER.info("wrote %s comparison to %s", args.metric, args.out)


if __name__ == "__main__":
    main()
