"""Plot saved experiment results without retraining models."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOGGER = logging.getLogger(__name__)
DEFAULT_FIGSIZE = (7.0, 4.0)


def configure_fonts(font_size: float) -> None:
    """Set plot text sizes as fixed ratios of one master size in points."""
    if font_size <= 0:
        raise ValueError("font_size must be positive")
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": 1.1 * font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": 0.8 * font_size,
            "ytick.labelsize": 0.8 * font_size,
            "legend.fontsize": 0.75 * font_size,
            "legend.title_fontsize": 0.8 * font_size,
        }
    )


def validate_figsize(figsize: tuple[float, float]) -> tuple[float, float]:
    """Validate and return a Matplotlib figure size in inches."""
    if len(figsize) != 2 or any(size <= 0 for size in figsize):
        raise ValueError("figsize width and height must both be positive")
    return figsize


def load_runs(root: Path) -> list[tuple[str, list[dict]]]:
    """Load every saved policy seed under ``root`` in deterministic order."""
    runs = [
        (path.parent.name, json.loads(path.read_text()))
        for path in sorted(root.glob("seed_*/records.json"))
    ]
    if not runs:
        raise FileNotFoundError(f"no seed runs under {root}")
    return runs


def metric_matrix(runs: list[tuple[str, list[dict]]], key: str) -> np.ndarray:
    """Build a seed-by-observation-count matrix for one saved metric."""
    return np.array([[record[key] for record in records] for _, records in runs])


def draw_traces(ax, x: np.ndarray, values: np.ndarray, color: str, label: str) -> None:
    """Draw all individual seed traces followed by their arithmetic mean."""
    for row in values:
        ax.plot(x, row, color=color, alpha=0.25, linewidth=0.9)
    ax.plot(x, np.nanmean(values, axis=0), color=color, linewidth=2, label=label)


def plot_summary(
    runs: list[tuple[str, list[dict]]],
    out: Path,
    max_iterations: int | None = None,
    gp_only: bool = False,
    show_titles: bool = True,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
) -> None:
    """Plot each summary metric in its own figure."""
    figsize = validate_figsize(figsize)
    if max_iterations is not None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        runs = [(name, records[:max_iterations]) for name, records in runs]
    counts = np.array([record["observation_count"] for record in runs[0][1]])
    gp = metric_matrix(runs, "gp_latent_variance_at_xstar")
    ensemble = metric_matrix(runs, "ensemble_variance_at_xstar")
    simple_regret = metric_matrix(runs, "simple_regret")
    cumulative_regret = metric_matrix(runs, "cumulative_regret")
    visits = metric_matrix(runs, "exact_optimum_visits")

    figures = [
        ("uncertainty", "uncertainty at x-star", "variance (log scale)", None),
        ("simple_regret", "simple regret", "regret", (simple_regret, "tab:green", "simple regret")),
        ("cumulative_regret", "cumulative regret", "regret", (cumulative_regret, "tab:purple", "cumulative regret")),
        ("optimum_visits", "optimum visits", "visits", (visits, "tab:red", "x-star visits")),
    ]
    for filename, title, ylabel, metric in figures:
        fig, ax = plt.subplots(figsize=figsize)
        if metric is None:
            draw_traces(ax, counts, gp, "tab:blue", "GP latent variance")
            if not gp_only:
                draw_traces(
                    ax, counts, ensemble, "tab:orange", "ensemble sample variance"
                )
            ax.set_yscale("log")
        else:
            values, color, label = metric
            draw_traces(ax, counts, values, color, label)
        ax.set(xlabel="observation count", ylabel=ylabel)
        if show_titles:
            ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{filename}.png", dpi=160)
        plt.close(fig)


def plot_snapshots(
    root: Path,
    run_name: str,
    out: Path,
    gp_only: bool = False,
    show_titles: bool = True,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
) -> None:
    """Plot saved 1D snapshots for the first seed as a representative run."""
    figsize = validate_figsize(figsize)
    arrays = np.load(root / run_name / "arrays.npz")
    grid = arrays["grid"]
    if grid.shape[1] != 1:
        LOGGER.info("skipping 1D snapshot plots for input_dim=%d", grid.shape[1])
        return

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(grid[:, 0], arrays["teacher_values"], label="teacher", linewidth=2)
    initial_count = len(arrays["initial_indices"])
    ax.scatter(
        arrays["x_obs"][:initial_count, 0],
        arrays["y_obs"][:initial_count],
        label="initial observations",
        color="tab:orange",
        marker="s",
        s=36,
        zorder=3,
    )
    ax.scatter(
        arrays["x_obs"][initial_count:, 0],
        arrays["y_obs"][initial_count:],
        label="sequential observations",
        color="black",
        marker="o",
        s=20,
        zorder=3,
    )
    ax.set(xlabel="x", ylabel="value")
    if show_titles:
        ax.set_title(f"saved data ({run_name})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "data.png", dpi=160)
    plt.close(fig)

    snapshots = np.load(root / run_name / "snapshots.npz")
    snapshot_count = len(
        [key for key in snapshots.files if key.startswith("s") and key.endswith("_teacher")]
    )
    for index in range(snapshot_count):
        try:
            teacher = snapshots[f"s{index}_teacher"]
            policy = snapshots[f"s{index}_policy"]
            gp_mean = snapshots[f"s{index}_gp_mean"]
            gp_variance = snapshots[f"s{index}_gp_variance"]
            observed_x = snapshots[f"s{index}_observed_x"][:, 0]
            observed_y = snapshots[f"s{index}_observed_y"]
            next_query = snapshots[f"s{index}_next_query"][0]
        except KeyError:
            continue

        fig, ax = plt.subplots(figsize=figsize)
        x = grid[:, 0]
        ax.plot(x, teacher, label="teacher", color="black")
        ax.plot(x, policy, label="policy network", color="tab:green")
        ax.plot(x, gp_mean, label="GP mean", color="tab:blue")
        ax.fill_between(
            x,
            gp_mean - 2 * np.sqrt(gp_variance),
            gp_mean + 2 * np.sqrt(gp_variance),
            color="tab:blue",
            alpha=0.18,
            label="GP latent ±2 sd",
        )
        if not gp_only and f"s{index}_ensemble_mean" in snapshots.files:
            ensemble_mean = snapshots[f"s{index}_ensemble_mean"]
            ensemble_variance = snapshots[f"s{index}_ensemble_variance"]
            ax.plot(x, ensemble_mean, label="ensemble mean", color="tab:orange")
            ax.fill_between(
                x,
                ensemble_mean - 2 * np.sqrt(ensemble_variance),
                ensemble_mean + 2 * np.sqrt(ensemble_variance),
                color="tab:orange",
                alpha=0.18,
                label="ensemble ±2 sample sd",
            )
        count_key = f"s{index}_observation_count"
        count = snapshots[count_key].item() if count_key in snapshots.files else index
        ax.scatter(observed_x, observed_y, label="observations", color="black", s=18)
        ax.axvline(next_query, color="tab:red", linestyle="--", label="next query")
        ax.set(xlabel="x", ylabel="value")
        if show_titles:
            ax.set_title(f"snapshot ({run_name}, n={count})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"snapshot_{index}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("root", help="Saved experiment directory.")
    parser.add_argument("--out", default=None, help="Output directory for plots.")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum number of observation-count iterations in summary plots.",
    )
    parser.add_argument(
        "--gp-only",
        action="store_true",
        help="Show only GP latent variance in the uncertainty summary plot.",
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
        "--no-titles",
        action="store_true",
        help="Omit titles from all generated plots.",
    )
    parser.add_argument(
        "--d-eff", type=float, default=None, help="Reserved effective dimension."
    )
    parser.add_argument(
        "--rate-window", default="2:20", help="Reserved rate-fitting window."
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

    root = Path(args.root)
    runs = load_runs(root)
    out = Path(args.out or root / "plots")
    out.mkdir(exist_ok=True)
    figsize = (args.figsize_width, args.figsize_height)
    plot_summary(
        runs, out, args.max_iterations, args.gp_only, not args.no_titles, figsize
    )
    plot_snapshots(root, runs[0][0], out, args.gp_only, not args.no_titles, figsize)
    LOGGER.info("wrote plots to %s for %d policy seeds", out, len(runs))


if __name__ == "__main__":
    main()
