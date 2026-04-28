"""compare_runs.py — load run artifacts and produce comparison plots + tables.

Usage:
    python scripts/benchmarking/compare_runs.py \\
        --runs results/logs/baseline_fp16/run_summary.json \\
               results/logs/bitnet_b158_125M/run_summary.json \\
        --output results/plots/comparison

Outputs written to --output/:
    loss_curves.png     — per-step loss for each run
    perplexity_bar.png  — final validation perplexity bar chart
    memory_bar.png      — peak memory usage (MB) bar chart
    training_time.png   — total training time (seconds) bar chart
    comparison.csv      — summary table (one row per run)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

# Matplotlib may not be installed in all environments; guard gracefully.
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend (works without a display)
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_summary(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_step_csv(summary_path: str | Path) -> list[dict[str, Any]]:
    """Load the step_metrics.csv that lives next to run_summary.json."""
    csv_path = Path(summary_path).parent / "step_metrics.csv"
    if not csv_path.exists():
        return []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({k: _try_float(v) for k, v in row.items()})
        return rows


def _try_float(v: str) -> Any:
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


def run_label(summary: dict[str, Any]) -> str:
    cfg = summary.get("model_config", {})
    return cfg.get("name", cfg.get("model_type", "unknown"))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _bar_chart(
    labels: list[str],
    values: list[float | None],
    title: str,
    ylabel: str,
    out_path: Path,
    color: str = "steelblue",
) -> None:
    if not HAS_MPL:
        print(f"  [skip] matplotlib not available — skipping {out_path.name}")
        return
    clean_vals = [v if v is not None else 0.0 for v in values]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 4))
    bars = ax.bar(labels, clean_vals, color=color, edgecolor="black", linewidth=0.7)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Run")
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] {out_path}")


def _loss_curves(
    runs: list[tuple[str, list[dict[str, Any]]]],
    out_path: Path,
) -> None:
    if not HAS_MPL:
        print(f"  [skip] matplotlib not available — skipping {out_path.name}")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, steps in runs:
        if not steps:
            continue
        xs = [s["step"] for s in steps if "step" in s]
        ys = [s["loss"] for s in steps if "loss" in s]
        ax.plot(xs, ys, label=label, linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] {out_path}")


# ---------------------------------------------------------------------------
# CSV summary table
# ---------------------------------------------------------------------------


def write_summary_csv(
    summaries: list[dict[str, Any]],
    labels: list[str],
    out_path: Path,
) -> None:
    fieldnames = [
        "run",
        "model_type",
        "num_layers",
        "hidden_size",
        "total_steps",
        "total_time_sec",
        "final_loss",
        "final_perplexity",
        "peak_memory_mb",
        "co2_kg",
    ]
    rows = []
    for label, s in zip(labels, summaries):
        cfg = s.get("model_config", {})
        rows.append({
            "run":              label,
            "model_type":       cfg.get("model_type", ""),
            "num_layers":       cfg.get("num_layers", ""),
            "hidden_size":      cfg.get("hidden_size", ""),
            "total_steps":      s.get("total_steps", ""),
            "total_time_sec":   s.get("total_time_sec", ""),
            "final_loss":       s.get("final_loss", ""),
            "final_perplexity": s.get("final_perplexity", ""),
            "peak_memory_mb":   s.get("peak_memory_mb", ""),
            "co2_kg":           s.get("co2_kg", ""),
        })
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [csv]  {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def compare(run_paths: list[str], output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summaries = [load_summary(p) for p in run_paths]
    step_data = [load_step_csv(p) for p in run_paths]
    labels = [run_label(s) for s in summaries]

    print(f"\nComparing {len(summaries)} run(s): {labels}")
    print(f"Output → {out}\n")

    # Loss curves
    _loss_curves(
        list(zip(labels, step_data)),
        out / "loss_curves.png",
    )

    # Perplexity bar chart
    ppls = [s.get("final_perplexity") for s in summaries]
    _bar_chart(labels, ppls, "Final Validation Perplexity", "Perplexity ↓",
               out / "perplexity_bar.png", color="coral")

    # Memory bar chart
    mems = [s.get("peak_memory_mb") for s in summaries]
    _bar_chart(labels, mems, "Peak Memory Usage", "Memory (MB)",
               out / "memory_bar.png", color="mediumseagreen")

    # Training time bar chart
    times = [s.get("total_time_sec") for s in summaries]
    _bar_chart(labels, times, "Total Training Time", "Time (sec)",
               out / "training_time.png", color="mediumpurple")

    # CSV summary table
    write_summary_csv(summaries, labels, out / "comparison.csv")

    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multiple training runs and generate plots."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        metavar="PATH",
        help="Paths to run_summary.json files (one per run).",
    )
    parser.add_argument(
        "--output",
        default="results/plots/comparison",
        help="Directory to write plots and CSV to.",
    )
    args = parser.parse_args()
    compare(args.runs, args.output)


if __name__ == "__main__":
    main()
