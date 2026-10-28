"""Phase 05 figure: training curves and the primary metric of each fair-comparison run.

Usage (from papers/power_of_QNN):
    python utils/plot_fair_comparison.py --runs results/phase05_runs.json --scope paper-accurate

One row per run (K1, X1, X2): left, mean training cross-entropy ± std over the fresh
seeds at each model's selected lr; right, the primary metric (training CE for K1, test
CE for X1/X2) with error bars of 3 standard errors, the resolution scale of the
predeclared decision rule. Writes results/phase05_fair_comparison.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from figure_io import RESULTS_DIR, load_metrics  # noqa: E402
from power_of_QNN.lib.plotting import _stamp  # noqa: E402

TITLES = {
    "fair_iris01_d8": "K1: Iris 0/1, d = 8, training CE (C-04)",
    "fair_iris01_d40": "X1: Iris 0/1, d = 40, held-out test CE",
    "fair_iris12_d40": "X2: Iris 1/2, d = 40, held-out test CE",
}


def _color(name: str) -> str:
    if name.startswith("A8"):
        return "tab:purple"
    if name.startswith(("Q", "easy")):
        return "tab:blue" if not name.startswith("easy") else "tab:cyan"
    return "tab:gray" if name == "classical_42" else "tab:orange"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--scope", required=True)
    args = parser.parse_args()
    runs = json.loads(args.runs.read_text(encoding="utf-8"))
    figure, axes = plt.subplots(
        len(runs), 2, figsize=(11, 3.6 * len(runs)), squeeze=False
    )
    for row, (experiment, run_dir) in enumerate(runs.items()):
        run_dir = Path(run_dir)
        metrics = load_metrics(run_dir)
        split = (
            "train" if metrics["comparison"]["selection_split"] == "train" else "test"
        )
        curve_axis, bar_axis = axes[row]
        names, means, errors, colors = [], [], [], []
        for name, entry in metrics["models"].items():
            mean = np.asarray(entry["loss_curve_mean"])
            std = np.asarray(entry["loss_curve_std"])
            steps = np.arange(len(mean))
            color = _color(name)
            # In the d=40 runs the linear [4,2] is an unmatched reference: draw it dotted.
            unmatched = experiment.endswith("d40") and entry["d"] != 40
            linestyle = ":" if unmatched else "-"
            curve_axis.plot(
                steps,
                mean,
                color=color,
                linestyle=linestyle,
                label=f"{name} (lr {entry['selected_lr']:g})",
            )
            curve_axis.fill_between(
                steps, mean - std, mean + std, color=color, alpha=0.12
            )
            values = np.load(run_dir / entry["arrays"])[f"{split}_loss"]
            names.append(name)
            means.append(values.mean())
            errors.append(3 * values.std() / np.sqrt(len(values)))
            colors.append(color)
        curve_axis.set_title(TITLES.get(experiment, experiment), fontsize=9)
        curve_axis.set_xlabel("training iteration")
        curve_axis.set_ylabel("training cross-entropy")
        curve_axis.legend(fontsize=7)
        positions = np.arange(len(names))
        bar_axis.bar(positions, means, yerr=errors, color=colors, capsize=3)
        bar_axis.set_xticks(positions, names, rotation=25, ha="right", fontsize=8)
        bar_axis.set_ylabel(f"final {split} cross-entropy (± 3 SE)")
        bar_axis.set_title(
            f"{metrics['comparison']['final_seeds']['count']} fresh seeds, per-model tuned lr",
            fontsize=9,
        )
    figure.tight_layout()
    _stamp(figure, args.scope)
    RESULTS_DIR.mkdir(exist_ok=True)
    figure.savefig(RESULTS_DIR / "phase05_fair_comparison.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
