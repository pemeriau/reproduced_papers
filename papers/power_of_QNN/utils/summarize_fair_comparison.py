"""Phase 05 comparison tables from the fair-comparison runs (K1, X1, X2).

Usage (from papers/power_of_QNN):
    python utils/summarize_fair_comparison.py --runs results/phase05_runs.json --output results/phase05_tables.md

``--runs`` maps an experiment name (``fair_iris01_d8``, ...) to its run directory. The
rules are the ones frozen in LOG.md "Phase 5 Comparison Plan" before any run:

- primary metric: final training CE when the lr was selected on the training split (K1),
  otherwise test CE (X1, X2); lower is better;
- X beats Y only if mean(Y) - mean(X) exceeds both 0.005 and
  3 * sqrt(std_X^2 / n_X + std_Y^2 / n_Y); otherwise there is no resolved difference.

Every value is read at full precision from metrics.json and the per-seed npz arrays,
which are checked for the expected seed count and finiteness.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from figure_io import load_metrics

TIE_TOLERANCE = 0.005
STANDARD_ERRORS = 3.0


def _primary_split(metrics: dict) -> str:
    return "train" if metrics["comparison"]["selection_split"] == "train" else "test"


def _checked_arrays(run_dir: Path, metrics: dict, name: str) -> dict:
    entry = metrics["models"][name]
    arrays = np.load(run_dir / entry["arrays"])
    expected = metrics["comparison"]["final_seeds"]["count"]
    for split in metrics["comparison"]["splits"]:
        for kind in ("loss", "accuracy"):
            values = arrays[f"{split}_{kind}"]
            if values.shape != (expected,) or not np.isfinite(values).all():
                raise ValueError(
                    f"{run_dir}/{entry['arrays']}: {split}_{kind} has shape {values.shape} "
                    f"(expected ({expected},)) or non-finite values"
                )
    grid_shape = (
        len(metrics["comparison"]["lr_grid"]),
        metrics["comparison"]["selection_seeds"]["count"],
    )
    if arrays["grid_losses"].shape != grid_shape:
        raise ValueError(
            f"{name}: grid_losses shape {arrays['grid_losses'].shape} != {grid_shape}"
        )
    return arrays


def _decision(a: np.ndarray, b: np.ndarray) -> tuple[float, float, str]:
    """Return mean(a) - mean(b), the 3-standard-error threshold, and the verdict."""
    gap = float(a.mean() - b.mean())
    threshold = STANDARD_ERRORS * math.sqrt(a.var() / len(a) + b.var() / len(b))
    if abs(gap) > max(threshold, TIE_TOLERANCE):
        verdict = "first lower (resolved)" if gap < 0 else "second lower (resolved)"
    else:
        verdict = "no resolved difference"
    return gap, threshold, verdict


def _run_tables(experiment: str, run_dir: Path) -> tuple[list[str], dict]:
    metrics = load_metrics(run_dir)
    split = _primary_split(metrics)
    splits = metrics["comparison"]["splits"]
    lines = [
        f"### {experiment} (`{run_dir.as_posix()}`)",
        "",
        f"Primary metric: final {split} cross-entropy, mean ± std over "
        f"{metrics['comparison']['final_seeds']['count']} fresh seeds "
        f"({metrics['final_seeds'][0]}-{metrics['final_seeds'][-1]}); lr selected on the "
        f"{metrics['comparison']['selection_split']} split of seeds "
        f"{metrics['selection_seeds'][0]}-{metrics['selection_seeds'][-1]}.",
        "",
        "| Model | d | Hardware | lr grid mean loss ("
        + " / ".join(f"{lr:g}" for lr in metrics["comparison"]["lr_grid"])
        + ") | Selected lr (tied) | "
        + " | ".join(f"{s} CE" for s in splits)
        + " | "
        + " | ".join(f"{s} acc" for s in splits)
        + " | Reach CE≤0.3 (median it.) | Wall-clock grid + final (s) | Seeds ok / expected |",
        "|---|---:|---|---|---|" + "---:|" * (2 * len(splits)) + "---|---:|---:|",
    ]
    per_seed, summary = {}, {}
    for name, entry in metrics["models"].items():
        arrays = _checked_arrays(run_dir, metrics, name)
        per_seed[name] = arrays[f"{split}_loss"]
        hardware = entry.get("hardware", {})
        hardware_text = (
            f"{hardware['modes']} modes / {hardware['photons']} photons"
            if hardware
            else entry["type"]
        )
        grid = " / ".join(
            f"{entry['grid_mean_loss'][str(lr)]:.4f}"
            for lr in metrics["comparison"]["lr_grid"]
        )
        tied = ", ".join(f"{lr:g}" for lr in entry["tied_lrs"])
        reach = entry["median_iterations_to_threshold"]
        seconds = sum(entry["grid_seconds"].values()) + entry["final_seconds"]
        count = metrics["comparison"]["final_seeds"]["count"]
        lines.append(
            f"| {name} | {entry['d']} | {hardware_text} | {grid} | {entry['selected_lr']:g} ({tied}) | "
            + " | ".join(
                f"{entry['final'][s]['loss_mean']:.4f} ± {entry['final'][s]['loss_std']:.4f}"
                for s in splits
            )
            + " | "
            + " | ".join(
                f"{entry['final'][s]['accuracy_mean']:.3f} ± {entry['final'][s]['accuracy_std']:.3f}"
                for s in splits
            )
            + f" | {entry['fraction_reaching_threshold']:.2f} ({'—' if reach is None else f'{reach:.0f}'})"
            + f" | {seconds:.0f} | {count} / {count} |"
        )
        summary[name] = {
            "d": entry["d"],
            "selected_lr": entry["selected_lr"],
            "tied_lrs": entry["tied_lrs"],
            "primary_split": split,
            "primary_loss_mean": float(per_seed[name].mean()),
            "primary_loss_std": float(per_seed[name].std()),
            "final": entry["final"],
            "wall_clock_seconds": seconds,
        }
    lines += [
        "",
        f"Pairwise decisions on {split} CE (gap = first − second; resolved if |gap| > max(3 SE, {TIE_TOLERANCE})):",
        "",
        "| First | Second | Gap | 3 SE | Verdict |",
        "|---|---|---:|---:|---|",
    ]
    decisions = []
    for first, second in itertools.combinations(per_seed, 2):
        gap, threshold, verdict = _decision(per_seed[first], per_seed[second])
        decisions.append(
            {
                "first": first,
                "second": second,
                "gap": gap,
                "threshold": threshold,
                "verdict": verdict,
            }
        )
        lines.append(
            f"| {first} | {second} | {gap:+.4f} | {threshold:.4f} | {verdict} |"
        )
    lines.append("")
    return lines, {
        "run_dir": run_dir.as_posix(),
        "models": summary,
        "decisions": decisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = json.loads(args.runs.read_text(encoding="utf-8"))
    lines = [
        "# Phase 05 fair comparison tables",
        "",
        "Generated by `utils/summarize_fair_comparison.py` from metrics.json and the per-seed",
        "npz arrays. Local CPU (Apple laptop), float64, exact simulation. MerLin models are",
        "simulated adaptations (lossless, ideal threshold detectors, no postselection).",
        "",
    ]
    structured = {}
    for experiment, run_dir in runs.items():
        run_lines, structured[experiment] = _run_tables(experiment, Path(run_dir))
        lines += run_lines
    args.output.write_text("\n".join(lines), encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(structured, indent=2), encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
