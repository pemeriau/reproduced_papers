"""Phase 05 fair comparison: the Iris split, the lr selection rule, and a smoke run."""

from __future__ import annotations

import json

import numpy as np
import pytest
from power_of_QNN.lib.data import load_arrays
from power_of_QNN.lib.fair_comparison import select_learning_rate
from test_cli import DATA_ROOT, _overlay


def _pair_cfg(classes):
    return {
        "data_root": str(DATA_ROOT),
        "dataset": {
            "name": "iris_pair",
            "classes": classes,
            "split_seed": 0,
            "per_class": {"train": 30, "validation": 10, "test": 10},
        },
    }


def test_iris_pair_split_is_stratified_disjoint_and_deterministic():
    cfg = _pair_cfg([1, 2])
    splits = {name: load_arrays(cfg, name) for name in ("train", "validation", "test")}
    for name, count in (("train", 30), ("validation", 10), ("test", 10)):
        x, y = splits[name]
        assert x.shape == (2 * count, 4)
        assert np.bincount(y).tolist() == [count, count]
        assert np.allclose(np.linalg.norm(x, axis=1), 1.0)
    # The three splits partition the 100 rows of classes 1 and 2 exactly (multiset equality).
    table = np.loadtxt(
        DATA_ROOT / "power_of_QNN" / "iris" / "iris.csv", delimiter=",", skiprows=1
    )
    source = table[np.isin(table[:, 4], [1, 2]), :4]
    source = source / np.linalg.norm(source, axis=1, keepdims=True)
    rows = np.concatenate([splits[name][0] for name in splits])
    assert np.allclose(_sorted_rows(rows), _sorted_rows(source))
    again = load_arrays(cfg, "test")
    assert np.array_equal(again[0], splits["test"][0])


def _sorted_rows(rows: np.ndarray) -> np.ndarray:
    return rows[np.lexsort(rows.T[::-1])]


def test_iris_pair_rows_match_the_source_classes():
    x01, y01 = load_arrays(_pair_cfg([0, 1]), "train")
    binary_x, binary_y = load_arrays(
        {"data_root": str(DATA_ROOT), "dataset": {"name": "iris_binary"}}, "train"
    )
    # Every 0/1 training row is a row of the paper's normalized classes 0/1 set, same label.
    for row, label in zip(x01, y01):
        match = np.flatnonzero(np.abs(binary_x - row).max(axis=1) < 1e-12)
        assert match.size and (binary_y[match] == label).all()


def test_iris_pair_rejects_unknown_split():
    with pytest.raises(ValueError):
        load_arrays(_pair_cfg([1, 2]), "holdout")


def test_selection_picks_minimum_and_reports_ties():
    selected, tied = select_learning_rate(
        {0.01: 0.40, 0.03: 0.302, 0.1: 0.300, 0.3: 0.50}, 0.005
    )
    assert selected == 0.1
    assert tied == [0.03, 0.1]
    with pytest.raises(ValueError):
        select_learning_rate({0.1: float("nan")}, 0.005)


def test_fair_comparison_smoke_run(tmp_path):
    from power_of_QNN.lib import runner as paper_runner

    cfg = _overlay("fair_smoke.json")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    paper_runner.train_and_evaluate(cfg, run_dir)
    status = json.loads((run_dir / "run_status.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert status["status"] == "COMPLETED"
    assert status["selection"]["split"] == "validation"
    for entry in metrics["models"].values():
        assert entry["selected_lr"] in (0.03, 0.1)
        best = min(entry["grid_mean_loss"].values())
        assert entry["grid_mean_loss"][str(entry["selected_lr"])] == best
        assert set(entry["final"]) == {"train", "validation", "test"}
        arrays = np.load(run_dir / entry["arrays"])
        assert arrays["test_loss"].shape == (2,)
        assert arrays["grid_losses"].shape == (2, 2)
        assert np.isclose(arrays["train_loss"].mean(), entry["loss_curve_mean"][-1])
