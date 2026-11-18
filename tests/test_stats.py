from __future__ import annotations

import pandas as pd
import pytest

from oas_experiments.analysis.stats import (
    friedman,
    holm,
    mean_ranks,
    wilcoxon_vs_reference,
)

LEVELS = {"A": 0.9, "B": 0.5, "C": 0.3, "D": 0.1}  # A best in every cell


@pytest.fixture
def table():
    rows = [
        {
            "stream": stream,
            "protocol": protocol,
            "method": method,
            "seed": seed,
            "closed_gap": level + 0.01 * i + 0.001 * seed,
        }
        for i, (stream, protocol) in enumerate(
            (s, p) for s in ["s0", "s1", "s2"] for p in ["warm", "cold"]
        )
        for method, level in LEVELS.items()
        for seed in [0, 1]
    ]
    return pd.DataFrame(rows)


def test_mean_ranks(table):
    ranks = mean_ranks(table)
    assert ranks["method"].tolist() == ["A", "B", "C", "D"]
    assert ranks["mean_rank"].tolist() == [1, 2, 3, 4]
    assert (ranks["n_cells"] == 6).all()
    assert mean_ranks(table, higher_is_better=False)["method"].iloc[0] == "D"


def test_incomplete_cells_are_dropped(table):
    table = table[~((table["method"] == "D") & (table["stream"] == "s0"))]
    assert (mean_ranks(table)["n_cells"] == 4).all()


def test_friedman(table):
    # rank sums 6, 12, 18, 24: 12 / (6 * 4 * 5) * 1080 - 3 * 6 * 5 = 18
    result = friedman(table)
    assert result["statistic"] == pytest.approx(18)
    assert result["p_value"] < 0.001
    assert (result["n_cells"], result["n_methods"]) == (6, 4)


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.6, 0.9]) == pytest.approx([1.0, 1.0])


def test_wilcoxon_vs_reference(table):
    out = wilcoxon_vs_reference(table, reference="A")
    assert out["method"].tolist() == ["B", "C", "D"]
    # every difference negative in all 6 cells: exact p = 2 / 2**6
    assert out["p_value"].tolist() == pytest.approx([1 / 32] * 3)
    assert out["p_holm"].tolist() == pytest.approx([3 / 32] * 3)
    assert out["median_diff"].tolist() == pytest.approx([-0.4, -0.6, -0.8])
    assert (out["n_cells"] == 6).all()
