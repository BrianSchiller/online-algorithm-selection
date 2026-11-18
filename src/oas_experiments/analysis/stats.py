"""Statistical comparison of methods across (stream, protocol) cells.

Each cell is one block: a method's value in a cell is its mean over seeds.
Only cells in which every method has a value are used, so filter the run
table to the methods (and protocols) you want to compare first; e.g. offline
baselines exist only for warm-start protocols.

With few cells the tests have little power: with 6 cells the smallest exact
two-sided Wilcoxon p-value is 0.031.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

CELLS = ("stream", "protocol")


def cell_means(
    table: pd.DataFrame, metric: str, cells: Sequence[str] = CELLS
) -> pd.DataFrame:
    """Mean of ``metric`` over seeds: one row per complete cell, one column per
    method."""
    wide = table.groupby([*cells, "method"])[metric].mean().unstack("method")
    return wide.dropna()


def mean_ranks(
    table: pd.DataFrame,
    metric: str = "closed_gap",
    higher_is_better: bool = True,
    cells: Sequence[str] = CELLS,
) -> pd.DataFrame:
    """Mean rank per method (1 = best, ties get the average rank).

    Returns
    -------
    pd.DataFrame
        Columns ``[method, mean_rank, n_cells]``, best method first.
    """
    wide = cell_means(table, metric, cells)
    ranks = wide.rank(axis=1, ascending=not higher_is_better).mean()
    out = pd.DataFrame({"mean_rank": ranks, "n_cells": len(wide)})
    return out.sort_values("mean_rank").rename_axis("method").reset_index()


def friedman(
    table: pd.DataFrame, metric: str = "closed_gap", cells: Sequence[str] = CELLS
) -> dict[str, float]:
    """Friedman test over complete cells (needs at least three methods).

    Returns
    -------
    dict
        ``statistic``, ``p_value``, ``n_cells``, ``n_methods``.
    """
    wide = cell_means(table, metric, cells)
    result = stats.friedmanchisquare(*wide.to_numpy().T)
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_cells": len(wide),
        "n_methods": wide.shape[1],
    }


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm-adjusted p-values, in the input order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def wilcoxon_vs_reference(
    table: pd.DataFrame,
    reference: str,
    metric: str = "closed_gap",
    cells: Sequence[str] = CELLS,
) -> pd.DataFrame:
    """Two-sided Wilcoxon signed-rank test of every method against ``reference``.

    Returns
    -------
    pd.DataFrame
        Columns ``[method, n_cells, median_diff, statistic, p_value, p_holm]``;
        ``median_diff`` is the median of ``method - reference`` over cells,
        ``p_holm`` is Holm-corrected over all compared methods.
    """
    wide = cell_means(table, metric, cells)
    others = [m for m in wide if m != reference]
    rows = []
    for method in others:
        result = stats.wilcoxon(wide[method], wide[reference])
        rows.append(
            {
                "method": method,
                "n_cells": len(wide),
                "median_diff": float((wide[method] - wide[reference]).median()),
                "statistic": float(result.statistic),
                "p_value": float(result.pvalue),
            }
        )
    out = pd.DataFrame(
        rows, columns=["method", "n_cells", "median_diff", "statistic", "p_value"]
    )
    out["p_holm"] = holm(out["p_value"])
    return out
