"""Hand-built frames in the shape of the D2 contract, for the D3 tests.

Streams: ``switch`` (two drift points, one transition interval) and ``calm``
(no drift). Protocols: ``warm`` and ``cold``. Methods: ``rook`` (detector,
signals in every run; the only method with ``uses_detector`` and detection
metrics, NaN elsewhere), ``ds-ts`` (no detector), the swept family ``unc``
(``unc-rate0.1``, ``unc-rate0.5``) and ``single`` (one seed only, so its
ci95 is NaN).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METRICS = [
    "closed_gap",
    "mean_user_cost",
    "solver_cpu",
    "extra_cpu",
    "overhead_seconds",
    "extra_run_rate",
    "n_drift_signals",
    "mean_delay",
    "n_missed",
    "n_false_alarms",
    "mean_recovery",
    "n_not_recovered",
    "mean_peak_regret",
]
METHODS = [  # method, family, seeds
    ("rook", "rook", (0, 1)),
    ("ds-ts", "ds-ts", (0, 1)),
    ("unc-rate0.1", "unc", (0, 1)),
    ("unc-rate0.5", "unc", (0, 1)),
    ("single", "single", (0,)),
]
STREAMS = {"switch": ([100, 200], [[100, 150]]), "calm": ([], [])}
PROTOCOLS = {"warm": "warm", "cold": "cold"}


def runs_frame() -> pd.DataFrame:
    rows = []
    for stream, (points, intervals) in STREAMS.items():
        for protocol, protocol_type in PROTOCOLS.items():
            for k, (method, family, seeds) in enumerate(METHODS):
                detector = method == "rook"
                for seed in seeds:
                    row = {m: 1.0 + k + 0.1 * seed for m in METRICS}
                    row.update(
                        run_id=f"{stream}-s0__{protocol}__{method}__seed{seed}",
                        method=method,
                        method_family=family,
                        stream=stream,
                        protocol=protocol,
                        protocol_type=protocol_type,
                        seed=seed,
                        closed_gap=0.1 * k + 0.01 * seed,
                        uses_detector=detector,
                        n_drift_signals=2 if detector else 0,
                        mean_delay=5.0 + seed if detector else np.nan,
                        n_missed=0.0 if detector else np.nan,
                        n_false_alarms=1.0 + seed if detector else np.nan,
                        drift_points=points,
                        transition_intervals=intervals,
                        drift_steps=[105, 210 + seed] if detector else [],
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


def agg_frame(runs: pd.DataFrame | None = None) -> pd.DataFrame:
    runs = runs_frame() if runs is None else runs
    keys = ["stream", "protocol", "protocol_type", "method", "method_family"]
    keys.append("uses_detector")
    grouped = runs.groupby(keys)[METRICS]
    mean, std, n = grouped.mean(), grouped.std(), grouped.size()
    ci = 1.96 * std.div(np.sqrt(n), axis=0)
    agg = pd.concat(
        [mean.add_suffix("_mean"), std.add_suffix("_std"), ci.add_suffix("_ci95")],
        axis=1,
    )
    agg["n"] = n
    return agg.reset_index()


def curves_frame(n_steps: int = 300) -> pd.DataFrame:
    rows = []
    for stream in STREAMS:
        for protocol in PROTOCOLS:
            for k, (method, family, seeds) in enumerate(METHODS):
                step = np.arange(n_steps)
                mean = np.full(n_steps, 0.1 * k)
                spread = np.nan if len(seeds) == 1 else 0.05
                rows.append(
                    pd.DataFrame(
                        dict(
                            stream=stream,
                            protocol=protocol,
                            method=method,
                            method_family=family,
                            uses_detector=method == "rook",
                            step=step,
                            mean=mean,
                            lo=mean - spread,
                            hi=mean + spread,
                        )
                    )
                )
    return pd.concat(rows, ignore_index=True)
