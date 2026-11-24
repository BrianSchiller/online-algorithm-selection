"""Generate the toy ASlib scenarios under ``tests/fixtures/aslib``.

    python tests/fixtures/make_toy_aslib.py

Two small runtime scenarios, ``toy_a`` (250 instances) and ``toy_b`` (150),
with 3 features and 3 algorithms (cutoff 100). ``toy_b``'s features are
shifted, and a different algorithm is best there, so playing ``toy_a`` then
``toy_b`` with the ``grouped`` drift builder gives one sudden covariate drift.
The tests and the example config use them; the script is deterministic.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import arff
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent / "aslib"
FEATURES = ["f0", "f1", "f2"]
ALGORITHMS = ["a0", "a1", "a2"]
CUTOFF = 100.0
# log runtime = 2 + x @ W + noise: a0 is best for small f0, a1 for large f0
W = np.array([[0.8, -0.8, 0.0], [0.0, 0.0, 0.3], [0.2, 0.2, -0.2]])
SCENARIOS = {"toy_a": (250, 0.0), "toy_b": (150, 3.0)}  # size, shift of f0


def _dump(path: Path, relation: str, attributes: list, data: list) -> None:
    text = arff.dumps({"relation": relation, "attributes": attributes, "data": data})
    path.write_text(text, encoding="utf-8")


def write_scenario(name: str, n: int, shift: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (n, len(FEATURES)))
    x[:, 0] += shift
    runtime = np.exp(2.0 + x @ W + rng.normal(0.0, 0.2, (n, len(ALGORITHMS))))
    ids = [f"{name}/{i:03d}" for i in range(n)]

    out = HERE / name
    out.mkdir(parents=True)
    description = {
        "scenario_id": name,
        "performance_measures": ["runtime"],
        "maximize": [False],
        "performance_type": ["runtime"],
        "algorithm_cutoff_time": CUTOFF,
        "algorithm_cutoff_memory": "?",
        "features_cutoff_time": "?",
        "features_cutoff_memory": "?",
        "features_deterministic": FEATURES,
        "features_stochastic": [],
        "algorithms_deterministic": ALGORITHMS,
        "algorithms_stochastic": [],
        "number_of_feature_steps": 1,
        "feature_steps": {"base": {"provides": FEATURES}},
    }
    (out / "description.txt").write_text(
        yaml.safe_dump(description, sort_keys=False), encoding="utf-8"
    )
    runs = [
        [iid, 1, algo, round(min(t, CUTOFF), 4), "ok" if t <= CUTOFF else "timeout"]
        for iid, row in zip(ids, runtime)
        for algo, t in zip(ALGORITHMS, row)
    ]
    _dump(
        out / "algorithm_runs.arff",
        f"{name}_algorithm_runs",
        [
            ("instance_id", "STRING"),
            ("repetition", "NUMERIC"),
            ("algorithm", "STRING"),
            ("runtime", "NUMERIC"),
            ("runstatus", ["ok", "timeout"]),
        ],
        runs,
    )
    _dump(
        out / "feature_values.arff",
        f"{name}_features",
        [("instance_id", "STRING"), ("repetition", "NUMERIC")]
        + [(f, "NUMERIC") for f in FEATURES],
        [[iid, 1, *np.round(row, 4).tolist()] for iid, row in zip(ids, x)],
    )
    _dump(
        out / "cv.arff",
        f"{name}_cv",
        [("instance_id", "STRING"), ("repetition", "NUMERIC"), ("fold", "NUMERIC")],
        [[iid, 1, i % 10 + 1] for i, iid in enumerate(ids)],
    )


def main() -> None:
    if HERE.exists():
        shutil.rmtree(HERE)
    for seed, (name, (n, shift)) in enumerate(SCENARIOS.items()):
        write_scenario(name, n, shift, seed)
    print(f"wrote {len(SCENARIOS)} toy ASlib scenarios to {HERE}")


if __name__ == "__main__":
    main()
