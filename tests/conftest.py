from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "example_toy_sudden.yaml"
FIXTURE_RESULTS = Path(__file__).resolve().parent / "fixtures" / "results"
TOY_ASLIB = Path(__file__).resolve().parent / "fixtures" / "aslib"


@pytest.fixture
def minimal_config() -> dict:
    """A small valid config (2 methods, 2 protocols, 2 seeds) to mutate in tests."""
    return {
        "name": "mini",
        "seeds": [0, 1],
        "protocols": [
            {"name": "warm20", "type": "warm", "n_warmstart": 20},
            {"name": "cold", "type": "cold"},
        ],
        "streams": [
            {
                "name": "sudden",
                "kind": "aslib",
                "seeds": [0],
                "params": {"root": str(TOY_ASLIB), "scenarios": ["toy_a", "toy_b"]},
                "drift": {
                    "builder": "grouped",
                    "params": {"schedule": ["toy_a", "toy_b"], "segment_length": 40},
                },
            }
        ],
        "methods": [
            {
                "name": "static",
                "protocols": ["warm20"],
                "selector": {
                    "class": "asf.online.selectors.StaticOnlineSelector",
                    "params": {
                        "selector": {
                            "class": "asf.selectors.PairwiseClassifier",
                            "params": {
                                "model_class": {
                                    "type": "sklearn.tree.DecisionTreeClassifier"
                                },
                                "random_state": "${seed}",
                            },
                        }
                    },
                },
            },
            {
                "name": "ds-ts",
                "selector": {
                    "class": "asf.online.selectors.DiscountedTSSelector",
                    "params": {"gamma": 0.95, "budget": "${stream.cutoff}"},
                },
            },
        ],
    }


@pytest.fixture
def minimal_config_file(tmp_path, minimal_config) -> Path:
    import yaml

    path = tmp_path / "mini.yaml"
    path.write_text(yaml.safe_dump(minimal_config, sort_keys=False), encoding="utf-8")
    return path
