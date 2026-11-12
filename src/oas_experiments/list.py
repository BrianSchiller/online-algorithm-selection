"""List the run ids of an experiment config, one per line (for job arrays).

    python -m oas_experiments.list <config>             # all run ids
    python -m oas_experiments.list <config> --missing   # runs without a result
    python -m oas_experiments.list <config> --count     # number of runs

The order is deterministic and is the order ``run --index`` refers to.
``--index`` always counts over all runs, not over the missing ones.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from oas_experiments import io
from oas_experiments.config import ConfigError, load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m oas_experiments.list",
        description="List the run ids of an experiment config.",
    )
    parser.add_argument("config", help="experiment config (YAML)")
    parser.add_argument(
        "--missing", action="store_true", help="only runs without a complete result"
    )
    parser.add_argument(
        "--count", action="store_true", help="print the number of runs instead"
    )
    parser.add_argument(
        "--results-dir",
        help=f"results root (default: ${io.RESULTS_DIR_ENV} or ./results)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
    except (ConfigError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    run_ids = config.run_ids
    if args.missing:
        root = io.results_root(args.results_dir)
        run_ids = [
            run_id
            for run_id in run_ids
            if not io.is_complete(io.run_dir(root, config.name, run_id))
        ]
    if args.count:
        print(len(run_ids))
    elif run_ids:
        print("\n".join(run_ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
