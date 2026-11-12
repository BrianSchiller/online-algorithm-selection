"""Run one cell of an experiment grid: one process, one run, one result.

    python -m oas_experiments.run <config> --run-id <id>
    python -m oas_experiments.run <config> --index <n>      # for job arrays

Exit codes: 0 = result written or already present, 2 = bad arguments or
config, 3 = a result exists for a different config, 4 = --final refused,
5 = the installed ASF cannot save results yet.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

from oas_experiments import io, provenance
from oas_experiments.build import BuildError, build_selector
from oas_experiments.config import (
    ConfigError,
    ExperimentConfig,
    RunSpec,
    canonical_json,
    load_config,
)
from oas_experiments.streams import build_stream

EXIT_OK, EXIT_USAGE, EXIT_CONFLICT, EXIT_NOT_FINAL, EXIT_NO_RESULT_IO = 0, 2, 3, 4, 5


def run_prequential(*args: Any, **kwargs: Any) -> Any:
    """``asf.online.evaluation.run_prequential`` (imported late: it is slow)."""
    from asf.online.evaluation import run_prequential as asf_run_prequential

    return asf_run_prequential(*args, **kwargs)


def build_metadata(
    config: ExperimentConfig,
    run: RunSpec,
    repo: dict[str, Any],
    final: bool,
) -> dict[str, Any]:
    """What the runner adds to the result (stored by ASF as ``meta.extra``)."""
    return {
        "experiment": {
            "name": config.name,
            "run_id": run.run_id,
            "config_file": config.path.name if config.path else None,
            "config_hash": config.config_hash,
        },
        "run": run.to_dict(),
        "repo": repo,
        "asf_pinned_commit": provenance.pinned_asf_commit(),
        "final": final,
        "environment": provenance.environment(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m oas_experiments.run",
        description="Run a single run of an experiment config and save its result.",
    )
    parser.add_argument("config", help="experiment config (YAML)")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--run-id", help="run id, as printed by oas_experiments.list")
    which.add_argument(
        "--index",
        type=int,
        help="0-based position in the output of oas_experiments.list",
    )
    parser.add_argument(
        "--results-dir",
        help=f"results root (default: ${io.RESULTS_DIR_ENV} or ./results)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing result"
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="refuse to run unless this repo and ASF are clean and ASF is at "
        "the pinned commit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.run_id is not None:
            run = config.get(args.run_id)
        elif 0 <= args.index < len(config.runs):
            run = config.runs[args.index]
        else:
            raise ConfigError(
                f"--index {args.index} is out of range: "
                f"{config.name} has {len(config.runs)} runs (0-based)."
            )
    except (ConfigError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if not io.asf_supports_results():
        print(
            "error: the installed ASF has no OnlineResult.save/load (F1 "
            "interfaces); nothing was run.",
            file=sys.stderr,
        )
        return EXIT_NO_RESULT_IO

    repo = provenance.repo_state()
    if args.final:
        try:
            provenance.check_final(
                repo, provenance.asf_state(), provenance.pinned_asf_commit()
            )
        except provenance.FinalRunError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_NOT_FINAL

    out = io.run_dir(io.results_root(args.results_dir), config.name, run.run_id)
    if io.is_complete(out) and not args.overwrite:
        stored = io.result_meta(io.load_result(out)).get("extra", {}).get("run")
        if canonical_json(stored) == canonical_json(run.to_dict()):
            print(f"skip {run.run_id}: result exists at {out}")
            return EXIT_OK
        print(
            f"error: {out} holds a result that was produced with a different "
            f"config for run id {run.run_id}. Rename the method/stream/protocol "
            "or pass --overwrite.",
            file=sys.stderr,
        )
        return EXIT_CONFLICT

    try:
        stream = build_stream(run)
        selector = build_selector(run, stream)
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    result = run_prequential(
        selector,
        stream,
        n_warmstart=run.n_warmstart,
        par_factor=run.par_factor,
        result_mode=run.result_mode,
        stream_seed=run.stream_seed,
        metadata=build_metadata(config, run, repo, args.final),
    )
    io.save_result(result, out, overwrite=args.overwrite)
    print(f"done {run.run_id}: {out}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
