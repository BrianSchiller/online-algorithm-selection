"""Run all missing runs of a config on this machine, several at a time.

    python -m oas_experiments.launch <config> --jobs 8
    python -m oas_experiments.launch <config> --jobs 8 --match "*__cold__*"

Every run is its own ``python -m oas_experiments.run`` process, exactly as on
a cluster, so a crash or a memory leak stays inside one run. Runs that already
have a result are not started. The output of each run goes to
``<results dir>/_logs/<experiment>/<run id>.log``; failed runs are listed at
the end and can simply be launched again.

Numerical libraries are limited to one thread per run (unless the variables
are already set), so ``--jobs`` is the number of cores used.

Exit code: 0 if every started run succeeded, 1 otherwise, 2 for bad arguments.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from oas_experiments import io
from oas_experiments.config import ConfigError, load_config

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def log_path(root: Path, experiment: str, run_id: str) -> Path:
    """Where the output of one run is written."""
    return Path(root) / "_logs" / experiment / f"{run_id}.log"


def _environment() -> dict[str, str]:
    env = dict(os.environ)
    for name in THREAD_VARIABLES:
        env.setdefault(name, "1")
    return env


def _run_one(command: list[str], log: Path, env: dict[str, str]) -> tuple[int, float]:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as out:
        code = subprocess.call(command, stdout=out, stderr=subprocess.STDOUT, env=env)
    return code, time.perf_counter() - started


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m oas_experiments.launch",
        description="Run all missing runs of an experiment config in parallel.",
    )
    parser.add_argument("config", help="experiment config (YAML)")
    parser.add_argument(
        "--jobs", type=int, default=1, help="runs at the same time (default 1)"
    )
    parser.add_argument(
        "--match",
        help="only run ids matching this glob pattern, e.g. '*__cold__*'",
    )
    parser.add_argument(
        "--results-dir",
        help=f"results root (default: ${io.RESULTS_DIR_ENV} or ./results)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="also re-run existing results"
    )
    parser.add_argument("--final", action="store_true", help="passed on to run")
    parser.add_argument(
        "--dry-run", action="store_true", help="only print the runs that would start"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.jobs < 1:
        print("error: --jobs must be >= 1", file=sys.stderr)
        return 2
    try:
        config = load_config(args.config)
    except (ConfigError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    root = io.results_root(args.results_dir)
    run_ids = config.run_ids
    if args.match:
        run_ids = [r for r in run_ids if fnmatch.fnmatchcase(r, args.match)]
    if not args.overwrite:
        run_ids = [
            r for r in run_ids if not io.is_complete(io.run_dir(root, config.name, r))
        ]
    if args.dry_run or not run_ids:
        print("\n".join(run_ids) if run_ids else "nothing to run")
        return 0

    base = [sys.executable, "-m", "oas_experiments.run", str(args.config)]
    base += ["--results-dir", str(root)]
    base += ["--overwrite"] * args.overwrite + ["--final"] * args.final
    env = _environment()
    total, failed = len(run_ids), []
    print(f"{config.name}: {total} runs, {args.jobs} at a time")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                _run_one,
                [*base, "--run-id", run_id],
                log_path(root, config.name, run_id),
                env,
            ): run_id
            for run_id in run_ids
        }
        for done, future in enumerate(as_completed(futures), start=1):
            run_id = futures[future]
            code, seconds = future.result()
            status = "ok" if code == 0 else f"FAILED (exit {code})"
            print(f"[{done}/{total}] {status} {run_id} ({seconds:.1f}s)", flush=True)
            if code != 0:
                failed.append(run_id)

    if failed:
        print(
            f"\n{len(failed)} of {total} runs failed; their logs are in "
            f"{log_path(root, config.name, '').parent}:"
        )
        print("\n".join(f"  {r}" for r in sorted(failed)))
        return 1
    print(f"all {total} runs done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
