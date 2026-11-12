"""Where results live, and the only place that touches ``OnlineResult`` IO.

Result files are never written by this repository: saving and loading go
through ``OnlineResult.save`` / ``OnlineResult.load`` (ASF, work package F1).
One run is one directory::

    <results dir>/<experiment>/<run id>/{meta.json, steps.parquet}

What this module assumes about ASF beyond the frozen file schema is kept in
:func:`result_meta`; adjust it there if F1 exposes the metadata differently.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

RESULTS_DIR_ENV = "OAS_RESULTS_DIR"
DEFAULT_RESULTS_DIR = "results"
_TMP_MARKER = ".tmp-"


class ResultIOUnavailable(RuntimeError):
    """The installed ASF has no ``OnlineResult.save`` / ``load`` yet."""


def results_root(cli_value: str | Path | None = None) -> Path:
    """``--results-dir``, else ``$OAS_RESULTS_DIR``, else ``./results``."""
    return Path(cli_value or os.environ.get(RESULTS_DIR_ENV) or DEFAULT_RESULTS_DIR)


def experiment_dir(root: str | Path, experiment: str) -> Path:
    return Path(root) / experiment


def run_dir(root: str | Path, experiment: str, run_id: str) -> Path:
    return experiment_dir(root, experiment) / run_id


def is_complete(path: Path) -> bool:
    """A run directory counts as a result once both files exist.

    Runs are saved to a temporary sibling directory and renamed, so a killed
    job never leaves a directory that passes this check.
    """
    return (path / "meta.json").is_file() and (path / "steps.parquet").is_file()


def is_temporary(path: Path) -> bool:
    return _TMP_MARKER in path.name


def result_class() -> Any:
    """``asf.online.evaluation.OnlineResult`` with save/load, or a clear error."""
    from asf.online.evaluation import OnlineResult

    if not (hasattr(OnlineResult, "save") and hasattr(OnlineResult, "load")):
        raise ResultIOUnavailable(
            "The installed ASF has no OnlineResult.save/load. Results need the "
            "F1 interfaces; install an ASF commit that contains them."
        )
    return OnlineResult


def asf_supports_results() -> bool:
    try:
        result_class()
    except (ResultIOUnavailable, ImportError):
        return False
    return True


def load_result(path: str | Path) -> Any:
    """``OnlineResult.load(path)``."""
    return result_class().load(Path(path))


def result_meta(result: Any) -> dict[str, Any]:
    """The contents of ``meta.json`` of a loaded result, as a dict."""
    meta = getattr(result, "meta", None)
    if not isinstance(meta, dict):
        raise ResultIOUnavailable(
            "Expected OnlineResult.meta to hold the contents of meta.json as a "
            "dict; adapt oas_experiments.io.result_meta to the F1 interface."
        )
    return meta


def save_result(result: Any, path: str | Path, overwrite: bool = False) -> Path:
    """Save via ``OnlineResult.save`` so that ``path`` appears only when complete."""
    path = Path(path)
    if is_complete(path) and not overwrite:
        raise FileExistsError(f"{path} already holds a result.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}{_TMP_MARKER}{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    result.save(tmp)
    if path.exists():  # an older result (overwrite) or an incomplete leftover
        shutil.rmtree(path)
    os.replace(tmp, path)
    return path
