"""Which code produced a result: commit and dirty flag of this repo and of ASF.

ASF records its own version in every result (``meta.asf``). The runner adds
the state of this repository and uses :func:`check_final` to refuse ``--final``
runs from code that cannot be traced to a commit.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import re
import subprocess
from pathlib import Path
from typing import Any

import oas_experiments

DISTRIBUTION = "oas-experiments"
_PIN = re.compile(r"asf(?:\[[^\]]*\])?\s*@\s*git\+[^\s\"']+@([0-9a-f]{40})\b")


class FinalRunError(RuntimeError):
    """A ``--final`` run was requested from code that is not traceable."""


def _git(path: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def git_state(
    path: Path, tracked_file: str, untracked_scope: str | None = None
) -> dict[str, Any]:
    """Commit and dirty flag of the git checkout that contains ``path``.

    ``tracked_file`` (relative to the checkout root) must be tracked there.
    This guards against mistaking an enclosing repository for the package's
    own one, e.g. when the package sits in a virtualenv inside another repo.

    Dirty means: tracked files differ from HEAD, or there are untracked,
    non-ignored files (only those under ``untracked_scope`` if given).
    """
    unknown = {"commit": None, "dirty": None, "commit_source": "unknown"}
    root = _git(path, "rev-parse", "--show-toplevel")
    if root is None:
        return unknown
    top = Path(root)
    if _git(top, "ls-files", "--error-unmatch", tracked_file) is None:
        return unknown
    commit = _git(top, "rev-parse", "HEAD")
    tracked = _git(top, "status", "--porcelain", "--untracked-files=no")
    scope = ["--", untracked_scope] if untracked_scope else []
    everything = _git(top, "status", "--porcelain", "--untracked-files=all", *scope)
    if commit is None or tracked is None or everything is None:
        return unknown
    untracked = any(line.startswith("??") for line in everything.splitlines())
    return {
        "commit": commit,
        "dirty": bool(tracked) or untracked,
        "commit_source": "git",
    }


def repo_root() -> Path | None:
    """Root of this repository if the package runs from a checkout of it."""
    root = Path(oas_experiments.__file__).resolve().parents[2]
    return root if (root / "pyproject.toml").is_file() else None


def repo_state() -> dict[str, Any]:
    """State of the experiments repository (needs an editable install)."""
    package_dir = Path(oas_experiments.__file__).resolve().parent
    return git_state(package_dir, "src/oas_experiments/__init__.py")


def asf_state() -> dict[str, Any]:
    """State of the imported ASF, with the same meaning as ``meta.asf``.

    ``git``: ASF is imported from a checkout; dirty covers tracked changes and
    untracked files under ``asf/``. ``direct_url``: ASF was pip-installed from
    a git URL pinned to a commit; dirty is ``None``. Otherwise ``unknown``.
    """
    import asf

    state = git_state(Path(asf.__file__).resolve().parent, "asf/__init__.py", "asf")
    if state["commit_source"] == "git":
        return state
    try:
        text = importlib.metadata.distribution("asf").read_text("direct_url.json")
        commit = json.loads(text or "{}").get("vcs_info", {}).get("commit_id")
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError):
        commit = None
    if commit:
        return {"commit": commit, "dirty": None, "commit_source": "direct_url"}
    return state


def pinned_asf_commit() -> str | None:
    """The ASF commit this repository pins in its ``pyproject.toml``."""
    root = repo_root()
    if root is not None:
        match = _PIN.search((root / "pyproject.toml").read_text(encoding="utf-8"))
        return match.group(1) if match else None
    try:
        requirements = importlib.metadata.requires(DISTRIBUTION) or []
    except importlib.metadata.PackageNotFoundError:
        return None
    for requirement in requirements:
        match = _PIN.search(requirement)
        if match:
            return match.group(1)
    return None


def check_final(
    repo: dict[str, Any], asf: dict[str, Any], pinned_commit: str | None
) -> None:
    """Raise :class:`FinalRunError` unless both code bases are clean and pinned."""
    problems: list[str] = []
    if repo["commit"] is None:
        problems.append(
            "the commit of this repository is unknown (run from a git checkout "
            "installed with `pip install -e .`)"
        )
    elif repo["dirty"]:
        problems.append("this repository has uncommitted or untracked files")
    if asf["commit"] is None:
        problems.append("the ASF commit is unknown")
    else:
        if asf["dirty"]:
            problems.append("the ASF checkout has uncommitted changes")
        if pinned_commit is None:
            problems.append("pyproject.toml does not pin ASF to a commit hash")
        elif asf["commit"] != pinned_commit:
            problems.append(
                f"ASF is at {asf['commit'][:10]} but pyproject.toml pins "
                f"{pinned_commit[:10]}"
            )
    if problems:
        raise FinalRunError(
            "Refusing to start a --final run:\n- " + "\n- ".join(problems)
        )


def environment() -> dict[str, Any]:
    """Interpreter, platform and versions of the numerical packages."""
    packages: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "pyarrow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }
