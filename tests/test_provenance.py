from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from oas_experiments import provenance
from oas_experiments.provenance import FinalRunError, check_final, git_state

CLEAN = {"commit": "a" * 40, "dirty": False, "commit_source": "git"}
PIN = "b" * 40
ASF_CLEAN = {"commit": PIN, "dirty": False, "commit_source": "git"}


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.org", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def checkout(tmp_path) -> Path:
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "__init__.py").write_text("")
    (repo / ".gitignore").write_text("ignored/\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def test_git_state_clean_and_dirty(checkout):
    state = git_state(checkout / "pkg", "pkg/__init__.py")
    assert state["commit_source"] == "git"
    assert len(state["commit"]) == 40
    assert state["dirty"] is False

    (checkout / "ignored").mkdir()
    (checkout / "ignored" / "result.txt").write_text("x")
    assert git_state(checkout / "pkg", "pkg/__init__.py")["dirty"] is False

    (checkout / "notes.txt").write_text("untracked")
    assert git_state(checkout / "pkg", "pkg/__init__.py")["dirty"] is True
    # untracked files outside the scope do not count when a scope is given
    assert git_state(checkout / "pkg", "pkg/__init__.py", "pkg")["dirty"] is False
    (checkout / "pkg" / "new.py").write_text("")
    assert git_state(checkout / "pkg", "pkg/__init__.py", "pkg")["dirty"] is True


def test_git_state_tracked_change(checkout):
    (checkout / "pkg" / "__init__.py").write_text("x = 1\n")
    assert git_state(checkout / "pkg", "pkg/__init__.py", "pkg")["dirty"] is True


def test_git_state_ignores_an_enclosing_repository(checkout):
    # e.g. a package installed into a virtualenv that sits inside another repo
    site = checkout / "ignored" / "site-packages" / "other"
    site.mkdir(parents=True)
    state = git_state(site, "other/__init__.py")
    assert state == {"commit": None, "dirty": None, "commit_source": "unknown"}


def test_git_state_outside_git(tmp_path):
    assert git_state(tmp_path, "x.py")["commit_source"] == "unknown"


def test_check_final_accepts_clean_pinned_code():
    check_final(CLEAN, ASF_CLEAN, PIN)
    # pip-installed from the pinned git URL: dirty is None
    check_final(CLEAN, {**ASF_CLEAN, "dirty": None, "commit_source": "direct_url"}, PIN)


@pytest.mark.parametrize(
    ("repo", "asf", "pin", "message"),
    [
        ({**CLEAN, "dirty": True}, ASF_CLEAN, PIN, "this repository has uncommitted"),
        ({**CLEAN, "commit": None}, ASF_CLEAN, PIN, "this repository is unknown"),
        (CLEAN, {**ASF_CLEAN, "dirty": True}, PIN, "ASF checkout has uncommitted"),
        (CLEAN, {**ASF_CLEAN, "commit": None}, PIN, "ASF commit is unknown"),
        (CLEAN, {**ASF_CLEAN, "commit": "c" * 40}, PIN, "pyproject.toml pins"),
        (CLEAN, ASF_CLEAN, None, "does not pin ASF"),
    ],
)
def test_check_final_refuses(repo, asf, pin, message):
    with pytest.raises(FinalRunError, match=message):
        check_final(repo, asf, pin)


def test_check_final_lists_every_problem():
    with pytest.raises(FinalRunError) as excinfo:
        check_final({**CLEAN, "dirty": True}, {**ASF_CLEAN, "dirty": True}, PIN)
    assert "this repository" in str(excinfo.value)
    assert "ASF checkout" in str(excinfo.value)


def test_states_of_the_real_checkouts():
    repo = provenance.repo_state()
    assert repo["commit_source"] == "git"
    assert len(repo["commit"]) == 40
    asf = provenance.asf_state()
    assert asf["commit_source"] in {"git", "direct_url"}
    assert len(asf["commit"]) == 40


def test_pinned_asf_commit_is_read_from_pyproject():
    pinned = provenance.pinned_asf_commit()
    assert pinned is not None and len(pinned) == 40
    pyproject = (provenance.repo_root() / "pyproject.toml").read_text()
    assert f"@{pinned}" in pyproject


def test_environment():
    env = provenance.environment()
    assert set(env) == {"python", "platform", "packages"}
    assert env["packages"]["numpy"]
