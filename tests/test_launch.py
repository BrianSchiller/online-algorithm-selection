from __future__ import annotations

import yaml

from oas_experiments import io, launch
from oas_experiments.config import load_config


def _launch(config, tmp_path, *extra):
    return launch.main(
        [str(config), "--results-dir", str(tmp_path / "results"), *extra]
    )


def test_launch_runs_the_missing_matching_runs(minimal_config_file, tmp_path, capsys):
    root = tmp_path / "results"
    config = load_config(minimal_config_file)
    cold = [r for r in config.run_ids if "__cold__" in r]

    assert (
        _launch(minimal_config_file, tmp_path, "--match", "*__cold__*", "--dry-run")
        == 0
    )
    assert capsys.readouterr().out.split() == cold
    assert not root.exists()

    assert (
        _launch(minimal_config_file, tmp_path, "--match", "*__cold__*", "--jobs", "2")
        == 0
    )
    out = capsys.readouterr().out
    assert "all 2 runs done" in out
    for run_id in cold:
        assert io.is_complete(io.run_dir(root, "mini", run_id))
        assert launch.log_path(root, "mini", run_id).is_file()

    assert _launch(minimal_config_file, tmp_path, "--match", "*__cold__*") == 0
    assert "nothing to run" in capsys.readouterr().out


def test_launch_reports_failed_runs(
    minimal_config, minimal_config_file, tmp_path, capsys
):
    minimal_config["methods"][1]["selector"]["class"] = "asf.online.selectors.Nope"
    minimal_config_file.write_text(yaml.safe_dump(minimal_config))
    run_id = "sudden-s0__cold__ds-ts__seed0"
    assert _launch(minimal_config_file, tmp_path, "--match", run_id) == 1
    assert "1 of 1 runs failed" in capsys.readouterr().out
    log = launch.log_path(tmp_path / "results", "mini", run_id)
    assert "Cannot import" in log.read_text()


def test_launch_bad_arguments(minimal_config_file, tmp_path):
    assert _launch(minimal_config_file, tmp_path, "--jobs", "0") == 2
    assert _launch(tmp_path / "missing.yaml", tmp_path) == 2
