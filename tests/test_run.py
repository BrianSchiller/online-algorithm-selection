from __future__ import annotations

import json

import pytest
import yaml

from oas_experiments import io, provenance, run
from oas_experiments import list as list_cli
from oas_experiments.config import load_config

RUN_ID = "sudden-s0__cold__ds-ts__seed0"


def _run(config, tmp_path, *extra):
    return run.main([str(config), "--results-dir", str(tmp_path / "results"), *extra])


def test_run_writes_one_result_with_provenance(minimal_config_file, tmp_path, capsys):
    assert _run(minimal_config_file, tmp_path, "--run-id", RUN_ID) == 0
    out = tmp_path / "results" / "mini" / RUN_ID
    assert sorted(p.name for p in out.iterdir()) == ["meta.json", "steps.parquet"]
    assert [p.name for p in out.parent.iterdir()] == [RUN_ID]  # no temp leftovers
    assert "done" in capsys.readouterr().out

    result = io.load_result(out)
    assert len(result.steps) == 80
    meta = io.result_meta(result)
    assert meta["protocol"] == "cold" and meta["n_warmstart"] == 0
    assert meta["seed"] == 0
    assert meta["stream"]["seed"] == 0 and meta["stream"]["name"] == "sudden"
    assert meta["result_mode"] == "best_of_scheduled"

    extra = meta["extra"]
    config = load_config(minimal_config_file)
    assert extra["run"] == config.get(RUN_ID).to_dict()
    assert extra["experiment"] == {
        "name": "mini",
        "run_id": RUN_ID,
        "config_file": "mini.yaml",
        "config_hash": config.config_hash,
    }
    assert extra["repo"] == provenance.repo_state()
    assert extra["asf_pinned_commit"] == provenance.pinned_asf_commit()
    assert extra["final"] is False
    assert extra["environment"]["python"]
    json.dumps(extra, allow_nan=False)


def test_warm_run_uses_the_warm_start_prefix(minimal_config_file, tmp_path):
    run_id = "sudden-s0__warm20__static__seed1"
    assert _run(minimal_config_file, tmp_path, "--run-id", run_id) == 0
    result = io.load_result(tmp_path / "results" / "mini" / run_id)
    meta = io.result_meta(result)
    assert len(result.steps) == 60
    assert (meta["protocol"], meta["n_warmstart"], meta["seed"]) == ("warm", 20, 1)
    assert meta["stream"]["n_instances"] == 80


def test_existing_result_is_skipped_unless_overwrite(
    minimal_config_file, tmp_path, capsys
):
    assert _run(minimal_config_file, tmp_path, "--run-id", RUN_ID) == 0
    meta_file = tmp_path / "results" / "mini" / RUN_ID / "meta.json"
    first = meta_file.stat().st_mtime_ns
    capsys.readouterr()

    assert _run(minimal_config_file, tmp_path, "--run-id", RUN_ID) == 0
    assert "skip" in capsys.readouterr().out
    assert meta_file.stat().st_mtime_ns == first

    assert _run(minimal_config_file, tmp_path, "--run-id", RUN_ID, "--overwrite") == 0
    assert "done" in capsys.readouterr().out
    assert meta_file.stat().st_mtime_ns != first


def test_existing_result_of_a_different_config_is_a_conflict(
    minimal_config, minimal_config_file, tmp_path, capsys
):
    assert _run(minimal_config_file, tmp_path, "--run-id", RUN_ID) == 0
    minimal_config["methods"][1]["selector"]["params"]["gamma"] = 0.5
    minimal_config_file.write_text(yaml.safe_dump(minimal_config))
    assert _run(minimal_config_file, tmp_path, "--run-id", RUN_ID) == 3
    assert "different config" in capsys.readouterr().err
    assert _run(minimal_config_file, tmp_path, "--run-id", RUN_ID, "--overwrite") == 0
    stored = io.result_meta(io.load_result(tmp_path / "results" / "mini" / RUN_ID))
    assert stored["extra"]["run"]["method"]["selector"]["params"]["gamma"] == 0.5


def test_incomplete_directory_is_not_a_result(minimal_config_file, tmp_path):
    out = tmp_path / "results" / "mini" / RUN_ID
    out.mkdir(parents=True)
    (out / "meta.json").write_text("{}")  # a job killed half-way through
    assert not io.is_complete(out)
    assert _run(minimal_config_file, tmp_path, "--run-id", RUN_ID) == 0
    assert io.is_complete(out)


def test_index_selects_the_nth_listed_run(minimal_config_file, tmp_path):
    config = load_config(minimal_config_file)
    assert _run(minimal_config_file, tmp_path, "--index", "4") == 0
    assert io.is_complete(tmp_path / "results" / "mini" / config.run_ids[4])
    assert _run(minimal_config_file, tmp_path, "--index", "6") == 2
    assert _run(minimal_config_file, tmp_path, "--index", "-1") == 2


def test_bad_arguments(minimal_config_file, tmp_path, capsys):
    assert _run(minimal_config_file, tmp_path, "--run-id", "nope") == 2
    assert "Unknown run id" in capsys.readouterr().err
    assert _run(tmp_path / "missing.yaml", tmp_path, "--run-id", RUN_ID) == 2
    with pytest.raises(SystemExit):  # neither --run-id nor --index
        _run(minimal_config_file, tmp_path)


def test_build_errors_are_reported(
    minimal_config, minimal_config_file, tmp_path, capsys
):
    minimal_config["methods"][1]["selector"]["class"] = "asf.online.selectors.Nope"
    minimal_config_file.write_text(yaml.safe_dump(minimal_config))
    assert _run(minimal_config_file, tmp_path, "--run-id", RUN_ID) == 2
    assert "Cannot import" in capsys.readouterr().err


def test_final_refuses_dirty_code(minimal_config_file, tmp_path, monkeypatch, capsys):
    clean = {"commit": "a" * 40, "dirty": False, "commit_source": "git"}
    monkeypatch.setattr(provenance, "pinned_asf_commit", lambda: "a" * 40)
    monkeypatch.setattr(provenance, "asf_state", lambda: clean)

    monkeypatch.setattr(provenance, "repo_state", lambda: {**clean, "dirty": True})
    assert _run(minimal_config_file, tmp_path, "--run-id", RUN_ID, "--final") == 4
    assert "Refusing to start a --final run" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()

    monkeypatch.setattr(provenance, "repo_state", lambda: clean)
    monkeypatch.setattr(provenance, "asf_state", lambda: {**clean, "dirty": True})
    assert _run(minimal_config_file, tmp_path, "--run-id", RUN_ID, "--final") == 4

    monkeypatch.setattr(provenance, "asf_state", lambda: clean)
    assert _run(minimal_config_file, tmp_path, "--run-id", RUN_ID, "--final") == 0
    stored = io.result_meta(io.load_result(tmp_path / "results" / "mini" / RUN_ID))
    assert stored["extra"]["final"] is True


def test_results_dir_from_environment(minimal_config_file, tmp_path, monkeypatch):
    monkeypatch.setenv(io.RESULTS_DIR_ENV, str(tmp_path / "elsewhere"))
    assert run.main([str(minimal_config_file), "--run-id", RUN_ID]) == 0
    assert io.is_complete(tmp_path / "elsewhere" / "mini" / RUN_ID)


def test_list_all_missing_and_count(minimal_config_file, tmp_path, capsys):
    results = ["--results-dir", str(tmp_path / "results")]
    config = load_config(minimal_config_file)

    assert list_cli.main([str(minimal_config_file)]) == 0
    assert capsys.readouterr().out.splitlines() == config.run_ids
    assert list_cli.main([str(minimal_config_file), "--count"]) == 0
    assert capsys.readouterr().out.strip() == "6"

    assert _run(minimal_config_file, tmp_path, "--run-id", RUN_ID) == 0
    capsys.readouterr()
    assert list_cli.main([str(minimal_config_file), "--missing", *results]) == 0
    missing = capsys.readouterr().out.splitlines()
    assert missing == [r for r in config.run_ids if r != RUN_ID]
    assert (
        list_cli.main([str(minimal_config_file), "--missing", "--count", *results]) == 0
    )
    assert capsys.readouterr().out.strip() == "5"

    assert list_cli.main([str(tmp_path / "missing.yaml")]) == 2
