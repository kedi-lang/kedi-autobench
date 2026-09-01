from __future__ import annotations

import importlib

# pyright: reportPrivateUsage=false
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from autobench import Case, replay_experiment

from kedi_autobench.terminal_bench.capture import (
    DEFAULT_MAX_EVIDENCE_FILE_BYTES,
    DEFAULT_MAX_EVIDENCE_TOTAL_BYTES,
    _load_json_if_present,
    discover_trials,
    record_harbor_job,
)
from kedi_autobench.terminal_bench.cli import main
from kedi_autobench.terminal_bench.models import AgentContext, HarborTrialResult, TimingInfo
from kedi_autobench.terminal_bench.runner import run_then_record
from kedi_autobench.terminal_bench.task import (
    _case_input,
    _looks_like_text,
    _positive_int,
    _require_descendant,
    _required_string,
    _sanitize_json,
    _sanitized_copy,
    _usage,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _job(tmp_path: Path, *, trials: int = 1) -> Path:
    job = tmp_path / "jobs" / "pilot"
    _write_json(job / "result.json", {"id": "job", "n_total_trials": trials})
    _write_json(job / "config.json", {"provider_api_key": "sk-configsecret123456"})
    _write_json(job / "lock.json", {"dataset": "terminal-bench@2.1"})
    _write_json(
        job / "kedi-manifest.json",
        {
            "material_digest": "a" * 64,
            "model": "openrouter/openai/gpt-test",
            "adapter": "pydantic",
            "effort": "high",
            "policy": {"history": True, "artifacts": True},
        },
    )
    for index in range(trials):
        trial = job / f"task-{index + 1}__1"
        _write_json(
            trial / "result.json",
            {
                "task_name": f"task-{index + 1}",
                "trial_name": f"task-{index + 1}__1",
                "trial_uri": trial.as_uri(),
                "source": "terminal-bench@2.1",
                "task_checksum": f"checksum-{index + 1}",
                "agent_result": {
                    "n_input_tokens": 100,
                    "n_cache_tokens": 60,
                    "n_output_tokens": 20,
                    "cost_usd": 0.04,
                },
                "verifier_result": {"rewards": {"reward": 1 if index == 0 else 0}},
                "started_at": "2026-08-31T10:00:00Z",
                "finished_at": "2026-08-31T10:00:12Z",
                "environment_setup": {
                    "started_at": "2026-08-31T10:00:00Z",
                    "finished_at": "2026-08-31T10:00:02Z",
                },
                "agent_setup": {
                    "started_at": "2026-08-31T10:00:02Z",
                    "finished_at": "2026-08-31T10:00:03Z",
                },
                "agent_execution": {
                    "started_at": "2026-08-31T10:00:03Z",
                    "finished_at": "2026-08-31T10:00:10Z",
                },
                "verifier": {
                    "started_at": "2026-08-31T10:00:10Z",
                    "finished_at": "2026-08-31T10:00:12Z",
                },
            },
        )
        _write_json(trial / "config.json", {"attempt": 1})
        _write_json(trial / "lock.json", {"task": f"task-{index + 1}"})
        _write_json(trial / "verifier" / "reward.json", {"reward": 1})
        _write_json(
            trial / "agent" / "kedi-result.json",
            {
                "state": "completed",
                "adapter": "pydantic",
                "usage": {
                    "requests": 3,
                    "tool_calls": 5,
                    "input_tokens": 100,
                    "cache_read_tokens": 60,
                    "cache_write_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 120,
                    "uncached_input_tokens": 40,
                    "cost_usd": 0.04,
                },
                "verification": {"state": "passed"},
                "terminal": {"commands": 4},
                "policy": {"history": True},
            },
        )
        (trial / "agent" / "kedi.txt").write_text(
            "Authorization: Bearer abcdefghijklmnopqrstuvwxyz\n"
            "OPENAI_API_KEY=sk-agentsecret123456\n",
            encoding="utf-8",
        )
        (trial / "agent" / ".env").write_text(
            "OPENROUTER_API_KEY=sk-extensionlesssecret123456\n",
            encoding="utf-8",
        )
        (trial / "artifacts").mkdir(parents=True, exist_ok=True)
        (trial / "artifacts" / "answer.bin").write_bytes(b"\x00answer")
    return job


def _observations(run: Any) -> dict[str, Any]:
    return {item.name: item.value for item in run.task_result.observations}


def test_record_harbor_job_preserves_metrics_artifacts_and_redacts_secrets(
    tmp_path: Path,
) -> None:
    job = _job(tmp_path, trials=2)
    first_trial = next(path.parent for path in job.glob("*/result.json"))
    (first_trial / "agent" / "ignored-link").symlink_to("kedi.txt")
    output = tmp_path / "records" / "pilot"

    assert record_harbor_job(job, output, concurrency=2) == output.resolve()

    experiment = replay_experiment(output)
    assert len(experiment.runs) == 2
    first = experiment.runs[0]
    observations = _observations(first)
    assert observations["terminal_bench.reward.reward"] == 1
    assert observations["kedi.cost_usd"] == 0.04
    assert observations["kedi.cache_read_ratio"] == 0.6
    assert observations["terminal_bench.evidence_complete"] is True
    assert first.task_result.output["kedi_state"] == "completed"
    assert first.task_result.output["evidence_file_count"] >= 8
    assert any(
        artifact.name == "terminal_bench.file.trial.agent.kedi-result.json"
        for artifact in first.task_result.artifacts
    )

    payload = b"\n".join(
        path.read_bytes()
        for path in output.rglob("*")
        if path.is_file() and path.stat().st_size < 2_000_000
    )
    assert b"sk-configsecret123456" not in payload
    assert b"sk-agentsecret123456" not in payload
    assert b"sk-extensionlesssecret123456" not in payload
    assert b"abcdefghijklmnopqrstuvwxyz" not in payload
    assert b"[REDACTED]" in payload


def test_record_harbor_job_marks_oversized_evidence_without_failing_trial(
    tmp_path: Path,
) -> None:
    job = _job(tmp_path)
    output = tmp_path / "record"

    record_harbor_job(job, output, max_evidence_file_bytes=8)

    run = replay_experiment(output).runs[0]
    assert run.task_result.output["skipped_file_count"] > 0
    assert _observations(run)["terminal_bench.evidence_complete"] is False


def test_terminal_bench_evidence_default_is_bounded() -> None:
    assert DEFAULT_MAX_EVIDENCE_FILE_BYTES == 20_000_000
    assert DEFAULT_MAX_EVIDENCE_TOTAL_BYTES == 50_000_000


def test_record_harbor_job_bounds_total_attached_evidence(tmp_path: Path) -> None:
    job = _job(tmp_path)
    trial = next(path.parent for path in job.glob("*/result.json"))
    (trial / "artifacts" / "large-a.txt").write_text("a" * 700, encoding="utf-8")
    (trial / "artifacts" / "large-b.txt").write_text("b" * 700, encoding="utf-8")
    (trial / "artifacts" / "compact.json").write_text(
        json.dumps({"values": [0] * 200}, separators=(",", ":")),
        encoding="utf-8",
    )
    output = tmp_path / "record"

    record_harbor_job(
        job,
        output,
        max_evidence_file_bytes=1_000,
        max_evidence_total_bytes=1_500,
    )

    run = replay_experiment(output).runs[0]
    assert run.task_result.output["evidence_bytes"] <= 1_500
    assert run.task_result.output["skipped_file_count"] > 0
    assert _observations(run)["terminal_bench.evidence_complete"] is False


def test_capture_imports_failure_and_harbor_usage_when_kedi_result_is_missing(
    tmp_path: Path,
) -> None:
    job = _job(tmp_path)
    trial = next(path.parent for path in job.glob("*/result.json"))
    (trial / "agent" / "kedi-result.json").unlink()
    result = json.loads((trial / "result.json").read_text())
    result["exception_info"] = {
        "exception_type": "AgentTimeoutError",
        "exception_message": "timed out",
    }
    result["verifier_result"] = None
    _write_json(trial / "result.json", result)

    output = tmp_path / "record"
    record_harbor_job(job, output)

    run = replay_experiment(output).runs[0]
    observations = _observations(run)
    assert run.task_result.output["harbor_exception"] == "AgentTimeoutError"
    assert run.task_result.output.get("kedi_state") is None
    assert run.task_result.output["usage"]["cache_read_tokens"] == 60
    assert observations["terminal_bench.grader_completed"] is False
    assert observations["kedi.trial_completed"] is False
    assert observations["terminal_bench.exception"] == "AgentTimeoutError"


def test_capture_records_kedi_failure_and_handles_zero_usage(tmp_path: Path) -> None:
    job = _job(tmp_path)
    trial = next(path.parent for path in job.glob("*/result.json"))
    kedi_path = trial / "agent" / "kedi-result.json"
    kedi = json.loads(kedi_path.read_text())
    kedi.update(
        {
            "state": "agent_failure",
            "error_type": "RuntimeError",
            "error_message": "token sk-runtimeerror123456",
            "failure_phase": "agent",
            "usage": {"input_tokens": 0, "cache_read_tokens": 0},
        }
    )
    _write_json(kedi_path, kedi)
    result_path = trial / "result.json"
    result = json.loads(result_path.read_text())
    result["agent_result"] = None
    _write_json(result_path, result)
    shutil.rmtree(trial / "artifacts")
    shutil.rmtree(trial / "verifier")

    output = tmp_path / "record"
    record_harbor_job(job, output)

    run = replay_experiment(output).runs[0]
    observations = _observations(run)
    assert observations["kedi.error"]["type"] == "RuntimeError"
    assert observations["kedi.error"]["message"] == "token [REDACTED]"
    assert "kedi.cache_read_ratio" not in observations


def test_discover_trials_validates_job_and_makes_duplicate_ids_unique(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="job result"):
        discover_trials(missing)

    job = _job(tmp_path / "empty", trials=0)
    with pytest.raises(ValueError, match="no recorded trials"):
        discover_trials(job)

    job = _job(tmp_path / "duplicate", trials=2)
    paths = sorted(job.glob("*/result.json"))
    second = json.loads(paths[1].read_text())
    second["trial_name"] = "task-1__1"
    _write_json(paths[1], second)
    locations = discover_trials(job)
    assert [item.case_id for item in locations] == ["task-1__1", "task-1__1-2"]


def test_record_harbor_job_rejects_invalid_bounds(tmp_path: Path) -> None:
    job = _job(tmp_path)
    with pytest.raises(ValueError, match="max_evidence_file_bytes"):
        record_harbor_job(job, tmp_path / "record-a", max_evidence_file_bytes=0)
    with pytest.raises(ValueError, match="max_evidence_total_bytes"):
        record_harbor_job(job, tmp_path / "record-total", max_evidence_total_bytes=0)
    with pytest.raises(ValueError, match="concurrency"):
        record_harbor_job(job, tmp_path / "record-b", concurrency=0)


def test_run_then_record_is_post_run_and_fail_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job(tmp_path)
    output = tmp_path / "record"
    order: list[str] = []

    def fake_run(command: tuple[str, ...], *, check: bool) -> SimpleNamespace:
        assert command == ("harbor", "run")
        assert check is False
        order.append("harbor")
        return SimpleNamespace(returncode=7)

    def broken_recorder(*_args: Any, **_kwargs: Any) -> Path:
        assert order == ["harbor"]
        order.append("capture")
        raise RuntimeError("token sk-capturesecret123456")

    monkeypatch.setattr("kedi_autobench.terminal_bench.runner.subprocess.run", fake_run)
    assert (
        run_then_record(
            ["harbor", "run"],
            job_dir=job,
            output_dir=output,
            recorder=broken_recorder,
        )
        == 7
    )
    assert order == ["harbor", "capture"]
    error = json.loads((tmp_path / "record.capture-error.json").read_text())
    assert error["error_type"] == "RuntimeError"
    assert "sk-capturesecret123456" not in json.dumps(error)
    assert "[REDACTED]" in error["error_message"]


def test_run_then_record_skips_capture_when_harbor_created_no_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fake_run(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(returncode=0)

    def recorder(*_args: Any, **_kwargs: Any) -> Path:
        nonlocal called
        called = True
        return tmp_path

    monkeypatch.setattr("kedi_autobench.terminal_bench.runner.subprocess.run", fake_run)
    assert (
        run_then_record(
            ["harbor", "run"],
            job_dir=tmp_path / "absent",
            output_dir=tmp_path / "record",
            recorder=recorder,
        )
        == 0
    )
    assert called is False


def test_run_then_record_preserves_real_subprocess_exit_and_records_afterward(
    tmp_path: Path,
) -> None:
    job = _job(tmp_path)
    output = tmp_path / "record"

    return_code = run_then_record(
        [sys.executable, "-c", "raise SystemExit(9)"],
        job_dir=job,
        output_dir=output,
    )

    assert return_code == 9
    assert len(replay_experiment(output).runs) == 1


def test_cli_records_existing_job_and_requires_run_command(tmp_path: Path) -> None:
    job = _job(tmp_path)
    output = tmp_path / "record"
    assert main(["record", "--job-dir", str(job), "--record-dir", str(output)]) == 0
    assert (output / "experiment.yaml").is_file()

    with pytest.raises(ValueError, match="must not be empty"):
        main(
            [
                "run",
                "--job-dir",
                str(job),
                "--record-dir",
                str(tmp_path / "unused"),
                "--",
            ]
        )


def test_cli_run_accepts_command_without_separator(tmp_path: Path) -> None:
    job = _job(tmp_path)
    output = tmp_path / "record"

    assert (
        main(
            [
                "run",
                "--job-dir",
                str(job),
                "--record-dir",
                str(output),
                sys.executable,
                "-c",
                "raise SystemExit(0)",
            ]
        )
        == 0
    )
    assert len(replay_experiment(output).runs) == 1


def test_capture_helper_boundaries(tmp_path: Path) -> None:
    assert importlib.import_module("kedi_autobench.terminal_bench.__main__") is not None
    assert _load_json_if_present(tmp_path / "missing.json") == {}
    invalid_object = tmp_path / "list.json"
    invalid_object.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="JSON object"):
        _load_json_if_present(invalid_object)

    with pytest.raises(TypeError, match="case input"):
        _case_input(Case(id="bad", input="not-an-object"))
    with pytest.raises(TypeError, match="requires trial_dir"):
        _required_string({}, "trial_dir")
    with pytest.raises(TypeError, match="positive integer"):
        _positive_int(True)
    with pytest.raises(ValueError, match="escapes Harbor job"):
        _require_descendant(tmp_path.parent, tmp_path)

    assert _looks_like_text(b"plain text") is True
    assert _looks_like_text(b"\x00binary") is False
    assert _looks_like_text(b"\xff") is False

    broken_json = tmp_path / "broken.json"
    broken_json.write_text('{"token": sk-invalidsecret123456', encoding="utf-8")
    prepared = _sanitized_copy(broken_json, relative="broken.json", root=tmp_path / "copy")
    assert "sk-invalidsecret123456" not in prepared.read_text()
    undecodable = tmp_path / "payload.bin"
    undecodable.write_bytes(b"\xff\x00")
    prepared_binary = _sanitized_copy(
        undecodable,
        relative="payload.bin",
        root=tmp_path / "binary-copy",
    )
    assert prepared_binary.read_bytes() == b"\xff\x00"
    assert _sanitize_json({"password": "visible", "items": ["plain", 1]}) == {
        "password": "[REDACTED]",
        "items": ["plain", 1],
    }

    no_usage = HarborTrialResult(task_name="task", trial_name="trial")
    assert _usage(no_usage, None) == {"total_tokens": 0}
    partial_usage = HarborTrialResult(
        task_name="task",
        trial_name="trial",
        agent_result=AgentContext(n_input_tokens=3, n_output_tokens=2, metadata=None),
    )
    assert _usage(partial_usage, None)["total_tokens"] == 5

    assert TimingInfo().duration_seconds is None
    assert (
        TimingInfo(
            started_at=datetime(2026, 1, 2, tzinfo=UTC),
            finished_at=datetime(2026, 1, 1, tzinfo=UTC),
        ).duration_seconds
        == 0
    )
    assert no_usage.duration_seconds is None
