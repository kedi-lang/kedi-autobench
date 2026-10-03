from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from autobench import ExperimentResult, replay_experiment

from kedi_autobench.terminal_bench.models import (
    CapturedTrial,
    HarborTrialResult,
    KediTrialResult,
    TrialLocation,
)
from kedi_autobench.terminal_bench.usage import trial_usage


class CaptureValidationError(RuntimeError):
    """Raised when an Autobench record does not faithfully capture its Harbor source."""


def validate_capture_result(
    result: ExperimentResult,
    *,
    locations: Sequence[TrialLocation],
) -> None:
    expected = {location.case_id: location for location in locations}
    problems: list[str] = []
    if result.termination.status.value != "completed":
        problems.append(f"experiment status is {result.termination.status.value!r}")

    planned = set(result.termination.planned_run_ids)
    recorded = set(result.termination.recorded_run_ids)
    actual_run_ids = {run.run_id for run in result.runs}
    actual_cases = {run.case_id for run in result.runs}
    if len(actual_run_ids) != len(result.runs):
        problems.append("duplicate run IDs in captured results")
    if len(result.runs) != len(expected):
        problems.append(f"expected {len(expected)} runs, found {len(result.runs)}")
    if actual_cases != set(expected):
        problems.append(
            "case IDs differ: "
            f"missing={sorted(set(expected) - actual_cases)!r}, "
            f"unexpected={sorted(actual_cases - set(expected))!r}"
        )
    if planned != recorded:
        problems.append(
            "recorded run IDs differ from the plan: "
            f"missing={sorted(planned - recorded)!r}, "
            f"unexpected={sorted(recorded - planned)!r}"
        )
    if actual_run_ids != recorded:
        problems.append(
            "actual run IDs differ from the recorded run IDs: "
            f"missing={sorted(recorded - actual_run_ids)!r}, "
            f"unexpected={sorted(actual_run_ids - recorded)!r}"
        )

    for run in result.runs:
        location = expected.get(run.case_id)
        if location is None:
            continue
        prefix = f"case {run.case_id!r}"
        if run.error is not None:
            problems.append(
                f"{prefix} has task error {run.error.error_type!r}: {run.error.message}"
            )
            continue
        if run.partial:
            problems.append(f"{prefix} is partial")
        if run.end_reason.value != "completed":
            problems.append(f"{prefix} ended as {run.end_reason.value!r}")
        try:
            captured = CapturedTrial.model_validate(run.task_result.output)
        except Exception as exc:  # noqa: BLE001 - preserve validation context
            problems.append(f"{prefix} has invalid captured output: {exc}")
            continue
        _compare_source(captured, location, problems=problems, prefix=prefix)

    if problems:
        raise CaptureValidationError("Invalid Terminal-Bench capture:\n- " + "\n- ".join(problems))


def validate_harbor_record(record_dir: Path, *, job_dir: Path) -> Path:
    job_dir = job_dir.expanduser().resolve()
    record_dir = record_dir.expanduser().resolve()
    from kedi_autobench.terminal_bench.capture import discover_trials

    try:
        result = replay_experiment(record_dir)
    except Exception as exc:
        raise CaptureValidationError(f"Terminal-Bench record cannot be replayed: {exc}") from exc
    validate_capture_result(result, locations=discover_trials(job_dir))
    return record_dir


def capture_record_complete(
    record_dir: Path,
    *,
    locations: Sequence[TrialLocation],
) -> bool:
    try:
        validate_capture_result(replay_experiment(record_dir), locations=locations)
    except Exception:  # noqa: BLE001 - completeness is the explicit boolean boundary
        return False
    return True


def _compare_source(
    captured: CapturedTrial,
    location: TrialLocation,
    *,
    problems: list[str],
    prefix: str,
) -> None:
    harbor = HarborTrialResult.model_validate_json(
        (location.trial_dir / "result.json").read_text(encoding="utf-8")
    )
    kedi_path = location.trial_dir / "agent" / "kedi-result.json"
    kedi = (
        KediTrialResult.model_validate_json(kedi_path.read_text(encoding="utf-8"))
        if kedi_path.is_file()
        else None
    )
    rewards = dict(harbor.verifier_result.rewards or {}) if harbor.verifier_result else {}
    expected_usage = trial_usage(harbor, kedi)
    comparisons: tuple[tuple[str, Any, Any], ...] = (
        ("task name", captured.task_name, location.task_name),
        ("trial name", captured.trial_name, location.trial_name),
        (
            "Harbor exception",
            captured.harbor_exception,
            None if harbor.exception_info is None else harbor.exception_info.exception_type,
        ),
        ("Kedi state", captured.kedi_state, None if kedi is None else kedi.state),
        ("rewards", captured.rewards, rewards),
        ("usage", captured.usage, expected_usage),
    )
    for name, actual, expected in comparisons:
        if actual != expected:
            problems.append(f"{prefix} {name} differs from source: {actual!r} != {expected!r}")


__all__ = (
    "CaptureValidationError",
    "capture_record_complete",
    "validate_capture_result",
    "validate_harbor_record",
)
