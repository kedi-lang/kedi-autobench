from __future__ import annotations

import hashlib
import json
import mimetypes
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from autobench import (
    ArtifactOverflow,
    Case,
    Direction,
    ObservationRole,
    RunContext,
    Semantic,
)
from kedi.integrations.harbor.records import is_secret_name, redact_text

from kedi_autobench.terminal_bench.models import (
    CapturedTrial,
    EvidenceFile,
    HarborTrialResult,
    KediTrialResult,
)
from kedi_autobench.terminal_bench.usage import trial_usage

_TEXT_SUFFIXES = frozenset(
    {
        ".csv",
        ".json",
        ".jsonl",
        ".kedi",
        ".log",
        ".md",
        ".py",
        ".sh",
        ".stderr",
        ".stdout",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)
_JOB_FILES = ("config.json", "job.log", "kedi-manifest.json", "lock.json", "result.json")
_TRIAL_FILES = ("config.json", "exception.txt", "lock.json", "result.json", "trial.log")


def capture_trial(ctx: RunContext, case: Case) -> CapturedTrial:
    value = _case_input(case)
    job_dir = Path(_required_string(value, "job_dir")).resolve()
    trial_dir = Path(_required_string(value, "trial_dir")).resolve()
    _require_descendant(trial_dir, job_dir)
    max_file_bytes = _positive_int(
        ctx.factor("max_evidence_file_bytes"),
        name="max_evidence_file_bytes",
    )
    max_total_bytes = _positive_int(
        ctx.factor("max_evidence_total_bytes"),
        name="max_evidence_total_bytes",
    )

    harbor = HarborTrialResult.model_validate_json(
        (trial_dir / "result.json").read_text(encoding="utf-8")
    )
    kedi_path = trial_dir / "agent" / "kedi-result.json"
    kedi = (
        KediTrialResult.model_validate_json(kedi_path.read_text(encoding="utf-8"))
        if kedi_path.is_file()
        else None
    )

    with ctx.span(
        "capture terminal-bench trial",
        kind="workflow",
        input={"task_name": harbor.task_name, "trial_name": harbor.trial_name},
    ) as span:
        usage = trial_usage(harbor, kedi)
        rewards = dict(harbor.verifier_result.rewards or {}) if harbor.verifier_result else {}
        _record_metrics(ctx, harbor, kedi, usage, rewards, span_id=span.id)
        evidence = _attach_evidence(
            ctx,
            job_dir=job_dir,
            trial_dir=trial_dir,
            max_file_bytes=max_file_bytes,
            max_total_bytes=max_total_bytes,
            span_id=span.id,
        )
        ctx.check(
            "terminal_bench.evidence_complete",
            all(item.attached for item in evidence),
            reason=(
                "all discovered evidence files were attached"
                if all(item.attached for item in evidence)
                else "one or more evidence files exceeded the configured capture bounds"
            ),
            span_id=span.id,
        )
        ctx.artifact(
            "terminal_bench.evidence_index",
            [item.model_dump(mode="json") for item in evidence],
            media_type="application/json",
            span_id=span.id,
        )
        result = CapturedTrial(
            task_name=harbor.task_name,
            trial_name=harbor.trial_name,
            harbor_exception=(
                None if harbor.exception_info is None else harbor.exception_info.exception_type
            ),
            rewards=rewards,
            kedi_state=None if kedi is None else kedi.state,
            usage=usage,
            evidence_file_count=sum(item.attached for item in evidence),
            evidence_bytes=sum(item.attached_byte_count or 0 for item in evidence),
            skipped_file_count=sum(not item.attached for item in evidence),
        )
        span.set_output(result.model_dump(mode="json"))
        return result


def _record_metrics(
    ctx: RunContext,
    harbor: HarborTrialResult,
    kedi: KediTrialResult | None,
    usage: Mapping[str, Any],
    rewards: Mapping[str, float | int],
    *,
    span_id: str,
) -> None:
    for name, value in rewards.items():
        ctx.metric(
            f"terminal_bench.reward.{name}",
            value,
            semantic_type=Semantic.QUALITY_CORRECTNESS,
            direction=Direction.MAXIMIZE,
            role=ObservationRole.OBJECTIVE,
            span_id=span_id,
        )
    if rewards:
        ctx.metric(
            "terminal_bench.reward.mean",
            sum(float(value) for value in rewards.values()) / len(rewards),
            semantic_type="quality.score",
            direction=Direction.MAXIMIZE,
            role=ObservationRole.OBJECTIVE,
            span_id=span_id,
        )
    grader_completed = (
        harbor.verifier_result is not None and harbor.verifier_result.rewards is not None
    )
    ctx.check(
        "terminal_bench.grader_completed",
        grader_completed,
        reason=(
            "official Harbor verifier produced rewards"
            if grader_completed
            else "official Harbor verifier did not produce rewards"
        ),
        span_id=span_id,
    )
    ctx.check(
        "kedi.trial_completed",
        kedi is not None and kedi.state == "completed",
        reason=("Kedi result missing" if kedi is None else f"Kedi state: {kedi.state}"),
        span_id=span_id,
    )
    _optional_metric(
        ctx,
        "terminal_bench.duration",
        harbor.duration_seconds,
        semantic_type=Semantic.TIME_LATENCY,
        unit="s",
        direction=Direction.MINIMIZE,
        span_id=span_id,
    )
    for name, timing in (
        ("environment_setup", harbor.environment_setup),
        ("agent_setup", harbor.agent_setup),
        ("agent_execution", harbor.agent_execution),
        ("verifier", harbor.verifier),
    ):
        _optional_metric(
            ctx,
            f"terminal_bench.{name}.duration",
            None if timing is None else timing.duration_seconds,
            semantic_type=Semantic.TIME_LATENCY,
            unit="s",
            direction=Direction.MINIMIZE,
            span_id=span_id,
        )

    metric_specs = (
        ("input_tokens", Semantic.LLM_TOKENS_INPUT, "token", None),
        ("cache_read_tokens", Semantic.LLM_TOKENS_CACHED_INPUT, "token", None),
        ("cache_write_tokens", Semantic.LLM_TOKENS_CACHE_WRITE, "token", None),
        ("output_tokens", Semantic.LLM_TOKENS_OUTPUT, "token", None),
        ("total_tokens", Semantic.LLM_TOKENS_TOTAL, "token", None),
        ("requests", Semantic.LLM_REQUEST_COUNT, "request", None),
        ("tool_calls", "tool.call.count", "call", None),
        ("cost_usd", Semantic.MONEY_COST, "USD", Direction.MINIMIZE),
    )
    for name, semantic_type, unit, direction in metric_specs:
        _optional_metric(
            ctx,
            f"kedi.{name}",
            usage.get(name),
            semantic_type=semantic_type,
            unit=unit,
            direction=direction,
            role=(ObservationRole.OBJECTIVE if name == "cost_usd" else ObservationRole.DIAGNOSTIC),
            span_id=span_id,
        )
    input_tokens = _number(usage.get("input_tokens"))
    cache_tokens = _number(usage.get("cache_read_tokens"))
    if input_tokens and cache_tokens is not None:
        ctx.metric(
            "kedi.cache_read_ratio",
            min(1.0, max(0.0, cache_tokens / input_tokens)),
            semantic_type="coverage.ratio",
            unit="ratio",
            direction=Direction.MAXIMIZE,
            role=ObservationRole.DIAGNOSTIC,
            span_id=span_id,
        )
    if harbor.exception_info is not None:
        ctx.diagnostic(
            "terminal_bench.exception",
            harbor.exception_info.exception_type,
            semantic_type=Semantic.ERROR_TYPE,
            span_id=span_id,
        )
    if kedi is not None and kedi.error_type is not None:
        ctx.diagnostic(
            "kedi.error",
            {
                "type": kedi.error_type,
                "phase": kedi.failure_phase,
                "message": redact_text(kedi.error_message or ""),
            },
            semantic_type=Semantic.ERROR_EXCEPTION,
            span_id=span_id,
        )


def _attach_evidence(
    ctx: RunContext,
    *,
    job_dir: Path,
    trial_dir: Path,
    max_file_bytes: int,
    max_total_bytes: int,
    span_id: str,
) -> tuple[EvidenceFile, ...]:
    candidates = _evidence_candidates(job_dir, trial_dir)
    evidence: list[EvidenceFile] = []
    attached_bytes = 0
    with tempfile.TemporaryDirectory(prefix="kedi-autobench-redacted-") as raw_temp:
        temp = Path(raw_temp)
        for source, relative in candidates:
            byte_count = source.stat().st_size
            digest = _sha256(source)
            if byte_count > max_file_bytes:
                evidence.append(
                    EvidenceFile(
                        path=relative,
                        source="harbor",
                        byte_count=byte_count,
                        sha256=digest,
                        attached=False,
                        reason=f"exceeds max_evidence_file_bytes={max_file_bytes}",
                    )
                )
                continue
            prepared = _sanitized_copy(source, relative=relative, root=temp)
            prepared_bytes = prepared.stat().st_size
            if prepared_bytes > max_file_bytes:
                evidence.append(
                    EvidenceFile(
                        path=relative,
                        source="harbor",
                        byte_count=byte_count,
                        sha256=digest,
                        attached=False,
                        reason=(
                            f"sanitized evidence exceeds max_evidence_file_bytes={max_file_bytes}"
                        ),
                    )
                )
                continue
            if attached_bytes + prepared_bytes > max_total_bytes:
                evidence.append(
                    EvidenceFile(
                        path=relative,
                        source="harbor",
                        byte_count=byte_count,
                        sha256=digest,
                        attached=False,
                        reason=f"exceeds max_evidence_total_bytes={max_total_bytes}",
                    )
                )
                continue
            media_type = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
            ctx.artifact_file(
                "terminal_bench.file." + relative.replace("/", "."),
                prepared,
                media_type=media_type,
                max_bytes=max_file_bytes,
                overflow=ArtifactOverflow.FAIL,
                filename=source.name,
                span_id=span_id,
                tags={"harbor.path": relative, "sha256.source": digest},
            )
            attached_bytes += prepared_bytes
            evidence.append(
                EvidenceFile(
                    path=relative,
                    source="harbor",
                    byte_count=byte_count,
                    sha256=digest,
                    attached=True,
                    attached_byte_count=prepared_bytes,
                )
            )
    return tuple(evidence)


def _evidence_candidates(job_dir: Path, trial_dir: Path) -> tuple[tuple[Path, str], ...]:
    found: list[tuple[Path, str]] = []
    seen: set[str] = set()

    def add(path: Path, relative: str) -> None:
        if relative in seen or not path.is_file() or path.is_symlink():
            return
        seen.add(relative)
        found.append((path, relative))

    for name in _JOB_FILES:
        add(job_dir / name, f"job/{name}")
    for name in _TRIAL_FILES:
        add(trial_dir / name, f"trial/{name}")
    for directory in ("agent", "artifacts", "verifier"):
        root = trial_dir / directory
        if not root.is_dir() or root.is_symlink():
            continue
        paths = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
        for path in paths:
            relative = path.relative_to(trial_dir).as_posix()
            add(path, f"trial/{relative}")
    return tuple(found)


def _sanitized_copy(source: Path, *, relative: str, root: Path) -> Path:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = source.read_bytes()
    if source.suffix.lower() not in _TEXT_SUFFIXES and not _looks_like_text(payload):
        target.write_bytes(payload)
        return target
    text = payload.decode("utf-8", errors="replace")
    if source.suffix.lower() == ".json":
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            text = (
                json.dumps(
                    _sanitize_json(value),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
    target.write_text(redact_text(text), encoding="utf-8")
    return target


def _looks_like_text(payload: bytes) -> bool:
    if b"\x00" in payload[:8192]:
        return False
    try:
        payload.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _sanitize_json(value: Any, *, key: str | None = None) -> Any:
    if key is not None and is_secret_name(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        mapping = cast(dict[Any, Any], value)
        return {str(name): _sanitize_json(item, key=str(name)) for name, item in mapping.items()}
    if isinstance(value, list):
        return [_sanitize_json(item) for item in cast(list[Any], value)]
    if isinstance(value, str):
        return redact_text(value)
    return value


def _optional_metric(
    ctx: RunContext,
    name: str,
    value: Any,
    *,
    semantic_type: str,
    unit: str,
    direction: Direction | None,
    span_id: str,
    role: ObservationRole = ObservationRole.DIAGNOSTIC,
) -> None:
    if value is None:
        return
    ctx.metric(
        name,
        value,
        semantic_type=semantic_type,
        unit=unit,
        direction=direction,
        role=role,
        span_id=span_id,
    )


def _case_input(case: Case) -> Mapping[str, Any]:
    if not isinstance(case.input, Mapping):
        raise TypeError("Terminal-Bench case input must be an object")
    return cast(Mapping[str, Any], case.input)


def _required_string(value: Mapping[str, Any], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str) or not item:
        raise TypeError(f"Terminal-Bench case input requires {name}")
    return item


def _positive_int(value: Any, *, name: str = "value") -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TypeError(f"{name} must be a positive integer")
    return value


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _require_descendant(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"trial directory escapes Harbor job: {path}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ("capture_trial",)
