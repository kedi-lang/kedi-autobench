from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, cast

from autobench import Benchmark, Case, FactorValue, FileRecorder, Variant, run_benchmark_spec

from kedi_autobench.terminal_bench.models import HarborTrialResult, TrialLocation

_CASE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")
DEFAULT_MAX_EVIDENCE_FILE_BYTES = 20_000_000


def record_harbor_job(
    job_dir: Path,
    output_dir: Path,
    *,
    max_evidence_file_bytes: int = DEFAULT_MAX_EVIDENCE_FILE_BYTES,
    concurrency: int = 4,
) -> Path:
    return asyncio.run(
        record_harbor_job_async(
            job_dir,
            output_dir,
            max_evidence_file_bytes=max_evidence_file_bytes,
            concurrency=concurrency,
        )
    )


async def record_harbor_job_async(
    job_dir: Path,
    output_dir: Path,
    *,
    max_evidence_file_bytes: int = DEFAULT_MAX_EVIDENCE_FILE_BYTES,
    concurrency: int = 4,
) -> Path:
    if max_evidence_file_bytes <= 0:
        raise ValueError("max_evidence_file_bytes must be positive")
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    job_dir = job_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    locations = discover_trials(job_dir)
    manifest = _load_json_if_present(job_dir / "kedi-manifest.json")
    variant = _variant(manifest, max_evidence_file_bytes=max_evidence_file_bytes)
    cases: list[Case | dict[str, Any]] = [
        Case(
            id=location.case_id,
            input={
                "job_dir": str(location.job_dir),
                "trial_dir": str(location.trial_dir),
                "trial_name": location.trial_name,
                "task_name": location.task_name,
            },
            metadata={"source": "harbor", "dataset": "terminal-bench@2.1"},
        )
        for location in locations
    ]
    benchmark = (
        Benchmark("kedi-terminal-bench-2.1")
        .description(
            "Import official Harbor Terminal-Bench trials as immutable Autobench evidence."
        )
        .dataset(
            cases,
            dataset_id="terminal-bench",
            version="2.1",
            metadata={
                "harbor_job": job_dir.name,
                "kedi_manifest_digest": manifest.get("material_digest"),
            },
        )
        .variants([variant])
        .task("kedi_autobench.terminal_bench.task:capture_trial")
    )
    source_files = tuple(
        path
        for name in ("config.json", "kedi-manifest.json", "lock.json", "result.json")
        if (path := job_dir / name).is_file()
    )
    recorder = FileRecorder(
        output_dir,
        source_files=source_files,
        path_root=job_dir,
        durability="synced",
    )
    await run_benchmark_spec(
        benchmark.to_spec(),
        concurrency_limit=min(concurrency, len(cases)),
        recorder=recorder,
    )
    return output_dir


def discover_trials(job_dir: Path) -> tuple[TrialLocation, ...]:
    job_dir = job_dir.expanduser().resolve()
    if not (job_dir / "result.json").is_file():
        raise FileNotFoundError(f"Harbor job result is missing: {job_dir / 'result.json'}")
    locations: list[TrialLocation] = []
    case_ids: set[str] = set()
    for result_path in sorted(job_dir.glob("*/result.json")):
        trial_dir = result_path.parent.resolve()
        trial = HarborTrialResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        case_id = _unique_case_id(trial.trial_name, case_ids)
        case_ids.add(case_id)
        locations.append(
            TrialLocation(
                case_id=case_id,
                trial_name=trial.trial_name,
                task_name=trial.task_name,
                job_dir=job_dir,
                trial_dir=trial_dir,
            )
        )
    if not locations:
        raise ValueError(f"Harbor job has no recorded trials: {job_dir}")
    return tuple(locations)


def _variant(
    manifest: dict[str, Any],
    *,
    max_evidence_file_bytes: int,
) -> Variant:
    digest = manifest.get("material_digest")
    suffix = str(digest)[:12] if digest else "unmanifested"
    raw_policy = manifest.get("policy")
    policy = cast(dict[str, Any], raw_policy) if isinstance(raw_policy, dict) else {}
    return Variant(
        id=f"kedi-{suffix}",
        factors=[
            FactorValue(name="model", value=manifest.get("model", "unknown")),
            FactorValue(name="adapter", value=manifest.get("adapter", "unknown")),
            FactorValue(name="effort", value=manifest.get("effort")),
            FactorValue(
                name="harness_policy",
                value=json.dumps(policy, ensure_ascii=False, sort_keys=True),
            ),
            FactorValue(name="max_evidence_file_bytes", value=max_evidence_file_bytes),
        ],
    )


def _unique_case_id(value: str, existing: set[str]) -> str:
    base = _CASE_COMPONENT.sub("-", value.strip()).strip("-").lower() or "trial"
    candidate = base
    index = 2
    while candidate in existing:
        candidate = f"{base}-{index}"
        index += 1
    return candidate


def _load_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return cast(dict[str, Any], value)


__all__ = (
    "DEFAULT_MAX_EVIDENCE_FILE_BYTES",
    "discover_trials",
    "record_harbor_job",
    "record_harbor_job_async",
)
