from __future__ import annotations

import subprocess
import traceback
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from kedi.integrations.harbor.records import atomic_write_json, redact_text

from kedi_autobench.terminal_bench.capture import record_harbor_job

Recorder = Callable[..., Path]


def run_then_record(
    command: Sequence[str],
    *,
    job_dir: Path,
    output_dir: Path,
    max_evidence_file_bytes: int = 512_000_000,
    concurrency: int = 4,
    recorder: Recorder = record_harbor_job,
) -> int:
    if not command:
        raise ValueError("Harbor command must not be empty")
    completed = subprocess.run(tuple(command), check=False)
    if job_dir.is_dir():
        try:
            recorder(
                job_dir,
                output_dir,
                max_evidence_file_bytes=max_evidence_file_bytes,
                concurrency=concurrency,
            )
        except Exception as exc:  # noqa: BLE001 - capture must not change Harbor's result
            _write_capture_error(output_dir, exc)
    return completed.returncode


def _write_capture_error(output_dir: Path, exc: Exception) -> None:
    path = output_dir.with_name(f"{output_dir.name}.capture-error.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "captured_at": datetime.now(UTC).isoformat(),
        "error_type": type(exc).__name__,
        "error_message": redact_text(str(exc)),
        "traceback": redact_text("".join(traceback.format_exception(exc))),
    }
    atomic_write_json(path, payload)


__all__ = ("run_then_record",)
