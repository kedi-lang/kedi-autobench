from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from kedi_autobench.terminal_bench.capture import (
    DEFAULT_MAX_EVIDENCE_FILE_BYTES,
    DEFAULT_MAX_EVIDENCE_TOTAL_BYTES,
    record_harbor_job,
)
from kedi_autobench.terminal_bench.runner import run_then_record
from kedi_autobench.terminal_bench.validation import validate_harbor_record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kedi-autobench-terminal-bench")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    record = subparsers.add_parser("record", help="Record an existing Harbor job.")
    _shared_options(record)

    validate = subparsers.add_parser(
        "validate",
        help="Validate a recorded job against its Harbor source without rerunning it.",
    )
    validate.add_argument("--job-dir", type=Path, required=True)
    validate.add_argument("--record-dir", type=Path, required=True)

    run = subparsers.add_parser(
        "run",
        help="Run Harbor first, then capture its completed job without changing its exit status.",
    )
    _shared_options(run)
    run.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def _shared_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument(
        "--max-evidence-file-bytes",
        type=int,
        default=DEFAULT_MAX_EVIDENCE_FILE_BYTES,
    )
    parser.add_argument(
        "--max-evidence-total-bytes",
        type=int,
        default=DEFAULT_MAX_EVIDENCE_TOTAL_BYTES,
    )
    parser.add_argument("--concurrency", type=int, default=4)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.subcommand == "record":
        record_harbor_job(
            args.job_dir,
            args.record_dir,
            max_evidence_file_bytes=args.max_evidence_file_bytes,
            max_evidence_total_bytes=args.max_evidence_total_bytes,
            concurrency=args.concurrency,
        )
        return 0
    if args.subcommand == "validate":
        validate_harbor_record(args.record_dir, job_dir=args.job_dir)
        return 0
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    return run_then_record(
        command,
        job_dir=args.job_dir,
        output_dir=args.record_dir,
        max_evidence_file_bytes=args.max_evidence_file_bytes,
        max_evidence_total_bytes=args.max_evidence_total_bytes,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
