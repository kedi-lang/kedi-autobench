from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TimingInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return max(0.0, (self.finished_at - self.started_at).total_seconds())


class ExceptionInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    exception_type: str
    exception_message: str = ""


class AgentContext(BaseModel):
    model_config = ConfigDict(extra="allow")

    n_input_tokens: int | None = None
    n_cache_tokens: int | None = None
    n_output_tokens: int | None = None
    cost_usd: float | None = None
    metadata: dict[str, Any] | None = None


class VerifierResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    rewards: dict[str, float | int] | None = None


class HarborTrialResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_name: str
    trial_name: str
    trial_uri: str | None = None
    source: str | None = None
    task_checksum: str | None = None
    agent_result: AgentContext | None = None
    verifier_result: VerifierResult | None = None
    exception_info: ExceptionInfo | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    environment_setup: TimingInfo | None = None
    agent_setup: TimingInfo | None = None
    agent_execution: TimingInfo | None = None
    verifier: TimingInfo | None = None

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return max(0.0, (self.finished_at - self.started_at).total_seconds())


class KediTrialResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    state: str
    adapter: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    verification: dict[str, Any] = Field(default_factory=dict)
    terminal: dict[str, Any] = Field(default_factory=dict)
    policy: dict[str, Any] = Field(default_factory=dict)
    failure_phase: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class CapturedTrial(BaseModel):
    task_name: str
    trial_name: str
    harbor_exception: str | None
    rewards: dict[str, float | int]
    kedi_state: str | None
    usage: dict[str, Any]
    evidence_file_count: int
    evidence_bytes: int
    skipped_file_count: int


class EvidenceFile(BaseModel):
    path: str
    source: str
    byte_count: int
    sha256: str
    attached: bool
    reason: str | None = None


class TrialLocation(BaseModel):
    case_id: str
    trial_name: str
    task_name: str
    job_dir: Path
    trial_dir: Path


__all__ = (
    "AgentContext",
    "CapturedTrial",
    "EvidenceFile",
    "ExceptionInfo",
    "HarborTrialResult",
    "KediTrialResult",
    "TimingInfo",
    "TrialLocation",
    "VerifierResult",
)
