from __future__ import annotations

import asyncio
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

import autobench.runtime.pipeline as pipeline_module
import pytest
from autobench import (
    BenchmarkInfo,
    BenchmarkSpec,
    CaptureLevel,
    CapturePolicy,
    Case,
    EndReason,
    FileRecorder,
    InstrumentationError,
    InstrumentationManager,
    RunContext,
    Semantic,
    TaskResult,
    TaskSpec,
    TaskStatus,
    Variant,
    finalize_staging,
    inspect_staging,
    recover_staging,
    replay_experiment,
    run_benchmark_spec,
)
from autobench.data.datasets import DatasetSpec
from autobench.instrumentation.pydantic_ai import PydanticAI
from autobench.runtime.instrumentation import reset_active_run_context, set_active_run_context
from kedi.agent_adapter import PydanticAdapter
from kedi.agent_adapter.telemetry import record_conversation_assets
from kedi.agent_profile import AgentProfile
from kedi.lang.compiler import compile_program
from kedi.lang.parser import parse_program
from kedi.telemetry import (
    Attributes,
    AttributeValue,
    CaptureKind,
    SpanKind,
    SpanLevel,
    TelemetryAsset,
    TelemetryScope,
    TelemetrySpan,
    add_counter,
    add_up_down_counter,
    current_span,
    get_backend,
    install_backend,
    record_asset,
    record_histogram,
    runtime_span,
    span,
)
from pydantic_ai.models.test import TestModel

import kedi_autobench.instrumentor as instrumentor_module
from kedi_autobench import Kedi
from kedi_autobench.backend import AutobenchSpan, CompositeSpan


@dataclass(frozen=True, slots=True)
class RecordedMetric:
    kind: str
    scope: TelemetryScope
    name: str
    value: float
    unit: str
    attributes: Attributes | None


class RecordingSpan:
    def __init__(self, name: str, attributes: Attributes | None = None) -> None:
        self.name = name
        self.attributes: dict[str, AttributeValue] = dict(attributes or {})
        self.events: list[tuple[str, Attributes | None]] = []
        self.exceptions: list[BaseException] = []
        self.recording = True

    def is_recording(self) -> bool:
        return self.recording

    def get_attribute(self, name: str) -> AttributeValue | None:
        return self.attributes.get(name)

    def set_attribute(self, name: str, value: AttributeValue) -> None:
        self.attributes[name] = value

    def add_event(self, name: str, attributes: Attributes | None = None) -> None:
        self.events.append((name, attributes))

    def record_exception(self, exc: BaseException) -> None:
        self.exceptions.append(exc)

    def update_name(self, name: str) -> None:
        self.name = name


class SilentSpan(RecordingSpan):
    def __init__(self) -> None:
        super().__init__("silent")
        self.recording = False


class RecordingBackend:
    def __init__(self) -> None:
        self.started: list[RecordingSpan] = []
        self.finished: list[RecordingSpan] = []
        self.metrics: list[RecordedMetric] = []
        self.assets: list[TelemetryAsset] = []
        self.native_owners: set[tuple[str, str]] = set()
        self.capture: set[CaptureKind] = set()
        self._current: ContextVar[RecordingSpan | None] = ContextVar(
            f"recording_backend_{id(self)}",
            default=None,
        )

    @contextmanager
    def start_span(
        self,
        *,
        scope: TelemetryScope,
        name: str,
        operation: str,
        kind: SpanKind,
        level: SpanLevel,
        attributes: Attributes | None,
    ) -> Generator[TelemetrySpan, None, None]:
        del scope, operation, kind, level
        active = RecordingSpan(name, attributes)
        self.started.append(active)
        token = self._current.set(active)
        try:
            yield active
        finally:
            active.recording = False
            self.finished.append(active)
            self._current.reset(token)

    def current_span(self) -> TelemetrySpan:
        return self._current.get() or SilentSpan()

    def enabled(self, scope: TelemetryScope, level: SpanLevel) -> bool:
        del scope, level
        return True

    def add_counter(
        self,
        *,
        scope: TelemetryScope,
        name: str,
        value: float,
        unit: str,
        attributes: Attributes | None,
    ) -> None:
        self.metrics.append(RecordedMetric("counter", scope, name, value, unit, attributes))

    def add_up_down_counter(
        self,
        *,
        scope: TelemetryScope,
        name: str,
        value: float,
        unit: str,
        attributes: Attributes | None,
    ) -> None:
        self.metrics.append(RecordedMetric("gauge", scope, name, value, unit, attributes))

    def record_histogram(
        self,
        *,
        scope: TelemetryScope,
        name: str,
        value: float,
        unit: str,
        attributes: Attributes | None,
    ) -> None:
        self.metrics.append(RecordedMetric("histogram", scope, name, value, unit, attributes))

    def owns_native_spans(self, adapter_shortname: str, operation: str) -> bool:
        return (adapter_shortname, operation) in self.native_owners

    def capture_enabled(self, kind: CaptureKind) -> bool:
        return kind in self.capture

    def record_asset(self, asset: TelemetryAsset) -> None:
        self.assets.append(asset)


def run_context() -> RunContext:
    return RunContext(
        benchmark_id="kedi",
        experiment_id="kedi-experiment",
        run_id="kedi-run",
        capture_policy=CapturePolicy(default_level=CaptureLevel.FULL),
        case=Case(id="case"),
        variant=Variant(id="variant"),
    )


def benchmark_spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        benchmark=BenchmarkInfo(id="kedi"),
        capture=CapturePolicy(default_level=CaptureLevel.FULL),
        dataset=DatasetSpec(cases=[Case(id="case")]),
        task=TaskSpec(kind="python", target="unused:run"),
        variants=[Variant(id="variant")],
    )


def test_instrumentor_contract_composes_restores_and_detects_install_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = RecordingBackend()
    install_backend(existing)
    manager = InstrumentationManager()
    instrumentor = Kedi(capture_content=True, capture_source_snippets=True)

    assert instrumentor.info.id == "kedi.autobench"
    assert instrumentor.info.capabilities.asset_discovery is True
    assert set(instrumentor.info.capabilities.asset_kinds) == {
        "agent_profile",
        "prompt",
        "schema",
        "skill",
        "source_identity",
        "template",
        "tool",
    }
    assert manager.check(instrumentor).installable is True

    handle = manager.install(instrumentor)
    installed = get_backend()
    assert installed is not existing
    assert installed.capture_enabled("content") is True
    assert installed.capture_enabled("source_path") is True
    assert installed.capture_enabled("source_snippet") is True
    assert installed.owns_native_spans("pydantic", "run_agent") is False
    existing.native_owners.add(("custom", "chat"))
    existing.capture.add("content")
    assert installed.owns_native_spans("custom", "chat") is True
    assert installed.capture_enabled("content") is True

    newer = RecordingBackend()
    install_backend(newer)
    handle.close()
    assert get_backend() is newer
    manager.close()

    original_install = instrumentor_module.install_backend

    def racing_install(backend: RecordingBackend) -> RecordingBackend:
        del backend
        replacement = RecordingBackend()
        original_install(replacement)
        return replacement

    monkeypatch.setattr(instrumentor_module, "install_backend", racing_install)
    with pytest.raises(InstrumentationError, match="changed during installation"):
        InstrumentationManager().install(Kedi())


def test_backend_noops_without_a_run_and_maps_every_span_kind() -> None:
    existing = RecordingBackend()
    install_backend(existing)
    context = run_context()
    manager = InstrumentationManager()
    handle = manager.install(Kedi())

    with span("runtime", "outside", "outside") as outside:
        assert outside.is_recording() is True
    assert context.spans == []

    token = set_active_run_context(context)
    try:
        expected: tuple[tuple[TelemetryScope, str, str], ...] = (
            ("agent", "run_agent", "agent"),
            ("agent", "chat", "llm"),
            ("agent", "call_tool", "tool"),
            ("runtime", "parse_program", "parser"),
            ("agent", "approval_wait", "approval"),
            ("artifacts", "store", "storage"),
            ("runtime", "dynamic_workflow", "workflow"),
            ("runtime", "other", "custom"),
        )
        for scope, operation, _kind in expected:
            with span(scope, operation, operation):
                pass
    finally:
        reset_active_run_context(token)
        handle.close()
        manager.close()

    trace = context.finalize()
    kinds = {
        record.operation: record.kind
        for record in trace.spans
        if record.operation.startswith("kedi.")
    }
    assert kinds == {f"kedi.{operation}": kind for _, operation, kind in expected}


def test_backend_maps_spans_metrics_assets_events_and_errors() -> None:
    existing = RecordingBackend()
    install_backend(existing)
    context = run_context()
    manager = InstrumentationManager()
    handle = manager.install(Kedi())
    token = set_active_run_context(context)
    try:
        with runtime_span(
            "kedi run fixture",
            "run_program",
            attributes={
                "code.file.path": "/project/main.kedi",
                "kedi.agent.profile": "reviewer",
                "gen_ai.agent.name": "reviewer",
                "gen_ai.request.model": "test:model",
                "gen_ai.tool.name": "review",
                "kedi.tool.source": "skill",
                "kedi.tool.risk": "low",
            },
        ) as active:
            assert active.is_recording() is True
            assert active.get_attribute("agent.name") == "reviewer"
            active.set_attribute("gen_ai.usage.input_tokens", 7)
            active.set_attribute("gen_ai.usage.output_tokens", 3)
            active.update_name("renamed run")
            active.add_event("ready", {"step": 1})
            current_span().set_attribute("kedi.result.type", "str")
            add_counter("runtime", "kedi.run.count")
            add_up_down_counter("runtime", "kedi.run.active", 1)
            record_histogram("runtime", "kedi.run.duration", 0.5, unit="s")
            record_histogram(
                "agent",
                "gen_ai.client.token.usage",
                7,
                unit="token",
                attributes={"gen_ai.token.type": "input"},
            )
            record_histogram(
                "agent",
                "gen_ai.client.token.usage",
                3,
                unit="token",
                attributes={"gen_ai.token.type": "output"},
            )
            add_counter("agent", "kedi.tool.calls")
            add_counter("agent", "kedi.approval.requests")
            add_counter("artifacts", "kedi.artifact.created")
            record_asset(
                TelemetryAsset(
                    kind="agent_profile",
                    local_id="reviewer",
                    name="reviewer",
                    source_locator="kedi:profile:reviewer",
                    content_fingerprint="a" * 64,
                    content={"name": "reviewer"},
                )
            )
            record_asset(
                TelemetryAsset(
                    kind="skill",
                    local_id="review",
                    name="review",
                    source_locator="kedi:skill:review",
                    content_fingerprint="b" * 64,
                    content={"name": "review"},
                )
            )
    finally:
        reset_active_run_context(token)
        handle.close()
        manager.close()

    trace = context.finalize()
    [root] = [record for record in trace.spans if record.operation == "kedi.run_program"]
    assert root.kind == "workflow"
    assert root.attributes[Semantic.AGENT_NAME] == "reviewer"
    assert root.attributes[Semantic.LLM_MODEL_REQUESTED] == "test:model"
    assert root.attributes["display.name"] == "renamed run"
    assert root.usage == {"input_tokens": 7, "output_tokens": 3}
    assert any(event.name == "ready" for event in root.events)
    assert {asset.source_locator for asset in context.asset_uses} == {
        "kedi:profile:reviewer",
        "kedi:skill:review",
        "kedi:source:/project/main.kedi",
    }
    semantics = {observation.semantic_type for observation in context.observations}
    assert {
        Semantic.OPERATION_COUNT,
        Semantic.OPERATION_PARALLELISM,
        Semantic.TIME_LATENCY,
        Semantic.LLM_TOKENS_INPUT,
        Semantic.LLM_TOKENS_OUTPUT,
        Semantic.TOOL_CALL_COUNT,
        Semantic.APPROVAL_COUNT,
        "kedi.artifact.created",
    } <= semantics
    assert len(existing.finished) == 1
    assert {asset.kind for asset in existing.assets} == {"agent_profile", "skill"}
    assert {
        "kedi.run.count",
        "kedi.run.active",
        "kedi.run.duration",
        "gen_ai.client.token.usage",
        "kedi.tool.calls",
        "kedi.approval.requests",
        "kedi.artifact.created",
    } <= {metric.name for metric in existing.metrics}


def test_asset_delivery_does_not_require_the_previous_backend_to_support_assets() -> None:
    context = run_context()
    manager = InstrumentationManager()
    handle = manager.install(Kedi(capture_content=True))
    token = set_active_run_context(context)
    try:
        with context.span("asset owner"):
            record_asset(
                TelemetryAsset(
                    kind="template",
                    local_id="template",
                    name="template",
                    source_locator="kedi:template:template",
                    content_fingerprint="c" * 64,
                    content="Hello",
                )
            )
    finally:
        reset_active_run_context(token)
        handle.close()
        manager.close()

    context.finalize()
    assert [asset.source_locator for asset in context.asset_uses] == ["kedi:template:template"]


def test_behavioral_assets_deduplicate_and_version_without_leaking_content() -> None:
    context = run_context()
    manager = InstrumentationManager()
    handle = manager.install(Kedi(capture_content=False))
    token = set_active_run_context(context)
    profile = AgentProfile(name="versioned-profile", model="test:model")
    try:
        with context.span("asset owner"):
            for prompt, schema in (
                ("first secret prompt", {"answer": "str"}),
                ("first secret prompt", {"answer": "str"}),
                ("second secret prompt", {"answer": "str", "reason": "str"}),
            ):
                record_conversation_assets(
                    prompt=prompt,
                    instructions=None,
                    profile=profile,
                    tools=(),
                    output_schema=schema,
                    call_kind="produce",
                    source_path="workflow.kedi",
                    source_line=7,
                    source_snippet=None,
                )
    finally:
        reset_active_run_context(token)
        handle.close()
        manager.close()

    context.finalize()
    template_locator = "kedi:template:workflow.kedi:7:produce"
    schema_locator = "kedi:schema:template:workflow.kedi:7:produce:output"
    template_uses = [
        asset for asset in context.asset_uses if asset.source_locator == template_locator
    ]
    schema_uses = [asset for asset in context.asset_uses if asset.source_locator == schema_locator]
    assert len(template_uses) == 2
    assert len({asset.version for asset in template_uses}) == 2
    assert len(schema_uses) == 2
    assert len({asset.version for asset in schema_uses}) == 2

    recorded_template = manager.runtime.registry.resolve_locator(template_locator)
    rendered = repr(recorded_template)
    assert "first secret prompt" not in rendered
    assert "second secret prompt" not in rendered


def test_span_facades_cover_noop_current_direct_and_composite_behavior() -> None:
    noop = AutobenchSpan(None)
    assert noop.is_recording() is False
    assert noop.get_attribute("missing") is None
    noop.set_attribute("name", "value")
    noop.add_event("event")
    noop.record_exception(RuntimeError("ignored"))
    noop.update_name("ignored")

    context = run_context()
    token = set_active_run_context(context)
    try:
        with context.span("direct") as raw:
            direct = AutobenchSpan(raw)
            direct.set_attribute("scalar", 1)
            direct.set_attribute("sequence", ("a", "b"))
            raw.record.attributes["invalid"] = {"not": "an attribute value"}
            assert direct.get_attribute("scalar") == 1
            assert direct.get_attribute("sequence") == ("a", "b")
            assert direct.get_attribute("invalid") is None
            direct.add_event("direct event")
            direct.record_exception(ValueError("direct"))
            direct.update_name("direct display")
            current = InstrumentationManager().runtime.current_span(Kedi().info)
            assert current is not None
            facade = AutobenchSpan(current)
            facade.set_attribute("current", True)
            facade.set_attribute("gen_ai.agent.name", "current-agent")
            facade.set_attribute("gen_ai.usage.input_tokens", 2)
            facade.add_event("current event", {"active": True})
            facade.update_name("current display")
            assert facade.get_attribute("current") is True

            previous = RecordingSpan("previous")
            composite = CompositeSpan((previous, facade))
            assert composite.is_recording() is True
            composite.set_attribute("shared", "yes")
            composite.add_event("shared event")
            error = RuntimeError("shared")
            composite.record_exception(error)
            composite.update_name("shared name")
            assert composite.get_attribute("shared") == "yes"
            previous.set_attribute("previous-only", "fallback")
            assert composite.get_attribute("previous-only") == "fallback"
            assert previous.name == "shared name"
    finally:
        reset_active_run_context(token)

    context.finalize()
    assert direct.is_recording() is False
    assert facade.is_recording() is False
    empty = CompositeSpan(())
    assert empty.is_recording() is False
    assert empty.get_attribute("missing") is None


async def test_real_kedi_pydantic_run_is_durable_without_duplicate_native_spans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = RecordingBackend()
    install_backend(existing)

    async def task(
        target: str,
        *,
        ctx: RunContext,
        case: Case,
        search_paths: tuple[str, ...] = (),
    ) -> TaskResult:
        del target, case, search_paths
        token = set_active_run_context(ctx)
        try:
            with runtime_span("kedi dogfood", "run_program"):
                program = parse_program(">> Say hello [answer]\n= <answer>")
                adapter = PydanticAdapter(
                    TestModel(call_tools=[], custom_output_args={"answer": "hello"})
                )
                runtime = compile_program(program, adapter=adapter)
                output = runtime.run_main()
                await ctx.checkpoint("kedi-complete")
            return TaskResult(output=output, status=TaskStatus.PASSED)
        finally:
            reset_active_run_context(token)

    monkeypatch.setattr(pipeline_module, "run_python_task", task)
    output_dir = tmp_path / "kedi-record"
    result = await run_benchmark_spec(
        benchmark_spec(),
        experiment_id="kedi-dogfood",
        recorder=FileRecorder(output_dir),
        instrumentors=(PydanticAI(), Kedi()),
    )

    trace = result.runs[0].trace
    assert trace is not None
    operations = [record.operation for record in trace.spans]
    assert operations.count("pydantic_ai.agent.run") == 1
    assert operations.count("pydantic_ai.model.request") == 1
    assert "kedi.run_agent" not in operations
    native_agent = next(
        record for record in trace.spans if record.operation == "pydantic_ai.agent.run"
    )
    assert native_agent.attributes["kedi.operation.name"] == "run_agent"
    assert native_agent.attributes["operation.name"] == "run_agent"
    assert any(record.operation == "kedi.run_program" for record in trace.spans)
    assert result.runs[0].asset_versions
    kedi_assets = [
        asset for asset in result.runs[0].asset_uses if asset.source_locator.startswith("kedi:")
    ]
    assert len(kedi_assets) == len(
        {(asset.source_locator, asset.version, asset.span_id) for asset in kedi_assets}
    )
    assert {asset.source_locator.split(":", maxsplit=2)[1] for asset in kedi_assets} >= {
        "profile",
        "schema",
        "template",
    }
    replayed = replay_experiment(output_dir)
    replayed_trace = replayed.runs[0].trace
    assert replayed_trace is not None
    assert replayed_trace.trace_id == trace.trace_id
    replayed_operations = [record.operation for record in replayed_trace.spans]
    assert replayed_operations == operations
    replayed_agent = next(
        record for record in replayed_trace.spans if record.operation == "pydantic_ai.agent.run"
    )
    assert replayed_agent.attributes["kedi.operation.name"] == "run_agent"
    assert existing.finished


async def test_kedi_cancellation_survives_staging_recovery_and_partial_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = asyncio.Event()

    async def task(
        target: str,
        *,
        ctx: RunContext,
        case: Case,
        search_paths: tuple[str, ...] = (),
    ) -> TaskResult:
        del target, case, search_paths
        token = set_active_run_context(ctx)
        try:
            with runtime_span("kedi wait", "run_program"):
                record_conversation_assets(
                    prompt="Wait for completion.",
                    instructions="Remain deterministic.",
                    profile=AgentProfile(name="cancelled-profile"),
                    tools=(),
                    output_schema=None,
                    call_kind="invoke",
                    source_path="cancelled.kedi",
                    source_line=1,
                    source_snippet=None,
                )
                entered.set()
                await asyncio.Event().wait()
            return TaskResult(output="unreachable", status=TaskStatus.PASSED)
        finally:
            reset_active_run_context(token)

    monkeypatch.setattr(pipeline_module, "run_python_task", task)
    output_dir = tmp_path / "cancelled"
    recorder = FileRecorder(output_dir)
    execution = asyncio.create_task(
        run_benchmark_spec(
            benchmark_spec(),
            experiment_id="kedi-cancelled",
            recorder=recorder,
            instrumentors=(Kedi(),),
        )
    )
    await entered.wait()
    execution.cancel()
    with pytest.raises(asyncio.CancelledError):
        await execution

    inspection = inspect_staging(recorder.staging_dir)
    assert inspection.checkpointed_run_ids
    recovered = recover_staging(recorder.staging_dir)
    recovered_trace = recovered.checkpoints[0].trace
    assert recovered_trace is not None
    assert {asset.source_locator for asset in recovered.checkpoints[0].asset_uses} >= {
        "kedi:profile:cancelled-profile",
        "kedi:prompt:cancelled-profile:instructions",
        "kedi:template:cancelled.kedi:1:invoke",
    }
    kedi_span = next(
        record for record in recovered_trace.spans if record.operation == "kedi.run_program"
    )
    assert kedi_span.end_reason is EndReason.CANCELLED
    finalize_staging(recorder.staging_dir, output_dir, allow_partial=True)
    replayed = replay_experiment(output_dir)
    assert replayed.runs[0].end_reason is EndReason.CANCELLED
