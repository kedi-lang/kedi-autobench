from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Literal

from autobench import (
    CurrentSpan,
    Direction,
    InstrumentationRuntime,
    InstrumentorInfo,
    ObservationRole,
    Semantic,
    Span,
)
from autobench.tracking import (
    AssetCandidate,
    AssetProvenance,
    AssetSensitivity,
    canonical_asset_content,
)
from kedi.telemetry import (
    AssetBackend,
    Attributes,
    AttributeValue,
    CaptureKind,
    SpanKind,
    SpanLevel,
    TelemetryAsset,
    TelemetryBackend,
    TelemetryScope,
    TelemetrySpan,
)
from pydantic import ConfigDict, TypeAdapter, ValidationError

MetricKind = Literal["counter", "up_down_counter", "histogram"]

_ATTRIBUTE_ALIASES = {
    "error.type": Semantic.ERROR_TYPE,
    "gen_ai.agent.name": Semantic.AGENT_NAME,
    "gen_ai.request.model": Semantic.LLM_MODEL_REQUESTED,
    "gen_ai.response.model": Semantic.LLM_MODEL_RESPONSE,
    "gen_ai.tool.name": Semantic.TOOL_NAME,
    "kedi.operation.name": Semantic.OPERATION_NAME,
}
_USAGE_ALIASES = {
    "gen_ai.usage.input_tokens": "input_tokens",
    "gen_ai.usage.output_tokens": "output_tokens",
}
_PYDANTIC_NATIVE_OPERATIONS = frozenset({"run_agent", "chat", "call_tool"})
_ATTRIBUTE_VALUE: TypeAdapter[AttributeValue] = TypeAdapter(
    AttributeValue,
    config=ConfigDict(strict=True),
)


@dataclass(frozen=True, slots=True)
class Capture:
    content: bool = False
    source_paths: bool = True
    source_snippets: bool = False


class AutobenchSpan:
    def __init__(self, span: Span | CurrentSpan | None) -> None:
        self._span = span

    def is_recording(self) -> bool:
        if self._span is None:
            return False
        if isinstance(self._span, CurrentSpan):
            return self._span.is_recording()
        return self._span.record.ended_at is None

    def get_attribute(self, name: str) -> AttributeValue | None:
        if self._span is None:
            return None
        if isinstance(self._span, CurrentSpan):
            value = self._span.get_attribute(name)
        else:
            value = self._span.record.attributes.get(name)
        try:
            return _ATTRIBUTE_VALUE.validate_python(value)
        except ValidationError:
            return None

    def set_attribute(self, name: str, value: AttributeValue) -> None:
        if self._span is None:
            return
        canonical_name = _ATTRIBUTE_ALIASES.get(name)
        if isinstance(self._span, CurrentSpan):
            self._span.set_attribute(name, value)
            if canonical_name is not None:
                self._span.set_attribute(canonical_name, value)
            usage_name = _USAGE_ALIASES.get(name)
            if usage_name is not None:
                self._span.set_usage(usage_name, value)
            return
        self._span.set_attribute(name, value)
        if canonical_name is not None:
            self._span.set_attribute(canonical_name, value)
        usage_name = _USAGE_ALIASES.get(name)
        if usage_name is not None:
            self._span.set_usage(usage_name, value)

    def add_event(self, name: str, attributes: Attributes | None = None) -> None:
        if self._span is None:
            return
        value = True if attributes is None else dict(attributes)
        if isinstance(self._span, CurrentSpan):
            self._span.event(name, value, semantic_type=Semantic.EVENT_OCCURRENCE)
        else:
            self._span.event(name, value, semantic_type=Semantic.EVENT_OCCURRENCE)

    def record_exception(self, exc: BaseException) -> None:
        if self._span is None:
            return
        if isinstance(self._span, CurrentSpan):
            self._span.record_exception(exc)
        else:
            self._span.error(exc)

    def update_name(self, name: str) -> None:
        self.set_attribute("display.name", name)


class CompositeSpan:
    def __init__(self, spans: tuple[TelemetrySpan, ...]) -> None:
        self._spans = spans

    def is_recording(self) -> bool:
        return any(span.is_recording() for span in self._spans)

    def get_attribute(self, name: str) -> AttributeValue | None:
        for span in reversed(self._spans):
            value = span.get_attribute(name)
            if value is not None:
                return value
        return None

    def set_attribute(self, name: str, value: AttributeValue) -> None:
        for span in self._spans:
            span.set_attribute(name, value)

    def add_event(self, name: str, attributes: Attributes | None = None) -> None:
        for span in self._spans:
            span.add_event(name, attributes)

    def record_exception(self, exc: BaseException) -> None:
        for span in self._spans:
            span.record_exception(exc)

    def update_name(self, name: str) -> None:
        for span in self._spans:
            span.update_name(name)


class AutobenchBackend:
    def __init__(
        self,
        runtime: InstrumentationRuntime,
        info: InstrumentorInfo,
        *,
        target_version: str,
        capture: Capture,
    ) -> None:
        self._runtime = runtime
        self._info = info
        self._target_version = target_version
        self._capture = capture

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
        source_attributes = dict(attributes or {})
        canonical_attributes: dict[str, AttributeValue] = {
            **source_attributes,
            "kedi.scope": scope,
            "kedi.span.level": level,
            "kedi.span.kind": kind,
            Semantic.OPERATION_NAME: operation,
        }
        for source_name, canonical_name in _ATTRIBUTE_ALIASES.items():
            value = source_attributes.get(source_name)
            if value is not None:
                canonical_attributes[canonical_name] = value
        span = self._runtime.span(
            self._info,
            f"kedi.{operation}",
            kind=self._span_kind(operation, scope),
            attributes=canonical_attributes,
            tags={"kedi.display_name": name},
            target_version=self._target_version,
            suppression_keys=("kedi", scope),
        )
        if span is None:
            yield AutobenchSpan(None)
            return
        with span:
            self._discover_assets(span, source_attributes)
            try:
                yield AutobenchSpan(span)
            except BaseException as error:
                span.error(error)
                raise

    def current_span(self) -> TelemetrySpan:
        return AutobenchSpan(
            self._runtime.current_span(
                self._info,
                suppression_keys=("kedi",),
            )
        )

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
        self._metric("counter", scope, name, value, unit, attributes)

    def add_up_down_counter(
        self,
        *,
        scope: TelemetryScope,
        name: str,
        value: float,
        unit: str,
        attributes: Attributes | None,
    ) -> None:
        self._metric("up_down_counter", scope, name, value, unit, attributes)

    def record_histogram(
        self,
        *,
        scope: TelemetryScope,
        name: str,
        value: float,
        unit: str,
        attributes: Attributes | None,
    ) -> None:
        self._metric("histogram", scope, name, value, unit, attributes)

    def owns_native_spans(self, adapter_shortname: str, operation: str) -> bool:
        return (
            adapter_shortname == "pydantic"
            and operation in _PYDANTIC_NATIVE_OPERATIONS
            and self._runtime.is_installed("autobench.pydantic_ai")
        )

    def capture_enabled(self, kind: CaptureKind) -> bool:
        if kind == "content":
            return self._capture.content
        if kind == "source_path":
            return self._capture.source_paths
        return self._capture.source_snippets

    def record_asset(self, asset: TelemetryAsset) -> None:
        sensitivity = {
            "public": AssetSensitivity.PUBLIC,
            "internal": AssetSensitivity.INTERNAL,
            "sensitive": AssetSensitivity.SENSITIVE,
        }[asset.sensitivity]
        current = self._runtime.current_span(self._info)
        canonical_content = canonical_asset_content(
            {
                "metadata": dict(asset.metadata),
                "content": (
                    asset.content
                    if asset.content is not None
                    else {"omitted": True, "sha256": asset.content_fingerprint}
                ),
            }
        )
        self._runtime.asset(
            self._info,
            AssetCandidate(
                kind=asset.kind,
                local_id=asset.local_id,
                name=asset.name,
                source_locator=asset.source_locator,
                canonical_content=canonical_content,
                content_fingerprint=asset.content_fingerprint,
                metadata={"descriptor": canonical_asset_content(dict(asset.metadata))},
                provenance=AssetProvenance(
                    system="kedi",
                    key=asset.kind,
                    instrumentor=self._info.id,
                    instrumented_library_version=self._target_version,
                ),
                sensitivity=sensitivity,
            ),
            span_id=None if current is None else current.id,
        )

    def _metric(
        self,
        kind: MetricKind,
        scope: TelemetryScope,
        name: str,
        value: float,
        unit: str,
        attributes: Attributes | None,
    ) -> None:
        tags = dict(attributes or {})
        tags["kedi.scope"] = scope
        tags["kedi.metric.kind"] = kind
        semantic_type = name
        direction = None
        if name.endswith(".duration"):
            semantic_type = Semantic.TIME_LATENCY
            direction = Direction.MINIMIZE
        elif name == "gen_ai.client.token.usage":
            semantic_type = (
                Semantic.LLM_TOKENS_INPUT
                if tags.get("gen_ai.token.type") == "input"
                else Semantic.LLM_TOKENS_OUTPUT
            )
        elif name == "kedi.tool.calls":
            semantic_type = Semantic.TOOL_CALL_COUNT
        elif name == "kedi.approval.requests":
            semantic_type = Semantic.APPROVAL_COUNT
        elif name.endswith((".count", ".calls", ".invocations")):
            semantic_type = Semantic.OPERATION_COUNT
        elif name.endswith(".active"):
            semantic_type = Semantic.OPERATION_PARALLELISM
        self._runtime.metric(
            self._info,
            name,
            value,
            semantic_type=semantic_type,
            unit=unit,
            direction=direction,
            role=ObservationRole.DIAGNOSTIC,
            tags=tags,
            suppression_keys=("kedi", scope),
        )

    def _span_kind(self, operation: str, scope: TelemetryScope) -> str:
        if operation == "run_agent":
            return "agent"
        if operation == "chat":
            return "llm"
        if operation == "call_tool":
            return "tool"
        if operation in {"parse_program", "compile_program", "compile_module"}:
            return "parser"
        if operation.startswith("approval"):
            return "approval"
        if scope == "artifacts":
            return "storage"
        if operation in {"run_program", "dynamic_workflow", "run_subagent"}:
            return "workflow"
        return "custom"

    def _discover_assets(
        self,
        span: Span,
        attributes: Mapping[str, AttributeValue],
    ) -> None:
        source_path = attributes.get("code.file.path")
        if isinstance(source_path, str):
            self._runtime.asset(
                self._info,
                AssetCandidate(
                    kind="source_identity",
                    local_id=source_path,
                    name=source_path,
                    source_locator=f"kedi:source:{source_path}",
                    canonical_content={"path": source_path},
                    provenance=AssetProvenance(
                        system="kedi",
                        key="source_identity",
                        instrumentor=self._info.id,
                        instrumented_library_version=self._target_version,
                    ),
                    sensitivity=AssetSensitivity.INTERNAL,
                ),
                span_id=span.id,
            )


class CompositeBackend:
    def __init__(self, previous: TelemetryBackend, autobench: AutobenchBackend) -> None:
        self._previous = previous
        self._autobench = autobench

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
        with ExitStack() as stack:
            previous = stack.enter_context(
                self._previous.start_span(
                    scope=scope,
                    name=name,
                    operation=operation,
                    kind=kind,
                    level=level,
                    attributes=attributes,
                )
            )
            autobench = stack.enter_context(
                self._autobench.start_span(
                    scope=scope,
                    name=name,
                    operation=operation,
                    kind=kind,
                    level=level,
                    attributes=attributes,
                )
            )
            yield CompositeSpan((previous, autobench))

    def current_span(self) -> TelemetrySpan:
        return CompositeSpan((self._previous.current_span(), self._autobench.current_span()))

    def enabled(self, scope: TelemetryScope, level: SpanLevel) -> bool:
        return self._previous.enabled(scope, level) or self._autobench.enabled(scope, level)

    def add_counter(
        self,
        *,
        scope: TelemetryScope,
        name: str,
        value: float,
        unit: str,
        attributes: Attributes | None,
    ) -> None:
        self._previous.add_counter(
            scope=scope, name=name, value=value, unit=unit, attributes=attributes
        )
        self._autobench.add_counter(
            scope=scope, name=name, value=value, unit=unit, attributes=attributes
        )

    def add_up_down_counter(
        self,
        *,
        scope: TelemetryScope,
        name: str,
        value: float,
        unit: str,
        attributes: Attributes | None,
    ) -> None:
        self._previous.add_up_down_counter(
            scope=scope, name=name, value=value, unit=unit, attributes=attributes
        )
        self._autobench.add_up_down_counter(
            scope=scope, name=name, value=value, unit=unit, attributes=attributes
        )

    def record_histogram(
        self,
        *,
        scope: TelemetryScope,
        name: str,
        value: float,
        unit: str,
        attributes: Attributes | None,
    ) -> None:
        self._previous.record_histogram(
            scope=scope, name=name, value=value, unit=unit, attributes=attributes
        )
        self._autobench.record_histogram(
            scope=scope, name=name, value=value, unit=unit, attributes=attributes
        )

    def owns_native_spans(self, adapter_shortname: str, operation: str) -> bool:
        return self._previous.owns_native_spans(
            adapter_shortname, operation
        ) or self._autobench.owns_native_spans(adapter_shortname, operation)

    def capture_enabled(self, kind: CaptureKind) -> bool:
        return self._previous.capture_enabled(kind) or self._autobench.capture_enabled(kind)

    def record_asset(self, asset: TelemetryAsset) -> None:
        if isinstance(self._previous, AssetBackend):
            self._previous.record_asset(asset)
        self._autobench.record_asset(asset)


__all__ = ("AutobenchBackend", "Capture", "CompositeBackend")
