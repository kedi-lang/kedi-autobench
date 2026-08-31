from __future__ import annotations

from importlib.metadata import version

from autobench import (
    Compatibility,
    InstrumentationError,
    InstrumentationHandle,
    InstrumentationRuntime,
    InstrumentorCapabilities,
    InstrumentorInfo,
)
from autobench.protocol import AbstractionLayer, CaptureMechanism
from kedi.telemetry import (
    get_backend,
    install_backend,
    restore_backend_if_current,
)

from kedi_autobench.backend import AutobenchBackend, Capture, CompositeBackend

_VERSION = "0.1.0"


class Kedi:
    """Multiplex Kedi telemetry into ABP without replacing an existing backend."""

    def __init__(
        self,
        *,
        capture_content: bool = False,
        capture_source_paths: bool = True,
        capture_source_snippets: bool = False,
    ) -> None:
        self._capture = Capture(
            content=capture_content,
            source_paths=capture_source_paths,
            source_snippets=capture_source_snippets,
        )
        self._info = InstrumentorInfo(
            id="kedi.autobench",
            version=_VERSION,
            target_distribution="kedi",
            supported_versions=">=0.4,<0.5",
            mechanism=CaptureMechanism.CALLBACK,
            layer=AbstractionLayer.FRAMEWORK,
            span_kinds=(
                "workflow",
                "agent",
                "llm",
                "tool",
                "parser",
                "approval",
                "storage",
            ),
            semantic_families=(
                "operation",
                "workflow",
                "agent",
                "llm",
                "tool",
                "approval",
                "artifact",
                "time",
            ),
            source_convention="kedi-telemetry",
            source_convention_version="0.4",
            capabilities=InstrumentorCapabilities.model_validate(
                {
                    "sync": True,
                    "async": True,
                    "streaming": True,
                    "native_hooks": True,
                    "asset_discovery": True,
                    "asset_kinds": (
                        "agent_profile",
                        "prompt",
                        "schema",
                        "skill",
                        "source_identity",
                        "template",
                        "tool",
                    ),
                }
            ),
        )

    @property
    def info(self) -> InstrumentorInfo:
        return self._info

    def check(self) -> Compatibility:
        return Compatibility.compatible(target_version=version("kedi"))

    def install(self, runtime: InstrumentationRuntime) -> InstrumentationHandle:
        target_version = version("kedi")
        previous = get_backend()
        installed = CompositeBackend(
            previous,
            AutobenchBackend(
                runtime,
                self.info,
                target_version=target_version,
                capture=self._capture,
            ),
        )
        replaced = install_backend(installed)
        if replaced is not previous:
            restore_backend_if_current(expected=installed, replacement=replaced)
            raise InstrumentationError("Kedi telemetry backend changed during installation")

        def close() -> None:
            restore_backend_if_current(expected=installed, replacement=previous)

        return InstrumentationHandle(close, info=self.info)


__all__ = ("Kedi",)
