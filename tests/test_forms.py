from __future__ import annotations

from pathlib import Path
from typing import Any

import autobench.runtime.pipeline as pipeline_module
import pytest
from autobench import (
    BenchmarkInfo,
    BenchmarkSpec,
    CaptureLevel,
    CapturePolicy,
    Case,
    Direction,
    FactorValue,
    FileRecorder,
    ObservationRole,
    RunContext,
    Semantic,
    TaskResult,
    TaskSpec,
    TaskStatus,
    Variant,
    replay_experiment,
    run_benchmark_spec,
)
from autobench.data.datasets import DatasetSpec
from autobench.reports.inference import build_hypothesis_results
from autobench.reports.models import ComparisonHypothesis, ComparisonReportSpec
from autobench.reports.trajectory import build_asset_trajectory, compare_asset_versions
from autobench.runtime.instrumentation import reset_active_run_context, set_active_run_context
from kedi.agent_adapter import PydanticAdapter
from kedi.forms import Form, FormError
from kedi.groups import form_fingerprint
from kedi.lang import compile_program, parse_program
from pydantic_ai.models.test import TestModel

from kedi_autobench import FORM_FACTOR_PREFIX, Kedi, variant_forms

PROGRAM = """@solve(problem: str) -> str:
  > evolve: planner:
    [plan] << Plan <problem>.
  = <plan>

= <solve(`"ship it"`)>
"""
GROUP = "solve.planner"
ASSET = "kedi:group:main:solve.planner"
SOURCE = "[plan] << Plan <problem>."
V2 = "[plan] << Plan <problem> in steps."
V3 = "[plan] << Plan <problem> in numbered steps."
CASES = ("a", "b", "c", "d", "e", "f")
# Cases each form solves: every generation solves more of them.
SOLVED = {"source": {"a"}, "v2": {"a", "b", "c"}, "v3": set(CASES)}


def _variant(name: str, form: dict[str, Any] | None) -> Variant:
    factors = [] if form is None else [FactorValue(name=FORM_FACTOR_PREFIX + GROUP, value=form)]
    return Variant(id=name, factors=factors)


SOURCE_VARIANT = _variant("source", None)
V2_VARIANT = _variant("v2", {"text": V2, "parent": form_fingerprint(SOURCE), "note": "steps"})
V3_VARIANT = _variant("v3", {"text": V3, "parent": form_fingerprint(V2)})


def test_variant_factors_select_forms() -> None:
    variant = Variant(
        id="v",
        factors=[
            FactorValue(name="model", value="m"),
            FactorValue(name=FORM_FACTOR_PREFIX + GROUP, value=V2),
            FactorValue(name=FORM_FACTOR_PREFIX + "lib:g", value={"text": V3, "note": "n"}),
        ],
    )

    assert variant_forms(variant) == {
        GROUP: Form(V2, origin=f"<factor kedi.form.{GROUP}>"),
        "lib:g": Form(V3, None, "n", "<factor kedi.form.lib:g>"),
    }


@pytest.mark.parametrize(
    ("value", "detail"),
    [
        (1, "must be form text or an object with 'text'"),
        ({"note": "n"}, "must be form text or an object with 'text'"),
        ({"text": V2, "body": "f.kedi"}, "unknown fields body"),
        ({"text": V2, "note": 1}, "'parent' and 'note' must be strings"),
        ({"text": V2, "parent": "abc"}, "64-character lowercase hex fingerprint"),
    ],
)
def test_invalid_form_factors_are_errors(value: Any, detail: str) -> None:
    variant = Variant(id="v", factors=[FactorValue(name=FORM_FACTOR_PREFIX + GROUP, value=value)])
    with pytest.raises(FormError, match=detail):
        variant_forms(variant)


async def _record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, generation: str, variants: list[Variant]
) -> Path:
    async def task(
        target: str, *, ctx: RunContext, case: Case, search_paths: tuple[str, ...] = ()
    ) -> TaskResult:
        del target, search_paths
        token = set_active_run_context(ctx)
        try:
            runtime = compile_program(
                parse_program(PROGRAM),
                adapter=PydanticAdapter(TestModel(call_tools=[])),
                forms=variant_forms(ctx.variant),
            )
            output = runtime.run_main()
        finally:
            reset_active_run_context(token)
        solved = case.id in SOLVED[ctx.variant.id]
        ctx.metric(
            "success",
            solved,
            semantic_type=Semantic.RESULT_SUCCESS,
            direction=Direction.MAXIMIZE,
            role=ObservationRole.OBJECTIVE,
        )
        return TaskResult(output=output, status=TaskStatus.PASSED if solved else TaskStatus.FAILED)

    monkeypatch.setattr(pipeline_module, "run_python_task", task)
    output_dir = tmp_path / generation
    await run_benchmark_spec(
        BenchmarkSpec(
            benchmark=BenchmarkInfo(id="kedi-forms"),
            capture=CapturePolicy(default_level=CaptureLevel.FULL),
            dataset=DatasetSpec(cases=[Case(id=case) for case in CASES]),
            task=TaskSpec(kind="python", target="unused:run"),
            variants=variants,
        ),
        experiment_id=generation,
        recorder=FileRecorder(output_dir),
        instrumentors=(Kedi(capture_content=True),),
    )
    return output_dir


async def test_forms_across_records_form_one_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One record per generation: the source and its first revision, then a revision of that.
    first = await _record(tmp_path, monkeypatch, "generation-1", [SOURCE_VARIANT, V2_VARIANT])
    second = await _record(tmp_path, monkeypatch, "generation-2", [V3_VARIANT])
    records = [replay_experiment(first), replay_experiment(second)]

    versions = {
        version.version: version
        for record in records
        for run in record.runs
        for version in run.asset_versions
        if version.asset_id == ASSET
    }
    by_variant = {
        run.variant_id: {use.version for use in run.asset_uses if use.asset_id == ASSET}
        for record in records
        for run in record.runs
    }
    (source,) = by_variant["source"]
    (v2,) = by_variant["v2"]
    (v3,) = by_variant["v3"]
    assert versions[source].parent_version is None
    assert versions[v2].parent_version == source
    assert versions[v3].parent_version == v2

    trajectory = build_asset_trajectory(records, asset_id=ASSET)
    assert [(step.version, step.parent_version) for step in trajectory.steps] == [
        (source, None),
        (v2, source),
        (v3, v2),
    ]
    assert [step.run_count for step in trajectory.steps] == [len(CASES)] * 3

    spec = ComparisonReportSpec(
        asset=ASSET,
        baseline="root",
        candidate="leaf",
        hypothesis=ComparisonHypothesis(metric="pass_rate"),
    )
    comparison = compare_asset_versions(
        records,
        asset_id=ASSET,
        baseline=spec.baseline,
        candidate=spec.candidate,
        metrics=spec.resolved_metrics(),
        name=spec.label(),
    )
    assert comparison.paired_count == len(CASES)
    (hypothesis,) = build_hypothesis_results((spec,), [comparison])
    assert hypothesis.status == "supported"
