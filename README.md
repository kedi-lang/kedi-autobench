# kedi-autobench

Kedi-owned telemetry adapter for Autobench. It composes with the backend already installed in
Kedi, emits Kedi lifecycle evidence into ABP, and conditionally restores the backend slot it owns.

Requires Autobench `>=0.3,<0.4` and Kedi `>=0.4,<0.5`.

```bash
pip install kedi-autobench
```

For repository development, `uv sync` resolves Kedi and Autobench from the exact Git revisions in
`pyproject.toml`. This keeps a fresh checkout independent from the sibling-directory layout used
while the integration was developed.

```python
from autobench import Benchmark
from autobench.instrumentation.pydantic_ai import PydanticAI
from kedi_autobench import Kedi

benchmark = Benchmark("kedi-program").instrument(PydanticAI(), Kedi())
```

When Autobench's Pydantic AI instrumentor is installed, native Pydantic agent, model, and tool
spans remain authoritative. Kedi enriches those spans and continues to own Kedi-only runtime,
workflow, approval, and artifact lifecycle evidence.

## Behavioral assets

Kedi emits typed descriptors for the definitions that shaped an agent call:

- rendered templates and effective instructions;
- complete agent profile configuration;
- tool descriptions, risk metadata, and input/return schemas;
- structured output schemas;
- skill content when a skill is read;
- source identity and optional bounded source snippets.

Capture is double-gated. Kedi decides whether raw values may cross its telemetry seam; Autobench's
`CapturePolicy` then decides whether the value is retained as full content, a hash, metadata, a
redacted value, or nothing.

```python
from autobench import Benchmark, CaptureLevel, CapturePolicy
from kedi_autobench import Kedi

benchmark = (
    Benchmark("kedi-program")
    .capture(CapturePolicy(default_level=CaptureLevel.FULL))
    .instrument(
        Kedi(
            capture_content=True,
            capture_source_paths=True,
            capture_source_snippets=False,
        )
    )
)
```

`capture_content=False` still emits stable identities and content fingerprints, so changed
templates and schemas create new versions without exposing their raw values. Source paths and
snippets have independent flags. Full prompt, schema, and skill text is carried as asset candidate
content, never placed in generic span attributes.

## Backend ownership

Installation preserves the current Kedi telemetry backend by wrapping it with a composite backend.
Closing the Autobench instrumentor restores the previous backend only if the adapter still owns the
active slot. A backend installed later is never overwritten.

Progress and durable persistence remain Autobench responsibilities. Kedi evidence, asset uses, and
ABP spans flow through normal `RunResult`, checkpoint, staging, cancellation, final record, and
replay paths. Completed recorder mutations and artifact transfers settle before abort or close, so
Kedi evidence follows the same cancellation guarantees as native Autobench evidence.

## Terminal-Bench 2.1 capture

The Terminal-Bench integration is an external, post-run importer. Harbor remains the execution,
grading, and trial-record authority. The importer starts only after the wrapped command exits, maps
each Harbor trial to one Autobench run, and preserves the wrapped command's exit status even if
capture fails. It does not install Autobench into Kedi or patch Harbor's runtime.

Record an existing Harbor job:

```bash
kedi-autobench-terminal-bench record \
  --job-dir ./jobs/kedi-pilot \
  --record-dir ./records/kedi-pilot
```

Run a Harbor command and capture its job afterward:

```bash
kedi-autobench-terminal-bench run \
  --job-dir ./jobs/kedi-pilot \
  --record-dir ./records/kedi-pilot \
  -- harbor run --job-name kedi-pilot
```

The wrapper waits for the Harbor process before reading its files. Capture errors are written next
to the requested record directory as `<name>.capture-error.json`; they never replace Harbor's exit
code. For automation where even post-run recording latency is undesirable, invoke `record` as a
separate step after Harbor.

The record contains official rewards, phase durations, request/tool/token/cache/cost observations,
Kedi completion state, and bounded copies of Harbor job, agent, verifier, and artifact evidence.
Text and JSON evidence is redacted before it reaches Autobench, including extensionless UTF-8 files
such as `.env`. Binary evidence is copied unchanged. Files larger than
`--max-evidence-file-bytes` are represented in the evidence index but not copied. The default is
20 MB per file. Attached evidence is also limited to 50 MB per trial by
`--max-evidence-total-bytes`. Core job and trial records are considered before agent, artifact, and
verifier directories. Files omitted by either bound retain their path, source size, and SHA-256 in
the evidence index, so large Harbor payloads remain authoritative without being duplicated into
the Autobench record. Raise the limits explicitly when a self-contained copy is required.
