# Visual acquisition and verifier integration

## Interfaces

`AcquisitionBackend.acquire(action)` returns an `AcquiredView` with pixels,
action ID, delivered root box, root image size and actual acquisition cost.
`Verifier.verify(view, action)` returns a `Verification` with a modeled
outcome, actual cost and reason. Custom backends can implement other visual
tools using these interfaces.

`LocalImageBackend` opens an image, applies EXIF orientation, converts it to
RGB, clips the requested root-coordinate box and resizes the delivered crop
to a bounded maximum edge. Root coordinates refer to the oriented image.
The verifier receives that crop and its geometry, rather than the full image.

## JSON verification gate

`JSONVerifier` accepts a client returning `(payload, Cost)`; the client may
call a frozen visual model or a task-specific verifier. A payload looks like:

```json
{
  "primitive_id": "read_tile",
  "entity_anchor_ids": ["tile"],
  "value": "yes",
  "legibility_ordinal": 3,
  "binding_ordinal": 3,
  "scope_complete": true,
  "witness_regions": [[0, 0, 64, 64]]
}
```

`value` is yes/no/unknown; scores are ordinal 0..3. Defaults require legibility
and binding at least two. Ordinal thresholds and channels are supplied/frozen
by the task's calibration procedure. Invalid JSON, mismatched anchors,
insufficient scores, missing witness boxes or witnesses outside the delivered
crop produce UNKNOWN. `requires_scope` additionally demands complete scope,
as needed for negative existence or count predicates. Threshold four rejects
all outputs and can be used to disable acceptance.

These checks enforce the evidence interface. Semantic correctness of a
literal and actual domain completeness depend on the visual verifier; a
self-reported scope flag alone is not an independent correctness proof.
UNKNOWN and rejected outputs never authorize elimination. The accepted
outcome is intersected with the task's separate compatibility table.

## Optional model endpoint

`OpenAICompatibleClient` sends an actual PNG crop to `/chat/completions` with
the bundled literal-only prompt. Configure the model ID, provider base URL
and credential in your local environment:

```powershell
$env:PNI_CGA_MODEL = "YOUR_MODEL_ID"
$env:PNI_CGA_BASE_URL = "https://YOUR_PROVIDER/v1"
$env:PNI_CGA_API_KEY = "YOUR_LOCAL_KEY"
python examples/run_vlm_acquisition.py configs/visual_binary.json outputs/image-demo/scene.png --config configs/default.json
```

On POSIX shells, use `export PNI_CGA_MODEL=...` and the equivalent other
variables. The two-candidate binary task is a wiring example with an illustrative
0.05 UNKNOWN rate; use a black or white crop filling its 64-by-64 region.
Supply task-specific joint states and fitted channels for research use.
The six-configuration demo uses `0/1` outcomes and its offline tile verifier;
the JSON model client requires task outcomes YES/NO/UNKNOWN.

The compatibility client supports providers offering image input, JSON-mode
responses and split prompt/completion usage in this API format. Provider
adaptations can use another callable with the same return type. Installation,
offline examples and unit tests make no provider calls. Calling the VLM
example uses your configured service and its normal resource charges.

## Costs and failures

Each episode records planning time, acquisition views/time and model input
tokens, output tokens, calls and time separately. Attempts that fail are still
charged for observed costs. There are no implicit retries. Actual overruns
stop before the returned evidence can exclude configurations; consumed cost
remains in the trace. An in-flight provider request cannot be undone, so
reserve appropriate per-action upper bounds and provider output limits.

If a provider omits usage or the connection fails, exact consumed tokens may
be unavailable. The execution stops with `tool_failure` and marks
`accounting_complete=false`; known calls/time remain recorded. Such records
are partial accounting and must not be interpreted as zero-token requests.
Server response bodies, endpoint URLs and authorization headers are not
included in error text or exported traces.

`Episode.views` exposes raw acquired crops for a common downstream answerer.
The controller's candidate answer, frontier or rationale need not be passed
to that answerer. `Episode.summary()` emits portable JSON without raw pixel
arrays, local source-image paths or model credentials.
