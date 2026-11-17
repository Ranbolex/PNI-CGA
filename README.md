# PNI-CGA

Anonymous research code for **Policy-Induced Non-Identifiability** (PNI) and
**Counterexample-Guided Acquisition** (CGA).

A correct answer can leave an opposite-answer scene compatible with every
acquired view. CGA plans observations to close those alternatives within a
resource budget. This package implements the manuscript's complete-evidence
closure objective, joint finite frontier, verified updates, committed policy
execution, and exact-twin auditing.

[Algorithm](docs/ALGORITHM.md) ·
[Task format](docs/TASK_FORMAT.md) · [Visual integration](docs/VISUAL_INTEGRATION.md)

## Install

Python 3.10 or later is required. From the extracted `PNI-CGA` directory:

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
```

Runtime dependencies are NumPy, SciPy and Pillow. All commands below run on
CPU; the offline examples require no model weights, network access or API key.
You can also install without development dependencies using `pip install .`.
`requirements-tested.txt` records the versions used with Python 3.13.9 for
release validation; `VALIDATION.json` records the completed checks.

## Run the core algorithm

```sh
pni-cga demo --output outputs/analytic.json
pni-cga verify-certificate
pni-cga complementary --unknown-probability 0.2
pni-cga solve configs/six_configurations.json --views 2 --depth 2
pni-cga solve configs/six_configurations.json --method exact --views 2 --depth 2
pni-cga solve configs/six_configurations.json --method edge --views 2 --depth 2
```

`python -m pni_cga` can replace `pni-cga` in these commands. The six-configuration
example compares objectives over the **same budget-feasible policy class**:

| Objective | Worst-configuration closure | Mean closure |
|---|---:|---:|
| Complete closure | 1/2 | 7/12 |
| Residual pair mass, then best closure on its optimal face | 1/3 | 11/18 |

The residual-pair objective attains worst expected remaining mass 2/27.
These are analytic values for the included finite construction. The independent
checker uses rational arithmetic over all 101 depth-two binary trees, including
repeated reads and early stopping. The main planner removes repeated reads
and stops on single-answer frontiers. Both yield the stated optimum values.
For the complementary-read example with independent unreadability 0.2, the
closure probability is `(1 - 0.2)^2 = 0.64`; neither single read settles the answer.

### Actual PNG acquisition

```sh
pni-cga image-demo --world h0 --seed 11 --output-dir outputs/image-demo
```

This creates a synthetic four-tile image, plans without receiving the selected
world ID, acquires actual pixel crops and verifies their black/white values.
The output directory contains the scene, acquired crops and a metered episode
trace. Depending on the sampled policy and world, the result can remain
unresolved: the two-view budget does not guarantee closure for every execution.

### Python API

```python
from pni_cga import Budget, CGAPlanner, PlannerSettings, solve_exact
from pni_cga.examples import six_configuration_model

model = six_configuration_model()
budget = Budget(views=2)
plan = CGAPlanner(PlannerSettings(depth=2)).plan(model, budget)
print(plan.summary())
reference = solve_exact(model, budget, depth=2)
assert abs(plan.lower_bound - reference.lower_bound) < 1e-8
```

## Core implementation

| Module | Function |
|---|---|
| `src/pni_cga/model.py` | Joint configurations, predictive channels, independent compatibility tables and budgets |
| `src/pni_cga/frontier.py` | Opposite-answer witnesses, verified updates, exact same-answer compression |
| `src/pni_cga/policies.py` | Contingent trees, branch feasibility, fixed-world closure evaluation |
| `src/pni_cga/game.py` | Maximin closure LP; mean/cost tie-breaking; matched residual-pair control |
| `src/pni_cga/planning.py` | Counterexample-guided policy expansion and exact enumeration |
| `src/pni_cga/execution.py` | Sample-once execution, raw views, trace and actual resource accounting |
| `src/pni_cga/vision.py` | Root-coordinate crop tool, conservative JSON verifier and optional image-model client |
| `src/pni_cga/audit.py` | Trajectory TV, exact-twin coverage and independent rational certificates |

The implemented state is an explicit finite joint model. Configurations retain
cross-region bindings and anonymous alternatives through facts and persistent
IDs. Task-specific question compilation, candidate generation and calibration
feed the documented interfaces. The release provides the algorithm, its
analytic verification and visual integration; paper benchmark assets and
result tables are separate research artifacts.

## Real image/model integration

`examples/run_vlm_acquisition.py` connects task JSON, `LocalImageBackend`,
`JSONVerifier`, and an OpenAI-compatible image endpoint. See
[Visual integration](docs/VISUAL_INTEGRATION.md) for environment variables,
the YES/NO/UNKNOWN schema, coordinates and provider accounting. No provider
is contacted by installation, tests or the offline demos.

## Validate and package

```sh
python -m pytest -q
python tools/check_release.py --verify-manifest
python tools/package_release.py --output ../PNI-CGA.zip
```

The ZIP has one top-level `PNI-CGA/` directory. A whitelist excludes generated
outputs, environments, caches and version-control history. Entries use fixed
timestamps and normalized permissions. `MANIFEST.sha256` supports file-integrity
verification. Source and metadata use an anonymous contributor identity.

## License and citation

MIT; see `LICENSE`. Anonymous citation metadata is in `CITATION.cff`.
When referring to the method, cite the accompanying manuscript,
*Policy-Induced Non-Identifiability in Active Vision-Language Agents*.
