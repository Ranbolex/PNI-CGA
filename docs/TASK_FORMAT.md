# Task model format

Tasks are UTF-8 JSON. See `configs/six_configurations.json`,
`configs/complementary.json` and `configs/visual_binary.json` for complete inputs.

```json
{
  "configurations": [
    {"id": "absent", "answer": "no", "facts": {"object": "absent"}},
    {"id": "present", "answer": "yes", "facts": {"object": "present"}, "anonymous": true}
  ],
  "actions": [{
    "id": "inspect", "primitive": "The object exists in this cell.",
    "anchor_ids": ["cell"], "box": [0, 0, 64, 64], "requires_scope": true,
    "cost": {"views": 1, "output_tokens": 1024, "calls": 1, "seconds": 90},
    "predicted_cost": 1
  }],
  "channels": {
    "inspect": {
      "absent": {"YES": 0, "NO": 0.9, "UNKNOWN": 0.1},
      "present": {"YES": 0.9, "NO": 0, "UNKNOWN": 0.1}
    }
  },
  "compatibility": {
    "inspect": {"YES": ["present"], "NO": ["absent"], "UNKNOWN": ["absent", "present"]}
  }
}
```

`configurations` must be nonempty with unique string IDs and nonempty string
answers. `facts` is a map of joint fact keys to values; `mass` defaults to one
and must be positive. `anonymous` marks residual/unproposed completions and
does not change their retention or scoring rules. Include OTHER as a real
answer class where the task requires unenumerated answer alternatives.

Every action has a unique ID and at least one view of cost. Root `box` uses
integer `[x0, y0, x1, y1]` coordinates, with exclusive right/bottom edges.
It is required for actual local image acquisition. All action/world pairs
must provide a normalized row of nonnegative probabilities for exactly the
action's outcome alphabet. Every alphabet includes the reserved `UNKNOWN`.

Compatibility is an independently justified evidence constraint. It may
be more conservative than predictive likelihood support. Rows name all
configurations compatible with the literal; UNKNOWN must name all of them.
No task file contains code, a selected true-world ID, secret key or hidden
reference image path.

`cost` defines planning resource reservations; use upper bounds when hard
runtime feasibility is required. It includes views, input/output tokens,
calls and seconds. The hard `Budget` caps views, **output** tokens, calls and
seconds. Input tokens are metered separately. `predicted_cost` is the scalar
cost used only after closure and mean tie-breaking; choose its units in the
task adapter. It never replaces hard resource caps.

For Python users:

```python
from pni_cga.io import load_model, model_to_dict
from pni_cga import Budget, CGAPlanner

model = load_model("configs/visual_binary.json")
plan = CGAPlanner().plan(model, Budget(views=1))
print(plan.summary())
```

`configs/default.json` holds planner, budget, verifier and seed settings. The
VLM example accepts it with `--config`. The `solve` command uses its explicit
CLI flags; it loads a task model, not this separate settings file.
