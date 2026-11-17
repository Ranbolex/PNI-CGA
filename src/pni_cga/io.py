"""Portable JSON task format; no execution code is loaded from task files."""
from __future__ import annotations

import json
from pathlib import Path

from .model import Action, AcquisitionModel, Configuration, Cost


def model_from_dict(data: dict) -> AcquisitionModel:
    configurations = [Configuration(h["id"], h["answer"],
                                     tuple(sorted(h.get("facts", {}).items())),
                                     float(h.get("mass", 1.0)), h.get("anonymous", False))
                      for h in data["configurations"]]
    actions = [Action(a["id"], Cost(**a.get("cost", {"views": 1})),
                      float(a.get("predicted_cost", 1.0)), a.get("primitive", ""),
                      tuple(a.get("anchor_ids", [])),
                      tuple(a["box"]) if "box" in a else None,
                      a.get("requires_scope", False)) for a in data["actions"]]
    return AcquisitionModel(configurations, actions, data["channels"], data["compatibility"])


def load_model(path: str | Path) -> AcquisitionModel:
    return model_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def model_to_dict(model: AcquisitionModel) -> dict:
    from dataclasses import asdict
    return {"configurations": [{**asdict(h), "facts": dict(h.facts)}
                               for h in model.configurations.values()],
            "actions": [{k: v for k, v in asdict(a).items() if v is not None}
                        for a in model.actions.values()],
            "channels": {a: {h: dict(row) for h, row in rows.items()}
                         for a, rows in model.channels.items()},
            "compatibility": {a: {o: sorted(keep) for o, keep in rows.items()}
                              for a, rows in model.compatibility.items()}}
