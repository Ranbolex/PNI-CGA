"""Command-line examples and portable task solving."""
from __future__ import annotations

import argparse
from dataclasses import replace
from importlib.resources import files
import json
from pathlib import Path

from PIL import Image

from .audit import exact_twin_audit, verify_separation_certificate
from .examples import complementary_model, six_configuration_model
from .execution import run_episode
from .io import load_model
from .model import Budget
from .planning import CGAPlanner, PlannerSettings, solve_exact
from .vision import BinaryTileVerifier, LocalImageBackend


def certificate():
    data = json.loads(files("pni_cga").joinpath("fixtures/separation_certificate.json").read_text(encoding="utf-8"))
    return verify_separation_certificate(data)


def _emit(data, path=None):
    text = json.dumps(data, indent=2)
    if path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
    print(text)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pni-cga", description="PNI-CGA core algorithms and offline examples")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Solve the manuscript's six-configuration analytic model")
    demo.add_argument("--output")
    cert = sub.add_parser("verify-certificate", help="Independent rational check of Proposition 4")
    cert.add_argument("--output")
    comp = sub.add_parser("complementary", help="Two complementary reads with optional UNKNOWN outcomes")
    comp.add_argument("--unknown-probability", type=float, default=0.0)
    comp.add_argument("--output")
    visual = sub.add_parser("image-demo", help="Acquire and verify actual PNG crops without an API")
    visual.add_argument("--world", choices=[f"h{i}" for i in range(6)], default="h0")
    visual.add_argument("--seed", type=int, default=11)
    visual.add_argument("--output-dir", default="outputs/image-demo")
    solve = sub.add_parser("solve", help="Solve a task in the documented JSON format")
    solve.add_argument("task")
    solve.add_argument("--method", choices=("cga", "exact", "edge"), default="cga")
    solve.add_argument("--views", type=int, default=4)
    solve.add_argument("--depth", type=int, default=4)
    solve.add_argument("--rounds", type=int, default=16)
    solve.add_argument("--max-nodes", type=int, default=20000)
    solve.add_argument("--max-trees", type=int, default=100000)
    solve.add_argument("--output")
    args = parser.parse_args(argv)
    if args.command == "verify-certificate":
        _emit(certificate(), args.output)
    elif args.command == "demo":
        model, budget = six_configuration_model(), Budget(views=2)
        exact = solve_exact(model, budget, 2)
        edge = solve_exact(model, budget, 2, "edge")
        bounded = CGAPlanner(PlannerSettings(depth=2)).plan(model, budget)
        twin = exact_twin_audit(model, list(zip(exact.policies, exact.probabilities)), "h0", "h2")
        _emit({"analytic_example": "six configurations, four binary reads, two views",
               "exact_closure": exact.summary(), "matched_randomized_edge": edge.summary(),
               "counterexample_guided": bounded.summary(), "exact_twin_audit": twin,
               "rational_certificate": certificate()}, args.output)
    elif args.command == "complementary":
        model = complementary_model(args.unknown_probability)
        result = CGAPlanner(PlannerSettings(depth=2)).plan(model, Budget(views=2))
        _emit(result.summary(), args.output)
    elif args.command == "image-demo":
        model = six_configuration_model()
        directory = Path(args.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        # The environment selects the world. Only the crop tool opens the PNG.
        image = Image.new("RGB", (256, 64))
        facts = dict(model.configurations[args.world].facts)
        for i in range(4):
            image.paste("white" if facts[f"region{i}"] == "1" else "black",
                        (i * 64, 0, (i + 1) * 64, 64))
        source = directory / "scene.png"
        image.save(source)
        planner = CGAPlanner(PlannerSettings(depth=2))
        episode = run_episode(model, LocalImageBackend(source), BinaryTileVerifier(),
                              planner.plan, Budget(views=2), seed=args.seed, depth=2)
        for i, view in enumerate(episode.views):
            view.pixels.save(directory / f"acquired-{i + 1}.png")
        _emit(episode.summary(), directory / "episode.json")
    else:
        model, budget = load_model(args.task), Budget(views=args.views)
        if args.method == "cga":
            plan = CGAPlanner(PlannerSettings(depth=args.depth, rounds=args.rounds,
                                             max_nodes=args.max_nodes)).plan(model, budget)
        else:
            plan = solve_exact(model, budget, args.depth,
                               "edge" if args.method == "edge" else "closure", args.max_trees)
        _emit(plan.summary(), args.output)


if __name__ == "__main__":
    main()
