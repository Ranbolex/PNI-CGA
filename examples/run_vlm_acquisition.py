"""Use an explicit task model, image, and provider-compatible visual verifier.

Set PNI_CGA_MODEL, PNI_CGA_BASE_URL and PNI_CGA_API_KEY in your environment.
Each action's outcomes must be YES, NO and UNKNOWN with explicit compatibility.
The supplied prediction table is an offline fitted input to planning.
"""
import argparse
import json

from pni_cga.execution import run_episode
from pni_cga.io import load_model
from pni_cga.model import Budget
from pni_cga.planning import CGAPlanner, PlannerSettings
from pni_cga.vision import JSONVerifier, LocalImageBackend, OpenAICompatibleClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("image")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--config", help="Optional planner/budget/verifier JSON settings")
    args = parser.parse_args()
    settings = {}
    if args.config:
        from pathlib import Path
        settings = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model = load_model(args.task)
    if any(set(outcomes) != {"YES", "NO", "UNKNOWN"} for outcomes in model.outcomes.values()):
        parser.error("The JSON pixel verifier requires YES, NO and UNKNOWN task outcomes")
    planner = CGAPlanner(PlannerSettings(**settings.get("planner", {})))
    episode = run_episode(model, LocalImageBackend(args.image),
                          JSONVerifier(OpenAICompatibleClient(), **settings.get("verifier", {})),
                          planner.plan, Budget(**settings.get("budget", {})),
                          seed=args.seed, depth=planner.settings.depth)
    print(json.dumps(episode.summary(), indent=2))


if __name__ == "__main__":
    main()
