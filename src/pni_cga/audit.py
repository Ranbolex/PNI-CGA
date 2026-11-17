"""Exact trajectory laws, twin coverage, and independent rational certificates."""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from itertools import product
from math import isfinite

from .model import AcquisitionModel, UNKNOWN
from .policies import PolicyTree, validate_structure


def trajectory_law(model: AcquisitionModel, tree: PolicyTree, configuration: str) -> dict:
    validate_structure(model, tree)
    if configuration not in model.configurations:
        raise ValueError("Unknown configuration")
    law = defaultdict(float)
    def visit(node, prefix, probability):
        if node.action is None:
            law[prefix] += probability
            return
        for outcome, child in node.children:
            p = model.channels[node.action][configuration][outcome]
            if p:
                visit(child, prefix + ((node.action, outcome),), probability * p)
    visit(tree, (), 1.0)
    return dict(law)


def exact_twin_audit(model: AcquisitionModel, mixture, first: str, second: str) -> dict:
    mixture = tuple(mixture)
    if first == second or model.configurations[first].answer == model.configurations[second].answer:
        raise ValueError("Exact twins require different configurations with opposite answers")
    if (abs(sum(w for _, w in mixture) - 1) > 1e-8
            or any(not isfinite(w) or w < 0 for _, w in mixture)):
        raise ValueError("Policy mixture must be a probability distribution")
    discriminative = set()
    for aid in model.actions:
        values = []
        for h in (first, second):
            row = model.channels[aid][h]
            support = [o for o, p in row.items() if p > 0]
            if len(support) != 1 or abs(row[support[0]] - 1) > 1e-10:
                raise ValueError("This identity requires deterministic exact-twin observations")
            values.append(support[0])
        if values[0] != values[1]:
            discriminative.add(aid)
    laws = []
    for h in (first, second):
        total = defaultdict(float)
        for tree, weight in mixture:
            for path, p in trajectory_law(model, tree, h).items():
                total[path] += weight * p
        laws.append(total)
    tv = 0.5 * sum(abs(laws[0][p] - laws[1][p]) for p in laws[0].keys() | laws[1].keys())
    coverage = [sum(p for path, p in law.items() if any(a in discriminative for a, _ in path))
                for law in laws]
    if abs(coverage[0] - coverage[1]) > 1e-8 or abs(tv - coverage[0]) > 1e-8:
        raise AssertionError("Exact-twin identity failed")
    return {"trajectory_tv": tv, "coverage": coverage[0],
            "twin_accuracy_ceiling": (1 + tv) / 2,
            "discriminative_actions": sorted(discriminative)}


def verify_separation_certificate(data: dict) -> dict:
    """Independent exact arithmetic over all 101 budget-two binary trees.

    Includes constant/repeated reads and early stopping; uses delivered bit
    transcripts, not the main evaluator, LP solver, or candidate generator.
    """
    q = Fraction
    worlds, labels = data["world_features"], data["answers"]
    n = len(worlds)
    if (n != 6 or len(labels) != n or len(set(labels)) != 2
            or any(len(row) != 4 or any(v not in (0, 1) for v in row) for row in worlds)):
        raise ValueError("The separation certificate requires six four-bit configurations")
    def trees(state, depth):
        if not state or depth == 0:
            return [None]
        result = [None]
        for a in range(len(worlds[0])):
            parts = [tuple(h for h in state if worlds[h][a] == o) for o in (0, 1)]
            result.extend((a, left, right) for left, right in product(
                trees(parts[0], depth - 1), trees(parts[1], depth - 1)))
        return result
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n) if labels[a] != labels[b]]
    def evaluate(tree):
        closure, residual = [], []
        for h in range(n):
            path, node = [], tree
            while node is not None:
                a = node[0]
                path.append((a, worlds[h][a]))
                node = node[1 + worlds[h][a]]
            retained = {g for g in range(n) if all(worlds[g][a] == o for a, o in path)}
            closure.append(q(int(len({labels[g] for g in retained}) == 1)))
            residual.append(q(sum(a in retained and b in retained for a, b in pairs), len(pairs)))
        return closure, residual
    candidates = [evaluate(t) for t in trees(tuple(range(n)), 2)]
    cert = data["rational_certificate"]
    uc = list(map(q, cert["cert_dual_world_weights"]))
    we = list(map(q, cert["edge_dual_world_weights"]))
    uf = list(map(q, cert["edge_face_cert_dual_weights"]))
    vf = list(map(q, cert["edge_face_edge_dual_weights"]))
    for dist in (uc, we, uf):
        if len(dist) != n or min(dist) < 0 or sum(dist) != 1:
            raise ValueError("Invalid dual certificate")
    if len(vf) != n or min(vf) < 0:
        raise ValueError("Invalid edge-face dual weights")
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    checks = [max(dot(uc, d) for d, r in candidates) == q(1, 2),
              min(dot(we, r) for d, r in candidates) == q(2, 27),
              max(dot(uf, d) - dot(vf, r) for d, r in candidates) + q(2, 27) * sum(vf) == q(1, 3)]
    def primal(mixture):
        if sum(q(v["weight"]) for v in mixture) != 1 or any(q(v["weight"]) < 0 for v in mixture):
            raise ValueError("Invalid primal mixture")
        ds, rs = [q(0)] * n, [q(0)] * n
        def validate(node, depth):
            if node is None:
                return
            if (depth == 0 or not isinstance(node, (list, tuple)) or len(node) != 3
                    or not isinstance(node[0], int) or not 0 <= node[0] < 4):
                raise ValueError("Primal certificate exceeds the two-view policy class")
            validate(node[1], depth - 1)
            validate(node[2], depth - 1)
        for item in mixture:
            validate(item["policy"], 2)
            d, r = evaluate(item["policy"])
            ds = [old + q(item["weight"]) * new for old, new in zip(ds, d)]
            rs = [old + q(item["weight"]) * new for old, new in zip(rs, r)]
        return min(ds), max(rs), sum(ds) / n
    cga = primal(data["closure_mixture"])
    edge = primal(data["edge_mixture"])
    checks.extend([cga[0] == q(1, 2), cga[2] == q(7, 12),
                   edge == (q(1, 3), q(2, 27), q(11, 18))])
    if not all(checks):
        raise ValueError("Exact separation certificate failed")
    return {"all_passed": True, "all_trees_checked": len(candidates),
            "closure_optimum": "1/2", "edge_optimum": "2/27",
            "best_closure_on_edge_optimal_face": "1/3",
            "closure_mean": "7/12", "edge_mean": "11/18"}
