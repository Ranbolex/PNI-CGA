"""Anonymous-release content scan and portable source-integrity verification."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re


TOP_LEVEL = frozenset({
    "README.md", "LICENSE", "CITATION.cff", "pyproject.toml",
    "requirements.txt", "requirements-dev.txt", "requirements-tested.txt", ".gitignore",
    "MANIFEST.in", "MANIFEST.sha256", "VALIDATION.json",
})
TREES = {"src/pni_cga": {".py", ".json", ".txt"}, "configs": {".json"},
         "examples": {".py"}, "tests": {".py"}, "docs": {".md"}, "tools": {".py"}}
EXCLUDED = frozenset({"__pycache__", ".pytest_cache", ".git", ".venv", "outputs", "build", "dist"})


def release_paths(root: Path) -> list[Path]:
    root = root.resolve()
    result = {root / name for name in TOP_LEVEL if (root / name).is_file()}
    for directory, suffixes in TREES.items():
        for path in (root / directory).rglob("*"):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            relative = path.relative_to(root)
            if any(part in EXCLUDED or part.endswith(".egg-info") for part in relative.parts):
                continue
            result.add(path)
    for path in result:
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Release sources must be regular files inside the package")
    return sorted(result, key=lambda p: p.relative_to(root).as_posix())


def scan_release(root: Path) -> list[str]:
    """Return only relative names/categories, never the potentially sensitive text."""
    issues = []
    patterns = {
        "absolute-host-path": re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/](?!/)[^\s\"']+"),
        "private-home-path": re.compile(r"/(?:ho" + r"me|Us" + r"ers)/[^/\s]+"),
        "private-key-material": re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
        "credential-token": re.compile(r"(?:sk[-_]|ghp_|github_pat_)[A-Za-z0-9_-]{20,}"),
    }
    emails = re.compile(r"[A-Za-z0-9_.+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")
    key_literal = re.compile(r"\b(?:api_key|access_token|password)\s*[:=]\s*['\"]([^'\"]+)['\"]", re.I)
    dummy_values = {"test-only", "placeholder", "YOUR_LOCAL_KEY"}
    required = TOP_LEVEL - {"MANIFEST.sha256", "VALIDATION.json"}
    for name in required:
        if not (root / name).is_file():
            issues.append(name + ": missing-required-file")
    for path in release_paths(root):
        relative = path.relative_to(root).as_posix()
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeError:
            issues.append(relative + ": unexpected-non-text-source")
            continue
        for category, regex in patterns.items():
            if regex.search(content):
                issues.append(relative + ": " + category)
        if any(m.group(1).lower() not in {"example.invalid", "example.com"} for m in emails.finditer(content)):
            issues.append(relative + ": personal-email")
        if any(m.group(1) not in dummy_values for m in key_literal.finditer(content)):
            issues.append(relative + ": hard-coded-credential")
    return issues


def write_manifest(root: Path) -> None:
    records = []
    for path in release_paths(root):
        if path.name == "MANIFEST.sha256":
            continue
        records.append(sha256(path.read_bytes()).hexdigest() + "  " + path.relative_to(root).as_posix())
    (root / "MANIFEST.sha256").write_text("\n".join(records) + "\n", encoding="utf-8", newline="\n")


def verify_manifest(root: Path) -> None:
    source = root / "MANIFEST.sha256"
    if not source.is_file():
        raise ValueError("Integrity manifest is missing")
    records = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or name in records:
            raise ValueError("Invalid integrity manifest")
        records[name] = digest
    actual = {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest()
              for p in release_paths(root) if p.name != "MANIFEST.sha256"}
    if records != actual:
        raise ValueError("Integrity manifest does not match the release sources")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--verify-manifest", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    issues = scan_release(root)
    if args.verify_manifest:
        try:
            verify_manifest(root)
        except (ValueError, OSError):
            issues.append("MANIFEST.sha256: verification-failed")
    print(json.dumps({"status": "failed" if issues else "passed",
                      "files": len(release_paths(root)), "issues": issues,
                      "manifest_verified": args.verify_manifest and not issues}, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
