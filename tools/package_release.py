"""Create a deterministic anonymous source ZIP from an explicit whitelist."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from check_release import release_paths, scan_release, verify_manifest, write_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.suffix.lower() != ".zip":
        parser.error("The release output must be a .zip file")
    issues = scan_release(root)
    if issues:
        print(json.dumps({"status": "failed", "issues": issues}, indent=2))
        raise SystemExit(1)
    write_manifest(root)
    verify_manifest(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = release_paths(root)
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths:
            name = "PNI-CGA/" + path.relative_to(root).as_posix()
            info = ZipInfo(name, date_time=(2000, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    print(json.dumps({"status": "passed", "archive": output.name, "files": len(paths),
                      "sha256": sha256(output.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
