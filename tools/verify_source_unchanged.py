"""Compare the read-only source tree against the captured SHA256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=r"D:\Codex\WaveletResidualBridge3D")
    parser.add_argument("--manifest", default="artifacts/source_tree_sha256.json")
    args = parser.parse_args()
    source = Path(args.source)
    expected_rows = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    expected = {row["relative_path"]: row for row in expected_rows}
    missing = []
    modified = []
    for relative, row in expected.items():
        path = source / Path(relative)
        if not path.is_file():
            missing.append(relative)
        elif _sha256(path) != row["sha256"]:
            modified.append(relative)
    payload = {
        "status": "PASS" if not missing and not modified else "FAIL",
        "source": str(source.resolve()),
        "manifest_file_count": len(expected),
        "missing": missing,
        "modified": modified,
    }
    Path("artifacts/source_unchanged_audit.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise RuntimeError("read-only source tree changed")


if __name__ == "__main__":
    main()
