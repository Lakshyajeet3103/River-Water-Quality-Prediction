"""Verify and restore the committed small demo bundle without overwriting changes."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil


def restore(bundle, destination, verify_only=False):
    bundle, destination = Path(bundle).resolve(), Path(destination).resolve()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    planned = []
    for entry in manifest["files"]:
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or ":" in str(relative) or "\\" in str(relative):
            raise ValueError("Unsafe bundle path")
        if relative.parts[0] not in ("outputs", "data"):
            raise ValueError("Bundle may only restore demo data/output paths")
        source = (bundle / "files" / str(relative)).resolve()
        target = (destination / str(relative)).resolve()
        if not source.is_relative_to(bundle / "files") or not target.is_relative_to(destination):
            raise ValueError("Bundle path escapes its directory")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"Bundle hash mismatch: {relative}")
        if not verify_only and target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Refusing to replace a different existing file: {target}")
        planned.append((source, target))
    if not verify_only:
        for source, target in planned:
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    return len(planned)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=str(root / "demo_bundle"))
    parser.add_argument("--destination", default=str(root))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        count = restore(args.bundle, args.destination, args.verify_only)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Demo bundle failed: {exc}\n")
    print(f"Verified {count} files" + ("" if args.verify_only else f"; demo ready in {Path(args.destination).resolve()}"))


if __name__ == "__main__":
    main()
