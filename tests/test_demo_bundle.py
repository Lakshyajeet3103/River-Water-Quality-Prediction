"""Bundle integrity and protection of an existing recipient's files."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

root = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("restore_demo_bundle", root / "scripts/restore_demo_bundle.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_real_bundle_restores_with_identical_hashes(tmp_path):
    bundle = root / "demo_bundle"
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert module.restore(bundle, tmp_path) == len(manifest["files"])
    for entry in manifest["files"]:
        assert hashlib.sha256((tmp_path / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    assert module.restore(bundle, tmp_path) == len(manifest["files"])


def test_bundle_refuses_to_replace_modified_file(tmp_path):
    bundle = root / "demo_bundle"
    module.restore(bundle, tmp_path)
    path = tmp_path / "outputs/fusion_demo_001/example_inputs.csv"
    path.write_text("recipient changed this")
    with pytest.raises(ValueError, match="Refusing to replace"):
        module.restore(bundle, tmp_path)
    assert path.read_text() == "recipient changed this"


def test_unsafe_manifest_path_rejected(tmp_path):
    bundle = tmp_path / "bad_bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"files": [{"path": "../outside", "sha256": "bad"}]}))
    with pytest.raises(ValueError, match="Unsafe"):
        module.restore(bundle, tmp_path / "destination")
