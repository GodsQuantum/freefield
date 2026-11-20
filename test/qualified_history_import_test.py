"""Offline regression tests for importing qualified lab media without GPU work."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "tools" / "import_qualified_media_into_history.py"
spec = importlib.util.spec_from_file_location("freefield_qualified_history", SOURCE)
assert spec and spec.loader
imp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imp)


class QualifiedHistoryImportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ff-import-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.receipts = self.root / "receipts"
        self.apps = self.root / "apps"
        self.history = self.root / "out" / "history"
        self.outputs = self.root / "out" / "validation"
        for d in (self.receipts, self.apps / "test-image", self.history, self.outputs):
            d.mkdir(parents=True)
        (self.apps / "test-image" / "app.json").write_text(json.dumps({
            "id": "test-image",
            "inputs": [
                {"id": "prompt", "type": "textarea", "required": True},
                {"id": "seed", "type": "seed", "default": -1},
                {"id": "image", "type": "image", "required": False},
            ],
        }))
        original_receipts, original_apps = imp.RECEIPTS, imp.APP_ROOT
        imp.RECEIPTS = self.receipts
        imp.APP_ROOT = self.apps
        self.addCleanup(lambda: setattr(imp, "RECEIPTS", original_receipts))
        self.addCleanup(lambda: setattr(imp, "APP_ROOT", original_apps))

    def fixture(self, *, quality="PASS_SINGLE_SCENE", with_input=False,
                workflow_override=False, tamper_checksum=False):
        name = "lab-success.json"
        folder = self.outputs / "lab-success"
        folder.mkdir()
        image = folder / "render.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\nTEST")
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if tamper_checksum:
            digest = "f" * 64
        config = {
            "values": {"prompt": "a toy robot", "seed": 42},
            "files": {},
        }
        if workflow_override:
            config["workflow_override"] = {"graph": {}}
        if with_input:
            fixture = self.outputs / "source" / "portrait.png"
            fixture.parent.mkdir()
            fixture.write_bytes(b"\x89PNG\r\n\x1a\nSOURCE")
            config["files"]["image"] = "/inputs/source/portrait.png"
        receipt = {
            "summary": {
                "app": "test-image", "status": "SUCCESS",
                "started_at": "2026-11-20T12:00:01.321+00:00", "seconds": 1.25,
            },
            "config": config,
            "visual_review": {"status": quality},
            "output_files": [{"name": image.name, "bytes": image.stat().st_size}],
            "media": [{"path": str(image), "bytes": image.stat().st_size,
                       "sha256": digest}],
            "prompt_ids": ["generated-job-id"],
        }
        (self.receipts / name).write_text(json.dumps(receipt))
        return name, image

    def test_real_media_and_source_hardlinks_idempotent(self):
        name, image = self.fixture(with_input=True)
        result = imp.import_one(name, self.history, self.outputs, True)
        self.assertEqual("IMPORTED", result["result"])
        folder = self.history / result["id"]
        stored = json.loads((folder / "meta.json").read_text())
        self.assertEqual("succeeded", stored["status"])
        self.assertEqual("image", stored["inputs"][0]["type"])
        self.assertEqual("image", stored["outputs"][0]["type"])
        self.assertEqual(image.stat().st_ino, (folder / "out-01.png").stat().st_ino)
        source = self.outputs / "source" / "portrait.png"
        self.assertEqual(source.stat().st_ino, (folder / "input-image.png").stat().st_ino)
        self.assertEqual("ALREADY_IMPORTED",
                         imp.import_one(name, self.history, self.outputs, True)["result"])
        self.assertEqual(1, len(list(self.history.iterdir())))

    def test_dry_run_never_adds_history(self):
        name, _ = self.fixture()
        result = imp.import_one(name, self.history, self.outputs, False)
        self.assertEqual("DRY_RUN_OK", result["result"])
        self.assertEqual([], list(self.history.iterdir()))

    def test_failed_quality_rejected(self):
        name, _ = self.fixture(quality="VISUAL_QUALITY_FAIL")
        with self.assertRaisesRegex(ValueError, "missing positive visual review"):
            imp.import_one(name, self.history, self.outputs, True)
        self.assertEqual([], list(self.history.iterdir()))

    def test_temporary_workflow_override_rejected(self):
        name, _ = self.fixture(workflow_override=True)
        with self.assertRaisesRegex(ValueError, "temporary workflow override"):
            imp.import_one(name, self.history, self.outputs, False)

    def test_source_or_result_checksum_failure_rejected(self):
        name, _ = self.fixture(tamper_checksum=True)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            imp.import_one(name, self.history, self.outputs, False)

    def test_missing_original_input_rejected(self):
        name, _ = self.fixture(with_input=True)
        (self.outputs / "source" / "portrait.png").unlink()
        with self.assertRaises(FileNotFoundError):
            imp.import_one(name, self.history, self.outputs, True)


if __name__ == "__main__":
    unittest.main()
