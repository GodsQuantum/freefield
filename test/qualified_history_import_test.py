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

    def test_reviewed_override_as_read_only_benchmark(self):
        name, image = self.fixture(workflow_override=True)
        result = imp.import_one(name, self.history, self.outputs, True, benchmark=True)
        self.assertEqual("IMPORTED", result["result"])
        record = json.loads((self.history / result["id"] / "meta.json").read_text())
        self.assertTrue(record["validation"]["benchmark_only"])
        self.assertFalse(record["validation"]["replayable"])
        self.assertTrue(record["validation"]["workflow_override"])
        self.assertEqual(record["validation"]["quality_status"], "PASS_SINGLE_SCENE")
        self.assertEqual(image.stat().st_ino, (self.history / result["id"] / "out-01.png").stat().st_ino)
        again = imp.import_one(name, self.history, self.outputs, True, benchmark=True)
        self.assertEqual("ALREADY_IMPORTED", again["result"])

    def test_nested_cli_history_true_output_is_hardlinked_not_input_or_metadata(self):
        name, image = self.fixture(workflow_override=True, with_input=True)
        entry = "20261123-110000000-abc1"
        node_dir = self.outputs / "lab-success" / "history" / entry
        node_dir.mkdir(parents=True)
        actual = node_dir / "out-01.png"
        actual.write_bytes(image.read_bytes())
        (node_dir / "input-image.png").write_bytes(b"\\x89PNG\\r\\n\\x1a\\nFAKE_INPUT")
        (node_dir / "meta.json").write_text("{}")
        receipt_path = self.receipts / name
        data = json.loads(receipt_path.read_text())
        data["output_files"] = [
            {"name": f"history/{entry}/input-image.png",
             "bytes": (node_dir / "input-image.png").stat().st_size},
            {"name": f"history/{entry}/meta.json", "bytes": 2},
            {"name": f"history/{entry}/out-01.png", "bytes": actual.stat().st_size},
        ]
        data["media"] = [{"path": str(actual), "bytes": actual.stat().st_size,
                           "sha256": hashlib.sha256(actual.read_bytes()).hexdigest()}]
        receipt_path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "temporary workflow override|requires reviewed benchmark"):
            imp.import_one(name, self.history, self.outputs, False)
        response = imp.import_one(name, self.history, self.outputs, True, benchmark=True)
        self.assertEqual("IMPORTED", response["result"])
        folder = self.history / response["id"]
        record = json.loads((folder / "meta.json").read_text())
        self.assertEqual(1, len(record["outputs"]))
        self.assertEqual("out-01.png", record["outputs"][0]["file"])
        self.assertEqual(actual.stat().st_ino, (folder / "out-01.png").stat().st_ino)
        self.assertEqual("input-image.png", record["inputs"][0]["file"])
        self.assertNotEqual((node_dir / "input-image.png").stat().st_ino,
                            (folder / "out-01.png").stat().st_ino)
        self.assertTrue(record["validation"]["benchmark_only"])
        self.assertFalse(record["validation"]["replayable"])
        self.assertEqual("ALREADY_IMPORTED",
                         imp.import_one(name, self.history, self.outputs, True, benchmark=True)["result"])

    def test_actual_cli_nested_generated_media_imports_as_normal_replayable_library_item(self):
        name, image = self.fixture()
        entry = "20261123-220650715-8dd3"
        own = self.outputs / "lab-success" / "history" / entry
        own.mkdir(parents=True)
        source = own / "out-01.png"
        source.write_bytes(image.read_bytes())
        (own / "meta.json").write_text("{}")
        (own / "input-image.png").write_bytes(b"\\x89PNG\\r\\n\\x1a\\nFAKE")
        receipt = self.receipts / name
        proof = json.loads(receipt.read_text())
        proof["output_files"] = [
            {"name": f"history/{entry}/input-image.png", "bytes": (own / "input-image.png").stat().st_size},
            {"name": f"history/{entry}/meta.json", "bytes": 2},
            {"name": f"history/{entry}/out-01.png", "bytes": source.stat().st_size},
        ]
        proof["media"] = [{"path": str(source), "bytes": source.stat().st_size,
                           "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}]
        receipt.write_text(json.dumps(proof))
        result = imp.import_one(name, self.history, self.outputs, False)
        self.assertEqual("DRY_RUN_OK", result["result"])
        self.assertEqual(["out-01.png"], result["media"])
        self.assertEqual(0, len(list(self.history.iterdir())))
        result = imp.import_one(name, self.history, self.outputs, True)
        self.assertEqual("IMPORTED", result["result"])
        produced = self.history / result["id"]
        record = json.loads((produced / "meta.json").read_text())
        self.assertEqual("cloud9-cli-qualification", record["validation"]["origin"])
        self.assertEqual("PASS_SINGLE_SCENE", record["validation"]["visual_review"])
        self.assertEqual([], record["inputs"])
        self.assertEqual(1, len(record["outputs"]))
        self.assertNotIn("benchmark_only", record["validation"])
        self.assertNotEqual(record["validation"].get("replayable"), False)
        self.assertEqual(source.stat().st_ino, (produced / "out-01.png").stat().st_ino)
        self.assertNotEqual((own / "input-image.png").stat().st_ino, (produced / "out-01.png").stat().st_ino)
        self.assertEqual("ALREADY_IMPORTED",
                         imp.import_one(name, self.history, self.outputs, True)["result"])

    def test_nested_cli_history_without_generated_output_is_never_accepted(self):
        name, _ = self.fixture(workflow_override=True)
        entry = "20261123-110000000-abc1"
        file = self.receipts / name
        data = json.loads(file.read_text())
        data["output_files"] = [
            {"name": f"history/{entry}/input-image.png", "bytes": 100},
            {"name": f"history/{entry}/meta.json", "bytes": 200},
        ]
        file.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "no generated output"):
            imp.import_one(name, self.history, self.outputs, False, benchmark=True)
        self.assertEqual([], list(self.history.iterdir()))

    def test_benchmark_rejects_failed_qa_and_unreviewed_output(self):
        name, _ = self.fixture(quality="VISUAL_QUALITY_FAIL", workflow_override=True)
        with self.assertRaisesRegex(ValueError, "not a qualified"):
            imp.import_one(name, self.history, self.outputs, False, benchmark=True)
        payload = json.loads((self.receipts / name).read_text())
        payload["visual_review"] = {}
        (self.receipts / name).write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "not a qualified"):
            imp.import_one(name, self.history, self.outputs, False, benchmark=True)

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
