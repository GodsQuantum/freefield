"""Regression checks for the idempotent, non-GPU Freefield evidence ledger."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("reconcile_benchmark_index", HERE / "tools/reconcile_benchmark_index.py")
assert SPEC is not None and SPEC.loader is not None
RECONCILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECONCILE)


class BenchmarkLedgerTests(unittest.TestCase):
    def test_preflight_never_counted_as_gpu_generation(self):
        self.assertEqual(RECONCILE.classify(
            "preflight", "PASS_STATIC_NATIVE_GRAPH_NO_GPU_INFERENCE", "angles", [], [], {"output_files": [{"name": "fake.png"}]}),
            "PREQUALIFICATION_NOT_INFERENCE")

    def test_failed_and_aborted_are_persisted_as_run_evidence(self):
        self.assertEqual(RECONCILE.classify(
            "validation", "SAFETY_ABORT", "angles", ["owned-prompt"], [], {}),
            "ABORTED_GPU_RUN")
        self.assertEqual(RECONCILE.classify(
            "validation", "TIMEOUT", "h3-video", ["owned-prompt"], [], {}),
            "TIMED_OUT_GPU_RUN")
        self.assertEqual(RECONCILE.classify(
            "benchmarks", "NOT_SUBMITTED_RESOURCE_GUARD", "angles", [], [], {}),
            "NOT_SUBMITTED")

    def test_existing_freefield_cli_single_output_is_not_lost(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output = root / "out.mp4"
            output.write_bytes(b"existing video output marker")
            evidence = {"status": "PASS_REAL_FREEFIELD_CLI_E2E_VIDEO_AUDIO_IDENTICAL_TO_NATIVE",
                        "app": "extend-video-int4", "history_id": "prior-history-entry",
                        "cli_output": str(output), "cli_output_bytes": output.stat().st_size,
                        "cli_output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                        "wall_seconds": 1019.8}
            receipt = root / "benchmarks" / "previous.json"
            receipt.parent.mkdir()
            receipt.write_text(json.dumps(evidence))
            row = RECONCILE.parse_receipt(root, "benchmarks", receipt)
            self.assertEqual(row["generation_status"], "GENERATED_SUCCESS")
            self.assertEqual(row["media_files_verified"], 1)
            self.assertEqual(row["seconds"], 1019.8)

    def test_native_comfy_embedded_history_provenance(self):
        history = {"history": {"prompt": [0, "native-prompt", {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "model-Q4.gguf"}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "encoder.safetensors"}}
        }], "status": {"status_str": "success"}, "outputs": {}}}
        self.assertEqual(RECONCILE.prompt_ids(history), ["native-prompt"])
        self.assertEqual(RECONCILE.graph_models(history), ["encoder.safetensors", "model-Q4.gguf"])

    def test_successful_generated_image_can_be_rejected_for_quality(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output = root / "test.png"
            output.write_bytes(b"existing image marker")
            receipt = root / "validation" / "candidate.json"
            receipt.parent.mkdir()
            receipt.write_text(json.dumps({
                "summary": {"app": "moss-image", "status": "SUCCESS", "seconds": 303.54},
                "prompt_ids": ["known-job"],
                "media": [{"path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}],
                "visual_review": {"status": "REJECTED_TEXT_ARTIFACTS"},
            }))
            item = RECONCILE.parse_receipt(root, "validation", receipt)
            self.assertEqual(item["generation_status"], "GENERATED_SUCCESS")
            self.assertEqual(item["quality_gate"], "REJECTED_OR_NO_QUALIFIED_OUTPUT")
            self.assertEqual(item["media_checks"][0]["sha_check"], "MATCH")


if __name__ == "__main__":
    unittest.main()
