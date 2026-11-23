#!/usr/bin/env python3
"""Import verified Freefield CLI results into the existing gallery without new GPU work.

Default import requires a SUCCESS receipt with a positive visual review,
complete replayable inputs, and no temporary workflow override.
The explicit --benchmark option archives only reviewed, successful native
benchmarks with immutable provenance and replayable=False, including legitimate
temporary workflow overrides. Visually rejected samples are excluded.
Files remain in validation/ and are hard-linked without extra media bytes.
Supports CLI out/history/<run-id>/out-01 media from named benchmark receipts.
Deterministic IDs and atomic rename make retries safe.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

REPO = Path(__file__).resolve().parent.parent
# In an IA-AV-Studio checkout, receipts live two levels above the Freefield repo.
# Outside that layout, use this repo by default. Both roots can be overridden.
BASE = Path(os.environ.get(
    "FREEFIELD_STUDIO_ROOT",
    str(REPO.parents[1] if REPO.parent.name == "sources" else REPO),
)).resolve()
RECEIPTS = BASE / "validation"
OUT = Path(os.environ.get("FREEFIELD_OUT", str(REPO / "out")))
APP_ROOT = REPO / "apps"
VALID_VISUAL = {"PASS", "PASS_SINGLE_SCENE", "PASS_WITH_LIMITATION", "PASS_WITH_LIMITATIONS",
                "VISUAL_PARITY_ON_SINGLE_NEUTRAL_PROMPT", "VISION_PASS"}
EXT_KIND = {".png": "image", ".jpg": "image", ".jpeg": "image",
            ".webp": "image", ".mp4": "video", ".webm": "video",
            ".mp3": "audio", ".wav": "audio", ".flac": "audio"}
ENTRY_ID = re.compile(r"^[0-9]{8}-[0-9]{9}-[0-9a-f]{4}$")
# A benchmark's GPU output may have been generated using a temporary node
# substitution, a different quantization or a special GPU-memory profile.
# Keep it for human comparison, but never offer "Recreate" as if the saved
# values were a complete recipe. Exclude visual failures and unknown QA.
BENCHMARK_REVIEW = VALID_VISUAL | {
    "PASS_WITH_LIMITATION", "PASS_STATIC_CONTACT",
    "TECHNICAL_PASS_STATIC_CONTACT_PASS", "FUNCTIONAL_AUDIO_PASS_NO_LISTENING",
}



def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def plan(name, history, validation_out, benchmark=False):
    filename = Path(name).name
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*\.json", filename):
        raise ValueError("Invalid receipt filename")
    receipt = RECEIPTS / filename
    data = json.loads(receipt.read_text())
    summary = data.get("summary", {})
    if summary.get("status") != "SUCCESS":
        raise ValueError(f"{filename}: no successful GPU job")
    review = data.get("visual_review", {}) or {}
    effective_review = review or (data.get("review", {}) if benchmark else {})
    quality = str(review.get("status") or review.get("result") or "")
    if benchmark:
        # Music tests include structural proof but no perceptual listening;
        # explicitly label that limitation, never call it a quality pass.
        if not quality:
            other = data.get("review") or {}
            quality = str(other.get("status") or other.get("result") or "")
        if not quality and data.get("audio_review", {}).get("listened") is False:
            quality = "AUDIO_FUNCTIONAL_ONLY_NO_LISTENING"
        if quality not in BENCHMARK_REVIEW | {"AUDIO_FUNCTIONAL_ONLY_NO_LISTENING"}:
            raise ValueError(f"{filename}: benchmark is not a qualified, explicitly labeled success ({quality})")
    elif quality not in VALID_VISUAL:
        raise ValueError(f"{filename}: missing positive visual review ({quality})")
    config = data.get("config", {})
    app = summary["app"]
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", app) or not (APP_ROOT / app / "app.json").is_file():
        raise ValueError(f"{filename}: unknown app")
    if config.get("workflow_override") and not benchmark:
        raise ValueError(f"{filename}: temporary workflow override not reproducible from saved UI values")
    if not isinstance(config.get("values"), dict):
        raise ValueError(f"{filename}: no original run values")

    t = dt.datetime.fromisoformat(summary["started_at"].replace("Z", "+00:00"))
    t = t.astimezone(dt.timezone.utc)
    ident = t.strftime("%Y%m%d-%H%M%S") + f"{t.microsecond // 1000:03d}" + "-" + hashlib.sha256(filename.encode()).hexdigest()[:4]
    assert ENTRY_ID.fullmatch(ident)
    seconds = float(summary["seconds"])
    if not 0 < seconds < 3600:
        raise ValueError(f"{filename}: invalid measured seconds")
    finished = t + dt.timedelta(seconds=seconds)

    root = validation_out.resolve()
    media_dir = (root / Path(filename).stem).resolve()
    if not media_dir.is_relative_to(root):
        raise ValueError("Invalid media directory")
    media_descriptors = data.get("media") or []
    expected = {Path(v["path"]).name: v for v in media_descriptors if isinstance(v, dict) and v.get("path")}
    app_spec = json.loads((APP_ROOT / app / "app.json").read_text())
    input_types = {x["id"]: x.get("type") for x in app_spec.get("inputs", [])}
    inputs = []
    for key, source_path in (config.get("files") or {}).items():
        if not isinstance(source_path, str) or not source_path.startswith("/inputs/"):
            raise ValueError(f"{filename}: input must reference existing validation fixture: {key}")
        if input_types.get(key) not in ("image", "video", "audio"):
            raise ValueError(f"{filename}: invalid input kind: {key}")
        src = (root / source_path[len("/inputs/"):]).resolve(strict=True)
        if not src.is_relative_to(root) or not src.is_file():
            raise ValueError(f"{filename}: input outside validation fixtures")
        if src.suffix.lower() not in EXT_KIND or EXT_KIND[src.suffix.lower()] != input_types[key]:
            raise ValueError(f"{filename}: input media kind mismatch")
        checksum = sha(src)
        reference_sha = effective_review.get("source_sha256")
        if reference_sha and reference_sha != checksum:
            raise ValueError(f"{filename}: source-image SHA256 differs from visual review")
        filename_for_history = "input-" + key + src.suffix.lower()
        inputs.append((src, filename_for_history, {
            "id": key, "type": input_types[key], "file": filename_for_history,
            "name": src.name
        }, checksum))
    candidates = data.get("output_files") or [
        {"name": Path(x["path"]).name, "bytes": x["bytes"]}
        for x in media_descriptors if isinstance(x, dict) and x.get("path") and x.get("bytes")
    ]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"{filename}: no persisted output files")
    # CLI with FREEFIELD_OUT=/out writes input-image.png plus a meta.json
    # under the run's private history. Only its true generated out-* media
    # are eligible for benchmark import. Never mislabel an uploaded input
    # as a generated image.
    def is_cli_history_item(item):
        if not isinstance(item, dict):
            return False
        parts = Path(str(item.get("name") or "")).parts
        return len(parts) == 3 and parts[0] == "history"
    if any(is_cli_history_item(item) for item in candidates):
        if not benchmark:
            raise ValueError(f"{filename}: nested CLI history requires reviewed benchmark mode")
        candidates = [item for item in candidates if is_cli_history_item(item)
                      and re.fullmatch(r"out-[0-9]{2}\.(png|jpg|jpeg|webp|mp4|webm|mp3|wav|flac)",
                                       Path(str(item.get("name") or "")).name)]
        if not candidates:
            raise ValueError(f"{filename}: no generated output (only CLI input/meta files)")
    outputs = []
    for i, candidate in enumerate(candidates, 1):
        rel = Path(str(candidate.get("name") or ""))
        nested = len(rel.parts) == 3 and rel.parts[0] == "history" and benchmark
        if nested:
            if (not ENTRY_ID.fullmatch(rel.parts[1])
                    or not re.fullmatch(r"out-[0-9]{2}\.(png|jpg|jpeg|webp|mp4|webm|mp3|wav|flac)", rel.name)):
                raise ValueError(f"{filename}: unsafe nested CLI output")
        elif rel.name != str(rel):
            raise ValueError(f"{filename}: unsafe output file")
        if rel.is_absolute() or rel.suffix.lower() not in EXT_KIND:
            raise ValueError(f"{filename}: unsafe output file")
        source = (media_dir / rel).resolve(strict=True)
        if not source.is_relative_to(media_dir) or source.is_symlink() or not source.is_file():
            raise ValueError(f"{filename}: output outside owned directory")
        if source.stat().st_size != int(candidate.get("bytes", -1)):
            raise ValueError(f"{filename}: media size mismatch: {rel}")
        checksum = sha(source)
        if expected.get(rel.name, {}).get("sha256") not in (None, checksum):
            raise ValueError(f"{filename}: SHA256 mismatch: {rel}")
        if benchmark and len(candidates) == 1:
            reviewed_sha = (effective_review.get("output_sha256") or effective_review.get("new_output_sha256"))
            if reviewed_sha and reviewed_sha != checksum:
                raise ValueError(f"{filename}: verified review output SHA256 mismatch")
        output = f"out-{i:02d}{rel.suffix.lower()}"
        outputs.append((source, output, {
            "file": output, "type": EXT_KIND[rel.suffix.lower()],
            "name": rel.name, "node": "validated-media",
        }, checksum))
    receipt_digest = sha(receipt)
    record = {
        "id": ident, "app": app, "status": "succeeded",
        "job": (data.get("prompt_ids") or [None])[0],
        "created": t.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "started": t.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "finished": finished.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "seconds": seconds, "total": seconds,
        "values": config["values"], "preset": None, "inputs": [x[2] for x in inputs],
        "outputs": [x[2] for x in outputs], "error": None, "retries": 0,
        "validation": {"origin": "cloud9-cli-qualification",
                       "receipt": filename, "receipt_sha256": receipt_digest,
                       "visual_review": quality, "input_replay_complete": True,
                       "media_sha256": {x[1]: x[3] for x in outputs + inputs},
                       **({
                           "origin": "cloud9-native-benchmark",
                           "benchmark_only": True,
                           "replayable": False,
                           "quality_status": quality,
                           "workflow_override": bool(config.get("workflow_override")),
                           "source_graph_sha256": hashlib.sha256(
                               json.dumps(config["workflow_override"], sort_keys=True, default=str).encode("utf-8")
                           ).hexdigest() if config.get("workflow_override") else None,
                           "method": "verified-success-with-review; original media hardlinked",
                       } if benchmark else {})},
    }
    return record, outputs + inputs, history / ident


def import_one(name, history, validation_out, apply, benchmark=False):
    record, outputs, target = plan(name, history, validation_out, benchmark=benchmark)
    if target.exists():
        existing = json.loads((target / "meta.json").read_text())
        if (existing.get("validation", {}).get("receipt") == record["validation"]["receipt"]
            and existing.get("validation", {}).get("receipt_sha256") == record["validation"]["receipt_sha256"]):
            return {"receipt": name, "result": "ALREADY_IMPORTED", "id": record["id"]}
        raise FileExistsError(f"History id collision at {target}")
    if not apply:
        return {"receipt": name, "result": "DRY_RUN_OK", "id": record["id"],
                "media": [x[1] for x in outputs]}
    if any(source.stat().st_dev != history.stat().st_dev for source, _, _, _ in outputs):
        raise ValueError("Cross-filesystem import forbidden: would duplicate media")
    folder = Path(tempfile.mkdtemp(prefix=".import-freefield-", dir=history))
    try:
        for source, target_name, _, checksum in outputs:
            os.link(source, folder / target_name)
            if sha(folder / target_name) != checksum:
                raise ValueError("linked media differs from qualified source")
        m = folder / "meta.json"
        with m.open("x") as file:
            json.dump(record, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.rename(folder, target)
    except BaseException:
        for entry in folder.iterdir():
            entry.unlink()
        folder.rmdir()
        raise
    return {"receipt": name, "result": "IMPORTED", "id": record["id"],
            "history_meta_sha256": sha(target / "meta.json"), "hardlinks": len(outputs)}


def main():
    global RECEIPTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", nargs="+", help="Existing qualified validation receipts")
    parser.add_argument("--apply", action="store_true", help="Actually add history entries")
    parser.add_argument("--benchmark", action="store_true", help="Import named, reviewed technical benchmarks as non-replayable (never production-quality claims)")
    parser.add_argument("--receipts-dir", type=Path, default=RECEIPTS, help="Receipt directory, e.g. existing Freefield/ComfyUI/benchmarks; never a GPU runner")
    parser.add_argument("--history-dir", type=Path, default=OUT / "history")
    parser.add_argument("--validation-out", type=Path, default=OUT / "validation")
    a = parser.parse_args()
    if not a.history_dir.is_dir() or not a.validation_out.is_dir() or not a.receipts_dir.is_dir():
        parser.error("History, validation output or receipts folder is missing")
    RECEIPTS = a.receipts_dir.resolve()
    results = []
    for name in a.receipt:
        results.append(import_one(name, a.history_dir, a.validation_out, a.apply, benchmark=a.benchmark))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
