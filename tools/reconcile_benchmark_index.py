#!/usr/bin/env python3
"""Rebuild/check the ONE existing Freefield/ComfyUI/historique-benchmarks-index.json.

Read-only examination of previous receipts, model decisions, outputs and legacy logs.
No ComfyUI contact, no downloads, no GPU generation, no model deletion.
Use --write after recording test receipts, or --check before changing a selected model.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

SOURCES = {
    "validation": "validation",
    "benchmarks": "Freefield/ComfyUI/benchmarks",
    "preflight": "Freefield/ComfyUI/preflight",
    "campaigns": "Freefield/ComfyUI/batches",
}
MODEL_FIELDS = {
    "ckpt_name", "unet_name", "clip_name", "clip_name1", "clip_name2",
    "clip_name3", "vae_name", "lora_name", "model_name", "vae_name",
}
MODEL_SUFFIXES = (".safetensors", ".gguf", ".ckpt", ".pt", ".pth")
GOOD_VISUAL = {"PASS", "VISION_PASS", "PASS_SINGLE_SCENE", "PASS_WITH_LIMITATION",
               "PASS_WITH_LIMITATIONS", "VISUAL_PARITY_ON_SINGLE_NEUTRAL_PROMPT"}
REJECT_KEYS = ("REJECT", "FAIL", "NO_OUTPUT", "UNREVIEW", "NOT_QUALIFIED")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scalar_status(value):
    if isinstance(value, dict):
        return value.get("status_str") or value.get("status") or "STRUCTURED_STATUS"
    if isinstance(value, str):
        return value
    if value is None:
        return None
    return str(value)


def find_graphs(d):
    out = []
    if isinstance(d.get("graph"), dict):
        out.append(d["graph"])
    cfg = d.get("config")
    if isinstance(cfg, dict):
        ov = cfg.get("workflow_override")
        if isinstance(ov, dict):
            out.append(ov.get("graph", ov))
        if isinstance(cfg.get("graph"), dict):
            out.append(cfg["graph"])
    native = d.get("history")
    if isinstance(native, dict):
        prompt = native.get("prompt")
        if isinstance(prompt, list) and len(prompt) >= 3 and isinstance(prompt[2], dict):
            out.append(prompt[2])
    history = d.get("histories")
    if isinstance(history, dict):
        for item in history.values():
            if not isinstance(item, dict):
                continue
            prompt = item.get("prompt")
            if isinstance(prompt, list) and len(prompt) >= 3 and isinstance(prompt[2], dict):
                out.append(prompt[2])
    return out


def graph_models(d):
    found = set()
    for graph in find_graphs(d):
        if not isinstance(graph, dict):
            continue
        for node in graph.values():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs", {})
            if not isinstance(inputs, dict):
                continue
            for key in MODEL_FIELDS:
                name = inputs.get(key)
                if isinstance(name, str) and name.lower().endswith(MODEL_SUFFIXES):
                    found.add(Path(name).name)
    return sorted(found)


def declared_candidate_models(d):
    found = set()
    values = [d.get(key) for key in (
        "file", "model", "model_file", "candidate_model", "candidate_models",
        "checkpoint", "checkpoint_file", "selected_model", "target_model", "models"
    )]
    cfg = d.get("config")
    if isinstance(cfg, dict):
        values.extend(cfg.get(key) for key in ("candidate_model", "candidate_models", "model_file"))
    def walk(item, depth=0):
        if depth > 3:
            return
        if isinstance(item, str) and item.lower().endswith(MODEL_SUFFIXES):
            found.add(Path(item).name)
        elif isinstance(item, dict):
            for key in ("file", "name", "filename", "model", "model_file", "checkpoint"):
                if key in item:
                    walk(item[key], depth+1)
        elif isinstance(item, list):
            for x in item:
                walk(x, depth+1)
    for value in values:
        walk(value)
    return sorted(found)


def prompt_ids(d):
    ids = set()
    for key in ("prompt_id", "native_prompt_id"):
        val = d.get(key)
        if isinstance(val, str) and val:
            ids.add(val)
    for key in ("prompt_ids",):
        values = d.get(key)
        if isinstance(values, list):
            ids.update(x for x in values if isinstance(x, str) and x)
    for key in ("histories",):
        values = d.get(key)
        if isinstance(values, dict):
            ids.update(x for x in values if isinstance(x, str) and x)
    native = d.get("history")
    if isinstance(native, dict):
        prompt = native.get("prompt")
        if isinstance(prompt, list) and len(prompt) > 1 and isinstance(prompt[1], str):
            ids.add(prompt[1])
    p = d.get("owned_prompt")
    if isinstance(p, dict):
        for key in ("prompt_id", "id"):
            val = p.get(key)
            if isinstance(val, str) and val:
                ids.add(val)
    elif isinstance(p, str):
        ids.add(p)
    return sorted(ids)


def first_number(*values):
    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return round(float(value), 4)
    return None


def review_status(d):
    for key in ("visual_review", "review", "audio_review"):
        value = d.get(key)
        if isinstance(value, dict):
            text = value.get("status") or value.get("result") or value.get("quality_status")
            if isinstance(text, str) and text:
                return text
    return None


def decision_status(d):
    for key in ("selection", "technical_selection", "decision"):
        value = d.get(key)
        if isinstance(value, dict):
            status = value.get("status") or value.get("decision")
            if isinstance(status, str) and status:
                return status[:240]
        if isinstance(value, str) and value:
            return value[:240]
    return None


def media_evidence(d, source, stem):
    candidates = d.get("media") or d.get("output_files") or d.get("validated_files") or []
    if not isinstance(candidates, list):
        candidates = []
    candidates = list(candidates)
    # The native MP4 hash may live in the verified Library receipt, not in
    # the SaveVideo filename descriptor. Compare it with the actual bytes.
    archive = d.get("freefield_library_archive")
    if isinstance(archive, dict) and archive.get("benchmark_only") is True:
        digest, size = archive.get("sha256"), archive.get("bytes")
        if isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest):
            candidates = [
                {**item, **({"sha256": digest, "bytes": size}
                           if isinstance(item, dict)
                           and isinstance(item.get("filename"), str)
                           and item["filename"].lower().endswith(".mp4")
                           and not item.get("sha256")
                           else {})}
                for item in candidates
            ]
    # Other successful runners use a single CLI history path rather than an
    # output_files list. Preserve its independently recorded size and checksum.
    if isinstance(d.get("cli_output"), str):
        candidates.append({"path": d["cli_output"], "bytes": d.get("cli_output_bytes"),
                           "sha256": d.get("cli_output_sha256")})
    # Native ComfyUI receipts can embed their history and output descriptors.
    native = d.get("history")
    if isinstance(native, dict) and isinstance(native.get("outputs"), dict):
        root = Path("/srv/lxc/ia-compute/data/repos/ComfyUI/output")
        for node in native["outputs"].values():
            if not isinstance(node, dict):
                continue
            for output_items in node.values():
                if not isinstance(output_items, list):
                    continue
                for media_item in output_items:
                    if not isinstance(media_item, dict) or not media_item.get("filename"):
                        continue
                    fname = Path(media_item["filename"]).name
                    sub = str(media_item.get("subfolder", ""))
                    loc = (root / sub / fname).resolve()
                    if loc.is_relative_to(root) and fname == media_item["filename"]:
                        candidates.append({"path": str(loc)})
    rows = []
    emitted = set()
    for obj in candidates:
        if not isinstance(obj, dict):
            continue
        raw = obj.get("path")
        path = Path(raw) if isinstance(raw, str) and raw else None
        # Native benchmark receipts also declare a plain output basename.
        # Resolve it inside the corresponding validation run, never arbitrary cwd.
        if path is not None and not path.is_absolute() and path.name == raw:
            for root in (Path("/srv/lxc/ia-studio/appdata/freefield/out/validation") / stem,
                         Path("/srv/lxc/ia-compute/data/repos/ComfyUI/output/freefield/validation") / stem):
                if (root / path.name).is_file():
                    path = root / path.name
                    break
        # Early web benchmarks persist a stable authenticated Library URL,
        # not an absolute media path. Resolve to the existing mounted history
        # on this host rather than treating their MP3 as unverified.
        history_url = obj.get("url")
        if path is None and isinstance(history_url, str):
            m = re.fullmatch(r"/api/history/([0-9]{8}-[0-9]{9}-[a-z0-9]{4})/files/([A-Za-z0-9_.-]+)", history_url)
            if m:
                path = Path("/srv/lxc/ia-studio/appdata/freefield/out/history") / m.group(1) / m.group(2)
        # Native Comfy SaveVideo/SaveAudio nodes report filename + subfolder
        # in the history, without a full path.
        if path is None and isinstance(obj.get("filename"), str):
            name = obj["filename"]
            sub = obj.get("subfolder") or ""
            root = Path("/srv/lxc/ia-compute/data/repos/ComfyUI/output")
            resolved = (root / str(sub) / name).resolve()
            if Path(name).name == name and resolved.is_relative_to(root):
                path = resolved
        if path is None and isinstance(obj.get("name"), str):
            name = Path(obj["name"]).name
            if name == obj["name"] and name.lower().endswith((".mp4", ".png", ".jpg", ".jpeg", ".webp", ".mp3", ".wav", ".flac")):
                for root in (
                    Path("/srv/lxc/ia-studio/appdata/freefield/out/validation") / stem,
                    Path("/srv/lxc/ia-compute/data/repos/ComfyUI/output/freefield/validation") / stem,
                ):
                    if (root / name).is_file():
                        path = root / name
                        break
        if path is None:
            continue
        key = str(path)
        if key in emitted:
            continue
        emitted.add(key)
        actual = {"path": key, "exists": path.is_file()}
        if not path.is_file():
            actual["status"] = "DECLARED_MEDIA_MISSING"
        else:
            size = path.stat().st_size
            actual["bytes"] = size
            expected_size = obj.get("bytes")
            if isinstance(expected_size, int):
                actual["size_check"] = "MATCH" if expected_size == size else "MISMATCH"
            digest = sha(path)
            actual["sha256"] = digest
            declared = obj.get("sha256")
            if isinstance(declared, str) and len(declared) == 64:
                actual["sha_check"] = "MATCH" if digest.lower() == declared.lower() else "MISMATCH"
            else:
                actual["sha_check"] = "COMPUTED_ONLY"
        rows.append(actual)
    return rows


def classify(source, status, app, ids, media_rows, d):
    if source == "preflight":
        return "PREQUALIFICATION_NOT_INFERENCE"
    if source == "campaigns":
        return "CAMPAIGN_SUMMARY_NOT_INFERENCE"
    run_status = str(status or "").upper()
    if "NOT_SUBMITTED" in run_status or "RESOURCE_GUARD_BLOCKED_NO" in run_status:
        return "NOT_SUBMITTED"
    if run_status.startswith(("STAGED_", "HISTORICAL_", "BASELINES_IMPORTED_")):
        return "HISTORICAL_EVIDENCE_NOT_NEW_INFERENCE"
    if "SAFETY_ABORT" in run_status or "INTERRUPT" in run_status or "MEMORY_ABORT" in run_status:
        return "ABORTED_GPU_RUN" if ids else "ABORTED_OR_BLOCKED_UNCONFIRMED_GPU"
    if "TIMEOUT" in run_status:
        return "TIMED_OUT_GPU_RUN" if ids else "TIMED_OUT_UNCONFIRMED_GPU"
    if run_status.startswith(("FAILED", "ERROR")):
        return "FAILED_GPU_RUN" if ids else "FAILED_BEFORE_OR_WITHOUT_GPU_PROOF"
    outputs = d.get("output_files") or d.get("media") or d.get("validated_files") or []
    native = d.get("native_status", {})
    if not isinstance(native, dict):
        native = {}
    if not native and isinstance(d.get("history"), dict):
        native = d["history"].get("status", {})
    # Direct ComfyUI benchmarks may store the native /history status under
    # this key. They remain native GPU results, not Freefield runner tests.
    if not native and isinstance(d.get("native_history_status"), dict):
        native = d["native_history_status"]
    if not isinstance(native, dict):
        native = {}
    native_success = native.get("status_str") == "success" and native.get("completed", True) is True
    real_success = (
        run_status == "SUCCESS"
        or run_status.startswith(("PASS_REAL_", "PASS_NATIVE_REAL_", "PASS_REAL_VIDEO_", "SUCCESS_REAL_"))
        or (run_status.startswith("SUCCESS_FREEFIELD_"))
        or native_success
    )
    if real_success:
        if not (isinstance(outputs, list) and outputs) and not media_rows:
            return "SUCCESS_REPORTED_NO_OUTPUT_LIST"
        if not ids and not d.get("history_id") and not d.get("run_id"):
            return "SUCCESS_REPORTED_NO_PROMPT_ID"
        return "GENERATED_SUCCESS"
    if run_status.startswith("PASS_"):
        return "OTHER_PASS_NOT_GENERATION_PROVEN"
    return "METADATA_OR_UNCLASSIFIED_NOT_GENERATION"


def parse_receipt(root, source, p):
    rel = p.relative_to(root).as_posix()
    d = json.loads(p.read_text())
    if not isinstance(d, dict):
        raise ValueError(f"Receipt must be a dictionary: {rel}")
    summary = d.get("summary") if isinstance(d.get("summary"), dict) else {}
    config = d.get("config") if isinstance(d.get("config"), dict) else {}
    native_history_status = d.get("history", {}).get("status") if isinstance(d.get("history"), dict) else None
    status = scalar_status(summary.get("status") or d.get("status") or d.get("native_status") or native_history_status)
    archive = d.get("freefield_library_archive")
    native_benchmark_archived = (
        isinstance(archive, dict) and archive.get("benchmark_only") is True
        and isinstance(d.get("native_history_status"), dict)
    )
    app = summary.get("app") or d.get("app") or config.get("app")
    if not isinstance(app, str) and native_benchmark_archived:
        app = archive.get("app")
    if not isinstance(app, str):
        app = None
    ids = prompt_ids(d)
    media = media_evidence(d, source, p.stem) if source in ("validation", "benchmarks") else []
    samples = d.get("samples") if isinstance(d.get("samples"), list) else []
    samples = [x for x in samples if isinstance(x, dict)]
    wall = first_number(summary.get("seconds"), d.get("elapsed_seconds"),
                        d.get("runtime_seconds"), d.get("generation_elapsed_seconds"),
                        d.get("wall_seconds"), d.get("native_wall_s"), d.get("wall_s"),
                        d.get("total"), d.get("metrics", {}).get("wall_s") if isinstance(d.get("metrics"), dict) else None,
                        samples[-1].get("elapsed_s") if samples else None)
    reading = d.get("metrics", {}) if isinstance(d.get("metrics"), dict) else {}
    min_ram = first_number(summary.get("min_available_gib"), d.get("min_available_gib"),
                           d.get("min_mem_available_gib"), d.get("sample_mem_available_min_gib"),
                           reading.get("min_host_available_gib"), reading.get("min_available_gib"))
    max_gtt = first_number(summary.get("max_gtt_gib"), d.get("max_gtt_gib"),
                           reading.get("max_GTT_gib"), reading.get("max_gtt_gib"))
    psi = first_number(summary.get("max_psi_full"), d.get("max_psi_full"),
                       d.get("max_psi_full10"), d.get("sample_max_psi_full10_pct"),
                       reading.get("max_psi_full10_pct"))
    if samples:
        ram_samples = [x["host_available_gib"] for x in samples if isinstance(x.get("host_available_gib"), (int, float))]
        gtt_samples = [x["gtt_gib"] for x in samples if isinstance(x.get("gtt_gib"), (int, float))]
        psi_samples = [x["psi_full_avg10_pct"] for x in samples if isinstance(x.get("psi_full_avg10_pct"), (int, float))]
        if min_ram is None and ram_samples:
            min_ram = round(min(ram_samples), 4)
        if max_gtt is None and gtt_samples:
            max_gtt = round(max(gtt_samples), 4)
        if psi is None and psi_samples:
            psi = round(max(psi_samples), 4)
    declared_outputs = d.get("output_files") or d.get("media") or d.get("validated_files") or []
    if not declared_outputs and media:
        declared_outputs = media
    visual = review_status(d)
    decision = decision_status(d)
    if any(x in (decision or "").upper() for x in REJECT_KEYS) or any(x in (visual or "").upper() for x in REJECT_KEYS):
        quality_gate = "REJECTED_OR_NO_QUALIFIED_OUTPUT"
    elif visual and visual.upper() in GOOD_VISUAL:
        quality_gate = "VISUAL_PASS_WITH_SCOPE_LIMITS"
    elif visual and "PASS" in visual.upper():
        quality_gate = "OTHER_POSITIVE_REVIEW"
    else:
        quality_gate = "NO_POSITIVE_VISUAL_REVIEW_IN_THIS_RECEIPT"
    generation = classify(source, status, app, ids, media, d)
    graph_names = graph_models(d)
    submitted = graph_names if source in ("validation", "benchmarks") and (
        generation in ("GENERATED_SUCCESS", "ABORTED_GPU_RUN", "TIMED_OUT_GPU_RUN", "FAILED_GPU_RUN")
    ) else []
    # Legacy index fields retained for existing readers.
    row = {"file": p.name, "source": source, "relative_path": rel,
           "execution_surface": "NATIVE_COMFY_ONLY" if native_benchmark_archived
               else "RUN_OR_RECEIPT_SURFACE_NOT_EXPLICITLY_MARKED",
           "size": p.stat().st_size, "receipt_sha256": sha(p),
           "status": status, "app": app, "seconds": wall,
           "min_available_gib": min_ram, "max_gtt_gib": max_gtt,
           "max_psi_full": psi, "outputs_count": len(declared_outputs) if isinstance(declared_outputs, list) else 0,
           "prompt_ids": ids, "generation_status": generation,
           "models_in_submitted_graph": submitted,
           "models_in_prepared_graph": graph_names if not submitted else [],
           "candidate_models_declared": declared_candidate_models(d),
           "visual_review": visual, "quality_gate": quality_gate,
           "selection_decision": decision}
    if media:
        row["media_checks"] = media
        row["media_files_verified"] = sum(x["exists"] and x.get("size_check") != "MISMATCH" and x.get("sha_check") != "MISMATCH" for x in media)
    if isinstance(d.get("status"), dict):
        row["root_status"] = scalar_status(d["status"])
    if isinstance(summary.get("started_at"), str):
        row["started_at"] = summary["started_at"]
    elif isinstance(d.get("start_utc"), str):
        row["started_at"] = d["start_utc"]
    elif isinstance(d.get("started_utc"), str):
        row["started_at"] = d["started_utc"]
    error = d.get("abort_reason") or d.get("error") or d.get("safety_abort_reason")
    if isinstance(error, str) and error:
        row["failure_reason"] = error[:350]
    if samples:
        max_oom = [x.get("oom_kill") for x in samples if isinstance(x.get("oom_kill"), int)]
        if max_oom:
            row["oom_kill_max"] = max(max_oom)
    return row


def validate_lock(root, snapshot):
    p = root / "sources/freefield/model-selection.lock.json"
    if not p.is_file():
        raise FileNotFoundError(f"Missing canonical pinned model selection: {p}")
    lock = json.loads(p.read_text())
    lookup = {x["relative_path"]: x for src in ("validation", "benchmarks", "preflight", "campaigns")
              for x in snapshot[src]}
    chosen = []
    problems = []
    for feature, item in lock.get("choices", {}).items():
        evidence = item.get("evidence")
        recorded = lookup.get(evidence)
        if recorded is None:
            problems.append(f"{feature}: missing lock evidence receipt {evidence}")
        elif item.get("assurance") in ("full_cli_e2e", "native_generated") and recorded["generation_status"] != "GENERATED_SUCCESS":
            problems.append(f"{feature}: claimed {item['assurance']} but indexed as {recorded['generation_status']} ({evidence})")
        else:
            bad = [x for x in recorded.get("media_checks", []) if not x["exists"] or x.get("size_check") == "MISMATCH" or x.get("sha_check") == "MISMATCH"]
            if bad:
                problems.append(f"{feature}: evidence media integrity failure {evidence}")
        chosen.append({"feature": feature, "selected": item["selected"], "assurance": item["assurance"],
                       "evidence": evidence, "evidence_verified": recorded is not None,
                       "generation_status": recorded["generation_status"] if recorded else None})
    for candidate in lock.get("challengers_unqualified", []):
        if candidate.get("evidence") not in lookup:
            problems.append(f"{candidate.get('id')}: challenger evidence missing {candidate.get('evidence')}")
    return {"lock_path": "sources/freefield/model-selection.lock.json",
            "lock_sha256": sha(p),
            "locked_app_count": len(lock.get("recipes", {})),
            "preferred_family_choices": chosen,
            "unqualified_candidate_count": len(lock.get("challengers_unqualified", [])),
            "problems": problems}


def build(root, old):
    sources = {}
    for name, sub in SOURCES.items():
        files = sorted((root / sub).glob("*.json"), key=lambda x: x.name)
        sources[name] = [parse_receipt(root, name, p) for p in files]
    logs = []
    for sub in ("validation", "Freefield/ComfyUI/benchmarks"):
        for p in sorted((root / sub).glob("*.log"), key=lambda x: x.name):
            logs.append({"file": p.relative_to(root).as_posix(), "size": p.stat().st_size, "sha256": sha(p),
                         "purpose": "HISTORICAL_LOG_NOT_INDEPENDENT_GENERATION_PROOF"})
    all_runs = sources["validation"] + sources["benchmarks"]
    verdicts = collections.Counter(x["generation_status"] for x in all_runs)
    apps = {}
    for row in all_runs:
        if not row.get("app"):
            continue
        dest = apps.setdefault(row["app"], {"actual_success": [], "failed_or_aborted": [], "other": []})
        if row["generation_status"] == "GENERATED_SUCCESS":
            dest["actual_success"].append(row["relative_path"])
        elif row["generation_status"] in ("ABORTED_GPU_RUN", "TIMED_OUT_GPU_RUN", "FAILED_GPU_RUN"):
            dest["failed_or_aborted"].append(row["relative_path"])
        else:
            dest["other"].append(row["relative_path"])
    for app in apps.values():
        for key in app:
            app[key].sort()
    model_runs = {}
    for row in all_runs:
        if row["generation_status"] not in ("GENERATED_SUCCESS", "ABORTED_GPU_RUN",
                                            "TIMED_OUT_GPU_RUN", "FAILED_GPU_RUN"):
            continue
        for model in row["models_in_submitted_graph"]:
            dest = model_runs.setdefault(model, {"successful_run_receipts": [], "failed_or_aborted_run_receipts": []})
            key = "successful_run_receipts" if row["generation_status"] == "GENERATED_SUCCESS" else "failed_or_aborted_run_receipts"
            dest[key].append(row["relative_path"])
    for model in model_runs.values():
        for key in model:
            model[key].sort()
    candidate_rows = {}
    for group in SOURCES:
        for row in sources[group]:
            for model in row["candidate_models_declared"]:
                candidate_rows.setdefault(model, []).append({
                    "evidence": row["relative_path"], "status": row["status"],
                    "generation_status": row["generation_status"]
                })
    for rows in candidate_rows.values():
        rows.sort(key=lambda x: x["evidence"])
    ids = {}
    for row in all_runs:
        for ident in row["prompt_ids"]:
            ids.setdefault(ident, []).append(row["relative_path"])
    shared = {k: sorted(v) for k, v in ids.items() if len(v) > 1}
    broken_media = []
    for row in all_runs:
        for media in row.get("media_checks", []):
            if not media["exists"] or media.get("size_check") == "MISMATCH" or media.get("sha_check") == "MISMATCH":
                broken_media.append({"receipt": row["relative_path"], "media": media})
    manifest = [(source, row["relative_path"], row["receipt_sha256"])
                for source in SOURCES for row in sources[source]]
    manifest += [("legacy_log", row["file"], row["sha256"]) for row in logs]
    lock_check = validate_lock(root, sources)
    manifest.append(("model-lock", lock_check["lock_path"], lock_check["lock_sha256"]))
    sig = hashlib.sha256(json.dumps(sorted(manifest), ensure_ascii=False).encode()).hexdigest()
    if old.get("evidence_manifest_sha256") == sig:
        updated = old.get("refreshed_at_cet")
    else:
        updated = dt.datetime.now(dt.timezone(dt.timedelta(hours=1))).isoformat(timespec="seconds")
    snapshot = {
        "schema_version": 2,
        "created_at_cet": old.get("created_at_cet", "2026-11-22T01:59:46+01:00"),
        "refreshed_at_cet": updated,
        "source_root": str(root / "validation"),
        "purpose": "Canonical, idempotent receipt index: previous successful AND failed/aborted tests remain proofs. No GPU run or model switch; never infer success from preflight/label alone.",
        "evidence_manifest_sha256": sig,
        "files": sources["validation"],       # backward-compatible legacy key
        "benchmarks": sources["benchmarks"],
        "preflight": sources["preflight"],
        "campaigns": sources["campaigns"],
        "logs": logs,
        "model_selection": lock_check,
        "by_app": dict(sorted(apps.items())),
        "by_submitted_graph_model": dict(sorted(model_runs.items())),
        "by_declared_candidate_model": dict(sorted(candidate_rows.items())),
        "shared_prompt_ids_evidence_not_new_runs": dict(sorted(shared.items())),
        "media_integrity_issues": broken_media,
        "counts": {
            "validation_json": len(sources["validation"]),
            "benchmarks_json": len(sources["benchmarks"]),
            "preflight_json": len(sources["preflight"]),
            "campaign_json": len(sources["campaigns"]),
            "legacy_logs": len(logs),
            "generation_classifications": dict(sorted(verdicts.items())),
            "apps_with_executed_receipts": len(apps),
            "model_files_in_submitted_graph": len(model_runs),
            "model_files_with_declared_candidate_evidence": len(candidate_rows),
            "technical_success_receipts_with_negative_visual_or_selection_decision": sum(
                row["generation_status"] == "GENERATED_SUCCESS" and
                row["quality_gate"] == "REJECTED_OR_NO_QUALIFIED_OUTPUT" for row in all_runs),
            "technical_success_receipts_with_positive_visual_review": sum(
                row["generation_status"] == "GENERATED_SUCCESS" and
                row["quality_gate"] in ("VISUAL_PASS_WITH_SCOPE_LIMITS", "OTHER_POSITIVE_REVIEW") for row in all_runs),
            "distinct_prompt_ids": len(ids),
            "prompt_ids_referenced_in_multiple_receipts": len(shared),
            "media_files_integrity_checked_sha256": sum(
                1 for row in all_runs for x in row.get("media_checks", [])
                if x["exists"] and x.get("sha256") is not None),
            "generated_receipts_with_declared_output_but_no_locatable_media": sum(
                row["generation_status"] == "GENERATED_SUCCESS"
                and row["outputs_count"] > 0 and not row.get("media_checks") for row in all_runs),
            "recorded_media_integrity_issues": len(broken_media),
            "lock_evidence_problems": len(lock_check["problems"]),
        },
    }
    return snapshot


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help="Atomic update of existing canonical index")
    action.add_argument("--check", action="store_true", help="Read-only drift and integrity check; fail if index stale")
    ap.add_argument("--studio-root", type=Path, default=Path(__file__).resolve().parents[3])
    args = ap.parse_args()
    root = args.studio_root.resolve()
    index = root / "Freefield/ComfyUI/historique-benchmarks-index.json"
    if not index.exists():
        raise FileNotFoundError(f"Refusing to invent a second index, canonical missing: {index}")
    old = json.loads(index.read_text())
    new = build(root, old)
    if args.check:
        same = new == old
        print(json.dumps({"status": "PASS_CANONICAL_INDEX_FRESH" if same else "STALE_CANONICAL_INDEX",
                          "counts": new["counts"], "manifest_sha256": new["evidence_manifest_sha256"],
                          "integrity_issues": new["media_integrity_issues"][:15],
                          "missing_lock_evidence": new["model_selection"]["problems"][:15]},
                         ensure_ascii=False, indent=2))
        return 0 if same and not new["media_integrity_issues"] and not new["model_selection"]["problems"] else 1
    if new != old:
        fd, tmp = tempfile.mkstemp(prefix=".hist-index-", suffix=".json", dir=index.parent)
        try:
            with os.fdopen(fd, "w") as out:
                json.dump(new, out, ensure_ascii=False, indent=2)
                out.write("\n")
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp, index)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    print(json.dumps({"status": "INDEX_UPDATED_OR_ALREADY_FRESH",
                      "counts": new["counts"], "manifest_sha256": new["evidence_manifest_sha256"],
                      "integrity_issues": new["media_integrity_issues"][:15],
                      "lock_problems": new["model_selection"]["problems"][:15]},
                     ensure_ascii=False, indent=2))
    return 0 if not new["media_integrity_issues"] and not new["model_selection"]["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
