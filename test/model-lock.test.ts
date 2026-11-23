import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { buildRecipeLock } from "../server/model-lock.ts";
import { repoRoot } from "../server/registry.ts";

const lock = JSON.parse(readFileSync(join(repoRoot(), "model-selection.lock.json"), "utf8"));
const current = buildRecipeLock();

test("all Freefield app model files, LoRAs, loader classes, sampler profiles and runtime caps are explicitly pinned", () => {
  assert.equal(lock.schema_version, 1);
  assert.deepEqual(current, lock.recipes, "Unexpected model or inference profile drift: review evidence before re-locking");
});

test("preferred model selections refer to ready apps and documented fallback/proof", () => {
  assert.ok(Object.keys(lock.choices).length >= 9, "Expected supported Freefield model family choices");
  for (const [feature, choice] of Object.entries(lock.choices) as [string, any][]) {
    assert.equal(current[choice.selected]?.status, "ready", `${feature}: selected app must be ready`);
    if (choice.fallback) assert.equal(current[choice.fallback]?.status, "ready", `${feature}: fallback unavailable`);
    assert.ok(typeof choice.evidence === "string" && choice.evidence.length >= 8,
      `${feature}: no actual evidence or decision receipt`);
    assert.ok(["full_cli_e2e", "native_generated", "existing_e2e", "retained_no_better_proven"].includes(choice.assurance),
      `${feature}: invalid evidence grade`);
  }
  for (const candidate of lock.challengers_unqualified) {
    assert.ok(typeof candidate.id === "string" && candidate.id.length > 0);
    assert.ok(typeof candidate.reason === "string" && candidate.reason.length > 0);
  }
});

test("H3 compact selections stay present and older INT8 variants remain available as fallback", () => {
  const ids = ["h3-video", "h3-photo", "extend-video", "reshoot"];
  for (const id of ids) {
    const compact = current[`${id}-int4`];
    assert.equal(compact?.status, "ready");
    assert.equal(current[id]?.status, "ready");
    assert.ok(compact.models.some((model) => model.file.includes("int4")), id);
    assert.ok(current[id].models.some((model) => model.file.includes("int8")), id);
  }
  const qwen = lock.choices["qwen-image-general-edit"];
  assert.equal(qwen.selected, "qwen21-edit");
  assert.equal(qwen.assurance, "full_cli_e2e");
});

test("Qwen2511 specialized angle selection keeps real INT8 E2E; GGUF Q4 remains unqualified after two guarded runs", () => {
  const winner = lock.choices["qwen-specialized-angles"];
  assert.equal(winner.selected, "angles");
  assert.equal(winner.assurance, "full_cli_e2e");
  assert.equal(winner.evidence, "validation/qwen2511-angles-native-subquad-cli1024-20261119.json");
  const q4 = lock.challengers_unqualified.find((entry: { id: string }) => entry.id === "qwen-image-edit-2511-gguf-q4");
  assert.ok(q4, "rejected experimental model must be recorded for no-duplicate future benchmarks");
  assert.equal(q4.evidence,
    "Freefield/ComfyUI/benchmarks/00-qwen2511-q4_k_m-angles-1024-20261123-retry2.json");
  assert.match(q4.reason, /two|2.*(safety|abort)|both|both ended|attempted twice/i);
  assert.match(q4.reason, /0\s?OOM|zero\s?OOM/i);
  assert.ok(q4.additional_evidence.some((receipt: string) => receipt.endsWith("-retry1.json")));
  assert.ok(current.angles.models.some((m) => m.file.endsWith("/qwen_image_edit_2511_int8_convrot.safetensors")));
});
