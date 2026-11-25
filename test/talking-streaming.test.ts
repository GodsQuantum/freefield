import { test } from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { loadRegistry, missingRequiredModels, repoRoot } from "../server/registry.ts";

const root = repoRoot();
const { apps, problems } = loadRegistry(root);
const app = apps.find(a => a.id === "talking")!;
const runtimeRoot = join(root, "infra/comfy_nodes/freefield_flashhead_streaming");
const lock = JSON.parse(readFileSync(join(runtimeRoot, "runtime-lock.json"), "utf8"));
const sha = (name: string) => createHash("sha256").update(readFileSync(join(runtimeRoot, name))).digest("hex");

test("Talking uses original audio, not text-driven voice or independent clips", () => {
  assert.equal(app.pack, "video-flashhead-lite");
  assert.deepEqual(app.inputs.map(i => i.id), ["portrait", "audio", "seed"]);
  assert.deepEqual(app.inputs.find(i => i.id === "audio")?.limits?.seconds, [2, 45]);
  assert.equal(app.graph?.["1"].class_type, "LoadImage");
  assert.equal(app.graph?.["2"].class_type, "LoadAudio");
  assert.equal(app.graph?.["3"].class_type, "FreefieldFlashHeadStreaming");
  assert.deepEqual(app.graph?.["3"].inputs.portrait, ["1", 0]);
  assert.deepEqual(app.graph?.["3"].inputs.audio, ["2", 0]);
  assert.equal(app.graph?.["4"].class_type, "SaveVideo");
  assert.deepEqual(app.graph?.["4"].inputs.video, ["3", 0]);
});

test("Talking loader aliases match actual installed filenames and pinned assets", () => {
  assert.equal(problems.filter(p => p.app === "talking").length, 0);
  const installed = new Set<string>(lock.aliases.map((a: {directory:string;name:string}) => a.directory + "/" + a.name));
  assert.equal(installed.size, 3);
  assert.deepEqual(missingRequiredModels(app, installed), []);
  const first = [...installed][0]; installed.delete(first);
  assert.deepEqual(missingRequiredModels(app, installed), [first]);
  assert.equal(lock.files.length, 8);
  for (const f of lock.files) assert.match(f.sha256, /^[0-9a-f]{64}$/);
  assert.equal(sha("__init__.py"), lock.adapter_sha256);
  assert.equal(sha("transformers5.patch"), lock.patch_sha256);
});
