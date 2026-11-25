import { test } from "node:test";
import assert from "node:assert/strict";
import { certifiedCatalog } from "../server/certified-catalog.ts";
import { loadRegistry, repoRoot } from "../server/registry.ts";
const root = repoRoot(), { apps } = loadRegistry(root);
test("certified production catalog exposes exactly reviewed applications, not installed candidates", () => {
  const c = certifiedCatalog(root, apps);
  assert.equal(c.apps.length, 30);
  assert.equal(c.proofs.size, c.apps.length);
  assert.ok(c.proofs.has("character-swap"));
  assert.ok(c.proofs.has("talking"));
  assert.ok(c.proofs.has("voice-clone"));
  assert.ok(c.proofs.has("upscale-video-hd"));
  assert.ok(c.proofs.has("ltx25-q3-t2va-lite"));
  assert.ok(c.proofs.has("recast"));
  assert.deepEqual(c.apps.find(a=>a.id==="recast")?.inputs.find(i=>i.id==="mode")?.options?.map(o=>o.value),[true]);
  for (const id of ["video", "video-t2v", "motion-transfer", "style-reference"]) assert.ok(!c.proofs.has(id), id);
  assert.equal(c.apps.find(a=>a.id==="moss-image")?.inputs.find(i=>i.id==="fast_mode")?.default,true);
  assert.equal(c.apps.find(a=>a.id==="styles")?.inputs.find(i=>i.id==="ultra_fast_mode")?.default,true);
});
test("explicit broader allowlist cannot expose unqualified apps; empty list fails closed", () => {
  assert.deepEqual(certifiedCatalog(root, apps, ["create-image","video","motion-transfer"]).apps.map(a=>a.id),["create-image"]);
  assert.throws(()=>certifiedCatalog(root, apps, ["video"]));
  assert.throws(()=>certifiedCatalog(root, apps, []));
});
test("workflow drift excludes a formerly certified recipe", () => {
  const altered = apps.map(a=>a.id==="create-image" ? {...a, workflowPath: apps.find(b=>b.id==="z-image")!.workflowPath}:a);
  const c=certifiedCatalog(root, altered);
  assert.ok(!c.proofs.has("create-image"));
  assert.equal(c.apps.length,29);
});
