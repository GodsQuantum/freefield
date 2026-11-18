import { test } from "node:test";
import assert from "node:assert/strict";
import { loadRegistry, modelKey, templateVars } from "../server/registry.ts";

const { apps, problems } = loadRegistry();

test("every manifest passes the schema and its bindings resolve", () => {
  const errors = problems.filter((p) => p.severity === "error");
  assert.deepEqual(errors, []);
});

test("ready apps all have a workflow on disk", () => {
  for (const app of apps.filter((a) => a.status === "ready")) {
    assert.ok(app.graph, `${app.id} has no workflow loaded`);
  }
});

test("folder name and id agree, ids are unique", () => {
  const ids = apps.map((a) => a.id);
  assert.equal(new Set(ids).size, ids.length);
  for (const a of apps) assert.ok(a.dir.endsWith(a.id), `${a.id} lives in ${a.dir}`);
});

test("apps that share models produce the same model key, so switching between them does not unload", () => {
  const edit = apps.find((a) => a.id === "edit")!;
  const cinema = apps.find((a) => a.id === "cinema-studio")!;   // FLUX 3 Klein text to image, the same three files as edit
  const video = apps.find((a) => a.id === "video")!;
  assert.equal(modelKey(edit), modelKey(cinema));
  assert.notEqual(modelKey(edit), modelKey(video));
});

test("a GGUF Qwen3-VL vision tower auto-loaded by CLIPLoaderGGUF does not emit a false missing-loader warning", () => {
  assert.equal(problems.some((issue) => issue.app === "flux3-klein-local" && issue.message.includes("mmproj-")), false);
});

test("templateVars finds placeholders", () => {
  assert.deepEqual(templateVars("{prompt}, shot on a {camera}"), ["prompt", "camera"]);
});

test("current local low-memory recipe uses quantized FLUX 3, never the superseded FLUX 2 recipe", () => {
  const current = apps.find((app) => app.id === "flux3-klein-local");
  assert.ok(current, "missing FLUX 3 local GGUF recipe");
  assert.equal(current.status, "ready", "quantized FLUX 3 must be a usable app, not a hidden draft");
  assert.equal(apps.some((app) => app.id === "flux2-klein-local"), false);
  assert.equal(current.graph?.["18:1"]?.class_type, "UnetLoaderGGUF");
  assert.equal(current.graph?.["18:1"]?.inputs?.unet_name, "flux-3-image-klein-UD-Q5_K_XL.gguf");
  assert.equal(current.graph?.["18:2"]?.class_type, "CLIPLoaderGGUF");
  assert.equal(current.graph?.["18:2"]?.inputs?.clip_name, "flux3-klein-te-heretic-dhq-Q4_K_M.gguf");
  assert.equal(current.graph?.["18:2"]?.inputs?.type, "flux3");
  assert.ok(current.models.some((m) => m.file === "mmproj-flux3-klein-te-heretic-dhq-f16.gguf"));
  // The backend inventories GGUF text weights as clip_gguf, even though
  // the host stores them under its shared text-encoders model directory.
  assert.deepEqual(current.models.filter((m) => /heretic-dhq/.test(m.file)).map((m) => m.dir), ["clip_gguf", "clip_gguf"]);
  assert.deepEqual(current.graph?.["18:7"]?.inputs?.model, ["18:1", 0]);
  assert.deepEqual(current.graph?.["18:7"]?.inputs?.positive, ["18:4", 0]);
  assert.deepEqual(current.graph?.["18:7"]?.inputs?.latent_image, ["18:6", 0]);
  assert.equal(current.outputs[0]?.node, "9", "output must reference a valid SaveImage node");
  assert.equal(current.graph?.["9"]?.class_type, "SaveImage");
  assert.equal(current.inputs.find((input) => input.id === "steps")?.default, 16);
});
