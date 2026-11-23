import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { loadRegistry, modelKey, type ApiWorkflow } from "../server/registry.ts";
import { resolveValues, applyInputs } from "../server/runner.ts";

const { apps, problems } = loadRegistry();
const reference = apps.find((a) => a.id === "kandinsky6-lite");
const app = apps.find((a) => a.id === "kandinsky6-lite-int8");
const file = "Kandinsky6_Lite_distill_5s_int8_convrot.safetensors";

test("Kandinsky 6 INT8 native quant is independently selectable without changing BF16 source", () => {
  assert.ok(app?.graph && reference?.graph);
  assert.equal(problems.some((p) => p.app === app.id), false);
  assert.equal(app.section, "video");
  assert.equal(app.pack, "video-kandinsky6-lite-int8");
  assert.equal(app.status, "ready");
  assert.equal(app.graph["1"].class_type, "UNETLoader");
  assert.deepEqual(app.graph["1"].inputs, { unet_name: file, weight_dtype: "default" });
  assert.equal(reference.graph["1"].inputs.unet_name,
    "Kandinsky-6.0-Lite-distill-5s-Diffusers/transformer/diffusion_pytorch_model.safetensors");
  assert.equal(app.models[0].dir, "diffusion_models");
  assert.equal(app.models[0].repo, "shadwosii/Kandinsky-6.0");
  assert.equal(app.models[0].file, "diffusion_models/" + file);
  assert.equal(app.models[0].gb < reference.models[0].gb, true);
  assert.notEqual(modelKey(app), modelKey(reference));
  assert.deepEqual(app.inputs.map((i) => i.id), ["prompt", "sound", "negative", "seed"]);
  assert.deepEqual(app.models.slice(1), reference.models.slice(1));
  assert.equal(Object.keys(app.graph).length, Object.keys(reference.graph).length);
  for (const id of Object.keys(reference.graph)) {
    if (id === "1" || id === "12") continue;
    assert.deepEqual(app.graph[id], reference.graph[id], "shared deterministic graph "+id);
  }
  assert.equal(app.graph["12"].inputs.filename_prefix, "freefield/kandinsky6-lite-int8");
  assert.equal(app.graph["12"].inputs.format, "mp4");
  assert.equal(app.graph["12"].inputs.codec, "h264");
  assert.deepEqual(app.graph["7"].inputs, { width: 448, height: 256, length: 25, fps: 24, batch_size: 1 });
  assert.equal(app.graph["8"].inputs.steps, 10);
  assert.equal(app.graph["8"].inputs.cfg, 1);
  assert.equal(app.graph["9"].inputs.tile_size, 256);
  assert.equal(app.graph["9"].inputs.temporal_size, 16);
  assert.deepEqual(app.graph["11"].inputs.audio, ["10", 0]);
  const m = JSON.parse(readFileSync(new URL("../apps/kandinsky6-lite-int8/app.json", import.meta.url), "utf8"));
  assert.equal(m.source.notes.includes("523522487aeb9009a93e77f5d74bc1c817b1469aa6ce99b2b564c3da69c0f96e"), true);
});

test("Kandinsky INT8 text and sound bindings do not change memory-safe graph or audio", () => {
  assert.ok(app?.graph);
  const graph: ApiWorkflow = structuredClone(app.graph);
  const resolved = resolveValues(app, {
    prompt: "Small white dog chasing a falling leaf in a sunlit park",
    sound: "Rustling leaves and light footfalls, distant birds",
    negative: "Flicker, watermarks, still image",
    seed: 202611232340,
  });
  applyInputs(app, resolved, {
    setInput: (node, field, v) => { graph[node].inputs[field] = v; },
  }, () => { throw new Error("Pure text-to-video generation cannot upload files"); });
  assert.equal(graph["5"].inputs.video_caption, resolved.prompt);
  assert.equal(graph["5"].inputs.audio_caption, resolved.sound);
  assert.equal(graph["6"].inputs.video_caption, resolved.negative);
  assert.equal(graph["8"].inputs.seed, resolved.seed);
  for (const id of ["1","2","3","4","7","9","10","11","12"]) {
    assert.deepEqual(graph[id], app.graph[id], id + " must not change");
  }
});

test("portable INT8 pack provides native 4.166GB quant plus exact same 7 supporting weights", () => {
  assert.ok(app && reference);
  const p = JSON.parse(readFileSync(new URL("../packs/video-kandinsky6-lite-int8.json", import.meta.url), "utf8"));
  assert.deepEqual(p.apps, ["kandinsky6-lite-int8"]);
  assert.equal(p.models.length, 8);
  assert.ok(p.download_gb > 12 && p.download_gb < 13.2);
  assert.deepEqual(p.models.find((m: { name: string }) => m.name === file)?.repo, "shadwosii/Kandinsky-6.0");
  assert.equal(p.custom_nodes.some((n: { repo: string; install_subdirectory?: string }) =>
    n.repo === "https://github.com/kandinskylab/kandinsky-6" && n.install_subdirectory === "comfyui"), true);
  assert.deepEqual(p.models.filter((m: { name: string }) => m.name !== file).length, 7);
  assert.equal(p.models.some((m: { name: string }) =>
    m.name === "bigvgan_vocoder/bigvgan_generator.pt"), true);
});
