import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { loadRegistry, modelKey, type ApiWorkflow } from "../server/registry.ts";
import { resolveValues, applyInputs } from "../server/runner.ts";

const { apps, problems } = loadRegistry();
const app = apps.find((x) => x.id === "ltx25-q3-t2va-lite");

test("LTX2.5 Q3: separate reproducible 8-step 25-frame opt-in first-stage audio/video app", () => {
  assert.ok(app?.graph);
  assert.equal(app.status, "ready");
  assert.equal(app.section, "video");
  assert.equal(app.pack, "video-ltx25-q3-compact");
  assert.equal(problems.filter((p) => p.app === app.id).length, 0);
  assert.deepEqual(app.inputs.map((i) => i.id), ["prompt", "seed"]);
  assert.equal(Object.keys(app.graph).length, 29);
  assert.equal(app.graph["405:387"].inputs.device, "cpu");
  assert.deepEqual(app.graph["405:365"].inputs.positive, ["405:364", 0]);
  assert.deepEqual(app.graph["405:365"].inputs.negative, ["405:364", 0]);
  assert.equal(app.graph["405:373"], undefined);
  assert.equal(app.graph["405:388"].inputs.video_cfg, 1);
  assert.equal(app.graph["405:388"].inputs.audio_cfg, 1);
  assert.equal(Object.values(app.graph).filter((n) => n.class_type === "CLIPTextEncode").length, 1);

  assert.equal(app.graph["405:344"].class_type, "SamplerCustomAdvanced");
  assert.equal(app.graph["405:384"].class_type, "UnetLoaderGGUF");
  assert.equal(app.graph["405:384"].inputs.unet_name, "ltx-2.5-22b-distilled-UD-Q3_K_XL.gguf");
  assert.equal(app.graph["405:387"].inputs.clip_name, "gemma4-12b-ltx-2.5-w4a8.safetensors");
  assert.equal(app.graph["405:362"].inputs.value, 1);
  assert.equal(app.graph["405:361"].inputs.value, 24);
  assert.equal(app.graph["409"].inputs.megapixels, 0.12);
  assert.equal(app.graph["405:404"].inputs.sigmas, "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0");
  assert.deepEqual(app.graph["405:356"].inputs.width, ["405:353", 1]);
  assert.deepEqual(app.graph["405:356"].inputs.height, ["405:355", 1]);
  assert.deepEqual(app.graph["405:369"].inputs.av_latent, ["405:344", 0]);
  assert.deepEqual(app.graph["405:370"].inputs.audio, ["405:358", 0]);
  assert.deepEqual(app.graph["405:370"].inputs.images, ["405:374", 0]);
  assert.deepEqual(app.graph["405:364"].inputs.text, ["405:376", 0]);
  assert.deepEqual(app.graph["405:374"].inputs, {
    samples: ["405:369", 0], vae: ["405:385", 0],
    tile_size: 256, overlap: 32, temporal_size: 16, temporal_overlap: 4,
  });
  assert.deepEqual(app.graph["75"].inputs, {
    video: ["405:370", 0], filename_prefix: "freefield/ltx25-q3-t2va-lite",
    format: "mp4", codec: "h264",
  });
  assert.equal(Object.values(app.graph).filter((n) => n.class_type === "SamplerCustomAdvanced").length, 1);
  assert.equal(Object.values(app.graph).filter((n) => n.class_type === "CLIPLoader").length, 1);
  assert.equal(Object.values(app.graph).filter((n) => n.class_type === "LTXVLatentUpsampler").length, 0);
  assert.equal(Object.values(app.graph).filter((n) => n.class_type === "TextGenerateLTX2Prompt").length, 0);
  assert.ok(modelKey(app).includes("unet_gguf/ltx-2.5-22b-distilled-UD-Q3_K_XL.gguf"));
});

test("LTX2.5 Q3 seed and scene are bound without changing duration, resolution, audios, core model, or VAE", () => {
  assert.ok(app?.graph);
  const graph: ApiWorkflow = structuredClone(app.graph);
  const values = resolveValues(app, { prompt: "Golden dandelion nodding in the wind", seed: 456789 });
  applyInputs(app, values, {
    setInput: (node, field, value) => { graph[node].inputs[field] = value; },
  }, () => { throw new Error("Text to video must not upload files"); });
  assert.equal(graph["405:376"].inputs.value, values.prompt);
  assert.equal(graph["405:339"].inputs.noise_seed, 456789);
  for (const id of ["405:362", "409", "405:404", "405:374", "405:370", "405:387", "405:384", "405:386"]) {
    assert.deepEqual(graph[id].inputs, app.graph[id].inputs);
  }
});

test("LTX2.5 Q3 portable standalone pack preserves existing LTX2.5 baseline separately", () => {
  assert.ok(app);
  const pack = JSON.parse(readFileSync(new URL("../packs/video-ltx25-q3-compact.json", import.meta.url), "utf8"));
  assert.deepEqual(pack.apps, ["ltx25-q3-t2va-lite"]);
  assert.equal(pack.models.length, 4);
  assert.ok(pack.download_gb >= 24 && pack.download_gb < 25.5);
  assert.equal(pack.needs_hf_token, true);
  const original = apps.find((x) => x.id === "video-t2v");
  assert.ok(original?.graph);
  assert.equal(original.pack, "video-ltx25");
  assert.equal(original.graph["405:384"].class_type, "UNETLoader");
  assert.equal(original.graph["405:384"].inputs.unet_name, "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors");
  assert.equal(original.graph["405:368"].class_type, "SamplerCustomAdvanced");
  assert.equal(original.graph["405:348"].class_type, "LTXVLatentUpsampler");
});
