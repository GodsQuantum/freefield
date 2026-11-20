import { test } from "node:test";
import assert from "node:assert/strict";
import { loadRegistry } from "../server/registry.ts";
import { applyInputs, resolveValues } from "../server/runner.ts";

const { apps, problems } = loadRegistry();
const fast = apps.find((entry) => entry.id === "nitrofusion2");
const quality = apps.find((entry) => entry.id === "z-image");

test("NitroFusion2: new option resolves without affecting the existing quality app", () => {
  assert.ok(fast?.graph, "NitroFusion2 has a complete native ComfyUI graph");
  assert.ok(quality?.graph, "Z-Image Turbo quality remains installed");
  assert.equal(problems.some((p) => p.app === "nitrofusion2" && p.severity === "error"), false);
  assert.equal(fast.status, "ready");
  assert.equal(fast.pack, "image-nitro");
  assert.equal(quality.graph["57:28"].inputs.unet_name, "z_image_turbo_int8_convrot.safetensors");
});

test("NitroFusion2: one-step W6A8 GPTQ with existing FP8 encoder and ae VAE", () => {
  assert.ok(fast?.graph);
  const g = fast.graph;
  assert.equal(g["1"].class_type, "UNETLoader");
  assert.equal(g["1"].inputs.unet_name, "nitrofusion2_w6a8_gptq.safetensors");
  assert.equal(g["3"].inputs.clip_name, "qwen_3_4b_fp8_mixed.safetensors");
  assert.equal(g["3"].inputs.type, "lumina2");
  assert.equal(g["6"].inputs.vae_name, "ae.safetensors");
  assert.equal(g["8"].class_type, "KSampler");
  assert.equal(g["8"].inputs.steps, 1);
  assert.equal(g["8"].inputs.cfg, 1);
  assert.equal(g["8"].inputs.sampler_name, "euler");
  assert.equal(g["8"].inputs.scheduler, "simple");
  assert.equal(g["2"].inputs.shift, 3);
  assert.equal(g["10"].class_type, "SaveImage");
  assert.equal(fast.outputs[0].node, "10");
  assert.equal(fast.models.length, 3, "one distinct DiT, two existing shared weights");
});

test("NitroFusion2: prompt, dimensions and seed have explicit reversible bindings", () => {
  assert.ok(fast?.graph);
  const binds = Object.fromEntries(fast.inputs.map((item) => [item.id, item.bind]));
  assert.deepEqual(binds["prompt"], [{ node: "4", field: "text" }]);
  assert.deepEqual(binds["width"], [{ node: "7", field: "width" }]);
  assert.deepEqual(binds["height"], [{ node: "7", field: "height" }]);
  assert.deepEqual(binds["seed"], [{ node: "8", field: "seed" }]);
  assert.equal(fast.graph["10"].inputs.filename_prefix, "freefield/nitrofusion2");
});
