import { test } from "node:test";
import assert from "node:assert/strict";
import { loadRegistry } from "../server/registry.ts";

const { apps, problems } = loadRegistry();
const moss = apps.find((a) => a.id === "moss-image");

test("MOSS fast: quality original graph preserved as default", () => {
  assert.ok(moss?.graph);
  assert.equal(problems.some((p) => p.app === "moss-image" && p.severity === "error"), false);
  assert.equal(moss.status, "ready");
  const g = moss.graph;
  assert.equal(g["18:1"].inputs.unet_name, "moss_image_sft_int8_convrot.safetensors");
  assert.equal(g["18:2"].inputs.clip_name, "qwen3.5_9b_w4a8_gptq.safetensors");
  assert.deepEqual(g["18:7"].inputs.model, ["18:24", 0]);
  assert.equal(g["18:7"].inputs.steps, 25);
  assert.equal(g["18:7"].inputs.cfg, 2.5);
  assert.equal(g["18:7"].inputs.sampler_name, "er_sde");
  assert.deepEqual(g["18:8"].inputs.samples, ["ff_moss_fast_switch", 0]);
  assert.equal(g["ff_moss_fast_switch"].inputs.switch, false);
  assert.deepEqual(g["ff_moss_fast_switch"].inputs.on_false, ["18:7", 0]);
  assert.equal(moss.test?.fast_mode, false);
});

test("MOSS fast: one 4step LoRA over existing INT8 model with native loader", () => {
  assert.ok(moss?.graph);
  const g = moss.graph;
  assert.equal(g["ff_moss_fast_lora"].class_type, "LoraLoaderModelOnly");
  assert.deepEqual(g["ff_moss_fast_lora"].inputs.model, ["18:24", 0]);
  assert.equal(g["ff_moss_fast_lora"].inputs.lora_name, "moss_image_sft_4step_v2.0.safetensors");
  assert.equal(g["ff_moss_fast_lora"].inputs.strength_model, 1);
  assert.equal(g["ff_moss_fast_sampler"].class_type, "KSampler");
  assert.deepEqual(g["ff_moss_fast_sampler"].inputs.model, ["ff_moss_fast_lora", 0]);
  assert.equal(g["ff_moss_fast_sampler"].inputs.steps, 4);
  assert.equal(g["ff_moss_fast_sampler"].inputs.cfg, 1);
  assert.equal(g["ff_moss_fast_sampler"].inputs.sampler_name, "euler");
  assert.equal(g["ff_moss_fast_sampler"].inputs.scheduler, "simple");
  assert.deepEqual(g["ff_moss_fast_switch"].inputs.on_true, ["ff_moss_fast_sampler", 0]);
  assert.equal(moss.models.filter((m) => m.file.includes("moss_image_sft_int8_convrot")).length, 1);
  assert.equal(moss.models.filter((m) => m.file === "loras/moss_image_sft_4step_v2.0.safetensors").length, 1);
});

test("MOSS fast: seed synchronizes both modes and the toggle binds only the switch", () => {
  assert.ok(moss?.graph);
  const inputs = Object.fromEntries(moss.inputs.map((x) => [x.id, x]));
  assert.deepEqual(inputs.fast_mode.bind, [{ node: "ff_moss_fast_switch", field: "switch" }]);
  assert.equal(inputs.fast_mode.default, false);
  assert.deepEqual(inputs.seed.bind, [{ node: "18:7", field: "seed" }, { node: "ff_moss_fast_sampler", field: "seed" }]);
  assert.deepEqual(inputs.steps.bind, [{ node: "18:7", field: "steps" }]);
  assert.deepEqual(inputs.rewrite.bind, [{ node: "18:16", field: "switch" }]);
  assert.deepEqual(inputs.width.bind, [{ node: "18:6", field: "width" }]);
  assert.deepEqual(inputs.height.bind, [{ node: "18:6", field: "height" }]);
});
