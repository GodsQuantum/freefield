import { test } from "node:test";
import assert from "node:assert/strict";
import { loadRegistry } from "../server/registry.ts";

const { apps, problems } = loadRegistry();
const edit = apps.find((app) => app.id === "qwen21-edit");
const older2511 = apps.find((app) => app.id === "angles");
const older2509 = apps.find((app) => app.id === "whats-next");
const optimizedRelight = apps.find((app) => app.id === "relight");

test("Qwen Image 2.1: general editor is independent of 2511 Relight and remaining 2509 Whats Next", () => {
  assert.ok(edit?.graph);
  assert.equal(edit.status, "ready");
  assert.equal(edit.pack, "image-qwen21");
  assert.equal(problems.some((p) => p.app === "qwen21-edit" && p.severity === "error"), false);
  assert.equal(older2511?.graph?.["10:37"].inputs.unet_name, "qwen_image_edit_2511_int8_convrot.safetensors");
  assert.equal(optimizedRelight?.graph?.["10:37"].inputs.unet_name, "qwen_image_edit_2511_int8_convrot.safetensors");
  assert.equal(optimizedRelight?.graph?.["10:89"].inputs.lora_name, "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors");
  assert.equal(older2509?.graph?.["10:37"].inputs.unet_name, "qwen_image_edit_2509_int8_convrot.safetensors");
  assert.equal(edit.graph["1"].class_type, "UnetLoaderGGUF");
  assert.equal(edit.graph["1"].inputs.unet_name, "qwen-image-2.1-turbo-Q4_K_M.gguf");
  assert.equal(edit.models.length, 3, "one Q4 diffusion model, unchanged 8B INT8 text encoder and VAE");
  assert.equal(edit.models[0].repo, "unsloth/Qwen-Image-2.1-Turbo-GGUF");
  assert.equal(edit.models[0].file, "diffusion_models/qwen-image-2.1-turbo-Q4_K_M.gguf");
  assert.equal(edit.models[1].file, "text_encoders/qwen3vl_8b_int8_convrot.safetensors");
  assert.equal(edit.models[2].file, "vae/qwen_image_2.1_vae_bf16.safetensors");
});

test("Qwen Image 2.1: native eight-step sigmas, cache, reference image and conditioning", () => {
  assert.ok(edit?.graph);
  const g = edit.graph;
  assert.equal(g["2"].class_type, "QwenImage21Cache");
  assert.deepEqual(g["2"].inputs.model, ["1", 0]);
  assert.equal(g["3"].inputs.clip_name, "qwen3vl_8b_int8_convrot.safetensors");
  assert.equal(g["4"].inputs.vae_name, "qwen_image_2.1_vae_bf16.safetensors");
  assert.equal(g["6"].class_type, "TextEncodeQwenImage21");
  assert.deepEqual(g["6"].inputs["images.image_1"], ["5", 0]);
  assert.deepEqual(g["8"].inputs.positive, ["6", 0]);
  assert.deepEqual(g["8"].inputs.negative, ["6", 1]);
  assert.deepEqual(g["11"].inputs.latent_image, ["6", 2]);
  assert.equal(g["9"].class_type, "ManualSigmas");
  assert.equal(String(g["9"].inputs.sigmas).split(",").length, 9, "eight steps with final zero sigma");
  assert.equal(g["10"].inputs.sampler_name, "euler");
  assert.equal(g["13"].class_type, "SaveImage");
  assert.equal(g["13"].inputs.filename_prefix, "freefield/qwen21-edit");
});

test("Qwen Image 2.1: exact reversible user input bindings and one image output", () => {
  assert.ok(edit?.graph);
  const binds = Object.fromEntries(edit.inputs.map((input) => [input.id, input.bind]));
  assert.deepEqual(binds.image, [{ node: "5", field: "image" }]);
  assert.deepEqual(binds.prompt, [{ node: "6", field: "prompt" }]);
  assert.deepEqual(binds.negative_prompt, [{ node: "6", field: "negative_prompt" }]);
  assert.deepEqual(binds.resolution, [{ node: "6", field: "resolution" }]);
  assert.deepEqual(binds.seed, [{ node: "7", field: "noise_seed" }]);
  assert.equal(edit.inputs.find((x) => x.id === "image")?.required, true);
  assert.equal(edit.inputs.find((x) => x.id === "resolution")?.default, 768);
  assert.deepEqual(edit.outputs, [{ node: "13", type: "image", label: "Edited image" }]);
});
