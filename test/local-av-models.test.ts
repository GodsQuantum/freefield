import { test } from "node:test";
import assert from "node:assert/strict";
import { loadRegistry } from "../server/registry.ts";

const { apps, problems } = loadRegistry();

test("native ACE-Step 1.5 Turbo music uses the official split 2B + two Qwen encoders and native MP3 output", () => {
  const music = apps.find((a) => a.id === "music-ace15-local");
  assert.ok(music, "missing low-memory native music app");
  assert.equal(music.pack, "audio-ace15");
  assert.equal(music.status, "draft", "new model must pass hardware E2E before production");
  assert.equal(music.graph?.["1"]?.class_type, "UNETLoader");
  assert.equal(music.graph?.["1"]?.inputs?.unet_name, "acestep_v1.5_turbo.safetensors");
  assert.equal(music.graph?.["2"]?.class_type, "DualCLIPLoader");
  assert.equal(music.graph?.["2"]?.inputs?.clip_name1, "qwen_0.6b_ace15.safetensors");
  assert.equal(music.graph?.["2"]?.inputs?.clip_name2, "qwen_1.7b_ace15.safetensors");
  assert.equal(music.graph?.["2"]?.inputs?.type, "ace");
  assert.equal(music.graph?.["7"]?.class_type, "ModelSamplingAuraFlow");
  assert.deepEqual(music.graph?.["8"]?.inputs?.model, ["7", 0]);
  assert.equal(music.graph?.["11"]?.class_type, "SaveAudioMP3");
  assert.equal(music.outputs[0]?.node, "11");
  assert.equal(music.inputs.find((i) => i.id === "duration")?.max, 1000, "do not add a needless short duration cap");
  assert.equal(music.inputs.find((i) => i.id === "generate_audio_codes")?.default, true);
  assert.ok((music.inputs.find((i) => i.id === "language")?.options?.length ?? 0) >= 50);
  assert.equal(music.models.length, 4);
  assert.equal(music.models.every((m) => m.license === "Apache-2.0"), true);
  assert.equal(problems.some((p) => p.app === music.id && p.severity === "error"), false);
});

test("Wan 2.2 5B Q6 local draft reuses the shared UMT5 encoder and Wan 2.2 VAE", () => {
  const wan = apps.find((a) => a.id === "wan22-5b-local");
  assert.ok(wan);
  assert.equal(wan.status, "draft", "hardware E2E required before exposing as usable");
  assert.equal(wan.graph?.["1"]?.class_type, "UnetLoaderGGUF");
  assert.equal(wan.graph?.["1"]?.inputs?.unet_name, "Wan2.2-TI2V-5B-Q6_K.gguf");
  assert.equal(wan.graph?.["6"]?.class_type, "Wan22ImageToVideoLatent");
  assert.equal(wan.graph?.["11"]?.class_type, "SaveVideo");
  assert.equal(wan.outputs[0]?.node, "11");
  assert.equal(wan.inputs.find((i) => i.id === "frames")?.max, 16381);
  assert.ok(wan.models.some((m) => m.file.endsWith("wan2.2_vae.safetensors")));
  assert.ok(wan.models.some((m) => m.file.endsWith("umt5_xxl_fp8_e4m3fn_scaled.safetensors")));
  assert.equal(problems.some((p) => p.app === wan.id && p.severity === "error"), false);
});

test("direct SFX does not require the optional Qwen prompt model by default", () => {
  const sfx = apps.find((a) => a.id === "sfx");
  assert.ok(sfx);
  const promptExpand = sfx.inputs.find((i) => i.id === "reprompt");
  assert.equal(promptExpand?.default, false);
  assert.equal(sfx.graph?.["52:35"]?.inputs?.value, false);
  assert.equal(sfx.graph?.["52:34"]?.class_type, "ComfySwitchNode");
  assert.ok(sfx.models.some((m) => m.file.endsWith("qwen3.5_2b_bf16.safetensors") && m.optional));
});
