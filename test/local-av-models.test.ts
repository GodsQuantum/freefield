import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { loadRegistry, modelKey, type ApiWorkflow } from "../server/registry.ts";
import { applyInputs, resolveValues } from "../server/runner.ts";

const { apps, problems } = loadRegistry();

test("native ACE-Step 1.5 Turbo music uses the official split 2B + two Qwen encoders and native MP3 output", () => {
  const music = apps.find((a) => a.id === "music-ace15-local");
  assert.ok(music, "missing low-memory native music app");
  assert.equal(music.pack, "audio-ace15");
  assert.equal(music.status, "ready", "30-second tiled decode passed the actual Freefield CLI without a memory-guard trip");
  assert.equal(music.graph?.["1"]?.class_type, "UNETLoader");
  assert.equal(music.graph?.["1"]?.inputs?.unet_name, "acestep_v1.5_turbo.safetensors");
  assert.equal(music.graph?.["2"]?.class_type, "DualCLIPLoader");
  assert.equal(music.graph?.["2"]?.inputs?.clip_name1, "qwen_0.6b_ace15.safetensors");
  assert.equal(music.graph?.["2"]?.inputs?.clip_name2, "qwen_1.7b_ace15.safetensors");
  assert.equal(music.graph?.["2"]?.inputs?.type, "ace");
  assert.equal(music.graph?.["7"]?.class_type, "ModelSamplingAuraFlow");
  assert.deepEqual(music.graph?.["8"]?.inputs?.model, ["7", 0]);
  assert.equal(music.graph?.["9"]?.class_type, "VAEDecodeAudioTiled");
  assert.deepEqual([music.graph?.["9"]?.inputs?.tile_size, music.graph?.["9"]?.inputs?.overlap], [128, 32], "preserve the validated bounded audio decoder");
  assert.deepEqual(music.graph?.["11"]?.inputs?.audio, ["9", 0]);
  assert.equal(music.graph?.["11"]?.class_type, "SaveAudioMP3");
  assert.equal(music.outputs[0]?.node, "11");
  assert.equal(music.inputs.find((i) => i.id === "duration")?.max, 1000, "do not add a needless short duration cap");
  assert.equal(music.inputs.find((i) => i.id === "generate_audio_codes")?.default, false, "use the validated 30-second direct-conditioning path; the musical planner remains optional");
  assert.equal(music.graph?.["4"]?.inputs?.generate_audio_codes, false);
  assert.deepEqual([music.test?.duration, music.test?.steps, music.test?.generate_audio_codes], [30, 8, false]);
  assert.ok((music.inputs.find((i) => i.id === "language")?.options?.length ?? 0) >= 50);
  assert.equal(music.models.length, 4);
  assert.equal(music.models.every((m) => m.license === "Apache-2.0"), true);
  assert.equal(problems.some((p) => p.app === music.id && p.severity === "error"), false);
});

test("Wan 2.2 5B Q6 local validated preview reuses shared UMT5 encoder and VAE", () => {
  const wan = apps.find((a) => a.id === "wan22-5b-local");
  assert.ok(wan);
  assert.equal(wan.status, "ready", "256x144 9-frame 4-step E2E MP4 is verified on Radeon 780M");
  assert.deepEqual([wan.test?.width,wan.test?.height,wan.test?.frames,wan.test?.steps,wan.test?.fps],[256,144,9,4,24]);
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

test("Wan2.2 image-to-video draft reuses the T2V weights and attaches a native LoadImage", () => {
  const t2v = apps.find((a) => a.id === "wan22-5b-local");
  const i2v = apps.find((a) => a.id === "wan22-5b-i2v-local");
  assert.ok(t2v && i2v);
  assert.equal(i2v.status, "ready", "image upload and Freefield gateway E2E confirmed on Radeon 780M");
  assert.match(i2v.summary ?? "", /qualité non validée pour publication/i,
    "technical MP4 compatibility must not be presented as production-quality video");
  const wanCard = JSON.parse(readFileSync(new URL("../apps/wan22-5b-i2v-local/app.json", import.meta.url), "utf8"));
  assert.match(wanCard.source.notes, /validation\/wan22-i2v-turbo-q6-res576x768-9f-5steps-20261121\.json/);

  assert.deepEqual(i2v.modes, t2v.modes);
  assert.equal(i2v.pack, t2v.pack);
  assert.deepEqual(i2v.models, t2v.models, "no duplicate downloads");
  assert.equal(i2v.graph?.["12"]?.class_type, "LoadImage");
  assert.deepEqual(i2v.graph?.["6"]?.inputs?.start_image, ["12", 0]);
  assert.deepEqual(i2v.inputs.find((x) => x.id === "image")?.bind, [{ node: "12", field: "image" }]);
  assert.equal(problems.some((p) => p.app === i2v.id && p.severity === "error"), false);
});

test("direct SFX does not require the optional Qwen prompt model by default", () => {
  const sfx = apps.find((a) => a.id === "sfx");
  assert.ok(sfx);
  const promptExpand = sfx.inputs.find((i) => i.id === "reprompt");
  assert.equal(promptExpand?.default, false);
  assert.equal(promptExpand?.type, "select");
  const graphFor = (choice: string) => {
    const sets: Record<string, unknown> = {};
    const wf = { setInput: (node: string, field: string, value: unknown) => { sets[`${node}.${field}`] = value; } };
    applyInputs(sfx, resolveValues(sfx, { description: "gentle rain", seconds: 3, reprompt: choice, seed: 42 }), wf, () => null);
    return sets;
  };
  const direct = graphFor("false");
  assert.deepEqual(direct["52:34.on_true"], ["52:31", 0], "unused Qwen branch must be unreachable");
  assert.equal(direct["52:35.value"], false);
  const expanded = graphFor("true");
  assert.deepEqual(expanded["52:34.on_true"], ["52:28", 0], "expanded mode must preserve Qwen branch");
  assert.equal(expanded["52:35.value"], true);
  assert.equal(sfx.graph?.["52:35"]?.inputs?.value, false);
  assert.equal(sfx.graph?.["52:34"]?.class_type, "ComfySwitchNode");
  assert.ok(sfx.models.some((m) => m.file.endsWith("qwen3.5_2b_bf16.safetensors") && m.optional));
});

for (const [id, selectorId, choices] of [["angles", "move", 10], ["shots", "shot", 9], ["zooms", "zoom", 5]] as const) {
  test(`${id}: all camera choices use native Qwen Edit 2511 INT8 with the matching fal and Lightning LoRAs`, () => {
    const app = apps.find((a) => a.id === id);
    assert.ok(app?.graph);
    assert.equal(app.graph["10:37"]?.class_type, "UNETLoader", "native quantized tensors must use the core loader");
    assert.deepEqual(app.graph["10:37"].inputs, {
      unet_name: "qwen_image_edit_2511_int8_convrot.safetensors", weight_dtype: "default",
    });
    assert.ok(app.models.some((m) => m.repo === "Comfy-Org/Qwen-Image-Edit"
      && m.file === "split_files/diffusion_models/qwen_image_edit_2511_int8_convrot.safetensors"
      && m.dir === "diffusion_models"));
    assert.equal(app.models.some((m) => m.dir === "unet_gguf"), false);
    assert.equal(app.vram?.peak_gb, null, "the native profile has no measured VRAM peak yet");
    assert.equal(problems.some((p) => p.app === id && p.severity === "error"), false);

    const selector = app.inputs.find((i) => i.id === selectorId);
    assert.equal(selector?.options?.length, choices);
    for (const option of selector!.options!) {
      const graph: ApiWorkflow = structuredClone(app.graph);
      applyInputs(app, resolveValues(app, {
        image: { bytes: new Uint8Array([1, 2, 3]), name: "portrait.png" },
        [selectorId]: option.value, details: "Keep the blue jacket.", seed: 42,
      }), { setInput: (node, field, value) => { graph[node].inputs[field] = value; } }, () => "uploaded.png");
      assert.match(String(graph["15"].inputs.value), /^<sks> /);
      assert.match(String(graph["15"].inputs.value), /Keep the blue jacket\.$/);
      assert.doesNotMatch(String(graph["15"].inputs.value), /\{details\}/);
      assert.deepEqual(graph["10:89"].inputs, {
        lora_name: "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        strength_model: 1, model: ["10:37", 0],
      });
      assert.deepEqual(graph["10:118"].inputs, {
        lora_name: "qwen-image-edit-2511-multiple-angles-lora.safetensors",
        strength_model: 1, model: ["10:89", 0],
      });
      assert.equal(graph["10:119"]?.class_type, "ModelAttentionBackend", "the sampler must retain the validated per-model attention override");
      assert.deepEqual(graph["10:119"].inputs, {
        model: ["10:118", 0], attention: "sub-quadratic attention",
      });
      assert.deepEqual(graph["10:66"].inputs.model, ["10:119", 0]);
      assert.deepEqual(graph["10:75"].inputs.model, ["10:66", 0]);
      assert.deepEqual(graph["10:3"].inputs.model, ["10:75", 0]);
      assert.deepEqual([graph["10:3"].inputs.steps, graph["10:3"].inputs.cfg, graph["10:3"].inputs.seed], [4, 1, 42]);
    }
  });
}

for (const [id, selectorId, lora, version] of [
  ["relight", "light", "Qwen-Image-Edit-2509-Relight.safetensors", "2511"],
  ["whats-next", "next", "next-scene_lora-v2-3000.safetensors", "2509"],
] as const) {
  test(id + ": native Qwen Edit " + version + " preserves all eight choices, LoRAs and prompt bindings", () => {
    const app = apps.find((a) => a.id === id);
    assert.ok(app?.graph);
    assert.equal(app.graph["10:37"].class_type, "UNETLoader");
    assert.deepEqual(app.graph["10:37"].inputs, {
      unet_name: "qwen_image_edit_" + version + "_int8_convrot.safetensors", weight_dtype: "default",
    });
    assert.ok(app.models.some((m) => m.repo === "Comfy-Org/Qwen-Image-Edit"
      && m.file === "split_files/diffusion_models/qwen_image_edit_" + version + "_int8_convrot.safetensors"
      && m.dir === "diffusion_models" && m.gb === 20.5));
    assert.equal(app.vram?.peak_gb, null, "the INT8 profile must not inherit a historical FP8 measurement");
    assert.equal(problems.some((p) => p.app === id && p.severity === "error"), false);
    assert.equal(app.inputs.find((i) => i.id === "seed")?.default, -1);
    const selector = app.inputs.find((i) => i.id === selectorId);
    assert.equal(selector?.options?.length, 8);
    assert.equal(selector?.default, id === "relight"
      ? "golden hour sunlight through a low window, warm color temperature, long soft shadows"
      : "the camera pulls back to reveal the wider surroundings");
    for (const option of selector!.options!) {
      const graph: ApiWorkflow = structuredClone(app.graph);
      applyInputs(app, resolveValues(app, {
        image: { bytes: new Uint8Array([1, 2, 3]), name: "portrait.png" },
        [selectorId]: option.value, details: "Keep the blue jacket.", seed: 42,
        ...(id === "relight" ? {
          direction: ", the light coming from the left", quality: ", soft light with blurred shadow edges",
          brightness: ", dim and low-key", color: ", blue colored light",
        } : {}),
      }), { setInput: (node, field, value) => { graph[node].inputs[field] = value; } }, () => "uploaded.png");
      assert.equal(graph["11"].inputs.image, "uploaded.png");
      assert.deepEqual(graph["10:89"].inputs, {
        model: ["10:37", 0], lora_name: "Qwen-Image-Edit-" + version + "-Lightning-4steps-V1.0-bf16.safetensors", strength_model: 1,
      });
      assert.deepEqual(graph["10:118"].inputs, { model: ["10:89", 0], lora_name: lora, strength_model: 1 });
      assert.equal(graph["10:119"]?.class_type, "ModelAttentionBackend");
      assert.deepEqual(graph["10:119"].inputs, { model: ["10:118", 0], attention: "sub-quadratic attention" });
      assert.deepEqual(graph["10:66"].inputs, { model: ["10:119", 0], shift: 3 });
      assert.deepEqual(graph["10:75"].inputs, { model: ["10:66", 0], strength: 1, pre_cfg: false });
      assert.deepEqual(graph["10:3"].inputs.model, ["10:75", 0]);
      assert.deepEqual([graph["10:3"].inputs.steps, graph["10:3"].inputs.cfg, graph["10:3"].inputs.seed], [4, 1, 42]);
      assert.deepEqual(graph["10:38"].inputs, {
        clip_name: "qwen_2.5_vl_7b_fp8_scaled.safetensors", type: "qwen_image", device: "default",
      });
      assert.equal(graph["10:39"].inputs.vae_name, "qwen_image_vae.safetensors");
      assert.equal(graph["10:117"].class_type, "FluxKontextImageScale");
      assert.deepEqual(graph["10:111"].inputs.prompt, ["14", 0]);
      assert.deepEqual(graph["14"].inputs.string_b, ["15", 0]);
      assert.equal(graph["14"].inputs.delimiter, "");
      if (id === "relight") {
        assert.equal(graph["14"].inputs.string_a, "重新照明,");
        assert.equal(graph["15"].inputs.value,
          "relight the image using " + option.value + ", the light coming from the left, soft light with blurred shadow edges, dim and low-key, blue colored light. Keep the blue jacket.");
      } else {
        assert.equal(graph["14"].inputs.string_a, "");
        assert.equal(graph["15"].inputs.value, "Next Scene: " + option.value + " Keep the blue jacket.");
      }
    }
  });
}

test("the Qwen edit pack keeps the native 2509 and 2511 weights and LoRAs separate", () => {
  const pack = JSON.parse(readFileSync(new URL("../packs/image-qwen-edit.json", import.meta.url), "utf8"));
  for (const [version, usedBy] of [["2509", ["whats-next"]], ["2511", ["angles", "relight", "shots", "zooms"]]] as const) {
    const name = "qwen_image_edit_" + version + "_int8_convrot.safetensors";
    const model = pack.models.find((m: { name: string }) => m.name === name);
    assert.ok(model, "the pack must install the native " + version + " weight file");
    assert.equal(model.dir, "diffusion_models");
    assert.equal(model.repo, "Comfy-Org/Qwen-Image-Edit");
    assert.equal(model.file, "split_files/diffusion_models/" + name);
    assert.deepEqual(model.used_by, usedBy);
  }
  assert.equal(pack.models.some((m: { name: string }) => m.name === "qwen_image_edit_2509_fp8_e4m3fn.safetensors"), false);
  assert.equal(pack.models.some((m: { dir: string }) => m.dir === "unet_gguf"), false);
  assert.equal(pack.custom_nodes.some((c: { repo: string }) => c.repo === "https://github.com/city96/ComfyUI-GGUF"), false);
  const index = JSON.parse(readFileSync(new URL("../packs/index.json", import.meta.url), "utf8"));
  assert.deepEqual(index["image-qwen-edit"], {
    apps: pack.apps, download_gb: pack.download_gb, needs_hf_token: pack.needs_hf_token,
  });
});

test("both native Qwen 2509 and 2511 VRAM maps leave unmeasured peaks unknown", () => {
  const map = JSON.parse(readFileSync(new URL("../docs/vram-map.json", import.meta.url), "utf8"));
  for (const id of ["relight", "whats-next"]) {
    const row = map.apps.find((r: { id: string }) => r.id === id);
    assert.equal(row.biggest_file.name, "qwen_image_edit_" + (id === "relight" ? "2511" : "2509") + "_int8_convrot.safetensors");
    assert.equal(row.measured_peak_gb, null);
    assert.equal(row.measured_on, null);
  }
});

test("SFX small checkpoint is opt-in, bounded to short effects, and unloads on model change", () => {
  const app = apps.find((a) => a.id === "sfx");
  assert.ok(app?.graph);
  assert.equal(app.graph["52:25"].inputs.ckpt_name, "stable_audio_3_medium.safetensors",
    "never change the original quality/music default");
  const renderer = app.inputs.find((i) => i.id === "renderer");
  assert.equal(renderer?.default, "medium");
  assert.equal(renderer?.options?.length, 2);
  assert.ok(app.models.some((m) => m.optional && m.file.endsWith("stable_audio_3_small_sfx.safetensors")));
  const medium = resolveValues(app, { category: "Music", seconds: 180, renderer: "medium", seed: 42 });
  const small = resolveValues(app, { category: "SFX", seconds: 3, renderer: "small_sfx", seed: 42 });
  assert.equal(modelKey(app, medium), modelKey(app),
    "existing model key should not change for original Medium default");
  assert.notEqual(modelKey(app, small), modelKey(app, medium),
    "gateway must unload Medium when the opt-in Small base model is selected");
  assert.match(modelKey(app, small), /active:52:25\.ckpt_name=stable_audio_3_small_sfx/);
  const graph: ApiWorkflow = structuredClone(app.graph);
  applyInputs(app, small, { setInput(node, field, value) { graph[node].inputs[field] = value; } }, () => null);
  assert.equal(graph["52:25"].inputs.ckpt_name, "stable_audio_3_small_sfx.safetensors");
  assert.deepEqual(graph["52:3"].inputs.model, ["52:25", 0]);
  assert.equal(resolveValues(app, { category: "One-shot", seconds: 120, renderer: "small_sfx", seed: 42 }).renderer, "small_sfx");
  assert.throws(() => resolveValues(app, { category: "Music", seconds: 30, renderer: "small_sfx", seed: 42 }), /renderer: category must be one of SFX, One-shot/);
  assert.throws(() => resolveValues(app, { category: "Instrument", seconds: 30, renderer: "small_sfx", seed: 42 }), /renderer: category must be one of/);
  assert.throws(() => resolveValues(app, { category: "SFX", seconds: 121, renderer: "small_sfx", seed: 42 }), /renderer: seconds must be <= 120/);
  assert.equal(problems.some((p) => p.app === app.id && p.severity === "error"), false);
  const pack = JSON.parse(readFileSync(new URL("../packs/audio.json", import.meta.url), "utf8"));
  assert.ok(pack.models.some((m: { name: string; optional?: boolean }) =>
    m.name === "stable_audio_3_small_sfx.safetensors" && m.optional));
  assert.equal(pack.models.find((m: { name: string }) => m.name === "stable_audio_3_medium.safetensors").optional, undefined);
});

test("What's Next: opt-in Qwen 2511 preserves original 2509 and the 2509 Next-Scene LoRA", () => {
  const app = apps.find((a) => a.id === "whats-next");
  assert.ok(app?.graph);
  const engine = app.inputs.find((i) => i.id === "engine");
  assert.equal(engine?.default, "2509");
  assert.deepEqual(engine?.options?.map((o) => o.value), ["2509", "2511-preview"]);
  assert.equal(app.inputs.find((i) => i.id === "next")?.options?.length, 8, "preserve every camera continuation");
  assert.ok(app.models.some((m) => m.optional && m.file.endsWith("qwen_image_edit_2511_int8_convrot.safetensors")));
  assert.ok(app.models.some((m) => m.optional && m.file.endsWith("Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")));
  assert.equal(app.graph["10:37"].inputs.unet_name, "qwen_image_edit_2509_int8_convrot.safetensors");
  assert.equal(app.graph["10:89"].inputs.lora_name, "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors");
  const vals = (choice: string) => resolveValues(app, {
    image: { bytes: new Uint8Array([1, 2, 3]), name: "portrait.png" },
    engine: choice, next: "the camera pulls back to reveal the wider surroundings",
    details: "", seed: 202611191712,
  });
  const original = vals("2509");
  const preview = vals("2511-preview");
  assert.equal(modelKey(app, original), modelKey(app), "default key remains backward compatible");
  assert.notEqual(modelKey(app, preview), modelKey(app, original), "gateway must unload the previous base DiT");
  assert.match(modelKey(app, preview), /active:10:37\.unet_name=qwen_image_edit_2511_int8_convrot\.safetensors/);
  const graphFor = (values: ReturnType<typeof resolveValues>) => {
    const graph: ApiWorkflow = structuredClone(app.graph!);
    applyInputs(app, values, { setInput(node, field, value) { graph[node].inputs[field] = value; } }, () => "portrait.png");
    return graph;
  };
  const g0 = graphFor(original);
  const g1 = graphFor(preview);
  assert.equal(g0["10:37"].inputs.unet_name, "qwen_image_edit_2509_int8_convrot.safetensors");
  assert.equal(g0["10:89"].inputs.lora_name, "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors");
  assert.equal(g1["10:37"].inputs.unet_name, "qwen_image_edit_2511_int8_convrot.safetensors");
  assert.equal(g1["10:89"].inputs.lora_name, "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors");
  for (const g of [g0, g1]) {
    assert.equal(g["10:118"].inputs.lora_name, "next-scene_lora-v2-3000.safetensors");
    assert.equal(g["10:3"].inputs.seed, 202611191712);
    assert.equal(g["10:3"].inputs.steps, 4);
    assert.equal(g["10:119"].inputs.attention, "sub-quadratic attention");
  }
  assert.equal(problems.some((p) => p.app === app.id && p.severity === "error"), false);
});


test("H3 INT4 compact: verified native model pair, safe default, old INT8 unchanged", () => {
  const baseline = apps.find((a) => a.id === "h3-video");
  const compact = apps.find((a) => a.id === "h3-video-int4");
  assert.ok(baseline?.graph && compact?.graph);
  assert.deepEqual(baseline.modes, ["h3-video", "h3-video-int4"]);
  assert.deepEqual(compact.modes, baseline.modes);
  assert.equal(Object.keys(compact.graph).length, 18, "no experimental SaveLatent node");
  assert.equal(compact.graph["140:127"].inputs.unet_name, "minimax_h3_fl2va_pruned_int4_convrot.safetensors");
  assert.equal(compact.graph["140:128"].inputs.clip_name, "qwen3vl_32b_minimax_h3_int4_convrot.safetensors");
  assert.equal(baseline.graph["140:127"].inputs.unet_name, "minimax_h3_fl2va_pruned_int8_convrot.safetensors");
  assert.equal(baseline.graph["140:128"].inputs.clip_name, "qwen3vl_32b_minimax_h3_int8_convrot.safetensors");
  assert.equal(compact.graph["115"].inputs.megapixels, 0.1);
  assert.equal(compact.graph["140:133"].inputs.value, 1);
  assert.equal(compact.graph["140:137"].inputs.value, 8);
  assert.deepEqual(compact.graph["140:122"].inputs.samples, ["140:125", 0]);
  assert.deepEqual(compact.graph["140:121"].inputs.samples, ["140:125", 0]);
  assert.deepEqual(compact.graph["140:130"].inputs.audio, ["140:121", 0], "audio output preserved");
  assert.equal(compact.graph["92"].class_type, "SaveVideo");
  const v = resolveValues(compact, { prompt: "red robot", duration: 1, steps: 8, seed: 202611201401 });
  const graph = structuredClone(compact.graph);
  applyInputs(compact, v, { setInput(node, field, value) { graph[node].inputs[field] = value; } }, () => "");
  assert.equal(graph["140:131"].inputs.prompt, "red robot");
  assert.equal(graph["140:133"].inputs.value, 1);
  assert.equal(graph["140:137"].inputs.value, 8);
  assert.equal(graph["140:129"].inputs.noise_seed, 202611201401);
  assert.notEqual(modelKey(compact), modelKey(baseline));
  assert.throws(() => resolveValues(compact, { prompt: "hi", duration: 3, steps: 8, seed: 1 }), /duration/);
  const pack = JSON.parse(readFileSync(new URL("../packs/video-h3.json", import.meta.url), "utf8"));
  for (const name of ["minimax_h3_fl2va_pruned_int4_convrot.safetensors", "qwen3vl_32b_minimax_h3_int4_convrot.safetensors"]) {
    const entry = pack.models.find((m: { name: string }) => m.name === name);
    assert.ok(entry?.optional, "compact weights must not bulk-download with old H3 pack");
  }
  assert.equal(problems.some((p) => p.app === compact.id && p.severity === "error"), false);
});


test("H3 Photo INT4: preserve first-frame + synchronized audio, native models, safe preview", () => {
  const old = apps.find((x) => x.id === "h3-photo");
  const app = apps.find((x) => x.id === "h3-photo-int4");
  assert.ok(old?.graph && app?.graph);
  assert.deepEqual(old.modes, ["h3-photo", "h3-photo-int4"]);
  assert.deepEqual(app.modes, old.modes);
  assert.equal(app.status, "ready");
  assert.equal(app.graph["105:6"].inputs.unet_name, "minimax_h3_fl2va_pruned_int4_convrot.safetensors");
  assert.equal(app.graph["105:13"].inputs.clip_name, "qwen3vl_32b_minimax_h3_int4_convrot.safetensors");
  assert.equal(old.graph["105:6"].inputs.unet_name, "minimax_h3_fl2va_pruned_int8_convrot.safetensors");
  assert.equal(old.graph["105:13"].inputs.clip_name, "qwen3vl_32b_minimax_h3_int8_convrot.safetensors");
  assert.equal(modelKey(app) !== modelKey(old), true);
  assert.equal(Object.keys(app.graph).length, 20);
  assert.equal(app.graph["114"].class_type, "LoadImage");
  assert.deepEqual(app.graph["105:104"].inputs.first_frame, ["119", 0]);
  assert.deepEqual(app.graph["119"].inputs.image, ["114", 0]);
  assert.deepEqual(app.graph["120"].inputs.image, ["119", 0]);
  assert.equal(app.graph["119"].inputs.megapixels, 0.1);
  assert.equal(app.graph["105:111"].inputs.value, 1);
  assert.equal(app.graph["105:124"].inputs.value, 8);
  assert.deepEqual(app.graph["105:10"].inputs.samples, ["105:14", 0]);
  assert.deepEqual(app.graph["105:23"].inputs.samples, ["105:14", 0]);
  assert.deepEqual(app.graph["105:91"].inputs.audio, ["105:23", 0]);
  assert.equal(app.graph["92"].class_type, "SaveVideo");
  assert.deepEqual(app.inputs.find((x) => x.id === "image")?.bind, [{ node: "114", field: "image" }]);
  assert.equal(app.test?.image, "web/examples/idol/input-image.jpg");
  const mockImage = { name: "robot.png", bytes: new Uint8Array([137,80,78,71]), type: "image/png" };
  const values = resolveValues(app, { image: mockImage, prompt: "robot waves", duration: 1, steps: 8, seed: 202611201401 });
  const graph = structuredClone(app.graph);
  applyInputs(app, values, { setInput(node, field, value) { graph[node].inputs[field] = value; } }, () => "mock-asset-id");
  assert.equal(graph["114"].inputs.image, "mock-asset-id");
  assert.equal(graph["105:104"].inputs.prompt, "robot waves");
  assert.equal(graph["105:111"].inputs.value, 1);
  assert.equal(graph["105:124"].inputs.value, 8);
  assert.equal(graph["105:15"].inputs.noise_seed, 202611201401);
  assert.throws(() => resolveValues(app, { prompt: "robot", duration: 1, steps: 8, seed: 2 }), /image: required/);
  assert.throws(() => resolveValues(app, { image: mockImage, prompt: "robot", duration: 3, steps: 8, seed: 2 }), /duration/);
  const pack = JSON.parse(readFileSync(new URL("../packs/video-h3.json", import.meta.url), "utf8"));
  for (const name of ["minimax_h3_fl2va_pruned_int4_convrot.safetensors", "qwen3vl_32b_minimax_h3_int4_convrot.safetensors"]) {
    const weight = pack.models.find((m: { name: string }) => m.name === name);
    assert.ok(weight?.optional && weight.used_by.includes("h3-video-int4") && weight.used_by.includes(app.id));
  }
  assert.equal(problems.some((p) => p.app === app.id && p.severity === "error"), false);
});
