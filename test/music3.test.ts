import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { loadRegistry, type ApiWorkflow } from "../server/registry.ts";
import { applyInputs, resolveValues } from "../server/runner.ts";

const { apps, problems } = loadRegistry();

test("Music3 routes autoregressive encoding to the native GPU selector while preserving user bindings and sampling", () => {
  const app = apps.find((a) => a.id === "music");
  assert.ok(app?.graph);
  assert.equal(problems.some((p) => p.app === "music" && p.severity === "error"), false);
  assert.deepEqual(app.inputs.map((i) => i.id), ["style", "lyrics", "seconds", "seed"]);
  const duration = app.inputs.find((i) => i.id === "seconds")!;
  assert.deepEqual([duration.default, duration.min, duration.max, duration.step], [60, 10, 240, 5]);
  assert.equal(app.inputs.find((i) => i.id === "style")?.default, "upbeat cumbia with accordion and modern synths, 100 bpm");
  assert.equal(app.inputs.find((i) => i.id === "lyrics")?.default, "");
  assert.equal(app.inputs.find((i) => i.id === "seed")?.default, -1);
  assert.deepEqual(app.graph["37:44"], {
    class_type: "SelectCLIPDevice",
    inputs: { clip: ["37:3", 0], device: "gpu:0" },
    _meta: { title: "Music3 text encoder on gpu:0" },
  });
  assert.deepEqual(app.models.map((m) => [m.dir, m.file, m.gb]), [
    ["diffusion_models", "diffusion_models/MiniMax-Music3-Q6_K.gguf", 2.116],
    ["text_encoders", "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors", 9.2],
    ["vae", "vae/minimax_music3_dav.safetensors", 0.22],
  ]);

  for (const input of [
    { seed: 42 },
    { style: "Warm piano and brushed drums.", lyrics: "", seconds: 10, seed: 123 },
    { style: "Gentle acoustic folk.", lyrics: "[Verse]\nThe river carries us home.", seconds: 240, seed: 456 },
  ]) {
    const graph: ApiWorkflow = structuredClone(app.graph);
    const values = resolveValues(app, input);
    applyInputs(app, values, { setInput: (node, field, value) => { graph[node].inputs[field] = value; } }, () => {
      throw new Error("Music3 has no file input");
    });
    assert.deepEqual(graph["37:3"].inputs, {
      clip_name: "minimax_music3_text_encoder_pruned_int8_convrot.safetensors", type: "minimax", device: "default",
    });
    assert.deepEqual(graph["37:44"].inputs, { clip: ["37:3", 0], device: "gpu:0" });
    assert.deepEqual(graph["37:13"].inputs, {
      clip: ["37:44", 0], seed: ["37:38", 0], caption: values.style, lyrics: values.lyrics,
      max_duration: values.seconds, cfg_scale: 1.7, top_k: 50,
    });
    assert.equal(graph["37:38"].inputs.seed, input.seed);
    assert.equal(graph["37:6"].class_type, "UnetLoaderGGUF");
    assert.deepEqual(graph["37:6"].inputs, { unet_name: "MiniMax-Music3-Q6_K.gguf" });
    assert.equal(app.models[0].repo, "Abiray/MiniMax-Music3-GGUF");
    assert.equal(app.models[1].file, "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors");
    assert.equal(app.models[2].file, "vae/minimax_music3_dav.safetensors");
    assert.deepEqual(graph["37:9"].inputs, {
      model: ["37:6", 0], positive: ["37:13", 0], negative: ["37:10", 0], latent_image: ["37:15", 0],
      seed: ["37:38", 0], steps: 30, cfg: 1.7, sampler_name: "euler", scheduler: "simple", denoise: 1,
    });
    assert.deepEqual(graph["37:15"].inputs, { seconds: ["37:13", 1], batch_size: 1 });
    assert.deepEqual(graph["37:7"].inputs, { vae_name: "minimax_music3_dav.safetensors" });
    assert.deepEqual(graph["37:43"].inputs, { on_false: ["37:12", 0], on_true: ["37:42", 0], switch: false });
    assert.deepEqual(graph["37:12"].inputs, { samples: ["37:9", 0], vae: ["37:7", 0] });
    assert.deepEqual(graph["35"].inputs, {
      audio: ["37:43", 0], filename_prefix: "freefield/music", format: "mp3", "format.quality": "V0",
    });
  }
});

test("the offline catalogue declares the native SelectCLIPDevice input and output contracts", () => {
  const info = JSON.parse(readFileSync(new URL("../tools/object_info.json", import.meta.url), "utf8"));
  const node = info.SelectCLIPDevice;
  assert.ok(node, "SelectCLIPDevice must be available to workflow conversion");
  assert.equal(node.python_module, "comfy_extras.nodes_multigpu");
  assert.deepEqual(node.input.required.clip, ["CLIP", {}]);
  assert.equal(node.input.required.device[0], "COMBO");
  assert.deepEqual(node.input_order.required, ["clip", "device"]);
  assert.deepEqual(node.output, ["CLIP"]);
});

test("Music3 leaves the new native GPU profile's dedicated VRAM peak unmeasured", () => {
  const app = apps.find((a) => a.id === "music")!;
  assert.equal(app.vram?.peak_gb, null);
  const map = JSON.parse(readFileSync(new URL("../docs/vram-map.json", import.meta.url), "utf8"));
  const row = map.apps.find((r: { id: string }) => r.id === "music");
  assert.equal(row.measured_peak_gb, null);
  assert.equal(row.measured_on, null);
});
