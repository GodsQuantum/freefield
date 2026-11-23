import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { loadRegistry, modelKey, type ApiWorkflow } from "../server/registry.ts";
import { applyInputs, resolveValues } from "../server/runner.ts";

const { apps, problems } = loadRegistry();
const app = apps.find((a) => a.id === "kandinsky6-lite");

test("Kandinsky6 Lite is a separate opt-in video app with only the verified safe 25-frame graph", () => {
  assert.ok(app?.graph);
  assert.equal(app.status, "ready");
  assert.equal(app.pack, "video-kandinsky6-lite");
  assert.equal(app.section, "video");
  assert.equal(problems.some((p) => p.app === app.id), false);
  assert.equal(app.inputs.every((i) => !["duration", "frames", "width", "height", "steps", "fps"].includes(i.id)), true);
  assert.deepEqual(app.inputs.map((i) => i.id), ["prompt", "sound", "negative", "seed"]);
  assert.equal(Object.keys(app.graph).length, 12);
  assert.deepEqual(app.graph["7"].inputs, { width: 448, height: 256, length: 25, fps: 24, batch_size: 1 });
  assert.equal(app.graph["8"].class_type, "Kandinsky6Sampler");
  assert.equal(app.graph["8"].inputs.steps, 10);
  assert.equal(app.graph["8"].inputs.cfg, 1.0);
  assert.equal(app.graph["8"].inputs.sampler_name, "euler");
  assert.equal(app.graph["8"].inputs.scheduler, "simple");
  assert.deepEqual(app.graph["9"].inputs, {
    samples: ["8", 0], vae: ["3", 0], tile_size: 256, overlap: 32,
    temporal_size: 16, temporal_overlap: 4,
  });
  assert.equal(app.graph["9"].class_type, "VAEDecodeTiled");
  assert.equal(app.graph["10"].class_type, "Kandinsky6AudioVAEDecode");
  assert.equal(app.graph["11"].class_type, "CreateVideo");
  assert.deepEqual(app.graph["11"].inputs, {
    images: ["9", 0], audio: ["10", 0], fps: ["7", 1],
  });
  assert.equal(app.graph["12"].class_type, "SaveVideo");
  assert.equal(app.graph["12"].inputs.format, "mp4");
  assert.equal(app.graph["12"].inputs.codec, "h264");
  assert.deepEqual(app.outputs, [{ node: "12", type: "video", label: "Vidéo MP4 H264 + audio AAC" }]);
  const manifest = JSON.parse(readFileSync(new URL("../apps/kandinsky6-lite/app.json", import.meta.url), "utf8"));
  const pinnedPlugin = manifest.custom_nodes.find((x: { repo: string; commit: string }) => x.repo === "https://github.com/kandinskylab/kandinsky-6");
  assert.equal(pinnedPlugin?.commit, "73b892b730885e02ab58fbf7fc5cb10b78e1e3da");
  assert.ok(app.models.some((m) => m.dir === "audio_vae" && m.file.endsWith("/v1-44.pth")));
  assert.ok(app.models.some((m) => m.dir === "audio_vae" && m.file.endsWith("/bigvgan_generator.pt")));
  assert.ok(modelKey(app).includes("diffusion_models/Kandinsky-6.0-Lite-distill-5s-Diffusers/transformer/diffusion_pytorch_model.safetensors"));
});

test("Kandinsky6 Lite text/seed bindings never modify validated frame count, video decoder or audio path", () => {
  assert.ok(app?.graph);
  const graph: ApiWorkflow = structuredClone(app.graph);
  const values = resolveValues(app, {
    prompt: "A small fox running along a snowy forest path.",
    sound: "Soft footfalls on snow and distant rustling wind.",
    negative: "No text, blurry, flicker.",
    seed: 22672,
  });
  applyInputs(app, values, {
    setInput: (node, field, value) => { graph[node].inputs[field] = value; },
  }, () => { throw new Error("no uploaded files for this text-to-video app"); });
  assert.equal(graph["5"].inputs.video_caption, values.prompt);
  assert.equal(graph["5"].inputs.audio_caption, values.sound);
  assert.equal(graph["6"].inputs.video_caption, values.negative);
  assert.equal(graph["8"].inputs.seed, 22672);
  assert.deepEqual(graph["7"].inputs, app.graph["7"].inputs);
  assert.deepEqual(graph["9"].inputs, app.graph["9"].inputs);
  assert.deepEqual(graph["10"].inputs, app.graph["10"].inputs);
  assert.deepEqual(graph["11"].inputs, app.graph["11"].inputs);
  assert.equal(graph["12"].inputs.filename_prefix, "freefield/kandinsky6-lite");
});

test("Kandinsky6 model pack is isolated from LTX/Wan/H3 and totals explicit audio dependencies", () => {
  assert.ok(app);
  const pack = JSON.parse(readFileSync(new URL("../packs/video-kandinsky6-lite.json", import.meta.url), "utf8"));
  assert.deepEqual(pack.apps, ["kandinsky6-lite"]);
  assert.ok(pack.download_gb > 14 && pack.download_gb < 16);
  assert.equal(pack.models.length, 8);
  assert.ok(pack.models.some((m: { file: string }) => m.file.endsWith("bigvgan_vocoder/config.json")));
  const byFile = new Map(pack.models.map((m: { file: string; name: string }) => [m.file, m.name]));
  assert.equal(byFile.get("transformer/diffusion_pytorch_model.safetensors"),
    "Kandinsky-6.0-Lite-distill-5s-Diffusers/transformer/diffusion_pytorch_model.safetensors");
  assert.equal(byFile.get("mmproj-BF16.gguf"), "mmproj-Qwen2.5-VL-7B-Instruct-BF16.gguf");
  assert.equal(byFile.get("audio_vae/bigvgan_vocoder/bigvgan_generator.pt"),
    "bigvgan_vocoder/bigvgan_generator.pt");
  assert.equal(byFile.get("audio_vae/bigvgan_vocoder/config.json"),
    "bigvgan_vocoder/config.json");
  assert.equal(pack.custom_nodes.find((n: { repo: string }) =>
    n.repo === "https://github.com/kandinskylab/kandinsky-6")?.install_subdirectory, "comfyui");
  assert.equal(app.inputs.some((i) => i.type === "video" || i.type === "image"), false);
});
