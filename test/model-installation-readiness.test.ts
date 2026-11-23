import { test } from "node:test";
import assert from "node:assert/strict";
import { loadRegistry, missingRequiredModels } from "../server/registry.ts";
const { apps } = loadRegistry();
function app(id: string) {
  const found = apps.find((a) => a.id === id);
  assert.ok(found);
  return found;
}
function withoutAliases(id: string) {
  return app(id).models.filter((m) => !m.optional)
    .map((m) => m.dir + "/" + (m.install_name ?? m.file).split("/").pop());
}
test("actual Music3 and Qwen2.1 GGUF files in unet_gguf are accepted without fake missing packs", () => {
  for (const id of ["music", "qwen21-edit"]) {
    const names = withoutAliases(id);
    const g = app(id).models.find((m) => m.dir === "diffusion_models" && m.file.endsWith(".gguf"));
    assert.ok(g);
    const gguf = g.file.split("/").pop();
    assert.ok(gguf);
    const installed = new Set(names.filter((x) => !x.endsWith("/" + gguf)));
    installed.add("unet_gguf/" + gguf);
    assert.deepEqual(missingRequiredModels(app(id), installed), []);
  }
});
test("original LTX2.5 absent checkpoints do NOT count as installed when other quantized variants exist", () => {
  const ids = ["video", "video-t2v"];
  for (const id of ids) {
    const names = withoutAliases(id);
    const installed = new Set(names.filter((x) =>
      !x.includes("ltx-2.5-22b-distilled-transformer-comfy-int8") &&
      !x.includes("gemma4-12b-with-proj-ltx-2.5-comfy-int8")));
    installed.add("unet_gguf/ltx-2.5-22b-distilled-UD-Q3_K_XL.gguf");
    installed.add("text_encoders/gemma4-12b-ltx-2.5-w4a8.safetensors");
    const missing = missingRequiredModels(app(id), installed);
    assert.ok(missing.some((x) => x.includes("ltx-2.5-22b-distilled-transformer-comfy-int8")));
    assert.ok(missing.some((x) => x.includes("gemma4-12b-with-proj-ltx-2.5-comfy-int8")));
  }
});
test("Kandinsky nested loader paths and audio directory model inventory resolve without flattening", () => {
  const candidate = app("kandinsky6-lite");
  const installed = new Set(withoutAliases(candidate.id));
  assert.deepEqual(missingRequiredModels(candidate, installed), []);
  const removed = "kandinsky6_audio_vae/bigvgan_generator.pt";
  installed.delete(removed);
  assert.ok(missingRequiredModels(candidate, installed).includes(removed));
});

test("both Kandinsky 6 manifests use official registered audio VAE category, and never silently accept a missing mandatory vocoder", () => {
  for (const id of ["kandinsky6-lite", "kandinsky6-lite-int8"]) {
    const a = app(id);
    const list = a.models.filter((m) => m.dir === "kandinsky6_audio_vae");
    assert.equal(list.length, 3);
    assert.ok(list.some((m) => m.file.endsWith("v1-44.pth")));
    assert.ok(list.some((m) => m.file.endsWith("bigvgan_vocoder/bigvgan_generator.pt")));
    const installed = new Set(withoutAliases(id));
    assert.deepEqual(missingRequiredModels(a, installed), []);
    installed.delete("kandinsky6_audio_vae/bigvgan_generator.pt");
    assert.ok(missingRequiredModels(a, installed).includes("kandinsky6_audio_vae/bigvgan_generator.pt"));
    const wrongDir = new Set(withoutAliases(id).map((v) => v.replace("kandinsky6_audio_vae/", "audio_vae/")));
    assert.ok(missingRequiredModels(a, wrongDir).length>=2);
  }
});
