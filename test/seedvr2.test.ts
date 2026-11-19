import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { loadRegistry, type ApiWorkflow } from "../server/registry.ts";
import { applyInputs, resolveValues } from "../server/runner.ts";

const { apps, problems } = loadRegistry();
const kj = {
  repo: "https://github.com/kijai/ComfyUI-KJNodes",
  commit: "d3cfe21625e5170126ce06fbfcfe1d88108688c3",
};

for (const [id, temporalSize] of [["upscale", 4096], ["upscale-video", 64]] as const) {
  test(id + ": bound inputs retain the qualified GPU VAE and temporal settings", () => {
    const app = apps.find((a) => a.id === id);
    assert.ok(app?.graph);
    assert.equal(problems.some((p) => p.app === id && p.severity === "error"), false);
    for (const scale of id === "upscale" ? [2, 4] : [2]) {
      const graph: ApiWorkflow = structuredClone(app.graph);
      const file = { bytes: new Uint8Array([1, 2, 3]), name: id === "upscale" ? "portrait.png" : "clip.mp4" };
      const values = id === "upscale" ? { image: file, scale, seed: 42 } : { clip: file, seed: 42 };
      applyInputs(app, resolveValues(app, values),
        { setInput: (node, field, value) => { graph[node].inputs[field] = value; } },
        () => "uploaded-" + file.name);

      assert.equal(graph["66:51"].class_type, "VAELoaderKJ");
      assert.deepEqual(graph["66:51"].inputs, {
        vae_name: "seedvr2_ema_vae_fp16.safetensors", device: "main_device", weight_dtype: "fp16",
      });
      for (const node of ["66:48", "66:55"]) {
        assert.deepEqual(graph[node].inputs.vae, ["66:51", 0]);
        assert.deepEqual(
          [graph[node].inputs.tile_size, graph[node].inputs.overlap, graph[node].inputs.temporal_size, graph[node].inputs.temporal_overlap],
          [256, 64, temporalSize, 8],
        );
      }
      assert.deepEqual(graph["66:52"].inputs, {
        unet_name: "seedvr2_3b_int8_convrot.safetensors", weight_dtype: "default",
      });
      assert.deepEqual(
        [graph["66:54"].inputs.steps, graph["66:54"].inputs.cfg, graph["66:54"].inputs.seed],
        [1, 1, 42],
      );
      assert.equal(graph["66:57"].inputs["resize_type.multiplier"], scale);
      if (id === "upscale") {
        assert.equal(graph["1"].inputs.image, "uploaded-portrait.png");
        assert.equal(app.inputs.find((i) => i.id === "scale")?.default, 2);
      } else {
        assert.equal(graph["73"].inputs.file, "uploaded-clip.mp4");
        assert.equal(graph["66:105"].inputs.value, false);
        assert.deepEqual(graph["66:99"].inputs, { latent: ["66:48", 0], temporal_overlap: 0, chunking_mode: "auto" });
        assert.deepEqual(graph["66:100"].inputs.temporal_overlap, ["66:99", 1]);
        assert.deepEqual(graph["66:54"].inputs.latent_image, ["66:102", 0]);
        assert.deepEqual(graph["66:55"].inputs.samples, ["66:103", 0]);
        assert.deepEqual(graph["66:75"].inputs, {
          images: ["66:59", 0], audio: ["66:74", 1], fps: ["66:74", 2], bit_depth: ["66:74", 3],
        });
      }
    }
  });
}

test("the upscale pack installs the pinned KJ dependency declared by both SeedVR2 apps", () => {
  for (const id of ["upscale", "upscale-video"]) {
    const manifest = JSON.parse(readFileSync(new URL("../apps/" + id + "/app.json", import.meta.url), "utf8"));
    assert.deepEqual(manifest.custom_nodes, [kj]);
  }
  const pack = JSON.parse(readFileSync(new URL("../packs/upscale.json", import.meta.url), "utf8"));
  const index = JSON.parse(readFileSync(new URL("../packs/index.json", import.meta.url), "utf8"));
  assert.deepEqual(pack.custom_nodes, [kj]);
  assert.deepEqual(index.upscale, {
    apps: pack.apps, download_gb: pack.download_gb, needs_hf_token: pack.needs_hf_token,
  });
});

test("the offline node catalogue accepts the SeedVR2 VAE device and precision inputs", () => {
  const info = JSON.parse(readFileSync(new URL("../tools/object_info.json", import.meta.url), "utf8"));
  const loader = info.VAELoaderKJ;
  assert.ok(loader, "VAELoaderKJ must be available to workflow conversion");
  assert.ok(loader.input.required.vae_name[0].includes("seedvr2_ema_vae_fp16.safetensors"));
  assert.ok(loader.input.required.device[0].includes("main_device"));
  assert.ok(loader.input.required.weight_dtype[0].includes("fp16"));
  assert.deepEqual(loader.output, ["VAE"]);
});
