import { test } from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { loadRegistry, type ApiWorkflow } from "../server/registry.ts";
import { applyInputs, resolveValues } from "../server/runner.ts";

const { apps, problems } = loadRegistry();
const specs = [
  { id: "styles", consumers: ["30:16", "30:6"],
    graphBefore: "69de6ecef0cd28cebf48f6316f9105d53a9bce8c97a82bd9e3361abad7581731",
    manifest: "9c3a11977fb27245aa26383128ee6be6da0b5878430dac054334cb5ae3a27ef5" },

];
const sha = (value: string | Uint8Array) => createHash("sha256").update(value).digest("hex");

function gpuSelector(graph: ApiWorkflow): string {
  const selectors = Object.entries(graph).filter(([, node]) => node.class_type === "SelectCLIPDevice");
  assert.equal(selectors.length, 1, "one native GPU selector must serve both Krea CLIP consumers");
  const [id, selector] = selectors[0];
  assert.deepEqual(selector.inputs, { clip: ["30:11", 0], device: "gpu:0" });
  return id;
}

for (const spec of specs) {
  test(`${spec.id}: the native GPU selector preserves every pre-existing node and manifest control`, () => {
    const app = apps.find((a) => a.id === spec.id);
    assert.ok(app?.graph);
    assert.equal(problems.some((p) => p.app === spec.id && p.severity === "error"), false);
    const selector = gpuSelector(app.graph);
    assert.deepEqual(app.graph["30:11"].inputs, {
      clip_name: "qwen3vl_4b_int8_convrot.safetensors", type: "krea2", device: "default",
    });
    const actualConsumers = Object.entries(app.graph)
      .filter(([, node]) => JSON.stringify(node.inputs.clip) === JSON.stringify([selector, 0]))
      .map(([id]) => id).sort();
    assert.deepEqual(actualConsumers, [...spec.consumers].sort());
    const restored = structuredClone(app.graph);
    delete restored[selector];
    for (const id of spec.consumers) restored[id].inputs.clip = ["30:11", 0];
    // Restore only the opt-in Krea fast-path edits before comparing the old frozen baseline.
    if (spec.id === "styles") {
      for (const extra of [
        "ff_krea_fast_enabled", "ff_krea_fast_lora", "ff_krea_fast_model",
        "ff_krea_steps_quality", "ff_krea_steps_fast", "ff_krea_steps_selected",
        "ff_krea_ultra_enabled", "ff_krea_ultra_lora", "ff_krea_ultra_model",
        "ff_krea_steps_ultra", "ff_krea_steps_ultra_selected",
      ]) delete restored[extra];
      restored["30:15"].inputs.model = ["30:10", 0];
      restored["30:3"].inputs.steps = 8;
    }
    // Pin the complete pre-adoption graph at d9f87, not only a few sampler fields.
    assert.equal(sha(JSON.stringify(restored)), spec.graphBefore);
    assert.equal(sha(readFileSync(new URL(`../apps/${spec.id}/app.json`, import.meta.url))), spec.manifest);
  });
}

test("styles: all nine looks retain their triggers, LoRAs and user settings through GPU routing", () => {
  const app = apps.find((a) => a.id === "styles")!;
  assert.ok(app.graph);
  const options = app.inputs.find((i) => i.id === "style")!.options!;
  assert.equal(options.length, 9);
  assert.equal(app.inputs.find((i) => i.id === "enhance")?.default, true);
  for (const option of options) for (const enhance of [false, true]) {
    const graph: ApiWorkflow = structuredClone(app.graph);
    const selector = gpuSelector(graph);
    const input = { prompt: "A piano beside a window.", style: option.value,
      strength: enhance ? 1.2 : 0.3, enhance, aspect: "16:9 (Widescreen)",
      megapixels: enhance ? 2 : 0.5, seed: 20261119 };
    const values = resolveValues(app, input);
    applyInputs(app, values, { setInput: (node, field, value) => { graph[node].inputs[field] = value; } },
      () => { throw new Error("styles has no file input"); });
    assert.equal(graph["30:19"].inputs.value, input.prompt);
    assert.equal(graph["30:27"].inputs.string_b, option.value);
    assert.equal(graph["30:15"].inputs.lora_name, option.sets![0].value);
    assert.equal(graph["30:15"].inputs.strength_model, input.strength);
    assert.equal(graph["30:24"].inputs.value, enhance);
    assert.equal(graph["49"].inputs.aspect_ratio, input.aspect);
    assert.equal(graph["49"].inputs.megapixels, input.megapixels);
    assert.equal(graph["30:3"].inputs.seed, input.seed);
    assert.deepEqual(graph["30:16"].inputs.clip, [selector, 0]);
    assert.deepEqual(graph["30:6"].inputs.clip, [selector, 0]);
    assert.deepEqual(graph["30:20"], app.graph["30:20"]); // live text-conversion branch
  }
});



test("styles: optional three-step LoRA is lazy, model/steps stay synchronized, defaults remain quality", () => {
  const app = apps.find((a) => a.id === "styles")!;
  assert.ok(app.graph);
  const graph = app.graph;
  const expectedLoRA = "krea2_turbo_3step_rank_64_lora_comfyui.safetensors";
  assert.equal(app.inputs.find((field) => field.id === "fast_mode")?.type, "toggle");
  assert.equal(app.inputs.find((field) => field.id === "fast_mode")?.default, false);
  assert.deepEqual(app.inputs.find((field) => field.id === "fast_mode")?.bind, [
    { node: "ff_krea_fast_enabled", field: "value" },
  ]);
  assert.deepEqual(app.inputs.map((field) => field.id).filter((id) =>
    id !== "fast_mode" && id !== "ultra_fast_mode"), [
    "prompt", "style", "strength", "enhance", "aspect", "megapixels", "seed",
  ]);
  assert.deepEqual(graph["ff_krea_fast_lora"].inputs, {
    model: ["30:10", 0], lora_name: expectedLoRA, strength_model: 1,
  });
  assert.deepEqual(graph["ff_krea_fast_model"].inputs, {
    on_false: ["30:10", 0], on_true: ["ff_krea_fast_lora", 0],
    switch: ["ff_krea_fast_enabled", 0],
  });
  assert.deepEqual(graph["30:15"].inputs.model, ["ff_krea_ultra_model", 0]);
  assert.deepEqual(graph["ff_krea_steps_quality"].inputs, { value: 8 });
  assert.deepEqual(graph["ff_krea_steps_fast"].inputs, { value: 3 });
  assert.deepEqual(graph["ff_krea_steps_selected"].inputs, {
    on_false: ["ff_krea_steps_quality", 0],
    on_true: ["ff_krea_steps_fast", 0],
    switch: ["ff_krea_fast_enabled", 0],
  });
  assert.deepEqual(graph["30:3"].inputs.steps, ["ff_krea_steps_ultra_selected", 0]);
  assert.equal(graph["30:3"].inputs.cfg, 1);
  assert.equal(graph["30:3"].inputs.scheduler, "simple");
  assert.equal(graph["30:3"].inputs.sampler_name, "euler");
  const metadata = app.models.find((m) => m.file.endsWith(expectedLoRA));
  assert.equal(metadata?.optional, true, "quality path must not require extra experimental weights");

  for (const fast_mode of [false, true]) {
    const copy: ApiWorkflow = structuredClone(graph);
    const values = resolveValues(app, {
      prompt: "A piano beside a window.", style: "Art Deco watercolor style",
      enhance: false, fast_mode, seed: 20261120,
    });
    applyInputs(app, values, {
      setInput: (node, field, value) => { copy[node].inputs[field] = value; },
    }, () => { throw new Error("unexpected image upload"); });
    assert.equal(copy["ff_krea_fast_enabled"].inputs.value, fast_mode);
    assert.deepEqual(copy["ff_krea_fast_model"].inputs.switch, ["ff_krea_fast_enabled", 0]);
    assert.deepEqual(copy["ff_krea_steps_selected"].inputs.switch, ["ff_krea_fast_enabled", 0]);
    assert.equal(copy["30:3"].inputs.seed, 20261120);
  }
});

test("styles: native 2-step Ultra preview remains optional, lazily overrides 3-step Fast, and preserves 8-step defaults", () => {
  const app = apps.find((a) => a.id === "styles")!;
  assert.ok(app.graph);
  const g = app.graph;
  const ultra = app.inputs.find((x) => x.id === "ultra_fast_mode")!;
  assert.equal(ultra.type, "toggle");
  assert.equal(ultra.default, false);
  assert.deepEqual(ultra.bind, [{ node: "ff_krea_ultra_enabled", field: "value" }]);
  assert.deepEqual(g["ff_krea_ultra_lora"].inputs, {
    model: ["30:10", 0],
    lora_name: "krea2_turbo_2step_rank_64_lora_comfyui.safetensors",
    strength_model: 1,
  });
  assert.deepEqual(g["ff_krea_ultra_model"].inputs, {
    on_false: ["ff_krea_fast_model", 0],
    on_true: ["ff_krea_ultra_lora", 0],
    switch: ["ff_krea_ultra_enabled", 0],
  });
  assert.deepEqual(g["ff_krea_steps_ultra"].inputs, { value: 2 });
  assert.deepEqual(g["ff_krea_steps_ultra_selected"].inputs, {
    on_false: ["ff_krea_steps_selected", 0],
    on_true: ["ff_krea_steps_ultra", 0],
    switch: ["ff_krea_ultra_enabled", 0],
  });
  assert.deepEqual(g["30:15"].inputs.model, ["ff_krea_ultra_model", 0]);
  assert.deepEqual(g["30:3"].inputs.steps, ["ff_krea_steps_ultra_selected", 0]);
  assert.equal(app.models.filter(m => m.file.endsWith("2step_rank_64_lora_comfyui.safetensors")).length, 1);
  assert.equal(app.models.find(m => m.file.endsWith("2step_rank_64_lora_comfyui.safetensors"))?.optional, true);
  for (const fast_mode of [false, true]) for (const ultra_fast_mode of [false, true]) {
    const graph: ApiWorkflow = structuredClone(g);
    const values = resolveValues(app, {
      prompt: "A jazz club with a grand piano.", fast_mode, ultra_fast_mode, seed: 202611201600,
    });
    applyInputs(app, values,
      { setInput: (node, field, value) => { graph[node].inputs[field] = value; } },
      () => { throw new Error("unexpected media upload"); });
    assert.equal(graph["ff_krea_fast_enabled"].inputs.value, fast_mode);
    assert.equal(graph["ff_krea_ultra_enabled"].inputs.value, ultra_fast_mode);
    // Both true means Ultra wins, without loading the inactive 3-step branch.
    const steps = ultra_fast_mode ? 2 : (fast_mode ? 3 : 8);
    assert.equal(steps, ultra_fast_mode ? g["ff_krea_steps_ultra"].inputs.value :
      fast_mode ? g["ff_krea_steps_fast"].inputs.value : g["ff_krea_steps_quality"].inputs.value);
    assert.deepEqual(graph["30:15"].inputs.model, ["ff_krea_ultra_model", 0]);
    assert.deepEqual(graph["30:3"].inputs.steps, ["ff_krea_steps_ultra_selected", 0]);
    assert.equal(graph["30:3"].inputs.seed, 202611201600);
  }
});

test("style-reference: native lazy switch defaults to verified vision-only conditioning", () => {
  const app = apps.find((a) => a.id === "style-reference")!;
  assert.ok(app.graph);
  assert.equal(problems.some((p) => p.app === "style-reference" && p.severity === "error"), false);
  const graph = app.graph;
  assert.equal(Object.values(graph).filter((node) => node.class_type === "SelectCLIPDevice").length, 0);
  assert.deepEqual(graph["30:11"].inputs, {
    clip_name: "qwen3vl_4b_int8_convrot.safetensors", type: "krea2", device: "default",
  });
  assert.deepEqual(graph["30:16"].inputs.clip, ["30:11", 0]);

  // The original high-detail image+VAE conditioning remains unchanged for GPUs with sufficient RAM.
  assert.deepEqual(graph["30:52"].inputs, {
    clip: ["30:11", 0], vae: ["30:12", 0], image1: ["69", 0], prompt: ["30:20", 0],
  });
  const vision = graph["ff_krea_ref_vision"];
  assert.equal(vision.class_type, "TextEncodeQwenImageEditPlus");
  assert.deepEqual(vision.inputs, {
    clip: ["30:11", 0], image1: ["69", 0], prompt: ["30:20", 0],
  });
  assert.equal(graph["ff_krea_ref_full"].class_type, "PrimitiveBoolean");
  assert.deepEqual(graph["ff_krea_ref_full"].inputs, { value: false });
  assert.equal(graph["ff_krea_ref_switch"].class_type, "ComfySwitchNode");
  assert.deepEqual(graph["ff_krea_ref_switch"].inputs, {
    switch: ["ff_krea_ref_full", 0], on_false: ["ff_krea_ref_vision", 0], on_true: ["30:52", 0],
  });
  assert.deepEqual(graph["30:53"].inputs.conditioning, ["ff_krea_ref_switch", 0]);

  // Remove only the new nodes to recover the pinned, already released 26-node graph.
  const restored = structuredClone(graph);
  delete restored["ff_krea_ref_vision"];
  delete restored["ff_krea_ref_full"];
  delete restored["ff_krea_ref_switch"];
  restored["30:53"].inputs.conditioning = ["30:52", 0];
  assert.equal(Object.keys(restored).length, 26);
  assert.equal(sha(JSON.stringify(restored)), "742307660e9f34570d8639f2ba6eed70a1b5d0e310829fc196614653d6bd13ba");

  const toggle = app.inputs.find((i) => i.id === "full_reference");
  assert.ok(toggle, "the costly mode must remain opt-in");
  assert.equal(toggle.type, "toggle");
  assert.equal(toggle.default, false);
  assert.deepEqual(toggle.bind, [{ node: "ff_krea_ref_full", field: "value" }]);
  assert.match(toggle.help!, /(?:memory|mémoire|GPU)/i);
  const controls = app.inputs.map((i) => i.id);
  assert.deepEqual(controls.filter((id) => id !== "full_reference"), [
    "style", "prompt", "enhance", "aspect", "megapixels", "seed",
  ]);
  assert.equal(app.models.length, 4, "same diffusion, encoder, VAE and style LoRA weights");
});

test("style-reference: opt-in flag maps only to the branch boolean", () => {
  const app = apps.find((a) => a.id === "style-reference")!;
  assert.ok(app.graph);
  for (const full of [false, true]) {
    const graph: ApiWorkflow = structuredClone(app.graph);
    const values = resolveValues(app, {
      prompt: "a red teapot",
      style: { name: "reference.png", bytes: new Uint8Array([0x89, 0x50, 0x4e, 0x47]), type: "image/png" },
      enhance: false, aspect: "1:1 (Square)", megapixels: 1, seed: 20261120, full_reference: full,
    });
    applyInputs(app, values, {
      setInput: (node, field, value) => { graph[node].inputs[field] = value; },
    }, (f) => f.name);
    assert.equal(graph["ff_krea_ref_full"].inputs.value, full);
    assert.deepEqual(graph["ff_krea_ref_switch"].inputs.on_false, ["ff_krea_ref_vision", 0]);
    assert.deepEqual(graph["ff_krea_ref_switch"].inputs.on_true, ["30:52", 0]);
    assert.deepEqual(graph["30:52"].inputs.vae, ["30:12", 0]);
  }
});
