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
    manifest: "22a783e9826973c74c4903961d06abb44d264411636e407e0e5fef56acbe9a83" },

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

test("style-reference remains on its original CPU/default encoder path", () => {
  const app = apps.find((a) => a.id === "style-reference")!;
  assert.ok(app.graph);
  assert.equal(problems.some((p) => p.app === "style-reference" && p.severity === "error"), false);
  assert.equal(Object.values(app.graph).filter((node) => node.class_type === "SelectCLIPDevice").length, 0);
  assert.deepEqual(app.graph["30:11"].inputs, {
    clip_name: "qwen3vl_4b_int8_convrot.safetensors", type: "krea2", device: "default",
  });
  assert.deepEqual(app.graph["30:16"].inputs.clip, ["30:11", 0]);
  assert.deepEqual(app.graph["30:52"].inputs.clip, ["30:11", 0]);
  assert.equal(sha(JSON.stringify(app.graph)), "742307660e9f34570d8639f2ba6eed70a1b5d0e310829fc196614653d6bd13ba");
  assert.equal(sha(readFileSync(new URL("../apps/style-reference/app.json", import.meta.url))),
    "3c106125bcbc464776307605993a869624302ae2ed17d850de82ad0c85d27443");
});
