import { test } from "node:test";
import assert from "node:assert/strict";
import { loadRegistry, type ApiWorkflow } from "../server/registry.ts";
import { resolveValues, applyInputs } from "../server/runner.ts";

const { apps, problems } = loadRegistry();
const app = apps.find((item) => item.id === "z-image");
const encoder = "Z-Image-AbliteratedV1.Q6_K.gguf";

test("z-image: native FP8 encoder stays default and abliterated Q6 is lazy and opt-in", () => {
  assert.ok(app?.graph);
  assert.equal(problems.some((problem) => problem.app === "z-image" && problem.severity === "error"), false);
  const graph = app.graph;
  const option = app.inputs.find((input) => input.id === "freer_encoder");
  assert.ok(option);
  assert.equal(option.type, "toggle");
  assert.equal(option.default, false);
  assert.deepEqual(option.bind, [{ node: "ff_z_ablit_enabled", field: "value" }]);
  assert.deepEqual(graph["57:30"].inputs, {
    clip_name: "qwen_3_4b_fp8_mixed.safetensors", type: "lumina2", device: "default",
  });
  assert.equal(graph["ff_z_ablit_clip"].class_type, "CLIPLoaderGGUF");
  assert.deepEqual(graph["ff_z_ablit_clip"].inputs, {
    clip_name: encoder, type: "lumina2",
  });
  assert.equal(graph["ff_z_ablit_enabled"].class_type, "PrimitiveBoolean");
  assert.deepEqual(graph["ff_z_ablit_enabled"].inputs, { value: false });
  assert.deepEqual(graph["ff_z_clip_router"].inputs, {
    switch: ["ff_z_ablit_enabled", 0],
    on_false: ["57:30", 0],
    on_true: ["ff_z_ablit_clip", 0],
  });
  assert.deepEqual(graph["57:27"].inputs.clip, ["ff_z_clip_router", 0]);
  assert.deepEqual(graph["57:28"].inputs, {
    unet_name: "z_image_turbo_int8_convrot.safetensors", weight_dtype: "default",
  });
  assert.equal(graph["57:3"].inputs.steps, 8);
  assert.equal(graph["57:3"].inputs.cfg, 1);
  assert.equal(graph["57:3"].inputs.sampler_name, "res_multistep");
  assert.equal(app.models.find((model) => model.file === encoder)?.optional, true);
  const restored: ApiWorkflow = structuredClone(graph);
  for (const id of ["ff_z_ablit_enabled", "ff_z_ablit_clip", "ff_z_clip_router"]) delete restored[id];
  restored["57:27"].inputs.clip = ["57:30", 0];
  assert.equal(Object.keys(restored).length, 10);
  assert.deepEqual(restored["57:27"].inputs.clip, ["57:30", 0]);
  assert.deepEqual(app.inputs.map((input) => input.id).filter((id) => id !== "freer_encoder"), [
    "prompt", "width", "height", "steps", "seed",
  ]);
});

test("z-image: UI toggle switches the CLIP model without changing default sampler, seed, or image size", () => {
  assert.ok(app?.graph);
  for (const freer_encoder of [false, true]) {
    const graph: ApiWorkflow = structuredClone(app.graph);
    const values = resolveValues(app, {
      prompt: "A fictional adult photographer holding a small camera",
      freer_encoder, width: 768, height: 768, steps: 8, seed: 20261120,
    });
    applyInputs(app, values, {
      setInput: (node, field, value) => { graph[node].inputs[field] = value; },
    }, () => { throw new Error("z-image has no input files"); });
    assert.equal(graph["ff_z_ablit_enabled"].inputs.value, freer_encoder);
    assert.deepEqual(graph["57:27"].inputs.clip, ["ff_z_clip_router", 0]);
    assert.deepEqual(graph["ff_z_clip_router"].inputs.switch, ["ff_z_ablit_enabled", 0]);
    assert.equal(graph["57:3"].inputs.seed, 20261120);
    assert.equal(graph["57:3"].inputs.steps, 8);
    assert.equal(graph["57:13"].inputs.width, 768);
    assert.equal(graph["57:13"].inputs.height, 768);
  }
});
