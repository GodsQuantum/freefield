// Source-controlled model and sampling choices; changes require explicit re-lock.
import { loadRegistry, type App } from "./registry.ts";

const PROFILE_FIELDS = new Set([
  "ckpt_name", "unet_name", "clip_name", "clip_name1", "clip_name2",
  "clip_name3", "vae_name", "lora_name", "model_name", "model_file",
  "checkpoint", "filename", "weights", "weight_dtype", "dtype",
  "scheduler", "sampler_name", "sigmas", "steps", "cfg", "denoise",
  "duration", "megapixels", "width", "height", "frames",
]);
const PROFILE_INPUT_IDS = new Set(["duration", "steps", "resolution", "frames"]);

export function recipeSnapshot(app: App) {
  return {
    status: app.status,
    pack: app.pack,
    models: app.models.map((m) => ({
      dir: m.dir, file: m.file, repo: m.repo, phase: m.phase, optional: m.optional === true,
    })).sort((a, b) => (a.dir + "/" + a.file).localeCompare(b.dir + "/" + b.file)),
    workflow: Object.entries(app.graph ?? {}).flatMap(([id, node]) => {
      const values = Object.fromEntries(
        Object.entries(node.inputs).filter(([key]) => PROFILE_FIELDS.has(key))
      );
      if (!Object.keys(values).length) return [];
      return [{ id, class_type: node.class_type, inputs: values }];
    }).sort((a, b) => a.id.localeCompare(b.id)),
    model_overrides: app.inputs.flatMap((input) =>
      (input.options ?? []).flatMap((option) =>
        (option.sets ?? []).filter((item) => PROFILE_FIELDS.has(item.field)).map((item) => ({
          input: input.id, option: String(option.value),
          node: item.node, field: item.field, value: item.value,
        }))
      )
    ).sort((a, b) =>
      (a.input + "/" + a.option + "/" + a.node + "/" + a.field)
        .localeCompare(b.input + "/" + b.option + "/" + b.node + "/" + b.field)),
    limits: app.inputs.filter((input) => PROFILE_INPUT_IDS.has(input.id)).map((input) => ({
      id: input.id, default: input.default ?? null, min: input.min ?? null, max: input.max ?? null,
    })).sort((a, b) => a.id.localeCompare(b.id)),
  };
}

export function buildRecipeLock(): Record<string, ReturnType<typeof recipeSnapshot>> {
  const { apps, problems } = loadRegistry();
  if (problems.some((p) => p.severity === "error")) {
    throw new Error("Cannot lock invalid Freefield model manifests");
  }
  return Object.fromEntries(apps.map((app): [string, ReturnType<typeof recipeSnapshot>] => [app.id, recipeSnapshot(app)])
    .sort(([a], [b]) => a.localeCompare(b)));
}
