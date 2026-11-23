// Regenerate the explicit model lock only when a selection has been approved.
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { buildRecipeLock } from "../server/model-lock.ts";
import { repoRoot } from "../server/registry.ts";

const file = join(repoRoot(), "model-selection.lock.json");
const current = JSON.parse(readFileSync(file, "utf8"));
if (current.schema_version !== 1 || !current.choices) {
  throw new Error("Selection lock metadata missing; do not overwrite");
}
const next = { ...current, recipes: buildRecipeLock() };
if (process.argv.includes("--check")) {
  const before = JSON.stringify(current.recipes);
  const after = JSON.stringify(next.recipes);
  if (before !== after) throw new Error("Model lock drift: regenerate intentionally after evidence review");
  console.log(`LOCK PASS: ${Object.keys(next.recipes).length} applications pinned`);
} else if (process.argv.includes("--refresh")) {
  writeFileSync(file, JSON.stringify(next, null, 2) + "\n");
  console.log(`Refreshed pinned recipes: ${Object.keys(next.recipes).length}`);
} else {
  throw new Error("Usage: node ... tools/lock_models.ts --check | --refresh");
}
