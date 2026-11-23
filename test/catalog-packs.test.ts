import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { basename, join } from "node:path";
import { loadRegistry, repoRoot } from "../server/registry.ts";

const root = repoRoot();
const { apps, problems } = loadRegistry();

test("every current Freefield app is present in exactly its generated model pack", () => {
  assert.equal(problems.filter((p) => p.severity === "error").length, 0);
  const index = JSON.parse(readFileSync(join(root, "packs/index.json"), "utf8"));
  const expectedPacks = [...new Set(apps.map((a) => a.pack))].sort();
  assert.deepEqual(Object.keys(index).sort(), expectedPacks);
  const packFiles = readdirSync(join(root, "packs"))
    .filter((f) => f.endsWith(".json") && f !== "index.json").map((f) => f.slice(0, -5)).sort();
  assert.deepEqual(packFiles, expectedPacks, "Missing or obsolete generated pack metadata");
  for (const packName of expectedPacks) {
    const manifest = JSON.parse(readFileSync(join(root, "packs", packName + ".json"), "utf8"));
    const subset = apps.filter((app) => app.pack === packName);
    const appIds = subset.map((a) => a.id).sort();
    assert.deepEqual(manifest.apps, appIds, packName + ": app consumers");
    assert.deepEqual(index[packName].apps, appIds, packName + ": index consumers");
    assert.equal(manifest.needs_hf_token, index[packName].needs_hf_token);
    assert.equal(manifest.download_gb, index[packName].download_gb);
    const expected = new Map<string, { repo: string; file: string; name: string }>();
    for (const app of subset) {
      for (const model of app.models) {
        const dest = model.install_name ?? basename(model.file);
        assert.ok(dest && !dest.startsWith("/") &&
          !dest.split("/").some((part) => part === "." || part === ".." || !part));
        const key = model.dir + "/" + dest;
        const existing = expected.get(key);
        if (existing) {
          assert.equal(existing.repo, model.repo, packName + ": conflicting repo at " + key);
          assert.equal(existing.file, model.file, packName + ": conflicting HF path at " + key);
        }
        expected.set(key, { repo: model.repo, file: model.file, name: dest });
      }
    }
    assert.equal(manifest.models.length, expected.size, packName + ": missing or obsolete models");
    for (const m of manifest.models as { dir: string; name: string; repo: string; file: string }[]) {
      const key = m.dir + "/" + m.name;
      const e = expected.get(key);
      assert.ok(e, packName + ": unreferenced file " + key);
      assert.equal(m.repo, e.repo);
      assert.equal(m.file, e.file);
      assert.equal(m.name, e.name);
    }
  }
});
