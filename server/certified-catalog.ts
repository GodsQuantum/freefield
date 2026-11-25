import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { App } from "./registry.ts";

export type Qualification = {
  evidence: string; evidence_sha256: string; app_sha256: string; workflow_sha256: string;
  measured_seconds: number; scope: string; review: "visual" | "limited" | "audio";
  defaults: Record<string, unknown>;
  reference_settings?: Record<string, unknown>;
  guidance?: string;
};
export type CertifiedManifest = { schema_version: number; revision: string; apps: Record<string, Qualification> };
const digest = (file: string) => createHash("sha256").update(readFileSync(file)).digest("hex");

/** Opt-in production policy. A new recipe or changed workflow is never implicitly certified. */
export function certifiedCatalog(root: string, catalog: App[], enabledIds?: string[]) {
  const manifest = JSON.parse(readFileSync(join(root, "production-catalog.json"), "utf8")) as CertifiedManifest;
  if (manifest.schema_version !== 1 || !manifest.revision || !manifest.apps || !Object.keys(manifest.apps).length)
    throw new Error("Invalid or empty production certification catalog");
  const selected = new Set(enabledIds ?? Object.keys(manifest.apps));
  const apps: App[] = [];
  const proofs = new Map<string, Qualification>();
  for (const app of catalog) {
    const q = manifest.apps[app.id];
    if (!Object.hasOwn(manifest.apps, app.id) || !selected.has(app.id) || !q) continue;
    if (app.status !== "ready" || !app.workflowPath || !q.evidence || !q.scope ||
        !["visual", "limited", "audio"].includes(q.review) ||
        !Number.isFinite(q.measured_seconds) || q.measured_seconds <= 0 ||
        digest(join(app.dir, "app.json")) !== q.app_sha256 || digest(app.workflowPath) !== q.workflow_sha256) {
      console.error("[certification] Excluded changed or invalid recipe:", app.id);
      continue;
    }
    const defaults = q.defaults ?? {};
    const inputs = app.inputs.map(input => Object.hasOwn(defaults, input.id) ? { ...input, default: defaults[input.id] } : input);
    apps.push({ ...app, inputs });
    proofs.set(app.id, q);
  }
  if (!apps.length) throw new Error("No certified production recipes; refusing unrestricted catalog");
  return { apps, proofs, revision: manifest.revision };
}
