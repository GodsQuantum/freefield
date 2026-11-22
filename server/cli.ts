// Command line for the registry and the runner.
//   npm run validate                          check every app.json and its bindings
//   npm run apps                              list apps with status and pack
//   npm run app -- <id> [--set k=v] [--file k=path] [--preset id] [--out dir] [--draft] [--no-history]
import { copyFileSync, linkSync, mkdirSync, readFileSync } from "node:fs";
import { basename, join } from "node:path";
import { loadRegistry } from "./registry.ts";
import { explainSubmitError, resolveValues, Runner, type Values } from "./runner.ts";
import { History } from "./history.ts";

const [cmd, ...rest] = process.argv.slice(2);
const { apps, problems } = loadRegistry();

function printProblems(): number {
  let errors = 0;
  for (const p of problems) {
    if (p.severity === "error") errors++;
    console.log(`${p.severity === "error" ? "✗" : "!"} ${p.app}: ${p.message}`);
  }
  return errors;
}

if (cmd === "validate") {
  const errors = printProblems();
  const ready = apps.filter((a) => a.status === "ready").length;
  console.log(`${apps.length} apps (${ready} ready, ${apps.length - ready} draft) · ${errors} errors · ${problems.length - errors} warnings`);
  process.exit(errors ? 1 : 0);
} else if (cmd === "apps") {
  for (const a of apps) console.log(`${a.status.padEnd(5)}  ${a.section.padEnd(7)}  ${a.pack.padEnd(14)}  ${a.id}`);
} else if (cmd === "app") {
  const id = rest[0];
  const app = apps.find((a) => a.id === id);
  if (!app) {
    console.error(`unknown app "${id}". Try: npm run apps`);
    process.exit(2);
  }
  const values: Values = {};
  let preset: string | undefined;
  let out = "out";
  let allowDrafts: boolean | undefined;
  let keepHistory = process.env.FREEFIELD_CLI_HISTORY !== "0";
  for (let i = 1; i < rest.length; i++) {
    const flag = rest[i];
    if (flag === "--draft") {
      allowDrafts = true;
      continue;
    }
    if (flag === "--no-history") {
      keepHistory = false;
      continue;
    }
    const arg = rest[++i] ?? "";
    if (flag === "--set") {
      const [k, ...v] = arg.split("=");
      values[k] = v.join("=");
    } else if (flag === "--file") {
      const [k, ...v] = arg.split("=");
      const path = v.join("=");
      values[k] = { bytes: new Uint8Array(readFileSync(path)), name: basename(path) };
    } else if (flag === "--preset") preset = arg;
    else if (flag === "--out") out = arg;
  }
  // The UI's own History format is shared with CLI generations; no polling bridge
  // or separate catalogue. Recording starts before GPU submission, just as on web.
  const history = keepHistory ? new History(join(process.env.FREEFIELD_OUT || "out", "history"), {
    appInfo: (id) => id === app.id && app.graph ? {
      outputs: app.outputs.map((o) => o.node),
      nodes: Object.keys(app.graph).length,
      classOf: (node) => app.graph?.[node]?.class_type ?? null,
    } : undefined,
  }) : null;
  const runner = new Runner({ allowDrafts });
  const t0 = Date.now();
  let entry: ReturnType<History["start"]> | undefined;
  let job;
  try {
    const resolved = resolveValues(app, values);
    entry = history?.start(app, resolved, preset);
    job = await runner.submit(app, resolved, preset);
  } catch (e) {
    const message = explainSubmitError(app, e) ?? (e as Error).message;
    if (history && entry) history.fail(entry, message);
    console.error(`could not submit ${app.id}: ${message}`);
    process.exit(1);
  }
  console.log(`submitted ${app.id} → job ${job.id}`);
  try {
    // History.follow uses the same resilient, indefinite job wait as the web UI
    // and saves verified SDK outputs to the persistent shared history directory.
    if (history && entry) entry = await history.follow(entry, job);
    else await job.wait();
  } catch (e) {
    if (history && entry) history.fail(entry, (e as Error).message);
    console.error(`job ${job.id} ended ${job.status}:`, (e as Error).message);
    process.exit(1);
  }
  if (job.status !== "succeeded" || (entry && (entry.status !== "succeeded" || !entry.outputs.length || entry.error))) {
    const err = job.error as { message?: string; class_type?: string; node_id?: string } | null | undefined;
    const where = err?.class_type ? ` in ${err.class_type} (node ${err.node_id})` : "";
    const details = entry?.error?.message ?? (err ? `${(err.message ?? JSON.stringify(err)).trim()}` : "no saved outputs");
    if (history && entry && entry.status === "succeeded") history.fail(entry, details);
    console.error(`job ${job.id} ended ${entry?.status ?? job.status}${where}: ${details}`);
    process.exit(1);
  }
  console.log(`status ${job.status} in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  mkdirSync(out, { recursive: true });
  if (history && entry) {
    // Preserve the CLI's --out contract without duplicating large video files
    // when both destinations are on the same filesystem.
    for (const output of entry.outputs) {
      const source = join(history.dir, entry.id, output.file);
      const destination = join(out, `${app.id}-${basename(output.name)}`);
      try {
        linkSync(source, destination);
      } catch (error) {
        // A separate filesystem cannot hard-link; an existing destination was
        // already created by a previous CLI call. Both can use the saved file.
        if (!["EXDEV", "EEXIST"].includes((error as NodeJS.ErrnoException).code ?? "")) throw error;
        copyFileSync(source, destination);
      }
      console.log(`saved ${destination}`);
    }
    console.log(`library /api/history/${entry.id}`);
  } else {
    for (const o of app.outputs) {
      for (const output of job.getOutputs(o.node)) {
        const path = join(out, `${app.id}-${basename(output.name)}`);
        await output.toFile(path);
        console.log(`saved ${path}`);
      }
    }
  }

} else {
  console.log("usage: cli.ts validate | apps | app <id> [--set k=v] [--file k=path] [--preset id] [--out dir] [--draft] [--no-history]");
  process.exit(cmd ? 2 : 0);
}
