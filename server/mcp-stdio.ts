// MCPProxy stdio entrypoint. It runs inside the Freefield Web container via
// Docker exec, so no credentials are copied into MCPProxy and no extra port
// or model server is needed. All operations use Freefield's authenticated API.
import { createHmac } from "node:crypto";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { serveStdio } from "@modelcontextprotocol/server/stdio";
import { createFreefieldMcp, type McpBackend } from "./mcp.ts";
import type { FileValue, Values } from "./runner.ts";

const passcode = process.env.FREEFIELD_PASSCODE;
if (!passcode) throw new Error("FREEFIELD_PASSCODE missing");
const cookie = "ff_key=" + createHmac("sha256", passcode).update("freefield-visitor").digest("hex");
const base = "http://127.0.0.1:" + (process.env.PORT || "3000");
const outputDir = process.env.FREEFIELD_OUT || "/app/out";

async function request(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  headers.set("Cookie", cookie);
  const r = await fetch(base + path, {
    ...init,
    headers,
    signal: AbortSignal.timeout(15_000),
  });
  if (!r.ok) {
    const detail = await r.text().catch(() => "");
    throw new Error("Freefield API " + r.status + ": " + detail.slice(0, 300));
  }
  return r;
}
async function api(path: string, init: RequestInit = {}): Promise<any> {
  return (await request(path, init)).json();
}
async function maybe(path: string): Promise<any | null> {
  try { return await api(path); }
  catch (e) { if (String(e).includes("Freefield API 404:")) return null; throw e; }
}
function fileValue(x: unknown): x is FileValue {
  return typeof x === "object" && x !== null && (x as FileValue).bytes instanceof Uint8Array;
}

const backend: McpBackend = {
  apps: async () => (await api("/api/apps")).apps,
  app: async (id) => (await api("/api/apps")).apps.find((a: {id: string}) => a.id === id) ?? null,
  run: async (id: string, values: Values, preset?: string) => {
    const form = new FormData();
    for (const [key, value] of Object.entries(values)) {
      if (fileValue(value)) {
        const upload = new Blob([new Uint8Array(value.bytes)], { type: value.type || "application/octet-stream" });
        form.append(key, upload, value.name);
      } else {
        form.append(key, Array.isArray(value) ? JSON.stringify(value) : String(value));
      }
    }
    if (preset) form.append("preset", preset);
    return api("/api/apps/" + encodeURIComponent(id) + "/run", { method: "POST", body: form });
  },
  runs: async (id?: string, limit = 10) => {
    const path = "/api/history" + (id ? "?app=" + encodeURIComponent(id) : "");
    return (await api(path)).entries.slice(0, limit);
  },
  job: async (id: string) => maybe("/api/history/" + encodeURIComponent(id)),
  cancel: async (id: string) => (await api("/api/history/" + encodeURIComponent(id) + "/cancel", {method: "POST"})).canceled === true,
  output: async (id: string, filename: string) => {
    if (!/^[0-9]{8}-[0-9]{9}-[0-9a-f]{4}$/.test(id)) return null;
    if (!/^out-[a-z0-9_-]+\.[a-z0-9]{1,5}$/.test(filename)) return null;
    const job = await maybe("/api/history/" + id);
    if (!job?.outputs?.some((x: {file: string}) => x.file === filename)) return null;
    const path = join(outputDir, "history", id, filename);
    return existsSync(path) ? path : null;
  },
  gpu: () => api("/api/gpu"),
};

await serveStdio(() => createFreefieldMcp(backend));
