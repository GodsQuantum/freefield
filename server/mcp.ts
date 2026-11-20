// MCP tools are a native façade over Freefield's registry, queue and history.
// They never access ComfyUI directly, keep the normal run limits, and use the same
// passcode/owner checks as the web API. No additional inference service is launched.
import { createMcpHandler, McpServer } from "@modelcontextprotocol/server";
import { toNodeHandler } from "@modelcontextprotocol/node";
import * as z from "zod/v4";
import { readFileSync, statSync, openSync, readSync, closeSync } from "node:fs";
import { extname } from "node:path";
import type { IncomingMessage, ServerResponse } from "node:http";
import type { FileValue, Values } from "./runner.ts";

export type McpBackend = {
  apps(): Promise<Record<string, unknown>[]>;
  app(id: string): Promise<Record<string, unknown> | null>;
  run(id: string, values: Values, preset?: string): Promise<unknown>;
  runs(app?: string, limit?: number): unknown[] | Promise<unknown[]>;
  job(id: string): unknown | null | Promise<unknown | null>;
  cancel(id: string): Promise<boolean>;
  output(id: string, filename: string): string | null | Promise<string | null>;
  gpu(): Promise<unknown>;
};

function text(data: unknown, error = false) {
  return { content: [{ type: "text" as const, text: JSON.stringify(data) }], ...(error ? { isError: true } : {}) };
}

const fileSchema = z.object({
  name: z.string().min(1).max(128),
  mime_type: z.string().max(100).optional(),
  base64: z.string().min(1),
});

function decodeFile(encoded: z.infer<typeof fileSchema>): FileValue {
  if (!/^[A-Za-z0-9+/]+={0,2}$/.test(encoded.base64) || encoded.base64.length % 4 !== 0) {
    throw new Error("Invalid base64 file upload");
  }
  // Streamable HTTP has a bounded body size; larger media belongs to the multipart API.
  if (encoded.base64.length > 3_000_000) throw new Error("MCP input exceeds 2 MiB; use Freefield's multipart HTTP API");
  const bytes = Buffer.from(encoded.base64, "base64");
  if (bytes.length > 2_250_000 || bytes.length === 0) throw new Error("Invalid or oversized MCP input");
  if (/[/\\]/.test(encoded.name) || encoded.name === "." || encoded.name === "..") throw new Error("Invalid filename");
  return { bytes: new Uint8Array(bytes), name: encoded.name, type: encoded.mime_type };
}

const sectionSchema = z.enum(["all", "image", "video", "audio", "utility"]);

/** One tool registry, used by both HTTP and stdio transports. */
export function createFreefieldMcp(backend: McpBackend): McpServer {
    const mcp = new McpServer({ name: "freefield", version: "1.0.0" }, { capabilities: { tools: {} } });

    mcp.registerTool("list_apps", {
      title: "List Freefield applications",
      description: "Discover the installed image, video, audio and utility generators. Defaults to runnable models only. No GPU wake-up.",
      inputSchema: z.object({
        section: sectionSchema.default("all"),
        query: z.string().max(120).optional(),
        offset: z.number().int().min(0).default(0),
        limit: z.number().int().min(1).max(100).default(30),
        available_only: z.boolean().default(true),
      }),
      annotations: { readOnlyHint: true, openWorldHint: false },
    }, async ({ section, query, offset, limit, available_only }) => {
      const q = (query ?? "").trim().toLowerCase();
      const all = (await backend.apps()).filter((a) => {
        if (section !== "all" && a.section !== section) return false;
        if (available_only && (!a.runnable || a.missing_pack)) return false;
        return !q || String(a.id).toLowerCase().includes(q) || String(a.name).toLowerCase().includes(q)
          || String(a.summary ?? "").toLowerCase().includes(q);
      });
      return text({ total: all.length, offset, limit, apps: all.slice(offset, offset + limit).map((a) => ({
        id: a.id, name: a.name, section: a.section, summary: a.summary,
        runnable: a.runnable, missing_pack: a.missing_pack, modes: a.modes,
      })) });
    });

    mcp.registerTool("get_app", {
      title: "Inspect Freefield generator",
      description: "Find exact parameters, option values, required input files, presets, installed models and output types for one app.",
      inputSchema: z.object({ app_id: z.string().min(1).max(100) }),
      annotations: { readOnlyHint: true, openWorldHint: false },
    }, async ({ app_id }) => {
      const app = await backend.app(app_id);
      return app ? text(app) : text({ error: "Unknown application", app_id }, true);
    });

    mcp.registerTool("generate", {
      title: "Generate image, video or audio in Freefield",
      description: "Start an asynchronous generation using a registered Freefield app. Returns history id immediately. Use get_job to poll. "
        + "Inputs use app manifest IDs (get_app). files maps image/video/audio input IDs to base64-encoded upload objects; for files >2 MiB use the multipart HTTP API. "
        + "Same memory guard, model catalog, queue, authentication and history as Freefield UI.",
      inputSchema: z.object({
        app_id: z.string().min(1).max(100),
        inputs: z.record(z.string(), z.union([z.string(), z.number(), z.boolean(), z.array(z.string())])).default({}),
        files: z.record(z.string(), fileSchema).optional(),
        preset: z.string().max(100).optional(),
      }),
      annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false },
    }, async ({ app_id, inputs, files, preset }) => {
      try {
        const values: Values = { ...inputs };
        for (const [id, file] of Object.entries(files ?? {})) {
          if (Object.prototype.hasOwnProperty.call(values, id)) throw new Error("Duplicate input: " + id);
          values[id] = decodeFile(file);
        }
        const run = await backend.run(app_id, values, preset);
        return text({ accepted: true, run, next: "get_job" });
      } catch (e) { return text({ error: (e as Error).message }, true); }
    });

    mcp.registerTool("list_runs", {
      title: "List Freefield history",
      description: "List recent Freefield inference jobs with status, run IDs and output names. Use get_job for details. Read-only, no GPU wake-up.",
      inputSchema: z.object({
        app_id: z.string().max(100).optional(),
        offset: z.number().int().min(0).max(250).default(0),
        limit: z.number().int().min(1).max(50).default(10),
      }),
      annotations: { readOnlyHint: true, openWorldHint: false },
    }, async ({ app_id, offset, limit }) => {
      // MCPProxy has a small text-response budget. Full history entries
      // contain source prompts and inputs; leave those to get_job.
      const all = await backend.runs(app_id, offset + limit + 1) as Record<string, any>[];
      const entries = all.slice(offset, offset + limit).map((run) => ({
        id: run.id, app: run.app, status: run.status, created: run.created,
        seconds: run.seconds ?? null, total: run.total ?? null,
        outputs: (run.outputs || []).map((o: Record<string, unknown>) => ({
          file: o.file, type: o.type, name: o.name,
        })),
        ...(run.validation?.benchmark_only ? { benchmark: {
          quality_status: run.validation.quality_status, replayable: false,
        } } : {}),
        ...(run.error ? { error: run.error.message } : {}),
      }));
      return text({ offset, limit, entries, next_offset: all.length > offset + limit ? offset + limit : null });
    });

    mcp.registerTool("get_job", {
      title: "Check Freefield generation",
      description: "Get progress, final media filenames, timing and errors from Freefield's persistent history. Does not wake GPU.",
      inputSchema: z.object({ run_id: z.string().min(1).max(100) }),
      annotations: { readOnlyHint: true, openWorldHint: false },
    }, async ({ run_id }) => {
      const job = await backend.job(run_id);
      return job ? text(job) : text({ error: "Run not found", run_id }, true);
    });

    mcp.registerTool("get_output", {
      title: "Read Freefield media output",
      description: "Return small images inline as MCP image content. Large images and audio/video over 10KB return a get_output_chunk hint to avoid MCPProxy response truncation.",
      inputSchema: z.object({ run_id: z.string().min(1).max(100), filename: z.string().min(1).max(160) }),
      annotations: { readOnlyHint: true, openWorldHint: false },
    }, async ({ run_id, filename }) => {
      const path = await backend.output(run_id, filename);
      if (!path) return text({ error: "Output not found or not authorized" }, true);
      const size = statSync(path).size;
      if (size > 2_000_000) {
        return text({ too_large_for_inline: true, bytes: size, next_tool: "get_output_chunk",
          api_path: "/api/history/" + run_id + "/files/" + filename,
          note: "MCP clients can fetch bounded chunks; authenticated HTTP clients can stream directly." });
      }
      const mime = ({ ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".gif": "image/gif", ".mp3": "audio/mpeg",
        ".wav": "audio/wav", ".mp4": "video/mp4", ".webm": "video/webm",
        ".ogg": "audio/ogg", ".flac": "audio/flac" } as Record<string, string>)[extname(filename).toLowerCase()] || "application/octet-stream";
      if (!mime.startsWith("image/") && size > 10_000) {
        return text({ too_large_for_inline: true, bytes: size, mime_type: mime,
          next_tool: "get_output_chunk",
          note: "Use bounded get_output_chunk calls or authenticated Freefield HTTP media streaming." });
      }
      const data = readFileSync(path).toString("base64");
      if (mime.startsWith("image/")) {
        return { content: [{ type: "image" as const, mimeType: mime, data }] };
      }
      return text({ mime_type: mime, bytes: size, base64: data });
    });

    mcp.registerTool("get_output_chunk", {
      title: "Fetch a bounded Freefield media chunk",
      description: "Retrieve up to 8192 raw bytes of an authenticated image, video or audio output as base64. Use next_offset until done. Bounded to avoid MCPProxy truncation.",
      inputSchema: z.object({
        run_id: z.string().min(1).max(100),
        filename: z.string().min(1).max(160),
        offset: z.number().int().min(0).max(1_000_000_000_000).default(0),
        max_bytes: z.number().int().min(1).max(8192).default(8192),
      }),
      annotations: { readOnlyHint: true, openWorldHint: false },
    }, async ({ run_id, filename, offset, max_bytes }) => {
      const path = await backend.output(run_id, filename);
      if (!path) return text({ error: "Output not found or not authorized" }, true);
      const size = statSync(path).size;
      if (offset > size) return text({ error: "Offset exceeds media size", size }, true);
      const bytes = Buffer.alloc(Math.min(max_bytes, size - offset));
      if (bytes.length > 0) {
        const fd = openSync(path, "r");
        try {
          const read = readSync(fd, bytes, 0, bytes.length, offset);
          if (read !== bytes.length) return text({ error: "Incomplete media read" }, true);
        } finally {
          closeSync(fd);
        }
      }
      return text({ run_id, filename, offset, bytes: bytes.length, size,
        next_offset: offset + bytes.length, done: offset + bytes.length >= size,
        base64: bytes.toString("base64") });
    });

    mcp.registerTool("cancel_job", {
      title: "Cancel Freefield generation",
      description: "Ask Freefield to cancel one running generation. Does not delete completed history or checkpoints.",
      inputSchema: z.object({ run_id: z.string().min(1).max(100) }),
      annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: true, openWorldHint: false },
    }, async ({ run_id }) => text({ canceled: await backend.cancel(run_id) }));

    mcp.registerTool("gpu_status", {
      title: "Freefield GPU availability",
      description: "Read backend status and queue utilization without starting a new GPU task.",
      inputSchema: z.object({}),
      annotations: { readOnlyHint: true, openWorldHint: false },
    }, async () => text(await backend.gpu()));

    return mcp;
}

/** An isolated stateless MCP handler is created per HTTP request to preserve each visitor's owner scope. */
export async function handleFreefieldMcp(req: IncomingMessage, res: ServerResponse, backend: McpBackend): Promise<void> {
  const handler = toNodeHandler(createMcpHandler(() => createFreefieldMcp(backend)));
  await handler(req, res);
}
