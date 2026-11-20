import { test } from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { once } from "node:events";
import { handleFreefieldMcp, type McpBackend } from "../server/mcp.ts";

test("MCP Streamable HTTP discovers and runs Freefield tools through one backend", async () => {
  const calls: unknown[] = [];
  const backend: McpBackend = {
    apps: async () => [{ id: "nitrofusion2", name: "NitroFusion2 Fast", section: "image", runnable: true, missing_pack: null }],
    app: async (id) => id === "nitrofusion2" ? { id, inputs: [{ id: "prompt", type: "textarea", required: true }] } : null,
    run: async (id, values, preset) => {
      calls.push({ id, values, preset });
      return { id: "test-id", status: "starting", app: id };
    },
    runs: () => [{ id: "test-id", status: "starting" }],
    job: (id) => id === "test-id" ? { id, status: "succeeded", outputs: [{ file: "out-0.png" }] } : null,
    output: () => null,
    cancel: async () => true,
    gpu: async () => ({ configured: true, running: 0 }),
  };
  const http = createServer((req, res) => {
    void handleFreefieldMcp(req, res, backend).catch((err: Error) => {
      if (!res.headersSent) res.writeHead(500);
      res.end(err.message);
    });
  });
  http.listen(0, "127.0.0.1");
  await once(http, "listening");
  const address = http.address();
  assert(address && typeof address !== "string");
  const url = "http://127.0.0.1:" + address.port;
  const call = async (method: string, params: unknown = {}, id = 1) => {
    const response = await fetch(url + "/mcp", {
      method: "POST",
      headers: {
        accept: "application/json, text/event-stream",
        "content-type": "application/json",
        "mcp-protocol-version": "2025-11-25",
      },
      body: JSON.stringify({ jsonrpc: "2.0", id, method, params }),
      signal: AbortSignal.timeout(5000),
    });
    const body = await response.text();
    assert.equal(response.status, 200, body);
    const data = body.startsWith("event:") ? /^data:\s*(.+)$/m.exec(body)?.[1] : body;
    assert(data, body);
    return JSON.parse(data);
  };
  try {
    const init = await call("initialize", { protocolVersion: "2025-11-25", capabilities: {}, clientInfo: { name: "test", version: "1" } });
    assert.equal(init.result.serverInfo.name, "freefield");
    const list = await call("tools/list");
    assert.equal(list.result.tools.length, 8);
    assert.equal(list.result.tools.find((x: {name: string}) => x.name === "generate").annotations.readOnlyHint, false);
    const found = await call("tools/call", { name: "list_apps", arguments: { section: "image" } });
    assert.match(found.result.content[0].text, /nitrofusion2/);
    const run = await call("tools/call", { name: "generate", arguments: { app_id: "nitrofusion2", inputs: { prompt: "test scene" } } });
    assert.match(run.result.content[0].text, /test-id/);
    assert.deepEqual(calls, [{ id: "nitrofusion2", values: { prompt: "test scene" }, preset: undefined }]);
    const done = await call("tools/call", { name: "get_job", arguments: { run_id: "test-id" } });
    assert.match(done.result.content[0].text, /succeeded/);
    const unknown = await call("tools/call", { name: "get_app", arguments: { app_id: "bad-app" } });
    assert.equal(unknown.result.isError, true);
    const gpu = await call("tools/call", { name: "gpu_status", arguments: {} });
    assert.match(gpu.result.content[0].text, /configured/);
  } finally {
    http.close();
    await once(http, "close");
  }
});
