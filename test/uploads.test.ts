import { test } from "node:test";
import assert from "node:assert/strict";
import type { IncomingMessage } from "node:http";
import { toRequest, uploadLimitBytes } from "../server/uploads.ts";

Object.defineProperty(globalThis, "document", { configurable: true, value: { documentElement: { lang: "en" } } });
Object.defineProperty(globalThis, "localStorage", { configurable: true, value: { getItem: () => null, setItem: () => {} } });
Object.defineProperty(globalThis, "location", { configurable: true, value: { search: "", href: "http://upload.test/" } });
// Visibility observers are unrelated to submitting: no card is rendered in this contract test.
Object.defineProperty(globalThis, "IntersectionObserver", { configurable: true, value: class {
  observe() {} unobserve() {} disconnect() {}
} });
const { loadCatalog } = await import(new URL("../web/js/catalog.js", import.meta.url).href);
const { feed } = await import(new URL("../web/js/feed.js", import.meta.url).href);

test("the shared submission path refuses files above the server body budget before any POST", async () => {
  const originalFetch = globalThis.fetch;
  const posts: unknown[] = [];
  globalThis.fetch = (async (input: unknown, init?: RequestInit) => {
    if (input === "/api/apps") return Response.json({ apps: [], drafts: false, hosted: true, upload: { max_body_bytes: 512 } });
    posts.push({ input, init });
    return Response.json({ id: "upload-should-not-start", app: "smooth", status: "succeeded" });
  }) as typeof fetch;
  try {
    await loadCatalog();
    const form = new FormData();
    form.append("clip", new File([new Uint8Array(1024)], "clip.mp4", { type: "video/mp4" }));
    await assert.rejects(feed.run("smooth", form), /upload|request|limit|MiB/i);
    assert.equal(posts.length, 0, "the request must not leave the browser");
    assert.equal(feed.get("upload-should-not-start"), undefined);
  } finally {
    globalThis.fetch = originalFetch;
    feed.entries.delete("upload-should-not-start");
  }
});

const { prepareUploadRequest, UploadBudgetError } = await import(new URL("../web/js/uploads.js", import.meta.url).href);

function multipartCase(kind: string): FormData {
  const form = new FormData();
  if (kind === "empty") return form;
  form.append('prompt"\n雪', 'Été 🐈\r\nligne 2\nligne 3\rfin');
  form.append("clip", new File([new Uint8Array(37)], 'clip"\r\n雪.mp4', { type: "video/mp4" }));
  if (kind === "multiple") {
    form.append("clip", new File([new Uint8Array(41)], "restored.mp4", { type: "video/mp4" }));
    form.append("mask", new File([new Uint8Array(23)], "mask.png", { type: "image/png" }));
    form.append("empty", new File([], "empty", { type: "" }));
    form.append("prompt", "another value");
    form.append("prompt", "repeated value");
  }
  if (kind === "replacement") form.set("clip", new File([new Uint8Array(7)], "new.mp4", { type: "video/mp4" }));
  return form;
}

for (const kind of ["empty", "single", "multiple", "replacement"]) {
  test(`native multipart ${kind}: exact body limit passes, one byte below fails`, async () => {
    const form = multipartCase(kind);
    const expected = (await new Request("http://upload.test/run", { method: "POST", body: form }).arrayBuffer()).byteLength;
    const request = await prepareUploadRequest("http://upload.test/run", form, expected);
    assert.equal(request.bodyUsed, false, "sizing must not consume the real upload");
    assert.equal((await request.arrayBuffer()).byteLength, expected);
    await assert.rejects(prepareUploadRequest("http://upload.test/run", form, expected - 1), UploadBudgetError);
  });
}

test("all fields, restored files and masks share the budget in feed.run; it sends the measured Request", async () => {
  const originalFetch = globalThis.fetch;
  const form = multipartCase("multiple");
  const expected = (await new Request("http://upload.test/run", { method: "POST", body: form }).arrayBuffer()).byteLength;
  let limit: number | null = expected - 1;
  const posts: Request[] = [];
  globalThis.fetch = (async (input: unknown) => {
    if (input === "/api/apps") return Response.json({ apps: [], upload: { max_body_bytes: limit } });
    assert.ok(input instanceof Request);
    assert.equal(input.url, "http://upload.test/api/apps/smooth/run");
    assert.equal(input.method, "POST");
    posts.push(input);
    return Response.json({ id: "upload-accepted", app: "smooth", status: "succeeded" });
  }) as typeof fetch;
  try {
    await loadCatalog();
    await assert.rejects(feed.run("smooth", form), /total limit/i);
    assert.equal(posts.length, 0, "aggregate body is rejected before fetch");
    limit = expected;
    await loadCatalog();
    await feed.run("smooth", form);
    assert.equal(posts.length, 1);
    assert.equal((await posts[0].arrayBuffer()).byteLength, limit);
    for (const disabled of [0, null]) {
      limit = disabled;
      await loadCatalog();
      await feed.run("smooth", form);
    }
    assert.equal(posts.length, 3, "zero and an absent/null cap preserve local uploads");
  } finally {
    globalThis.fetch = originalFetch;
    feed.entries.delete("upload-accepted");
  }
});

test("text-only fields and native multipart overhead count against the cap", async () => {
  const form = new FormData();
  form.append("prompt", "雪".repeat(200));
  await assert.rejects(prepareUploadRequest("http://upload.test/run", form, 600), UploadBudgetError);
  for (const cap of [0, null, undefined]) {
    const request = await prepareUploadRequest("http://upload.test/run", form, cap);
    assert.equal(request.bodyUsed, false);
    assert.ok((await request.arrayBuffer()).byteLength > 600);
  }
});

test("the operator body cap retains its hosted, local, zero and custom MiB semantics", () => {
  assert.equal(uploadLimitBytes(undefined, true), 104857600);
  assert.equal(uploadLimitBytes(undefined, false), null);
  assert.equal(uploadLimitBytes("0", true), null);
  assert.equal(uploadLimitBytes("0", false), null);
  assert.equal(uploadLimitBytes("0.5", true), 524288);
  assert.equal(uploadLimitBytes("200", false), 209715200);
});

function incoming(chunks: number[], headers: Record<string, string> = {}, read?: () => void): IncomingMessage {
  return {
    url: "/api/apps/smooth/run", method: "POST", headers,
    async *[Symbol.asyncIterator]() {
      for (const size of chunks) { read?.(); yield Buffer.alloc(size); }
    },
  } as unknown as IncomingMessage;
}

test("oversized Content-Length is refused before reading or retaining any chunk", async () => {
  let reads = 0;
  await assert.rejects(toRequest(incoming([9], { "content-length": "9" }, () => reads++), 8), /MiB total/);
  assert.equal(reads, 0);
});

test("streamed bodies still enforce the exact cap without trusting Content-Length", async () => {
  for (const headers of [{}, { "content-length": "1" }] as Record<string, string>[]) {
    const atLimit = await toRequest(incoming([3, 5], headers), 8);
    assert.equal((await atLimit.arrayBuffer()).byteLength, 8);
    await assert.rejects(toRequest(incoming([3, 5, 1], headers), 8), /MiB total/);
  }
  const unlimited = await toRequest(incoming([3, 5, 1], { "content-length": "9" }), null);
  assert.equal((await unlimited.arrayBuffer()).byteLength, 9);
});
