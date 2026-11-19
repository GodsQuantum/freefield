/** A body budget is shared by every file, field and multipart header in one run. */
export class UploadBudgetError extends Error {
  constructor(maxBodyBytes) {
    super(`Upload exceeds the ${maxBodyBytes / 2 ** 20} MiB total request limit`);
    this.name = "UploadBudgetError";
    this.maxBodyBytes = maxBodyBytes;
  }
}

function boundaryLength(request) {
  const type = request.headers.get("content-type");
  const match = /(?:^|;)\s*boundary=(?:"([^"]+)"|([^;\s]+))/i.exec(type ?? "");
  if (!match) throw new Error("Missing native multipart boundary");
  return (match[1] ?? match[2]).length;
}

/** Keep the native Request we measured, so fetch sends exactly that boundary.
 * Only empty file placeholders are serialized for sizing: real payloads stay in
 * their original Blobs, even for large uploads or files restored from history. */
export async function prepareUploadRequest(url, formData, maxBodyBytes) {
  const request = new Request(url, { method: "POST", body: formData });
  if (!Number.isFinite(maxBodyBytes) || maxBodyBytes <= 0) return request;

  const empty = new FormData();
  let fileBytes = 0;
  let entries = 0;
  for (const [name, value] of formData) {
    entries++;
    if (typeof value === "string") empty.append(name, value);
    else {
      fileBytes += value.size;
      if (fileBytes > maxBodyBytes) throw new UploadBudgetError(maxBodyBytes);
      empty.append(name, new File([], value.name, { type: value.type }));
    }
  }
  const envelope = new Request(url, { method: "POST", body: empty });
  // Native encoders may choose different boundary lengths. There is one boundary
  // per entry, plus the closing boundary; all escaping/newlines remain native.
  let size = fileBytes + (boundaryLength(request) - boundaryLength(envelope)) * (entries + 1);
  const reader = envelope.body.getReader();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > maxBodyBytes) throw new UploadBudgetError(maxBodyBytes);
    }
  } finally {
    await reader.cancel();
    reader.releaseLock();
  }
  return request;
}
