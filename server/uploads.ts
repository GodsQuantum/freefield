import type { IncomingMessage } from "node:http";
import { InputError } from "./runner.ts";

/** FREEFIELD_MAX_UPLOAD_MB has always used binary MiB; zero disables the body cap. */
export function uploadLimitBytes(value: string | undefined, hosted: boolean): number | null {
  const mb = Number(value || (hosted ? 100 : 0));
  return mb ? mb * 2 ** 20 : null;
}

export async function toRequest(req: IncomingMessage, maxBodyBytes: number | null): Promise<Request> {
  const checkSize = (size: number) => {
    if (maxBodyBytes !== null && size > maxBodyBytes) {
      throw new InputError(`uploads are limited to ${maxBodyBytes / 2 ** 20} MiB total here`);
    }
  };
  // Reject a known oversized body before retaining chunks. The counter below also
  // enforces the same budget for chunked requests and untrusted declared lengths.
  checkSize(Number(req.headers["content-length"]));
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const c of req) {
    size += (c as Buffer).byteLength;
    checkSize(size);
    chunks.push(c as Buffer);
  }
  return new Request(`http://local${req.url}`, {
    method: req.method,
    headers: req.headers as Record<string, string>,
    body: chunks.length ? Buffer.concat(chunks) : undefined,
  });
}
