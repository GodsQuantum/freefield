/** Coalesce model discovery, without pinning a cold-start failure for 30s. */
export class ModelInventory {
  private cached: { at: number; files: Set<string> | null } | undefined;
  private pending: Promise<Set<string> | null> | undefined;
  private readonly load: (timeoutMs: number) => Promise<Set<string> | null>;
  private readonly now: () => number;
  constructor(load: (timeoutMs: number) => Promise<Set<string> | null>, now = Date.now) { this.load = load; this.now = now; }
  async get(wait = false): Promise<Set<string> | null> {
    const c = this.cached;
    if (c && c.files !== null && this.now() - c.at < 30_000) return c.files;
    if (!wait && c && c.files === null && this.now() - c.at < 2_000) return null;
    if (this.pending) {
      const files = await this.pending;
      if (files || !wait) return files;
      // A short discovery may have timed out while the lazy backend started.
      // Fall through to one longer, explicitly requested attempt.
      if (this.pending) return this.pending;
    }
    const p = Promise.resolve().then(() => this.load(wait ? 35_000 : 5_000))
      .catch(() => null).then(files => {
        this.cached = {at: this.now(), files};
        return files;
      });
    this.pending = p;
    try { return await p; } finally { if (this.pending === p) this.pending = undefined; }
  }
}
