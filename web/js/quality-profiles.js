const NEXT = {
  image: ["h3-photo-int4", "talking", "recast", "character-swap", "upscale", "qwen21-edit", "headshot", "angles", "relight", "inpaint", "remove-bg", "depth-map"],
  video: ["upscale-video-hd", "smooth", "upscale-video", "recast", "extend-video-int4", "reshoot-int4", "video-remove-bg"],
  audio: ["talking", "voice-clone"]
};
/** Unknown inventory is retryable, not an assertion that models are installed. */
export function canAttempt(app) {
  return !app.missing_pack && (app.runnable || (app.status === "ready" && !!app.qualification && app.inventory_status === "unknown"));
}
export function nextTargets(apps, type) {
  const order = NEXT[type] || [];
  const rank = app => order.includes(app.id) ? order.indexOf(app.id) : order.length;
  return [...apps].filter(canAttempt).sort((a,b) => rank(a)-rank(b) || a.name.localeCompare(b.name))
    .flatMap(app => app.inputs.filter(slot => slot.type === type && !slot.paint).map(slot => ({app,slot,recommended:order.includes(app.id)})));
}
export function referenceSettings(app) {
  return Object.entries(app.qualification?.reference_settings ?? {}).filter(([id]) =>
    app.inputs.some(i => i.id === id && ["slider","number","select","toggle"].includes(i.type)));
}
export function restoreReference(app, values) {
  for (const [id,value] of referenceSettings(app)) values[id] = structuredClone(value);
  return values;
}
