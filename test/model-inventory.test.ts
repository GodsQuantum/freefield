import {test} from "node:test";
import assert from "node:assert/strict";
import {ModelInventory} from "../server/model-inventory.ts";

test("cold inventory can be retried immediately on submission; successful cache expires", async () => {
  let now=0;
  const calls:number[]=[];
  const models=new Set(["diffusion_models/a"]);
  const i=new ModelInventory(async ms=>{calls.push(ms);return ms===5000?null:models},()=>now);
  assert.equal(await i.get(),null);
  assert.equal(await i.get(),null);
  assert.deepEqual(calls,[5000]);
  assert.equal(await i.get(true),models);
  assert.deepEqual(calls,[5000,35000]);
  assert.equal(await i.get(),models);
  now=31_000;
  assert.equal(await i.get(),null);
});
test("simultaneous cold discovery shares request and submission retries once",async()=>{
  const calls:number[]=[];
  let release:(files:Set<string>|null)=>void=()=>{};
  const i=new ModelInventory(ms=>{calls.push(ms);return ms===5000?new Promise(resolve=>{release=resolve}):Promise.resolve(new Set(["a"]))});
  const a=i.get(), b=i.get(), c=i.get(true),d=i.get(true);
  await Promise.resolve();release(null);
  assert.equal(await a,null);assert.equal(await b,null);
  assert.deepEqual(await c,new Set(["a"]));assert.deepEqual(await d,new Set(["a"]));
  assert.deepEqual(calls,[5000,35000]);
});
