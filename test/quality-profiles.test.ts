import {test} from "node:test";
import assert from "node:assert/strict";
import {loadRegistry,repoRoot} from "../server/registry.ts";
import {certifiedCatalog} from "../server/certified-catalog.ts";
import {resolveValues} from "../server/runner.ts";
const {canAttempt,nextTargets,referenceSettings,restoreReference}=await import(new URL("../web/js/quality-profiles.js",import.meta.url).href);
const c=certifiedCatalog(repoRoot(),loadRegistry().apps);
const apps=c.apps.map(a=>({...a,runnable:true,qualification:c.proofs.get(a.id)}));
test("all 32 reference profiles resolve to valid technical inputs and preserve creative work",()=>{
  assert.equal(apps.length,32);
  for(const a of apps){
    assert.ok(a.qualification?.guidance,a.id);
    const values={prompt:"keep me",seed:1234,portrait:"my portrait",audio:"my audio",reference:"my reference"};
    const before=structuredClone(values);
    restoreReference(a,values);
    for(const [key,value] of Object.entries(before))assert.equal((values as any)[key],value,a.id+":"+key);
    for(const [id,value] of referenceSettings(a)){
      const input=a.inputs.find(i=>i.id===id)!;
      assert.ok(input);
      if(input.type==="select")assert.ok(input.options?.some(o=>o.value===value),a.id+":"+id);
      if(input.min!==undefined)assert.ok(Number(value)>=input.min);
      if(input.max!==undefined)assert.ok(Number(value)<=input.max);
    }
  }
});
test("media reuse prioritizes real certified destinations and exposes every non-mask slot",()=>{
  assert.equal(nextTargets(apps,"video")[0].app.id,"upscale-video-hd");
  assert.equal(nextTargets(apps,"audio")[0].app.id,"talking");
  assert.equal(nextTargets(apps,"audio")[0].slot.id,"audio");
  const swap=apps.find(a=>a.id==="character-swap")!;
  assert.equal(nextTargets(apps,"image").filter((t:any)=>t.app.id===swap.id).length,swap.inputs.filter(i=>i.type==="image"&&!i.paint).length);
  assert.ok(nextTargets(apps,"image").every((t:any)=>!t.slot.paint));
  assert.equal(canAttempt({...apps[0],runnable:false,inventory_status:"unknown"}),true);
  assert.equal(canAttempt({...apps[0],runnable:false,inventory_status:"known"}),false);
  assert.equal(canAttempt({...apps[0],runnable:true,missing_pack:"missing"}),false);
});
test("random seeds respect the native 32-bit limits; ACE Turbo prevents excessive ranges",()=>{
  const f={bytes:new Uint8Array([1]),name:"test"};
  for(const id of ["talking","voice-clone"]){
    const a=apps.find(a=>a.id===id)!;
    const inputs=Object.fromEntries(a.inputs.filter(i=>["image","video","audio"].includes(i.type)).map(i=>[i.id,f]));
    for(let n=0;n<50;n++){const v=Number(resolveValues(a,{...inputs,seed:-1}).seed);assert.ok(v>=0&&v<=4294967295);}
    assert.throws(()=>resolveValues(a,{...inputs,seed:4294967296}));
  }
  const ace=apps.find(a=>a.id==="music-ace15-local")!;
  assert.equal(resolveValues(ace,{}).duration,30);
  assert.throws(()=>resolveValues(ace,{steps:10000}));
  assert.throws(()=>resolveValues(ace,{duration:1000}));
});


test("certified quality instructions reflect effective native guidance and fast branches",()=>{
  const qwen=apps.find(a=>a.id==="qwen21-edit")!;
  assert.equal(qwen.graph!["8"].inputs.cfg,8);
  assert.match(qwen.inputs.find(i=>i.id==="negative_prompt")!.help!,/CFG 8/);
  assert.match(qwen.qualification!.scope,/CFG 8/);
  const krea=apps.find(a=>a.id==="styles")!;
  assert.equal(krea.inputs.find(i=>i.id==="ultra_fast_mode")!.default,true);
  assert.deepEqual(krea.graph!["ff_krea_effective_enhance"].inputs.on_true,["ff_krea_ultra_no_enhance",0]);
  assert.equal(krea.graph!["ff_krea_ultra_no_enhance"].inputs.value,false);
  assert.match(krea.qualification!.scope,/rewrite bypassed/);
  assert.match(krea.inputs.find(i=>i.id==="enhance")!.help!,/ignoré/);
  const moss=apps.find(a=>a.id==="moss-image")!;
  assert.equal(moss.inputs.find(i=>i.id==="fast_mode")!.default,true);
  assert.equal(moss.graph!["ff_moss_fast_sampler"].inputs.steps,4);
  assert.match(moss.inputs.find(i=>i.id==="steps")!.help!,/qualité uniquement/);
});
