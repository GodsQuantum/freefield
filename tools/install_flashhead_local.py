#!/usr/bin/env python3
"""Install the already provisioned, SHA-pinned Cloud9 FlashHead backend.
Run only while ComfyUI is idle and stopped. No service or package mutation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file,"sha256").hexdigest()

def write_same_or_new(path, content):
    if path.exists():
        if path.read_bytes()!=content:
            raise RuntimeError(f"Refuse overwrite: {path}")
        return
    with path.open("xb") as f: f.write(content)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("comfy","engine","models","python","deps","cache"):
        p.add_argument("--"+name, required=True,type=Path)
    p.add_argument("--verify-only",action="store_true")
    a=p.parse_args()
    src=Path(__file__).resolve().parents[1]/"infra/comfy_nodes/freefield_flashhead_streaming"
    lock=json.loads((src/"runtime-lock.json").read_text())
    for key in ("comfy","engine","models","python","deps","cache"):
        path=getattr(a,key)
        if not path.is_absolute() or not path.exists(): raise RuntimeError(f"Missing absolute path: {key}")
    if subprocess.check_output(["git","-C",str(a.engine),"rev-parse","HEAD"],text=True).strip()!=lock["engine_commit"]:
        raise RuntimeError("Unexpected FlashHead engine commit")
    patch=subprocess.check_output(["git","-C",str(a.engine),"diff","--","flash_head/audio_analysis/wav2vec2.py"])
    if hashlib.sha256(patch).hexdigest()!=lock["patch_sha256"]:
        raise RuntimeError("Unexpected compatibility patch")
    if subprocess.check_output(["git","-C",str(a.engine),"diff","--name-only"],text=True).splitlines()!=["flash_head/audio_analysis/wav2vec2.py"]:
        raise RuntimeError("Unexpected tracked engine changes")
    if sha(src/"__init__.py")!=lock["adapter_sha256"]: raise RuntimeError("Unexpected adapter digest")
    for file in lock["files"]:
        path=a.models/file["relative_path"]
        if path.stat().st_size!=file["bytes"] or sha(path)!=file["sha256"]: raise RuntimeError(f"Unexpected model: {path}")
    # Resolve all collisions before touching anything.
    dest=a.comfy/"custom_nodes/freefield_flashhead_streaming"
    runtime=(json.dumps({"source":str(a.engine),"python":str(a.python),"deps":str(a.deps),"cache":str(a.cache)},indent=2)+"\n").encode()
    payload={"__init__.py":(src/"__init__.py").read_bytes(),"runtime.json":runtime}
    if dest.exists():
        for name,content in payload.items():
            path=dest/name
            if not path.is_file() or path.read_bytes()!=content: raise RuntimeError(f"Refuse overwrite: {path}")
    pairs=[]
    for item in lock["aliases"]:
        link=a.comfy/"models"/item["directory"]/item["name"]
        target=a.models/item["relative_path"]
        if not link.parent.is_dir(): raise RuntimeError(f"Missing model directory: {link.parent}")
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve()!=target.resolve(): raise RuntimeError(f"Refuse replacement: {link}")
        pairs.append((link,target))
    if a.verify_only:
        if not dest.is_dir() or any(not link.is_symlink() for link, _ in pairs):
            raise RuntimeError("FlashHead adapter or model alias missing from installed runtime")
        print("FlashHead pinned engine, weights, installed adapter and aliases verified")
        return
    created=[]; newdir=False
    try:
        if not dest.exists():
            dest.mkdir();newdir=True
            for name,content in payload.items():
                path=dest/name; write_same_or_new(path,content);created.append(path)
        for link,target in pairs:
            if not link.is_symlink(): link.symlink_to(target);created.append(link)
    except BaseException:
        for path in reversed(created): path.unlink()
        if newdir: dest.rmdir()
        raise
    print("FlashHead local node and aliases installed; existing runtime unchanged")

if __name__=="__main__": main()
