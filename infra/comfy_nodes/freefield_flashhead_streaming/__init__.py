"""Freefield's local adapter for the pinned upstream FlashHead streaming CLI.

No service, no second GPU lock: the existing ComfyUI worker already owns the
native GPU lock. Model generation and video encoding are upstream FlashHead.
"""
import gc
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time

import numpy as np
from PIL import Image
import soundfile as sf
import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar
from comfy_api.input_impl.video_types import VideoFromFile

NAMES = (
    ("diffusion_models", "soulx-flashhead-lite-dit.safetensors"),
    ("vae", "soulx-flashhead-lite-vae-ltx.safetensors"),
    ("audio_encoders", "soulx-flashhead-wav2vec2-base-960h.safetensors"),
)

class _OwnedVideo(VideoFromFile):
    def __init__(self, filename, work):
        super().__init__(str(filename))
        self._work = work

class FlashHeadStreaming:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "portrait": ("IMAGE",), "audio": ("AUDIO",),
            "model_name": ([NAMES[0][1]],),
            "vae_name": ([NAMES[1][1]],),
            "audio_encoder_name": ([NAMES[2][1]],),
            "seed": ("INT", {"default":42,"min":0,"max":4294967295}),
        }}
    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "generate"
    CATEGORY = "Freefield/video"

    def generate(self, portrait, audio, model_name, vae_name, audio_encoder_name, seed):
        cfg = json.loads(Path(__file__).with_name("runtime.json").read_text())
        source = Path(cfg["source"])
        weights = [Path(folder_paths.get_full_path_or_raise(category,name)).resolve()
                   for (category,_),name in zip(NAMES,(model_name,vae_name,audio_encoder_name))]
        root = weights[0].parent.parent
        wavdir = weights[2].parent
        if weights[0] != root/"Model_Lite/diffusion_pytorch_model.safetensors" or weights[1] != root/"VAE_LTX/diffusion_pytorch_model.safetensors" or weights[2].name != "model.safetensors":
            raise ValueError("FlashHead checkpoint layout mismatch")
        for p in (root/"Model_Lite/config.json",root/"VAE_LTX/config.json",wavdir/"config.json",wavdir/"preprocessor_config.json"):
            if not p.is_file(): raise FileNotFoundError(str(p))
        wave = audio["waveform"].detach().cpu().numpy()
        rate = int(audio["sample_rate"])
        if portrait.ndim != 4 or len(portrait)!=1 or wave.ndim!=3 or len(wave)!=1 or wave.shape[1] not in (1,2) or rate<8000 or rate>192000:
            raise ValueError("One portrait and mono/stereo audio required")
        duration = wave.shape[2]/rate
        if not 2 <= duration <= 45.001 or not np.isfinite(wave).all():
            raise ValueError("Audio must be finite and 2-45 seconds")
        work = tempfile.TemporaryDirectory(prefix="freefield-flashhead-",dir=folder_paths.get_temp_directory())
        w = Path(work.name); target = w/"video.mp4"
        child=None
        try:
            Image.fromarray((portrait[0].detach().cpu().numpy()[...,:3].clip(0,1)*255).round().astype(np.uint8)).save(w/"portrait.png")
            sf.write(w/"audio.wav", wave[0].T, rate, subtype="FLOAT")
            del wave
            mm.unload_all_models(); gc.collect(); mm.soft_empty_cache()
            env=dict(os.environ,HF_HUB_OFFLINE="1",HF_HOME=cfg["cache"],PYTHONPATH=cfg["deps"]+os.pathsep+str(source),OMP_NUM_THREADS="4",MKL_NUM_THREADS="4")
            cmd=[cfg["python"],"generate_video.py","--ckpt_dir",str(root),"--wav2vec_dir",str(wavdir),"--model_type","lite",
                "--cond_image",str(w/"portrait.png"),"--audio_path",str(w/"audio.wav"),"--save_file",str(target),
                "--audio_encode_mode","stream","--dtype","float16","--attention-backend","sdpa","--no-compile-model","--no-compile-vae",
                "--async-video-write","--rope-implementation","legacy","--motion-cache-mode","reencode","--base_seed",str(seed)]
            chunks=math.ceil(duration*25/24)
            bar=ProgressBar(chunks)
            limit=time.monotonic()+min(2100,180+chunks*38)
            log=w/"engine.log"
            with log.open("w") as f:
                child=subprocess.Popen(cmd,cwd=source,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                seen=0
                while child.poll() is None:
                    mm.throw_exception_if_processing_interrupted()
                    if time.monotonic()>limit: raise TimeoutError("FlashHead rendering deadline")
                    count=len(re.findall(r"Generate video chunk-\d+ done",log.read_text(errors="replace")))
                    if count>seen: bar.update(count-seen); seen=count
                    time.sleep(1)
                if child.returncode or not target.is_file():
                    raise RuntimeError("FlashHead failed: "+log.read_text(errors="replace")[-3000:])
                if seen<chunks: bar.update(chunks-seen)
            # Keep the native file-backed VIDEO alive until SaveVideo remuxes it.
            (w/"portrait.png").unlink(); (w/"audio.wav").unlink(); log.unlink()
            return (_OwnedVideo(target,work),)
        except BaseException:
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL); child.wait(timeout=10)
            work.cleanup()
            raise

NODE_CLASS_MAPPINGS={"FreefieldFlashHeadStreaming":FlashHeadStreaming}
NODE_DISPLAY_NAME_MAPPINGS={"FreefieldFlashHeadStreaming":"FlashHead Lite - streaming talking avatar"}
