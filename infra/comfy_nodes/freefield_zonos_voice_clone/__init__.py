"""Existing native ZONOS3-Mini service; no model loading or extra daemon."""
import base64
import gc
import io
import json
from pathlib import Path
import urllib.request
import numpy as np
import soundfile as sf
import torch
import folder_paths
import comfy.model_management as mm

paths, extensions=folder_paths.folder_names_and_paths["audio_encoders"]
folder_paths.folder_names_and_paths["audio_encoders"]=(paths,set(extensions)|{".gguf"})
NAMES=("zonos3-backbone-ptq.gguf","zonos3-spk-encoder.gguf","zonos3-zac-full-mr.gguf")
class ZonosVoiceClone:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"reference":("AUDIO",),"text":("STRING",{"multiline":True}),"seed":("INT",{"default":42,"min":0,"max":4294967295}),"backbone":([NAMES[0]],),"speaker_encoder":([NAMES[1]],),"decoder":([NAMES[2]],)}}
    RETURN_TYPES=("AUDIO",)
    FUNCTION="generate"
    CATEGORY="Freefield/audio"
    def generate(self,reference,text,seed,backbone,speaker_encoder,decoder):
        cfg=json.loads(Path(__file__).with_name("runtime.json").read_text())
        for name in (backbone,speaker_encoder,decoder):
            actual=Path(folder_paths.get_full_path_or_raise("audio_encoders",name)).resolve()
            if actual!=Path(cfg["weights"][name]).resolve():raise ValueError("Unexpected voice model path")
        text=text.strip()
        if not 1<=len(text)<=200:raise ValueError("French text must contain 1-200 characters")
        wave=reference["waveform"].detach().cpu().numpy()
        rate=int(reference["sample_rate"])
        if wave.ndim!=3 or len(wave)!=1 or wave.shape[1] not in (1,2) or not 8000<=rate<=192000 or not 2<=wave.shape[2]/rate<=30 or not np.isfinite(wave).all():raise ValueError("One finite 2-30 second reference recording required")
        # Lossless PCM from native LoadAudio. Do not add a persistent speaker.
        buf=io.BytesIO()
        sf.write(buf,wave[0].T,rate,format="WAV",subtype="PCM_16")
        del wave
        mm.unload_all_models();gc.collect();mm.soft_empty_cache()
        mm.throw_exception_if_processing_interrupted()
        req={"text":text,"language":"fr_fr","speaker_audio_base64":base64.b64encode(buf.getvalue()).decode("ascii"),"speaker_mode":"embedding","decoder":"full_mr","stream":False,"format":"wav","seed":int(seed),"laci":True,"max_tokens":360,"temperature":0.75,"topk":80,"top_p":0.69,"repetition_penalty":2.4,"repetition_window":96,"repetition_codebooks":6}
        # Fixed administrator-configured service, no user-controlled network target.
        with urllib.request.urlopen(urllib.request.Request(cfg["base_url"]+"/v2/generate",data=json.dumps(req).encode(),headers={"Content-Type":"application/json"}),timeout=100) as response:
            raw=response.read(6_000_001)
            if len(raw)>6_000_000 or response.headers.get("X-Decoder")!="full_mr":raise ValueError("Unexpected voice decoder or response size")
        mm.throw_exception_if_processing_interrupted()
        y,sr=sf.read(io.BytesIO(raw),dtype="float32",always_2d=True)
        if sr!=44100 or y.shape[1]!=1 or not 0.25<=len(y)/sr<=25 or not np.isfinite(y).all():raise ValueError("Invalid voice output")
        return ({"waveform":torch.from_numpy(y.T.copy()).unsqueeze(0),"sample_rate":sr},)
NODE_CLASS_MAPPINGS={"FreefieldZonosVoiceClone":ZonosVoiceClone}
NODE_DISPLAY_NAME_MAPPINGS={"FreefieldZonosVoiceClone":"ZONOS3-Mini native voice clone"}
