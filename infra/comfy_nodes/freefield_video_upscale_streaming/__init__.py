"""Bounded streaming RealESRGAN x2 video restoration on the existing GPU worker."""
import gc,json,os,signal,subprocess,sys,tempfile,time
from pathlib import Path
import folder_paths
import comfy.model_management as mm
from comfy_api.input_impl.video_types import VideoFromFile
class OwnedVideo(VideoFromFile):
    def __init__(self,file,work):
        super().__init__(str(file));self.work=work
class VideoUpscale2x:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"clip":("VIDEO",),"model_name":(["RealESRGAN_x2plus.pth"],)}}
    RETURN_TYPES=("VIDEO",)
    FUNCTION="generate"
    CATEGORY="Freefield/video"
    def generate(self,clip,model_name):
        src=clip.get_stream_source()
        if not isinstance(src,str) or clip.get_active_trim_window()!=(0.,0.) or clip.get_dimensions()!=(512,512) or not 1<=clip.get_duration()<=10.05:raise ValueError("Use an untrimmed 512x512 25fps file, 1-10 seconds")
        model=folder_paths.get_full_path_or_raise("upscale_models",model_name)
        work=tempfile.TemporaryDirectory(prefix="freefield-upscale2x-",dir=folder_paths.get_temp_directory());w=Path(work.name);out=w/"result.mp4";log=w/"engine.log";child=None
        try:
            mm.unload_all_models();gc.collect();mm.soft_empty_cache()
            with log.open("w") as f:
                child=subprocess.Popen([sys.executable,str(Path(__file__).with_name("engine.py")),model,src,str(out)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                deadline=time.monotonic()+450
                while child.poll() is None:
                    mm.throw_exception_if_processing_interrupted()
                    if time.monotonic()>deadline:raise TimeoutError("Video upscale deadline")
                    time.sleep(1)
                if child.returncode or not out.is_file():raise RuntimeError(log.read_text(errors="replace")[-2000:])
            log.unlink()
            return (OwnedVideo(out,work),)
        except BaseException:
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
            work.cleanup();raise
NODE_CLASS_MAPPINGS={"FreefieldVideoUpscale2x":VideoUpscale2x}
NODE_DISPLAY_NAME_MAPPINGS={"FreefieldVideoUpscale2x":"RealESRGAN x2 streaming video"}
