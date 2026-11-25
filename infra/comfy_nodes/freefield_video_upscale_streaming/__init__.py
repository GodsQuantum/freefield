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

class VideoUpscale2xFlexible(VideoUpscale2x):
    def generate(self,clip,model_name):
        from comfy.utils import ProgressBar
        src=clip.get_stream_source()
        w,h=clip.get_dimensions(); fps=float(clip.get_frame_rate()); duration=clip.get_duration()
        if not isinstance(src,str) or clip.get_active_trim_window()!=(0.,0.) or not (128<=w<=960 and 128<=h<=960 and w%2==h%2==0 and w*h<=518400 and 8<=fps<=60 and 1<=duration<=20.05 and w*h*round(fps*duration)<=140000000):
            raise ValueError("Use an untrimmed CFR SDR clip, even dimensions up to 960px per edge, 1-20 seconds; maximum 140M pixel-frames")
        model=folder_paths.get_full_path_or_raise("upscale_models",model_name)
        work=tempfile.TemporaryDirectory(prefix="freefield-upscale2x-",dir=folder_paths.get_temp_directory()); p=Path(work.name);out=p/"result.mp4";log=p/"engine.log";child=None
        frames=round(fps*duration);bar=ProgressBar(frames);seen=0
        try:
            mm.unload_all_models();gc.collect();mm.soft_empty_cache()
            with log.open("w") as f:
                child=subprocess.Popen([sys.executable,str(Path(__file__).with_name("engine_flexible.py")),model,src,str(out)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                deadline=time.monotonic()+min(1100,120+frames*w*h/180000)
                while child.poll() is None:
                    mm.throw_exception_if_processing_interrupted()
                    if time.monotonic()>deadline:raise TimeoutError("Video upscale deadline")
                    for line in log.read_text(errors="replace").splitlines():
                        try:count=int(json.loads(line).get("frames",0))
                        except (ValueError,TypeError):continue
                        if count>seen:bar.update(count-seen);seen=count
                    time.sleep(1)
                if child.returncode or not out.is_file():raise RuntimeError(log.read_text(errors="replace")[-3000:])
            if seen<frames:bar.update(frames-seen)
            log.unlink();return (OwnedVideo(out,work),)
        except BaseException:
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
            work.cleanup();raise
NODE_CLASS_MAPPINGS["FreefieldVideoUpscale2xFlexible"]=VideoUpscale2xFlexible
NODE_DISPLAY_NAME_MAPPINGS["FreefieldVideoUpscale2xFlexible"]="RealESRGAN x2 flexible streaming video"
