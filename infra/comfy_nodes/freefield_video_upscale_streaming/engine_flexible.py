"""Bounded CFR video super-resolution; native RRDBNet via Spandrel, no frame accumulation."""
import argparse,json,math,subprocess as S,time
from fractions import Fraction
import torch,numpy as np
from spandrel import ModelLoader

def inspect(source):
    meta=json.loads(S.check_output(["ffprobe","-v","error","-show_streams","-of","json",source],timeout=20))
    v=next(s for s in meta["streams"] if s["codec_type"]=="video")
    w,h=int(v["width"]),int(v["height"]); fps=Fraction(v["avg_frame_rate"]); frames=int(v["nb_frames"])
    if not (128<=w<=960 and 128<=h<=960 and w%2==h%2==0 and w*h<=518400):raise ValueError("Even dimensions, 128-960 per edge, at most 518400 input pixels")
    if not (8<=fps<=60 and fps==Fraction(v["r_frame_rate"]) and 1<=frames/float(fps)<=20.05 and w*h*frames<=140000000):raise ValueError("CFR 8-60 fps; 1-20 seconds, max 140 million pixel-frames")
    if v.get("sample_aspect_ratio","N/A") not in ("N/A","1:1") or any(s.get("rotation",0) for s in v.get("side_data_list",[])):raise ValueError("Use upright square-pixel source")
    if v.get("color_transfer") in ("smpte2084","arib-std-b67") or v.get("pix_fmt") not in ("yuv420p","yuvj420p","yuv422p","yuv444p","rgb24"):raise ValueError("SDR 8-bit input required")
    packets=json.loads(S.check_output(["ffprobe","-v","error","-select_streams","v:0","-show_packets","-show_entries","packet=pts_time","-of","json",source],timeout=20))["packets"]
    pts=sorted(float(p["pts_time"]) for p in packets)
    if len(pts)!=frames or any(abs((b-a)-1/float(fps))>0.002 for a,b in zip(pts,pts[1:])):raise ValueError("Variable frame rate or incomplete frame timestamps; convert to CFR first")
    audio=next((s for s in meta["streams"] if s["codec_type"]=="audio"),None)
    return w,h,fps,frames,audio

def restore(model,x,w,h):
    if w*h<=512*512:return model(x)
    # Two long-axis tiles with context; stitch only each tile's central pixels.
    axis=3 if w>=h else 2; length=w if axis==3 else h
    split=(length//4)*2; pad=32
    pieces=[]
    for start,end in ((0,split),(split,length)):
        lo=max(0,start-pad);hi=min(length,end+pad)
        z=model(x.narrow(axis,lo,hi-lo))
        pieces.append(z.narrow(axis,2*(start-lo),2*(end-start)).cpu())
        del z
    return torch.cat(pieces,dim=axis)

def main():
    p=argparse.ArgumentParser();p.add_argument("model");p.add_argument("source");p.add_argument("output");a=p.parse_args()
    w,h,fps,frames,audio=inspect(a.source);duration=frames/float(fps)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    desc=ModelLoader().load_from_state_dict(torch.load(a.model,map_location="cpu",weights_only=True))
    assert desc.scale==2 and desc.supports_half
    model=desc.model.eval().to(device="cuda",dtype=torch.float16)
    t=time.monotonic();n=0
    raw=S.Popen(["ffmpeg","-nostdin","-v","error","-threads","2","-i",a.source,"-map","0:v:0","-fps_mode","passthrough","-frames:v",str(frames),"-f","rawvideo","-pix_fmt","rgb24","pipe:1"],stdout=S.PIPE)
    enc=None
    try:
        args=["ffmpeg","-nostdin","-v","error","-n","-f","rawvideo","-pix_fmt","rgb24","-s",f"{w*2}x{h*2}","-r",str(fps),"-i","pipe:0","-i",a.source,"-map","0:v:0","-map","1:a:0?","-c:v","libx264","-preset","fast","-crf","16","-pix_fmt","yuv420p","-threads","4"]
        if audio:args+=["-c:a","copy"] if audio["codec_name"]=="aac" else ["-c:a","aac","-b:a","192k"]
        enc=S.Popen(args+["-t",f"{duration:.9f}","-movflags","+faststart",a.output],stdin=S.PIPE)
        with torch.inference_mode():
            while n<frames:
                data=raw.stdout.read(w*h*3)
                if len(data)!=w*h*3:raise RuntimeError(f"Short decode at {n}/{frames}")
                x=torch.from_numpy(np.frombuffer(data,dtype=np.uint8).copy().reshape(h,w,3)).permute(2,0,1).unsqueeze(0).to("cuda",torch.float16)/255
                y=restore(model,x,w,h).clamp(0,1).mul(255).round().to(torch.uint8).cpu().squeeze(0).permute(1,2,0).contiguous().numpy()
                assert y.shape==(h*2,w*2,3)
                enc.stdin.write(y.tobytes());n+=1
                if n in (1,5,10,frames) or n%25==0:print(json.dumps({"frames":n,"total":frames,"seconds":round(time.monotonic()-t,2)}),flush=True)
        enc.stdin.close()
        if raw.wait(timeout=20) or enc.wait(timeout=30) or n!=frames:raise RuntimeError("Incomplete encode")
        print(json.dumps({"status":"SUCCESS","frames":n,"seconds":round(time.monotonic()-t,2)}),flush=True)
    finally:
        for proc in (raw,enc):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except S.TimeoutExpired:proc.kill();proc.wait()
if __name__=="__main__":main()
