import argparse, subprocess as S, json, time, torch, numpy as np
from pathlib import Path
from spandrel import ModelLoader
def main():
 p=argparse.ArgumentParser();p.add_argument("model");p.add_argument("source");p.add_argument("output");a=p.parse_args()
 info=json.loads(S.check_output(["ffprobe","-v","error","-show_streams","-of","json",a.source]))
 v=next(s for s in info["streams"] if s["codec_type"]=="video")
 w,h=v["width"],v["height"];fps=v["avg_frame_rate"]
 assert (w,h)==(512,512) and fps=="25/1",(w,h,fps)
 duration=float(v["duration"]);frames=round(duration*25);assert 25<=frames<=250 and abs(duration-frames/25)<0.02
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
 # PyTorch's safe weight-only deserialization; official release SHA verified outside.
 sd=torch.load(a.model,map_location="cpu",weights_only=True)
 desc=ModelLoader().load_from_state_dict(sd)
 assert desc.scale==2 and desc.supports_half
 model=desc.model.eval().to(device="cuda",dtype=torch.float16)
 t=time.monotonic();n=0
 raw=S.Popen(["ffmpeg","-nostdin","-v","error","-threads","2","-i",a.source,"-map","0:v:0","-frames:v",str(frames),"-f","rawvideo","-pix_fmt","rgb24","pipe:1"],stdout=S.PIPE)
 enc=S.Popen(["ffmpeg","-nostdin","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s","1024x1024","-r","25","-i","pipe:0","-i",a.source,"-map","0:v:0","-map","1:a:0?","-c:v","libx264","-preset","fast","-crf","16","-pix_fmt","yuv420p","-threads","4","-c:a","copy","-t",str(frames/25),a.output],stdin=S.PIPE)
 try:
  with torch.inference_mode():
   while n<frames:
    data=raw.stdout.read(w*h*3)
    if not data:break
    assert len(data)==w*h*3
    x=torch.from_numpy(np.frombuffer(data,dtype=np.uint8).copy().reshape(h,w,3)).permute(2,0,1).unsqueeze(0).to("cuda",torch.float16)/255
    y=model(x).clamp(0,1).mul(255).round().to(torch.uint8).cpu().squeeze(0).permute(1,2,0).contiguous().numpy()
    assert y.shape==(1024,1024,3)
    enc.stdin.write(y.tobytes());n+=1
    if n in (1,5,10) or n%25==0:print(json.dumps({"frames":n,"seconds":round(time.monotonic()-t,2)}),flush=True)
  enc.stdin.close();assert raw.wait(timeout=20)==0 and enc.wait(timeout=30)==0
  assert n==frames,n
  print(json.dumps({"status":"SUCCESS","frames":n,"seconds":round(time.monotonic()-t,2)}),flush=True)
 finally:
  for p in (raw,enc):
   if p.poll() is None:p.terminate();p.wait(timeout=10)
if __name__=="__main__":main()
