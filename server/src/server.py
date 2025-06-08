#!/usr/bin/env python3
import os, glob, cv2, numpy as np, threading, multiprocessing as mp, torch, time
import torch.multiprocessing as tmp
from PIL import Image
from collections import deque
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import torchvision.transforms as T
from torchvision.transforms.functional import normalize, rgb_to_grayscale
from torchvision.ops import sigmoid_focal_loss
from torch.amp import autocast, GradScaler
import grpc
from concurrent.futures import ThreadPoolExecutor
import inference_pb2, inference_pb2_grpc
from model import MarioGrayscaleMobileNetTSM

# ── multiprocess -------------------------------------------------
mp.set_start_method("spawn", force=True)
tmp.set_sharing_strategy("file_descriptor")
CTX = mp.get_context("spawn")

# ── constants ----------------------------------------------------
DATA_ROOT="/home/cryptor/gameAI/dataset/smbdataset/data-smb"
CKPT_DIR="./checkpoints"
BATCH=128; LR=1e-3; FRAMES=4
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
PORT=50051
CUTMIX_P=0.7; CUTMIX_BETA=1.0
VAL_GAIN=0.995; LOG_INT=100

# ── frame buffer -------------------------------------------------
def to_tensor(buf:bytes):
    a=np.frombuffer(buf,np.uint8); h=a.size//(256*3)
    img=a.reshape(h,256,3); img=cv2.copyMakeBorder(img,8,8,0,0,cv2.BORDER_CONSTANT)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    t=torch.from_numpy(img).unsqueeze(0).float()/255.
    return normalize(t,[0.5],[0.5])

class FrameBuf:
    def __init__(s,k): s.k=k; s.d={}
    def cat(s,peer,buf):
        x=to_tensor(buf).to(DEVICE).unsqueeze(0)
        dq=s.d.setdefault(peer,deque([x]*s.k,maxlen=s.k)); dq.append(x)
        return torch.cat(list(dq),1)

BUF=FrameBuf(FRAMES)

# ── transforms ---------------------------------------------------
class VCompose:  # list-aware
    def __init__(s,ts): s.ts=ts
    def __call__(s,ims):
        for t in s.ts: ims=t(ims)
        return ims
class Pad256:
    def __call__(s,ims):
        r=[];   # vertical center
        for im in ims:
            if im.height==256: r.append(im)
            else:
                tb=(256-im.height)//2
                r.append(T.functional.pad(im,(0,tb,0,256-im.height-tb),0))
        return r
class Gray:    # RGB→1ch
    def __call__(s,ims): return [rgb_to_grayscale(im,1) for im in ims]
class Bright:
    def __call__(s,ims):
        g=np.random.uniform(0.9,1.1)
        return [T.functional.adjust_brightness(im,g) for im in ims]
class Erase:
    def __init__(s,p=0.05,sc=(0.02,0.2)): s.p,s.sc=p,sc
    def __call__(s,ims):
        if np.random.rand()>s.p: return ims
        ref=T.ToTensor()(ims[0])
        i,j,h,w,v=T.RandomErasing.get_params(ref,scale=s.sc,ratio=(0.3,3.3),value=(0,))
        out=[]
        for im in ims:
            t=T.ToTensor()(im)
            t=T.functional.erase(t,i,j,h,w,v)
            out.append(T.ToPILImage()(t))
        return out

VIDEO_TFM=VCompose([Pad256(),Gray(),Bright(),Erase()])

# ── dataset ------------------------------------------------------
class SMBset(Dataset):
    def __init__(s,root,tfm,frames=4):
        s.tfm,s.frames=tfm,frames; s.samples=[]
        paths=sorted(glob.glob(root+"/**/*.png", recursive=True))
        seq={}
        for p in paths: seq.setdefault(os.path.dirname(p),[]).append(p)
        for pths in seq.values():
            pths.sort(); labs=[]
            import re
            for p in pths:
                m=re.search(r"_a(\d+)_",os.path.basename(p))
                labs.append([(int(m.group(1))>>i)&1 for i in reversed(range(8))] if m else None)
            val=[(p,l) for p,l in zip(pths,labs) if l]
            if len(val)<frames: continue
            for i in range(frames-1,len(val)):
                lab=val[i][1]
                if sum(lab)==0:        # ★ raw=0 → 全ビット0 を除外
                    continue
                window=[val[j][0] for j in range(i-frames+1,i+1)]
                s.samples.append((window,lab))
        lb=torch.tensor([l for _,l in s.samples],dtype=torch.float32)
        pos,neg=lb.sum(0),lb.size(0)-lb.sum(0)
        s.pos_weight=torch.clamp(neg/pos,max=50.)
    def __len__(s): return len(s.samples)
    def __getitem__(s,i):
        pths,lab=s.samples[i]
        ims=[Image.open(p).convert("RGB") for p in pths]
        ims=s.tfm(ims)
        tens=[T.Normalize([0.5],[0.5])(T.ToTensor()(im)) for im in ims]
        return torch.cat(tens,0), torch.tensor(lab,dtype=torch.float32)

# ── CutMix -------------------------------------------------------
def cutmix(x,y,beta=CUTMIX_BETA):
    lam=np.random.beta(beta,beta); bs=len(x); idx=torch.randperm(bs)
    H,W=x.size(2),x.size(3)
    rw,rh=int(W*np.sqrt(1-lam)),int(H*np.sqrt(1-lam))
    rx,ry=np.random.randint(W),np.random.randint(H)
    x1,y1=max(rx-rw//2,0),max(ry-rh//2,0)
    x2,y2=min(rx+rw//2,W),min(ry+rh//2,H)
    x[:,:,y1:y2,x1:x2]=x[idx,:,y1:y2,x1:x2]
    lam=1-((x2-x1)*(y2-y1)/(W*H))
    return x,y,y[idx],lam

# ── inference service -------------------------------------------
infer_model=MarioGrayscaleMobileNetTSM(FRAMES).to(DEVICE)
infer_lock=threading.Lock()

class Infer(inference_pb2_grpc.InferenceServicer):
    def Predict(s,req,ctx):
        with infer_lock:
            x=BUF.cat(ctx.peer(),req.frame)
            with torch.no_grad(),autocast(device_type="cuda"):
                bits=(infer_model(x).sigmoid()>0.5).int()[0].tolist()
        a,up,lf,b,st,ri,dn,se=bits
        return inference_pb2.InferenceResponse(action=[b,0,se,st,up,dn,lf,ri,a])

# ── DataLoader helper -------------------------------------------
def loader(ds,samp=None):
    return DataLoader(ds,BATCH,shuffle=(samp is None),sampler=samp,
                      num_workers=1,prefetch_factor=2,multiprocessing_context=CTX)

# ── train loop ---------------------------------------------------
def train(base):
    ds=SMBset(DATA_ROOT,VIDEO_TFM,FRAMES)
    N=len(ds); tr_len=int(N*0.7); va_len=int(N*0.2); te_len=N-tr_len-va_len
    tr_ds,va_ds,te_ds=random_split(ds,[tr_len,va_len,te_len])
    w=[(torch.tensor(l)*ds.pos_weight).sum().item() for _,l in ds.samples]
    samp=WeightedRandomSampler([w[i] for i in tr_ds.indices],len(tr_ds),True)
    tr_ld,va_ld=loader(tr_ds,samp),loader(va_ds)

    opt=torch.optim.AdamW(base.parameters(),lr=LR,weight_decay=5e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=30)
    scaler=GradScaler()
    ema=torch.optim.swa_utils.AveragedModel(base)
    crit=lambda p,t:sigmoid_focal_loss(p,t,gamma=2.0,alpha=0.25,reduction="mean")

    best=float("inf"); ep=0
    while True:
        ep+=1; base.train()
        run_loss=run_acc=0; t0=time.time()
        for bi,(x,y) in enumerate(tr_ld,1):
            x,y=x.to(DEVICE),y.to(DEVICE)
            use_cm=np.random.rand()<CUTMIX_P
            if use_cm: x,ya,yb,lam=cutmix(x,y)
            with autocast(device_type="cuda"):
                o=base(x)
                loss=(lam*crit(o,ya)+(1-lam)*crit(o,yb)) if use_cm else crit(o,y)
                tgt=ya*lam+yb*(1-lam) if use_cm else y
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad()
            run_loss+=loss.item()*x.size(0)
            pred=(o.sigmoid()>0.5).int()
            run_acc+=(pred==tgt.int()).all(dim=1).float().sum().item()
            ema.update_parameters(base)

            if bi%LOG_INT==0:
                print(f"  Ep{ep} B{bi}/{len(tr_ld)}  "
                      f"loss {run_loss/(bi*BATCH):.4f}  "
                      f"acc {run_acc/(bi*BATCH):.3f}")

        # ---- epoch end ----
        torch.optim.swa_utils.update_bn(tr_ld,ema.module,device=DEVICE)
        ema.module.eval(); val_loss=val_acc=0
        with torch.no_grad(),autocast(device_type="cuda"):
            for xv,yv in va_ld:
                xv,yv=xv.to(DEVICE),yv.to(DEVICE)
                out=ema.module(xv)
                val_loss+=crit(out,yv).item()*xv.size(0)
                val_acc+=( (out.sigmoid()>0.5).int()==yv.int()).all(dim=1).float().sum().item()
        val_loss/=va_len; val_acc/=va_len

        if val_loss<best*VAL_GAIN:
            best=val_loss
            with infer_lock:
                infer_model.load_state_dict(ema.module.state_dict(),strict=False)

        sch.step()
        print(f"[Ep{ep}] train_loss {run_loss/tr_len:.4f} "
              f"train_acc {run_acc/tr_len:.3f} "
              f"val_loss {val_loss:.4f} val_acc {val_acc:.3f} "
              f"best {best:.4f}  time {time.time()-t0:.1f}s")
        os.makedirs(CKPT_DIR,exist_ok=True)
        torch.save(ema.module.state_dict(),f"{CKPT_DIR}/model_ep{ep}.pth")

# ── main ---------------------------------------------------------
if __name__=="__main__":
    base=MarioGrayscaleMobileNetTSM(FRAMES).to(DEVICE)
    print("[Init] training from scratch (gray 1-ch, no-action removed)")
    threading.Thread(target=train,args=(base,),daemon=True).start()

    srv=grpc.server(ThreadPoolExecutor(max_workers=4))
    inference_pb2_grpc.add_InferenceServicer_to_server(Infer(),srv)
    srv.add_insecure_port(f"0.0.0.0:{PORT}")
    srv.start(); print(f"gRPC running on {PORT}")
    srv.wait_for_termination()
