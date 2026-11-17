import argparse, numpy as np, torch
from ever.core.config import import_config
from ever.core.builder import make_model
from module.baseline.CoDER import CoDER  # noqa
from tools.eval_robust import build_test_loader, confusion
ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True)
ap.add_argument("--config",default="baseline.CoDER_whu_moar_ml_r1"); ap.add_argument("--bs",type=int,default=16)
a=ap.parse_args(); torch.manual_seed(42)
cfg=import_config(a.config); K=int(cfg["model"]["params"]["num_classes"])
loader=build_test_loader(cfg,a.bs); m=make_model(cfg["model"]).cuda()
ck=torch.load(a.ckpt,map_location="cpu"); m.load_state_dict(ck["model"] if "model" in ck else ck,strict=False)
m.eval(); m.infer_mode="fuse"; cm=np.zeros((K,K),np.int64)
with torch.no_grad():
    for ms,y in loader:
        p=m(ms.cuda(),y).argmax(1).cpu().numpy(); cm+=confusion(p,y["cls"].numpy(),K)
def stats(cm):
    inter=np.diag(cm).astype(float); union=cm.sum(0)+cm.sum(1)-inter
    iou=inter/np.maximum(union,1); oa=inter.sum()/max(cm.sum(),1)
    return iou, oa
names=["farmland","city","village","water","forest","road","others"]
iou,oa=stats(cm)
print("per-class GT pixels:",{names[i]:int(cm[i].sum()) for i in range(K)})
print(f"ALL-7:        mIoU={iou.mean()*100:.2f}  OA={oa*100:.2f}")
for ex in [6]:  # exclude 'others'
    keep=[i for i in range(K) if i!=ex]
    cmx=cm[np.ix_(keep,keep)]  # drop class entirely (rare in lit) 
    ioux,oax=stats(cmx)
    print(f"EXCL-{names[ex]} (drop row+col): mIoU={ioux.mean()*100:.2f}  OA={oax*100:.2f}")
    # OA excluding GT==ex pixels only (treat 'others' as ignore in denom)
    mask=np.ones(K,bool); 
    num=sum(cm[i,i] for i in range(K) if i!=ex); den=sum(cm[i].sum() for i in range(K) if i!=ex)
    print(f"OA(GT!=others)= {num/max(den,1)*100:.2f}")
