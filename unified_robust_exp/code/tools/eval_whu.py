"""Self-contained WHU eval (7-class) for a CoDER checkpoint, both fuse & sar modes.

Offline-eval note: under missing_readout='head' (default), fuse and sar outputs
are softmax(logits) and do NOT depend on the cov/β machinery (which only feeds
last_uncertainty), so mIoU is reproduced exactly even though cov_matrix_layer
buffers are not in the checkpoint.
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from ever.core.config import import_config
from ever.core.builder import make_model
from module.baseline.CoDER import CoDER  # noqa: F401 — registers 'CoDER'
from data.WHU import WhuM3, WHU_CLASSES, WHU_NUM_CLASSES


def confusion(pred, gt, n, ignore=-1):
    m = gt != ignore
    p, g = pred[m].astype(np.int64), gt[m].astype(np.int64)
    return np.bincount(g * n + p, minlength=n * n).reshape(n, n)


@torch.no_grad()
def run_mode(model, loader, mode, device):
    model.eval()
    model.infer_mode = mode
    cm = np.zeros((WHU_NUM_CLASSES, WHU_NUM_CLASSES), dtype=np.int64)
    for ms, y in loader:
        ms = ms.to(device)
        prob = model(ms, y)
        pred = prob.argmax(1).cpu().numpy()
        cm += confusion(pred, y["cls"].numpy(), WHU_NUM_CLASSES)
    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    iou = inter / np.maximum(union, 1)
    oa = inter.sum() / np.maximum(cm.sum(), 1)
    return iou, float(iou.mean()), float(oa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="baseline.CoDER_whu")
    ap.add_argument("--modes", default="fuse,sar")
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()

    cfg = import_config(args.config)
    tp = cfg["data"]["test"]["params"]
    ds = WhuM3(tp["root"], "test", tp["patch"], tp["seed"], tp["train_frac"],
               tp["transforms"], cache_ram=True)
    loader = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=8, pin_memory=True)

    model = make_model(cfg["model"]).cuda()
    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck["model"] if "model" in ck else ck
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"loaded {args.ckpt}: missing={len(miss)} unexpected={len(unexp)}")

    for mode in args.modes.split(","):
        iou, miou, oa = run_mode(model, loader, mode, "cuda")
        print(f"\n=== mode={mode}  mIoU={miou:.4f}  OA={oa:.4f} ===")
        for name, v in zip(WHU_CLASSES, iou):
            print(f"  {name:10s} {v:.4f}")


if __name__ == "__main__":
    main()
