"""Per-image confusion matrices, the input of the paired bootstrap (paired_bootstrap.py).

SEN12MS-CR: label confusion of the clear, cloudy and SAR-only predictions and the in-cloud
consistency confusion, with the real cloud cover of every image.
WHU / EarthMiss: label confusion for clean, missing optical, missing SAR and the physical
optical conditions at severities 3-5, with the seeds of evaluate.py.

--split test reads the held-out split and is refused until revision_tgrs/TEST_FROZEN exists.

    python experiments/revision/dump_perimage.py --config revision.s12cr_moddrop_gp25 \
        --run-dir ../runs/revision_v1/s12cr_moddrop_gp25/seed_2333 --seed 2333 --split val
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[2]
REPO = CODE_ROOT.parents[1]
sys.path.insert(0, str(CODE_ROOT))

from ever.core.config import import_config  # noqa: E402
from experiments.revision.eval_realcloud import check_dataset_tag  # noqa: E402
from experiments.revision.evaluate import Corruptor, load_model, loader, sha256  # noqa: E402


def image_confusion(ref, pred, k, valid=None):
    """[B,k,k] confusion per image; pixels with valid == False are skipped."""
    b = ref.shape[0]
    idx = (torch.arange(b, device=ref.device)[:, None, None] * k + ref) * k + pred
    idx = idx[valid] if valid is not None else idx.flatten()
    return torch.bincount(idx.flatten(), minlength=b * k * k).reshape(b, k, k).cpu().numpy().astype(np.int32)


def real_cloud(model, config, k, split):
    keys = ("heldout_test", "heldout_cloudy") if split == "test" else ("val_clear", "val_cloudy")
    clear_dl, cloudy_dl = loader(config, keys[0], 8), loader(config, keys[1], 8)
    ds = clear_dl.dataset
    out = {n: [] for n in ("clear", "cloudy", "sar", "in_cloud")}
    cover, names, index = [], [], 0
    for (xc, tc), (xk, _) in zip(clear_dl, cloudy_dl):
        b = xc.shape[0]
        y = tc["cls"].cuda().long()
        valid = y != -1
        y = y.clamp_min(0)
        cloud = torch.from_numpy(np.stack([ds.cloud_mask(index + j) for j in range(b)])).cuda().bool()
        names += [ds.ids[index + j] for j in range(b)]
        index += b
        model.infer_mode = "fuse"
        p_clear, p_cloudy = model(xc.cuda(), tc).argmax(1), model(xk.cuda(), tc).argmax(1)
        model.infer_mode = "sar"
        p_sar = model(xk.cuda(), tc).argmax(1)
        out["clear"].append(image_confusion(y, p_clear, k, valid))
        out["cloudy"].append(image_confusion(y, p_cloudy, k, valid))
        out["sar"].append(image_confusion(y, p_sar, k, valid))
        out["in_cloud"].append(image_confusion(p_clear, p_cloudy, k, cloud))
        cover.append(cloud.flatten(1).float().mean(1).cpu().numpy())
    return {n: np.concatenate(v) for n, v in out.items()}, np.concatenate(cover), names


CONDITIONS = [("clean", "fuse", "none", 0), ("missing_optical", "sar", "none", 0), ("missing_sar", "rgb", "none", 0)] + [
    (f"{kind}@{s}", "fuse", kind, s) for kind in ("phys_haze", "phys_cloudshadow") for s in (3, 4, 5)]


def suite(model, config, k, split):
    dl = loader(config, "heldout_test" if split == "test" else "test", 8)
    corrupt = Corruptor(config)
    out, names = {}, []
    for c, (name, mode, kind, severity) in enumerate(CONDITIONS):
        model.infer_mode = mode
        rows = []
        for batch_index, (x, target) in enumerate(dl):
            generator = torch.Generator().manual_seed(10_000 + batch_index)
            x = corrupt(x.cuda(), kind, severity, generator)
            y = target["cls"].cuda().long()
            valid = (y >= 0) & (y < k)
            rows.append(image_confusion(y.clamp(0, k - 1), model(x, target).argmax(1), k, valid))
            if c == 0:
                names += list(target["fname"])
        out[name] = np.concatenate(rows)
    return out, None, names


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--dataset-tag", default="")
    args = ap.parse_args()
    check_dataset_tag(args.dataset_tag)
    if args.split == "test" and not (REPO / "revision_tgrs" / "TEST_FROZEN").exists():
        raise SystemExit("test split refused: revision_tgrs/TEST_FROZEN does not exist")
    name = args.config.split(".")[-1]
    config = import_config(args.config)
    k = int(config["model"]["params"]["num_classes"])
    run = Path(args.run_dir)
    ckpt = run / args.checkpoint if args.checkpoint else \
        run / "Final.pth" if (run / "Final.pth").exists() else run / "Best.pth"
    model = load_model(config, ckpt)
    conf, cover, names = (real_cloud if name.startswith("s12cr") else suite)(model, config, k, args.split)
    folder = f"perimg_{args.split}" + (f"_{args.dataset_tag}" if args.dataset_tag else "")
    out = REPO / "revision_tgrs" / "results_analysis" / folder / name / f"seed_{args.seed}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    extra = {} if cover is None else {"cover": cover}
    np.savez_compressed(out, names=np.array(names), checkpoint_sha256=sha256(ckpt),
                        evaluator_sha256=sha256(Path(__file__)), **extra, **{f"cm_{n}": v for n, v in conf.items()})
    print(out, len(names), flush=True)


if __name__ == "__main__":
    main()
