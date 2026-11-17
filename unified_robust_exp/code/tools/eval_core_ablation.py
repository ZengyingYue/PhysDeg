"""Evaluate trained ReBound core-ablation checkpoints into JSON and CSV.

The evaluator scores the deployed prediction for clean fusion, both missing
modalities, and representative severe RGB corruptions.  Accuracy, calibration,
selective ranking, mean evidential uncertainty, and effective beta values are
reported from one common implementation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from ever.core.builder import make_model
from ever.core.config import import_config

from module.baseline.CoDER import CoDER  # noqa: F401
from tools.eval_calib import Calib
from tools.eval_robust import build_test_loader, degrade_rgb


CONDITIONS = (
    ("clean_fuse", "fuse", "clean", 0),
    ("rgb_missing", "sar", "clean", 0),
    ("sar_missing", "rgb", "clean", 0),
    ("blur5_fuse", "fuse", "blur", 5),
    ("noise5_fuse", "fuse", "noise", 5),
    ("haze5_fuse", "fuse", "haze", 5),
)

# Noise must be identical for every checkpoint.  Seeding only once at process
# startup is insufficient because each model consumes the global RNG stream at
# a different point.  Resetting a dedicated CPU RNG scope for every batch makes
# the corruption a function of (condition, batch index), not evaluation order.
CORRUPTION_REALIZATION_ID = "whu_rgb_noise_s5_seed42005_per_batch_v1"
CORRUPTION_SEED = 42005


def confusion(pred, gt, classes, ignore=-1):
    valid = gt != ignore
    p = pred[valid].astype(np.int64)
    g = gt[valid].astype(np.int64)
    return np.bincount(g * classes + p, minlength=classes * classes).reshape(
        classes, classes
    )


@torch.no_grad()
def evaluate_condition(model, loader, classes, mode, kind, severity, device):
    model.eval()
    model.infer_mode = mode
    # The deployed label comes from ``prob``.  Score both its conventional
    # max-softmax confidence and the paper's reliability confidence (1-u).
    soft_metric = Calib()
    reliability_metric = Calib()
    cm = np.zeros((classes, classes), dtype=np.int64)
    uncertainty_sum = 0.0
    valid_count = 0
    nll_sum = 0.0
    brier_sum = 0.0
    reliable_wrong = 0
    beta_obs = []
    beta_rec = []
    for batch_index, (multispectral, target) in enumerate(loader):
        if kind == "noise":
            # Degradation runs on CPU.  fork_rng prevents corruption sampling
            # from perturbing any RNG state used by the evaluated model.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(CORRUPTION_SEED + batch_index)
                multispectral = degrade_rgb(multispectral, kind, severity)
        else:
            multispectral = degrade_rgb(multispectral, kind, severity)
        multispectral = multispectral.to(device)
        gt = target["cls"].to(device)
        prob = model(multispectral, target)
        pred = prob.argmax(1)
        soft_conf = prob.max(1).values
        valid = gt != -1
        correct = (pred == gt)[valid].float()
        soft_metric.update(
            soft_conf[valid].float().cpu().numpy(), correct.cpu().numpy()
        )
        uncertainty = model.last_uncertainty
        reliability_conf = (1.0 - uncertainty).clamp(0.0, 1.0)
        reliability_metric.update(
            reliability_conf[valid].float().cpu().numpy(), correct.cpu().numpy()
        )
        cm += confusion(pred.cpu().numpy(), gt.cpu().numpy(), classes)
        uncertainty_sum += float(uncertainty[valid].sum())
        n_valid = int(valid.sum())
        valid_count += n_valid
        true_prob = prob.permute(0, 2, 3, 1)[valid]
        true_label = gt[valid].long()
        nll_sum += float(-torch.log(
            true_prob.gather(1, true_label[:, None]).clamp_min(1e-8)
        ).sum())
        one_hot = torch.nn.functional.one_hot(true_label, classes).float()
        brier_sum += float(((true_prob - one_hot) ** 2).sum(1).sum())
        reliable_wrong += int(((reliability_conf[valid] >= 0.7) & (correct == 0)).sum())
        observed = model.last_beta_obs
        recovered = model.last_beta_rec
        if observed is None:
            observed = model._compute_beta(recovered=False)
        if recovered is None:
            recovered = model._compute_beta(recovered=True)
        beta_obs.append(float(observed.detach().mean()))
        beta_rec.append(float(recovered.detach().mean()))

    intersection = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - intersection
    iou = intersection / np.maximum(union, 1)
    oa = intersection.sum() / max(cm.sum(), 1)
    return {
        "miou": float(iou.mean()),
        "oa": float(oa),
        "per_class_iou": iou.tolist(),
        "ece": reliability_metric.ece(),
        "aurc_x1000": reliability_metric.aurc() * 1000.0,
        "sel_acc_80": reliability_metric.sel(0.8),
        "softmax_ece": soft_metric.ece(),
        "softmax_aurc_x1000": soft_metric.aurc() * 1000.0,
        "softmax_sel_acc_80": soft_metric.sel(0.8),
        "nll": nll_sum / max(valid_count, 1),
        "brier": brier_sum / max(valid_count, 1),
        "reliable_wrong_rate_07": reliable_wrong / max(valid_count, 1),
        "mean_uncertainty": uncertainty_sum / max(valid_count, 1),
        "mean_beta_obs": float(np.mean(beta_obs)),
        "mean_beta_rec": float(np.mean(beta_rec)),
    }


def parse_entry(raw):
    parts = raw.split("=", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("entry must be TAG=CONFIG,CKPT")
    tag, rest = parts
    try:
        config, checkpoint = rest.split(",", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("entry must be TAG=CONFIG,CKPT") from exc
    return tag, config, checkpoint


def write_csv(path, results):
    scalar_keys = [
        "miou", "oa", "ece", "aurc_x1000", "sel_acc_80",
        "softmax_ece", "softmax_aurc_x1000", "softmax_sel_acc_80",
        "nll", "brier", "reliable_wrong_rate_07",
        "mean_uncertainty", "mean_beta_obs", "mean_beta_rec",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["variant", "condition", *scalar_keys]
        )
        writer.writeheader()
        for variant, conditions in results.items():
            for condition, metrics in conditions.items():
                writer.writerow({
                    "variant": variant,
                    "condition": condition,
                    **{key: metrics[key] for key in scalar_keys},
                })


def write_results(base, results):
    """Atomically persist all completed conditions after every evaluation."""
    json_path = base.with_suffix(".json")
    json_tmp = json_path.with_suffix(".json.tmp")
    json_tmp.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    json_tmp.replace(json_path)

    csv_path = base.with_suffix(".csv")
    csv_tmp = csv_path.with_suffix(".csv.tmp")
    write_csv(csv_tmp, results)
    csv_tmp.replace(csv_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entry", action="append", type=parse_entry, required=True,
        help="repeatable TAG=CONFIG,CKPT",
    )
    parser.add_argument("--out", required=True, help="output path without suffix")
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--model-seed", type=int, default=2333)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)
    selected = CONDITIONS[:3] if args.quick else CONDITIONS
    print(
        "paired corruption protocol: "
        f"{CORRUPTION_REALIZATION_ID}",
        flush=True,
    )
    base = Path(args.out)
    base.parent.mkdir(parents=True, exist_ok=True)
    protocol_path = base.with_name(f"{base.name}_protocol").with_suffix(".json")
    protocol_tmp = protocol_path.with_suffix(".json.tmp")
    protocol_tmp.write_text(json.dumps({
        "checkpoint_selection_seed": args.model_seed,
        "evaluation_seed": 42,
        "corruption_realization_id": CORRUPTION_REALIZATION_ID,
        "corruption_seed": CORRUPTION_SEED,
        "batch_size": args.bs,
        "conditions": [
            {
                "condition": condition,
                "mode": mode,
                "corruption": kind,
                "severity": severity,
            }
            for condition, mode, kind, severity in selected
        ],
        "entries": [
            {"variant": tag, "config": config, "checkpoint": checkpoint}
            for tag, config, checkpoint in args.entry
        ],
    }, indent=2, sort_keys=True) + "\n")
    protocol_tmp.replace(protocol_path)

    results = {}
    for tag, config_name, checkpoint in args.entry:
        cfg = import_config(config_name)
        classes = int(cfg["model"]["params"]["num_classes"])
        loader = build_test_loader(cfg, args.bs)
        model = make_model(cfg["model"]).cuda()
        payload = torch.load(checkpoint, map_location="cpu")
        state = payload["model"] if "model" in payload else payload
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"loaded {tag}: missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
        results[tag] = {}
        for condition, mode, kind, severity in selected:
            metrics = evaluate_condition(
                model, loader, classes, mode, kind, severity, "cuda"
            )
            results[tag][condition] = metrics
            write_results(base, results)
            print(
                f"[{tag:14s}] {condition:14s} mIoU={metrics['miou']:.4f} "
                f"relECE={metrics['ece']:.4f} "
                f"relAURC={metrics['aurc_x1000']:.1f} "
                f"softECE={metrics['softmax_ece']:.4f} "
                f"beta={metrics['mean_beta_obs']:.3f}/"
                f"{metrics['mean_beta_rec']:.3f}",
                flush=True,
            )
        del model
        torch.cuda.empty_cache()

    write_results(base, results)
    print(f"wrote {base.with_suffix('.json')} and {base.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
