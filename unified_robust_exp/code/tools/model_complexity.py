"""Parameters and forward FLOPs of the compared models (one 512x512 or 256x256 input).

    python tools/model_complexity.py --out ../../revision_tgrs/results_analysis/complexity.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ever.core.builder import make_model  # noqa: E402
from ever.core.config import import_config  # noqa: E402
from module.baseline.MetaRS import MetaRS  # noqa: E402,F401 - registration
import module.baseline.revision_baselines  # noqa: E402,F401 - registration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", nargs="+", default=[
        "whu_moddrop", "whu_pad", "whu_shaspec", "whu_metars", "s12cr_moddrop", "s12cr_moddrop_relauxgp25uema"])
    args = ap.parse_args()
    torch.set_num_threads(4)
    out = {}
    for name in args.configs:
        config = import_config(f"revision.{name}")
        model = make_model(config["model"]).cuda().eval()
        patch = 256 if name.startswith("s12cr") else 512
        sar = int(config["model"]["params"].get("sar_in_channels", 1))
        x = torch.zeros(1, patch, patch, sar + 3).permute(0, 3, 1, 2).contiguous().cuda()
        y = dict(cls=torch.zeros(1, patch, patch, dtype=torch.long).cuda())
        with torch.inference_mode(), FlopCounterMode(display=False) as counter:
            model(x, y)
        out[name] = dict(params_m=sum(p.numel() for p in model.parameters()) / 1e6,
                         gflops=counter.get_total_flops() / 1e9, input=[1, sar + 3, patch, patch],
                         note="FLOPs counted by torch.utils.flop_counter (one multiply-add = 2 FLOPs)")
        print(name, out[name], flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
