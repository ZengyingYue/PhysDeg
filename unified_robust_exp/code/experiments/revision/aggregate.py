"""Summarise revision_tgrs/results/**/seed_*.json into revision_tgrs/LEDGER.md.

Numbers are only ever read from evaluator outputs.  mean ± sample std is
printed only when all three seeds exist; otherwise the individual seed values
are listed with n.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
RESULTS = REPO / "revision_tgrs" / "results"
LEDGER = REPO / "revision_tgrs" / "LEDGER.md"
SEEDS = (2333, 42, 1234)


def load() -> dict:
    runs = {}
    for path in sorted(RESULTS.glob("*/seed_*.json")):
        if not path.with_suffix(".done").exists():
            continue  # evaluation still in progress (or interrupted)
        runs.setdefault(path.parent.name, {})[int(path.stem.split("_")[1])] = json.loads(path.read_text())
    return runs


def cell(values: list[float], scale: float = 100.0, digits: int = 2) -> str:
    if not values:
        return "—"
    v = [x * scale for x in values]
    if len(v) == len(SEEDS):
        return f"{statistics.mean(v):.{digits}f} ± {statistics.stdev(v):.{digits}f}"
    return " / ".join(f"{x:.{digits}f}" for x in v) + f" (n={len(v)})"


def metric(runs: dict, name: str, condition: str, getter) -> list[float]:
    out = []
    for seed in SEEDS:
        record = runs.get(name, {}).get(seed)
        if record and condition in record["conditions"]:
            out.append(getter(record["conditions"][condition]))
    return out


def main() -> None:
    runs = load()
    lines = ["# Revision results ledger (auto-generated — do not edit by hand)", "",
             "Source: `revision_tgrs/results/<config>/seed_<s>.json` written by "
             "`experiments/revision/evaluate.py` on the protocol-v1 test split. "
             "mIoU in %, mean ± sample std over seeds 2333/42/1234 only when all three exist.", ""]
    lines += ["## Clean / missing-modality", "",
              "| config | clean | missing optical | missing SAR | mean missing | seeds |",
              "|---|---|---|---|---|---|"]
    for name in sorted(runs):
        miou = lambda c: metric(runs, name, c, lambda r: r["miou"])
        mean_missing = [
            (a + b) / 2 for a, b in zip(miou("missing_optical"), miou("missing_sar"))]
        lines.append(f"| {name} | {cell(miou('clean'))} | {cell(miou('missing_optical'))} | "
                     f"{cell(miou('missing_sar'))} | {cell(mean_missing)} | "
                     f"{sorted(runs[name])} |")
    lines += ["", "## Deployed-probability reliability (MSP)", "",
              "| config | condition | ECE15 | AURC×10³ | error AUROC |", "|---|---|---|---|---|"]
    for name in sorted(runs):
        for cond in ("clean", "missing_optical", "missing_sar"):
            ece = metric(runs, name, cond, lambda r: r["ece15"])
            aurc = metric(runs, name, cond, lambda r: r["msp"]["aurc"])
            auroc = metric(runs, name, cond, lambda r: r["msp"]["error_auroc"])
            if ece:
                lines.append(f"| {name} | {cond} | {cell(ece, 1, 4)} | {cell(aurc, 1000, 1)} | "
                             f"{cell(auroc, 1, 3)} |")
    lines += ["", "## Corruption sweeps (mIoU; gated readout / always-fuse where applicable)", ""]
    for name in sorted(runs):
        conds = [c for c in next(iter(runs[name].values()))["conditions"] if "@" in c]
        if not conds:
            continue
        lines += [f"### {name}", "", "| condition | gated | always fuse | SAR-head route |",
                  "|---|---|---|---|"]
        for cond in conds:
            gated = metric(runs, name, cond, lambda r: r["miou"])
            fused = metric(runs, name, cond, lambda r: r.get("always_fuse", r)["miou"])
            route = metric(runs, name, cond,
                           lambda r: r.get("route_fraction", {}).get("sar_head", 0.0))
            lines.append(f"| {cond} | {cell(gated)} | {cell(fused)} | {cell(route, 100, 1)} |")
        lines.append("")
    LEDGER.write_text("\n".join(lines) + "\n")
    print(f"wrote {LEDGER} ({len(runs)} configs)")


if __name__ == "__main__":
    main()
