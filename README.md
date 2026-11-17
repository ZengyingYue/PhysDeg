# PhysDeg: Cloud-Robust Optical–SAR Segmentation via Physically Grounded Degradation Training

Official experimental codebase, configuration files, spatial split definitions, and evaluation result files for **PhysDeg**.

---

## Overview

`PhysDeg` is a training-only recipe of physically grounded sensor degradations for dual-stream optical–SAR semantic segmentation with modality dropout:
- **Optical physics**: wavelength-dependent atmospheric scattering haze and fractal clouds with solar-geometry cast shadows, paired with generic corruptions;
- **SAR physics**: multiplicative Gamma speckle, radiometric calibration offset & incidence-angle range ramp, scalloped thermal noise floor (NESZ), and range-line radio-frequency interference (RFI);
- **Geometry**: sub-pixel optical–SAR co-registration shifts.

It requires **zero real cloudy training images**, leaves the baseline network, segmentation loss, and training schedule unchanged, and adds **zero parameters or inference cost** at deployment.

## Repository Structure

```text
.
├── unified_robust_exp/code/
│   ├── train.py                                # Main training entrypoint (ever framework, bf16)
│   ├── configs/revision/                       # Shared protocol (common.py, baselines.py) & all experiment configs
│   ├── data/                                   # Data loaders for SEN12MS-CR, WHU-OPT-SAR (PAD split), EarthMiss
│   ├── module/
│   │   ├── reliability/phys_degrade.py         # Physically grounded optical, SAR & co-registration simulator
│   │   ├── reliability/train_degrade.py        # Generic corruption training family
│   │   └── baseline/
│   │       ├── revision_baselines.py           # ModDrop, PAD, ShaSpec & physically supervised reliability head
│   │       ├── MetaRS.py                       # MetaRS adapted to the shared protocol
│   │       └── stars.py                        # STARS architecture + paper objective under shared protocol
│   ├── experiments/revision/
│   │   ├── evaluate.py                         # Clean / missing-modality / corruption / phys suite + ECE/AURC/AUROC
│   │   ├── eval_realcloud.py                   # Real Sentinel-2 cloud consistency & label evaluation
│   │   ├── eval_relmap.py                      # Pixel-level cloud localization AUROC for reliability head
│   │   ├── dump_perimage.py                    # Per-image confusion matrices for hierarchical bootstrap
│   │   └── paired_bootstrap.py                 # 10,000-resample paired hierarchical bootstrap
│   └── tools/                                  # Dataset preparation & model complexity utilities
└── revision_tgrs/
    ├── TEST_FROZEN                             # Protocol freeze record before the single test-split evaluation
    ├── splits/                                 # Spatially disjoint train/val/test split manifests
    ├── results/                                # Validation & test suite JSON outputs
    ├── results_realcloud/                      # Real-cloud evaluation JSON outputs (winter, spring, summer)
    ├── results_relmap/                         # Reliability-head cloud mask localization JSON outputs
    └── results_analysis/                       # Complexity, cloud-cover stats, fallback & bootstrap JSONs
```

## Datasets

All experiments use publicly available benchmarks with spatially disjoint splits:
1. **SEN12MS-CR** ([Ebel et al., IEEE TGRS 2021](https://doi.org/10.1109/TGRS.2020.3024744)) paired with **SEN12MS** MODIS land-cover labels and **ESA WorldCover 2020** (10 m): whole-ROI split (`19 / 3 / 5` ROIs in winter; spring and summer ROIs for cross-season validation).
2. **WHU-OPT-SAR** ([Li et al., JAG 2022](https://doi.org/10.1016/j.jag.2021.102638)): official source-image test split of **PAD** (`70 train / 10 val / 20 test` whole images; `4,900 / 700 / 1,400` patches of `512×512`).
3. **EarthMiss** ([Zhou et al., RSE 2026](https://doi.org/10.1016/j.rse.2025.115132)): official cross-city split (`7 train / 3 val / 3 test` cities, GSD = 0.6 m).

Set dataset paths via environment variables `S12CR_ROOT`, `WHU_ROOT`, and `EARTHMISS_ROOT` (see `unified_robust_exp/code/configs/revision/common.py`).

## Training & Evaluation

```bash
cd unified_robust_exp/code

# Train PhysDeg (gp25) on SEN12MS-CR, WHU-OPT-SAR (PAD split), or EarthMiss
python train.py --config_path revision.s12cr_moddrop_gp25 --model_dir ../../runs/s12cr_moddrop_gp25/seed_2333 --opt seed=2333
python train.py --config_path revision.whupad_moddrop_gp25 --model_dir ../../runs/whupad_moddrop_gp25/seed_2333 --opt seed=2333
python train.py --config_path revision.em_moddrop_gp25 --model_dir ../../runs/em_moddrop_gp25/seed_2333 --opt seed=2333

# Evaluate on held-out test splits
python experiments/revision/eval_realcloud.py --config revision.s12cr_moddrop_gp25 --ckpt ../../runs/s12cr_moddrop_gp25/seed_2333/Best.pth --split test
python experiments/revision/evaluate.py --config revision.whupad_moddrop_gp25 --ckpt ../../runs/whupad_moddrop_gp25/seed_2333/Best.pth --test-key heldout_test --suite phys
```

## License

Released under the [MIT License](LICENSE).
