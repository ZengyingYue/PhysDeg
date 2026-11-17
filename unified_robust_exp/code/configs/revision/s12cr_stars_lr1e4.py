"""STARS on s12cr with the authors' learning rate (CVEO/STARS train.py: AdamW lr 1e-4, wd 0.01).

Only the learning rate differs from revision.s12cr_stars (2e-4, the shared protocol); the poly
schedule, iterations, batch and losses are unchanged.  Pre-registered in PHYS_PLAN 9.s
(2026-11-10): seed 2333 of both settings is compared on the training-time validation selection
score (best_checkpoint.json, mean of fused / SAR-only / optical-only mIoU, the same score used to
select every model); the higher one is used for seeds 42 and 1234, ties go to 1e-4.
"""
import copy

from configs.revision.baselines import build

config = copy.deepcopy(build("s12cr", "stars"))
config["optimizer"]["params"]["lr"] = 1e-4
config["learning_rate"]["params"]["base_lr"] = 1e-4
