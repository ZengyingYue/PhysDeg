"""STARS on em (R3 priority iii: STARS retrained under matched settings on every dataset).

Identical to the shared protocol of every other baseline on em (512 px crops, batch 16 on one
GPU, same steps, loss, schedule, split and selection); architecture from the authors' release,
the paper's objective, lr 1e-4 as chosen on SEN12MS-CR validation (PHYS_PLAN 9.s).
Prepared 2026-11-11 for the rented 2x RTX PRO 6000 server; NOT queued until the author approves.
"""
import copy

from configs.revision.baselines import build

config = copy.deepcopy(build("em", "stars"))
config["optimizer"]["params"]["lr"] = 1e-4
config["learning_rate"]["params"]["base_lr"] = 1e-4
