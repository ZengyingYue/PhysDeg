"""Revision protocol-v1: shaspec on whupad = official PAD image-level split of WHU-OPT-SAR (see configs/revision/baselines.py)."""
from configs.revision.baselines import build

config = build("whupad", "shaspec")
