"""Revision protocol-v1: MetaRS on whupad (official PAD image split; R3: same baselines on every dataset)."""
from configs.revision.baselines import build

config = build("whupad", "metars")
