"""Revision protocol-v1: pad on s12cr (see configs/revision/baselines.py)."""
from configs.revision.baselines import build

config = build("s12cr", "pad")
