"""Revision protocol-v1: em_moddrop_gp25 as trained before 2026-11-08, with the wrong 5 m GSD
for the cloud-shadow offset (EarthMiss is 0.6 m).  Kept to evaluate / report those runs."""
from configs.revision.baselines import build

config = build("em", "moddrop_gp25")
config["model"]["params"]["degrade"]["gsd"] = 5.0
