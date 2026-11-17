"""CoDER_flow — HEAD-readout ablation arm (``missing_readout='head'``).

Identical to ``CoDER_flow.py`` (the dempster / headline arm) except the
missing-modality eval readout is the MetaRS-equivalent ``softmax(available
head)``; the σ² gate then only feeds the uncertainty map and does **not** drive
mIoU.  Two uses:

  1. the baseline cell of the ``{head,dempster} × {flow,det}`` 2×2 ablation;
  2. the faithful eval config for checkpoints that were *trained* with the head
     readout (e.g. the first 15k run's ``checkpoint-15000.pth``, produced before
     the #17 readout switch landed) — evaluating those with the dempster readout
     would be unfair since their α_dst path was only weakly EDL-supervised.
"""
import copy

from configs.baseline.CoDER_flow import config as _base

config = copy.deepcopy(_base)
config['model']['params']['missing_readout'] = 'head'
# the head-trained checkpoints never saw the dempster-fused CE term
config['model']['params']['loss']['lam_ce_dst_rec'] = 0.0
# seed-variance reliability (NO learned head) — required to load checkpoints
# trained before #22 (e.g. checkpoint-15000), and doubles as the seed-variance
# ablation arm.  Without this the model would build a reliability_head whose
# weights are absent from the old ckpt → load_state_dict failure.
config['model']['params']['reliability_source'] = 'seed'
config['model']['params']['loss']['lam_rel'] = 0.0
