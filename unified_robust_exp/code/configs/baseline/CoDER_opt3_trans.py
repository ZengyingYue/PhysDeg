"""CoDER Plan B (Opt3) + the author's transductive MMR statistics.

Identical to `CoDER_opt3.py` except `model.params.data` — the loader used by
`conduct_mask_matrix()` at `begin_mmr_iter` to compute the covariance /
whitening-mask statistics — points at the THREE TEST CITIES instead of the
seven train cities.  This replicates what the author's released run actually
did (author `config.pkl` `model.params.data` = NewYork/Hakodate/Callao
SAR_512/RGB_512), i.e. the whitening directions are adapted to the test
domain (transductive).  Use for parity-with-author experiments only; flag it
clearly when reporting.
"""
from configs.baseline.CoDER_opt3 import config as _base
from configs.base.EarthMiss import data as _data

_mmr_transductive = dict(
    type='EarthM3DALoader',
    params=dict(
        image_dir=_data['test']['params']['image_dir'],
        mask_dir=_data['test']['params']['mask_dir'],
        sensors=('SAR', 'RGB'),
        transforms=_data['mmr']['params']['transforms'],
        CV=dict(k=10, i=-1),
        distributed=False,
        training=False,
        batch_size=4,
        num_workers=4,
    ),
)

# Shallow-copy only along the mutated path so the imported base config module
# is left untouched.
config = dict(_base)
config['model'] = dict(_base['model'])
config['model']['params'] = dict(_base['model']['params'])
config['model']['params']['data'] = _mmr_transductive
