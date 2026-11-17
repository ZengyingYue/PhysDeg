"""Per-class IoU under fuse / SAR-only / RGB-only on OEM-SAR heldout.

Operationalises the *recoverable boundary* (contribution 1): classes where
SAR-only ~ RGB-only are SHARED (structural/geometric -> SAR-recoverable); classes
where SAR-only << RGB-only are OPTICAL-SPECIFIC (spectral -> unrecoverable from
SAR). The per-class (RGB-only - SAR-only) gap is the magnitude of optical-specific
content the model must NOT hallucinate but route to uncertainty.
"""
import numpy as np, torch
from ever.core.config import import_config
from ever.core.builder import make_model
from module.baseline.CoDER import CoDER  # noqa: F401
from torch.utils.data import DataLoader
from data.EarthMiss import EarthM3

K = 8
# OpenEarthMap canonical 8-class order (label 1..8 -> idx 0..7). VERIFY against
# cliffbb/OpenEarthMap-SAR before using names in the paper.
NAMES = ['bareland', 'rangeland', 'developed', 'road', 'tree', 'water', 'agri', 'building']

cfg = import_config('baseline.CoDER_oem')
tp = cfg['data']['test']['params']
ds = EarthM3(tp['image_dir'], tp['mask_dir'], tp['transforms'], tp['sensors'])
loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=8)
model = make_model(cfg['model']).cuda()
ck = torch.load('./log/OEM/CoDER_oem/Best.pth', map_location='cpu')
model.load_state_dict(ck['model'] if 'model' in ck else ck, strict=False)
model.eval()


def per_class(mode):
    model.infer_mode = mode
    cm = np.zeros((K, K), dtype=np.int64)
    with torch.no_grad():
        for ms, y in loader:
            pred = model(ms.cuda(), y).argmax(1).cpu().numpy()
            gt = y['cls'].numpy(); m = gt != -1
            cm += np.bincount(gt[m].astype(np.int64) * K + pred[m].astype(np.int64),
                              minlength=K * K).reshape(K, K)
    inter = np.diag(cm).astype(float)
    union = cm.sum(0) + cm.sum(1) - inter
    return inter / np.maximum(union, 1)


f, s, r = per_class('fuse'), per_class('sar'), per_class('rgb')
print(">>> class        fuse   SAR    RGB    RGB-SAR(=optical-specific)")
for i, n in enumerate(NAMES):
    print(">>> %-11s %.3f  %.3f  %.3f   %+.3f" % (n, f[i], s[i], r[i], r[i] - s[i]))
print(">>> mean        %.3f  %.3f  %.3f" % (f.mean(), s.mean(), r.mean()))
