import ever as er
from ever.core.builder import make_model, make_dataloader
import torch
import os
import random
from data.EarthMiss import COLOR_MAP
from tqdm import tqdm
from module.viz import VisualizeSegmm
from module.slide_test import slide_inference
from module.baseline.MetaRS import MetaRS  # legacy baseline, kept for ablation
from module.baseline.CoDER import CoDER  # noqa: F401 — side-effect: registers 'CoDER'
from module.evidential import expected_calibration_error
import logging
from ever.core.checkpoint import load_model_state_dict_from_ckpt
from ever.core.config import import_config
import numpy as np
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from datetime import datetime



def seed_torch(seed=2333):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
parser = argparse.ArgumentParser(description='Eval methods')
parser.add_argument('--ckpt_path',  type=str,
                    help='ckpt path', default='./log/deeplabv3p.pth')
parser.add_argument('--config_path',  type=str,
                    help='config path', default='baseline.deeplabv3p')
parser.add_argument('--vis_path',  type=str,
                    help='config path', default='baseline.deeplabv3p')
parser.add_argument('--modality',  type=str,
                    help='modality', default='SAR')
parser.add_argument('--tta', action='store_true',
                    help='enable horizontal+vertical flip TTA')
parser.add_argument('--ms_tta', type=str, default='',
                    help='multi-scale TTA: comma-separated scales, e.g. "0.75,1.0,1.25"')
parser.add_argument('--save_uncertainty', action='store_true',
                    help='dump CoDER per-pixel uncertainty u = K/S as 8-bit heatmaps next to vis')
parser.add_argument('--ece_bins', type=int, default=15,
                    help='number of bins for Expected Calibration Error (CoDER only)')
parser.add_argument('--mmr_stats', type=str, default='none',
                    choices=['none', 'train', 'test'],
                    help='recompute the MMR covariance/mask statistics at eval time '
                         'on the train cities ("train" = fair) or the test cities '
                         '("test" = author-style transductive). Default "none" keeps '
                         'beta=1 (no stats). Only affects CoDER dempster readouts.')

logger = logging.getLogger(__name__)

logging.getLogger().setLevel(logging.INFO) 


def parse_args():
    return parser.parse_args()


def _apply_modality_config(cfg, modality):
    params = cfg['model']['params']
    modality = modality.upper()

    if modality == 'SAR':
        params['modality'] = [1, 0]
        params['load_rgb'] = False
        params['infer_rgb'] = False
        params['infer_mode'] = 'sar'
    elif modality in ('RGB', 'OPTICAL'):
        params['modality'] = [0, 1]
        params['load_rgb'] = True
        params['infer_rgb'] = True
        params['infer_mode'] = 'rgb'
    elif modality in ('BOTH', 'ALL', 'SAR_RGB', 'RGB_SAR'):
        params['modality'] = [1, 1]
        params['load_rgb'] = False
        params['infer_rgb'] = False
        params['infer_mode'] = 'fuse'
    else:
        raise ValueError(f'Unsupported modality: {modality}')

    return cfg

def evaluate(ckpt_path, config_path='base.hrnetw32', vis_path='model',use_tta=False,modality='SAR',slide_test=False, ms_scales=None, save_uncertainty=False, ece_bins=15, mmr_stats='none'):
    cfg = import_config(config_path)

    model_state_dict = load_model_state_dict_from_ckpt(ckpt_path)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    log_dir = os.path.dirname(ckpt_path)
    
    _apply_modality_config(cfg, modality)
    model = make_model(cfg['model'])
    
    use_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    cfg['data']['test']['params']['distributed'] = use_distributed
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if torch.cuda.is_available() and use_distributed:
        dist.init_process_group(
            backend="nccl", init_method="env://" 
        )
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.load_state_dict(model_state_dict)
    model.to(device)
    
    if dist.is_available() and dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)
    
    test_dataloader = make_dataloader(cfg['data']['test'])
    model.eval()
    classes = cfg.model.params.num_classes if cfg.model.params.num_classes != 1 else 2
    metric_op = er.metric.PixelMetric(classes, logdir=log_dir, logger=logger,class_names=list(COLOR_MAP.keys()))
    vis_dir = os.path.join(log_dir,vis_path)
    palette = np.array(list(COLOR_MAP.values())).reshape(-1).tolist()
    viz_op = VisualizeSegmm(vis_dir, palette)

    # CoDER-specific bookkeeping: ECE accumulators + uncertainty heatmap dir.
    model_unwrapped = model.module if hasattr(model, 'module') else model
    is_coder = type(model_unwrapped).__name__ == 'CoDER'

    # Optional eval-time recomputation of the MMR covariance/mask statistics.
    # Without this, CovarianceReliability has no stats and falls back to
    # beta = 1.0 (full trust). 'test' replicates the author's transductive
    # setting (stats fitted on the test cities); 'train' is the fair variant.
    if mmr_stats != 'none' and hasattr(model_unwrapped, 'conduct_mask_matrix'):
        src = 'test' if mmr_stats == 'test' else 'mmr'
        base_block = cfg['data'][src]
        params = dict(base_block['params'])
        params['distributed'] = False
        params['training'] = False
        params['batch_size'] = 4
        model_unwrapped.config.data = dict(type=base_block['type'], params=params)
        logger.info(f'[mmr_stats={mmr_stats}] computing covariance mask matrix '
                    f'on the {src} loader ...')
        model_unwrapped.conduct_mask_matrix()
        logger.info('[mmr_stats] covariance statistics ready; beta discount active.')
    unc_dir = None
    if is_coder and save_uncertainty:
        if slide_test or use_tta or ms_scales:
            logger.warning('[CoDER] --save_uncertainty: slide_test/TTA active — '
                           'heatmaps will be approximate (last crop / last flip).')
        unc_dir = os.path.join(log_dir, vis_path + '_unc')
        os.makedirs(unc_dir, exist_ok=True)
    ece_sum, ece_count = 0.0, 0

    import torch.nn.functional as F

    def _infer(x):
        if slide_test:
            return slide_inference(model, x, gt, slide_test)
        return model(x, gt)

    def _multi_scale_infer(x, scales):
        H, W = x.shape[-2:]
        accum = None
        for s in scales:
            if abs(s - 1.0) < 1e-3:
                xs = x
            else:
                new_h = max(32, int(round(H * s / 32) * 32))
                new_w = max(32, int(round(W * s / 32) * 32))
                xs = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            ps = _infer(xs)
            if ps.shape[-2:] != (H, W):
                ps = F.interpolate(ps, size=(H, W), mode='bilinear', align_corners=False)
            accum = ps if accum is None else accum + ps
        return accum / len(scales)

    def _full_infer(x):
        if ms_scales:
            return _multi_scale_infer(x, ms_scales)
        return _infer(x)

    with torch.no_grad():
        for img, gt in tqdm(test_dataloader):
            img = img.to(device)
            y_true = gt['cls']
            y_true = y_true.cpu()
            pred = _full_infer(img)
            if use_tta:
                pred_h = _full_infer(torch.flip(img, dims=[-1]))
                pred = pred + torch.flip(pred_h, dims=[-1])
                pred_v = _full_infer(torch.flip(img, dims=[-2]))
                pred = pred + torch.flip(pred_v, dims=[-2])
                pred_hv = _full_infer(torch.flip(img, dims=[-2, -1]))
                pred = pred + torch.flip(pred_hv, dims=[-2, -1])
                pred = pred / 4.0
            # CoDER post-processing: compute ECE on probabilities and dump
            # uncertainty heatmap u = K/S before argmax collapses it.
            if is_coder:
                # ECE expects a true probability distribution; pred is already
                # alpha/S from CoDER (or softmax for MetaRS).
                prob_cpu = pred.detach().cpu()
                ece_batch = expected_calibration_error(
                    prob_cpu, y_true, num_bins=ece_bins, ignore_index=-1
                )
                ece_sum += float(ece_batch) * pred.shape[0]
                ece_count += pred.shape[0]
                if save_uncertainty and unc_dir is not None:
                    u_map = getattr(model_unwrapped, 'last_uncertainty', None)
                    if u_map is not None:
                        u_np = u_map.detach().cpu().numpy()
                        # normalize to [0, 255] for PNG; u in (0, 1].
                        u_np = np.clip(u_np, 0.0, 1.0)
                        for u_img, imname in zip(u_np, gt['fname']):
                            out = (u_img * 255.0).astype(np.uint8)
                            from PIL import Image
                            png_name = imname.replace('tif', 'png').replace('jpg', 'png')
                            Image.fromarray(out).save(os.path.join(unc_dir, png_name))

            pred = pred.argmax(dim=1).cpu()

            valid_inds = y_true != -1
            metric_op.forward(y_true[valid_inds], pred[valid_inds])

            for clsmap, imname in zip(pred, gt['fname']):
                viz_op(clsmap.cpu().numpy().astype(np.uint8), imname.replace('tif', 'png'))
    td = metric_op.summary_all()
    if not dist.is_initialized() or dist.get_rank() == 0:
        with open(os.path.join(log_dir, 'metric_summary.log'), 'a') as log_file:
            log_file.write(f"\n{'='*50}\n")
            log_file.write(f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Checkpoint: {os.path.basename(ckpt_path)}\n")
            log_file.write(str(td))
            log_file.write("\n")
            if is_coder and ece_count > 0:
                ece_mean = ece_sum / max(ece_count, 1)
                log_file.write(f"ECE ({ece_bins} bins): {ece_mean:.5f}\n")
                logger.info(f"[CoDER] ECE={ece_mean:.5f} over {ece_count} samples")
    torch.cuda.empty_cache()




if __name__ == '__main__':
    args = parse_args()
    seed_torch(2333)
    # ckpt_path = './log/deeplabv3p.pth'
    # config_path = 'baseline.deeplabv3p'
    slide_test = dict(
        crop_size=(512,512),
        stride=(341,341)
    )
    ms_scales = None
    if args.ms_tta:
        ms_scales = [float(s) for s in args.ms_tta.split(',') if s.strip()]
    model1 = evaluate(args.ckpt_path, args.config_path, args.vis_path,args.tta,args.modality,
                      slide_test=slide_test, ms_scales=ms_scales,
                      save_uncertainty=args.save_uncertainty, ece_bins=args.ece_bins,
                      mmr_stats=args.mmr_stats)
