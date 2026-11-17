import os

import cv2
# Limit OpenCV threads (inherited by forked data-loader workers) so that
# concurrent jobs do not oversubscribe the CPU.  Does not change any output.
cv2.setNumThreads(int(os.environ.get("CV2_THREADS", "2")))
import torch.distributed as _dist
_DIST_BACKEND = os.environ.get("DIST_BACKEND", "nccl")
if _DIST_BACKEND != "nccl":
    _orig_init_pg = _dist.init_process_group
    def _init_pg_override(*args, **kwargs):
        kwargs["backend"] = _DIST_BACKEND
        return _orig_init_pg(*args, **kwargs)
    _dist.init_process_group = _init_pg_override

# Per-param-group LR scaling.  The stock ever LR schedulers call
# `ever.opt.learning_rate.set_lr(optimizer, lr)` every step, which forces
# *every* param group to the same lr (clobbering any per-group lr we set).
# We patch it to honour a per-group `lr_scale` key so CoDER.custom_param_groups
# can slow down the pretrained TransOSS ViT trunk relative to the random-init
# taps / FPN / decoders.  Groups without `lr_scale` are unaffected (scale 1.0).
import ever.opt.learning_rate as _lr_mod
def _set_lr_with_scale(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr * pg.get("lr_scale", 1.0)
_lr_mod.set_lr = _set_lr_with_scale

import ever as er
import torch
import numpy as np
import json
from data.EarthMiss import COLOR_MAP
from tqdm import tqdm
import random
from module.slide_test import slide_inference
from module.viz import VisualizeSegmm
from module.baseline.MetaRS import MetaRS  # legacy baseline, kept for ablation
from module.baseline.CoDER import CoDER  # noqa: F401 — side-effect: registers 'CoDER'
import module.baseline.revision_baselines  # noqa: F401 — registers the revision baselines
import module.baseline.stars  # noqa: F401 — registers 'STARS' (revision, 2026-11-08)
from module.ema import EMAModel
from ever import registry
from ever.core.launcher import Launcher
import ever.core.launcher as launcher_module
from ever.trainer.trainer import Trainer as BaseTrainer
from ever.interface.callback import Callback


def patch_base_trainer_launcher():
    def build_launcher(self, model_fn=None, optimizer_fn=None, lr_fn=None):
        model = self.make_model(model_fn=model_fn).to(self.device)
        kwargs = dict(
            model_dir=self.args.model_dir,
            mixed_precision=self.args.mixed_precision,
            model=model,
        )
        lr_opt = self.make_lr_optimizer(model.custom_param_groups(),
                                        lr_fn=lr_fn,
                                        optimizer_fn=optimizer_fn)
        kwargs.update(lr_opt)

        # EMA hook — when EMA_DECAY > 0, attach a post-step hook on the
        # optimizer that updates a shadow copy of the model weights.  The
        # shadow is stashed on the model itself for retrieval at eval-time
        # (see evaluate_cls_fn) and is saved alongside each checkpoint.
        ema_decay = float(os.environ.get('EMA_DECAY', '0.0'))
        if ema_decay > 0.0:
            ema_warmup = int(os.environ.get('EMA_WARMUP', '500'))
            ema = EMAModel(model, decay=ema_decay)
            optimizer = lr_opt['optimizer']
            step_counter = {'n': 0}

            def _ema_post_step(*args, **kwargs):
                step_counter['n'] += 1
                if step_counter['n'] < ema_warmup:
                    return
                ema.update(model)

            if hasattr(optimizer, 'register_step_post_hook'):
                optimizer.register_step_post_hook(_ema_post_step)
            else:
                # PyTorch < 2.0 fallback: wrap step()
                _orig_step = optimizer.step

                def _wrapped_step(*a, **kw):
                    out = _orig_step(*a, **kw)
                    _ema_post_step()
                    return out

                optimizer.step = _wrapped_step
            model._ema = ema
            print(f'[EMA] enabled — decay={ema_decay}, warmup={ema_warmup}')

        tl = Launcher(**kwargs)
        return dict(config=self.config, launcher=tl)

    BaseTrainer.build_launcher = build_launcher


class DisabledSaveCheckpointCallback(Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(
            epoch_interval=1,
            only_master=True,
            prior=0,
            before_train=False,
            after_train=False
        )

    def func(self):
        return None

    def name(self):
        return 'DisabledSaveCheckpoint'


def patch_default_checkpoint_callback():
    launcher_module.SaveCheckpointCallback = DisabledSaveCheckpointCallback


def patch_thddp_launcher_ema():
    """Attach EMA under the *th_ddp* trainer.

    ``patch_base_trainer_launcher`` only patches ``Trainer.build_launcher``,
    but ``THDDPTrainer`` overrides ``build_launcher`` (and DDP-wraps the model),
    so the base EMA hook is shadowed and never runs.  We wrap the th_ddp
    ``build_launcher`` instead: call the original (keeps all DDP logic intact),
    then register the EMA post-step hook on the returned launcher's optimizer,
    tracking the *unwrapped* ``model.module`` so the shadow matches what
    ``evaluate_cls_fn`` swaps in at eval-time.
    """
    from ever.trainer.th_ddp_trainer import THDDPTrainer
    _orig = THDDPTrainer.build_launcher

    def build_launcher(self, model_fn=None, optimizer_fn=None, lr_fn=None):
        out = _orig(self, model_fn=model_fn, optimizer_fn=optimizer_fn, lr_fn=lr_fn)
        # config train.ema_decay (per experiment) takes precedence over the EMA_DECAY env var
        ema_decay = float(self.config['train'].get('ema_decay', os.environ.get('EMA_DECAY', '0.0')))
        if ema_decay <= 0.0:
            return out
        launcher = out['launcher']
        model = launcher.model
        model_ref = model.module if hasattr(model, 'module') else model
        optimizer = launcher.optimizer
        ema_warmup = int(self.config['train'].get('ema_warmup', os.environ.get('EMA_WARMUP', '500')))
        ema = EMAModel(model_ref, decay=ema_decay)
        step_counter = {'n': 0}

        def _ema_post_step(*args, **kwargs):
            step_counter['n'] += 1
            if step_counter['n'] < ema_warmup:
                ema.shadow = {k: v.detach().clone() for k, v in model_ref.named_parameters() if k in ema.shadow}
                return
            ema.update(model_ref)

        if hasattr(optimizer, 'register_step_post_hook'):
            optimizer.register_step_post_hook(_ema_post_step)
        else:
            _orig_step = optimizer.step

            def _wrapped_step(*a, **kw):
                r = _orig_step(*a, **kw)
                _ema_post_step()
                return r

            optimizer.step = _wrapped_step
        model_ref._ema = ema
        print(f'[EMA] enabled (th_ddp) — decay={ema_decay}, warmup={ema_warmup}',
              flush=True)
        return out

    THDDPTrainer.build_launcher = build_launcher


def extract_best_score(metric_table, metric_name='miou'):
    metric_name = metric_name.lower()
    if metric_name in ('miou', 'iou'):
        values = metric_table.iou()
        mean_row = next(row for row in metric_table._rows if row[1] == 'mean')
        return float(mean_row[2])
    if metric_name in ('mf1', 'f1'):
        mean_row = next(row for row in metric_table._rows if row[1] == 'mean')
        return float(mean_row[3])
    raise ValueError(f'Unsupported best checkpoint metric: {metric_name}')


def remove_stale_checkpoints(model_dir, keep_filename='Best.pth'):
    for filename in os.listdir(model_dir):
        if filename.startswith('checkpoint-') and filename.endswith('.pth') and filename != keep_filename:
            os.remove(os.path.join(model_dir, filename))


def load_best_record(model_dir, initial_score):
    record_path = os.path.join(model_dir, 'best_checkpoint.json')
    if os.path.exists(record_path):
        with open(record_path, 'r') as f:
            return json.load(f)
    return dict(score=float(initial_score), step=0, filename='')


def save_best_record(model_dir, record):
    with open(os.path.join(model_dir, 'best_checkpoint.json'), 'w') as f:
        json.dump(record, f)


def update_checkpoint_info_after_best(launcher, keep_filename):
    checkpoint_log = launcher.checkpoint._json_log
    for step, filename in list(checkpoint_log.items()):
        if step == launcher.checkpoint.LASTCHECKPOINT:
            continue
        if filename != keep_filename and str(filename).startswith('checkpoint-'):
            checkpoint_log.pop(step, None)
    checkpoint_log[launcher.checkpoint.LASTCHECKPOINT] = dict(
        step=launcher.checkpoint.global_step,
        name=keep_filename
    )
    launcher.checkpoint.save_checkpoint_info(launcher.model_dir, launcher.logger)


def maybe_save_best_checkpoint(launcher, metric_table, config):
    if config is None:
        return
    best_cfg = config.get('best_checkpoint', {})
    if not best_cfg or not best_cfg.get('enabled', False):
        return
    if not launcher.is_main_process:
        return

    metric_name = best_cfg.get('metric', 'miou')
    score = extract_best_score(metric_table, metric_name)
    initial_score = best_cfg.get('initial_score', float('-inf'))
    record = launcher.buffer.setdefault(
        'best_checkpoint',
        load_best_record(launcher.model_dir, initial_score)
    )
    best_score = float(record.get('score', float('-inf')))
    if score <= best_score:
        launcher.info(
            f'Best checkpoint not improved: {metric_name}={score:.5f}, best={best_score:.5f}'
        )
        return

    keep_filename = best_cfg.get('filename', 'Best.pth')
    launcher.checkpoint.save(keep_filename)
    if best_cfg.get('delete_old_checkpoints', True):
        remove_stale_checkpoints(launcher.model_dir, keep_filename)
        update_checkpoint_info_after_best(launcher, keep_filename)

    record.update(dict(
        score=score,
        step=launcher.checkpoint.global_step,
        filename=keep_filename,
        metric=metric_name,
    ))
    save_best_record(launcher.model_dir, record)
    launcher.info(
        f'Best checkpoint improved: {metric_name}={score:.5f}, saved {keep_filename}'
    )


def evaluate_select_modes_fn(self, test_dataloader, config):
    """Evaluate every deployment condition and select on their mean mIoU.

    Used by the revision protocol: ``config['select_modes']`` lists the
    model infer modes (e.g. fuse / sar / rgb = clean / missing-optical /
    missing-SAR).  The dataloader is the validation split; the checkpoint
    with the best mean validation mIoU is kept.
    """
    self.model.eval()
    model_ref = self.model.module if hasattr(self.model, 'module') else self.model
    # EMA (config train.ema_decay): select on, and save, the averaged weights; Last.pth always
    # holds the EMA weights of the latest evaluation (= end of training after the final eval).
    ema = getattr(model_ref, '_ema', None)
    ema_backup = ema.apply_to(model_ref) if ema is not None else None
    device = next(model_ref.parameters()).device
    classes = model_ref.config.num_classes
    original_mode = model_ref.infer_mode
    scores = {}
    with torch.no_grad():
        for mode in config['select_modes']:
            model_ref.infer_mode = mode
            confusion = torch.zeros(classes, classes, dtype=torch.long, device=device)
            for img, gt in tqdm(test_dataloader):
                pred = self.model(img.to(device), gt).argmax(dim=1)
                y_true = gt['cls'].to(device).long()
                valid = y_true != -1
                confusion += torch.bincount(
                    y_true[valid] * classes + pred[valid], minlength=classes * classes
                ).reshape(classes, classes)
            inter = confusion.diag().double()
            union = confusion.sum(0).double() + confusion.sum(1).double() - inter
            present = union > 0
            scores[mode] = float((inter[present] / union[present]).mean())
    model_ref.infer_mode = original_mode
    mean_score = float(np.mean(list(scores.values())))
    self.logger.info('[select] step %d val mIoU %s mean=%.5f' % (
        int(self.checkpoint.global_step),
        ' '.join(f'{m}={v:.5f}' for m, v in scores.items()), mean_score))

    class _Table:  # minimal stand-in so maybe_save_best_checkpoint can read the score
        _rows = [(None, "mean", mean_score)]

        @staticmethod
        def iou():
            return None
    maybe_save_best_checkpoint(self, _Table(), config)
    if ema is not None:
        if self.is_main_process:
            self.checkpoint.save('Last.pth')
        ema.restore(model_ref, *ema_backup)
    if self.is_main_process:
        with open(os.path.join(self._model_dir, 'selection_log.jsonl'), 'a') as f:
            f.write(json.dumps(dict(step=int(self.checkpoint.global_step),
                                    mean=mean_score, ema=ema is not None, **scores)) + '\n')
        # Resume point (2026-11-10): model + optimizer + step at every validation, so a run
        # killed by systemd-oomd restarts from here (ever's try_resume reads the latest entry
        # of checkpoint_info.json) instead of from step 0.  Saving does not touch any RNG or
        # the selection; Best.pth stays the selected checkpoint.  EMA runs already save
        # Last.pth above (the averaged weights), so they are left unchanged.
        if ema is None:
            self.checkpoint.save('Last.pth')
    torch.cuda.empty_cache()
    return None


def evaluate_cls_fn(self, test_dataloader, config=None):
    if config is not None and config.get('select_modes'):
        return evaluate_select_modes_fn(self, test_dataloader, config)
    self.model.eval()
    model_ref = self.model.module if hasattr(self.model, 'module') else self.model
    # If an EMA shadow exists on the model, swap it in for the duration of
    # the eval pass.  This evaluates from the averaged weights, which are
    # typically smoother and give +0.5-1 mIoU late in training.
    ema = getattr(model_ref, '_ema', None)
    ema_backup = None
    if ema is not None:
        ema_backup = ema.apply_to(model_ref)
        # Snapshot EMA state alongside the regular checkpoint dir so we
        # can re-evaluate offline with the EMA weights too.
        if self.is_main_process:
            step = int(self.checkpoint.global_step)
            ema_path = os.path.join(self._model_dir, f'ema-checkpoint-{step}.pth')
            try:
                ema.save(ema_path)
            except Exception as exc:
                self.logger.info(f'[EMA] save failed: {exc}')
    device = next(model_ref.parameters()).device
    classes = model_ref.config.num_classes if model_ref.config.num_classes != 1 else 2
    # class_names must match `classes` (PixelMetric asserts equality).  EarthMiss
    # uses the 8-class COLOR_MAP; other datasets (e.g. BraTS, 4-class) fall back.
    _cmap_names = list(COLOR_MAP.keys())
    if len(_cmap_names) == classes:
        _class_names = _cmap_names
    elif classes == 4:
        _class_names = ["bg", "NCR/NET", "edema", "ET"]      # BraTS tumour classes
    else:
        _class_names = [f"class_{i}" for i in range(classes)]
    metric_op = er.metric.PixelMetric(classes, logdir=self._model_dir, logger=self.logger,class_names=_class_names)

    vis_dir = os.path.join(self._model_dir, 'vis-{}'.format(self.checkpoint.global_step))

    palette = np.array(list(COLOR_MAP.values())).reshape(-1).tolist()
    viz_op = VisualizeSegmm(vis_dir, palette)

    with torch.no_grad():
        for img, gt in tqdm(test_dataloader):
            img = img.to(device)
            y_true = gt['cls']
            y_true = y_true.cpu()
            if config.get('slide_test', False):
                pred = slide_inference(self.model,img,gt, config['slide_test'])
            else:
                pred = self.model(img,gt)

            y_pred = pred.argmax(dim=1).cpu()

            valid_inds = y_true != -1
            metric_op.forward(y_true[valid_inds], y_pred[valid_inds])

            for clsmap, imname in zip(y_pred, gt['fname']):
                if "tif" in imname:
                    viz_op(clsmap.cpu().numpy().astype(np.uint8), imname.replace('tif', 'png'))
                elif "jpg" in imname:
                    viz_op(clsmap.cpu().numpy().astype(np.uint8), imname.replace('jpg', 'png'))
                else:
                    viz_op(clsmap.cpu().numpy().astype(np.uint8), imname)
    metric_table = metric_op.summary_all()
    maybe_save_best_checkpoint(self, metric_table, config)
    # Restore live (non-EMA) weights for the next training step.
    if ema is not None and ema_backup is not None:
        ema.restore(model_ref, *ema_backup)
    torch.cuda.empty_cache()
    return metric_table



def register_evaluate_fn(launcher):
    launcher.override_evaluate(evaluate_cls_fn)


def seed_torch(seed=2333):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


if __name__ == '__main__':
    seed_torch(int(os.environ.get('SEED', '2333')))
    if os.environ.get('DISABLE_CKPT_CALLBACK', '0') == '1':
        patch_default_checkpoint_callback()
    patch_base_trainer_launcher()
    patch_thddp_launcher_ema()
    trainer = er.trainer.get_trainer()()
    trainer.run(after_construct_launcher_callbacks=[register_evaluate_fn])
