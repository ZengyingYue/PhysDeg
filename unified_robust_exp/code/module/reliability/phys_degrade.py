"""Physically motivated optical / SAR degradations (training and evaluation).

Optical, in raw DN; ``unit`` scales radiance constants to the dataset (1 for
8-bit WHU / EarthMiss, cloud_raw / 235 for SEN12MS-CR reflectance):

  spectral_haze   scattering model I = J t + A (1 - t), t = exp(-tau(x) (lambda/0.55)^-alpha);
                  tau varies smoothly in space, alpha in [0, 2] spans thin cloud (grey) to
                  aerosol haze (stronger in blue).
  cloud_shadow    fractal cloud with its cast shadow: the shadow is the same cloud field shifted
                  by h tan(theta_s) / GSD along the solar azimuth, cut from a canvas larger than
                  the patch so clouds outside the view also cast shadows into it; shadowed pixels
                  keep only the diffuse skylight k_lambda = k0 (lambda/0.55)^-gamma (bluish).

SAR (``domain`` 'amp' = 8-bit amplitude DN, 'db' = backscatter in dB):

  speckle         multiplicative Gamma speckle (as ``train_degrade.degrade_sar``)
  calib           radiometric calibration offset + incidence-angle ramp across range
  noisefloor      thermal noise (NESZ) added in intensity, scalloped along azimuth:
                  dark targets (water, smooth surfaces) drown in noise
  rfi             radio-frequency interference: bright streaks along random range lines

Geometry:

  coreg           optical-SAR co-registration error: sub-pixel shift of the SAR image
                  relative to the optical image and the labels (labels follow the optical).

Every ``apply_*`` takes explicit parameters (evaluation); ``random_*`` samples
them per image (training) and works on normalised tensors like
``train_degrade``.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

LAMBDA = (0.665, 0.560, 0.490)  # R, G, B centre wavelengths (um)
OPTICAL_TYPES = ("spectral_haze", "cloud_shadow")
SAR_TYPES = ("speckle", "calib", "noisefloor", "rfi")


def _lam(ref: torch.Tensor) -> torch.Tensor:
    return torch.tensor(LAMBDA, dtype=ref.dtype, device=ref.device).view(1, 3, 1, 1) / 0.55


def _randn(shape, device, generator):
    if generator is None:
        return torch.randn(shape, device=device)
    return torch.randn(shape, generator=generator).to(device)


def _rand(generator=None) -> float:
    return float(torch.rand(1, generator=generator))


def fractal_field(b, h, w, device, beta=2.5, generator=None):
    """Random fields [B,1,H,W] with power spectrum ~ 1/f^beta, scaled to [0, 1]."""
    noise = _randn((b, 1, h, w), device, generator)
    fy = torch.fft.fftfreq(h, device=device).view(-1, 1)
    fx = torch.fft.rfftfreq(w, device=device).view(1, -1)
    f = torch.sqrt(fx ** 2 + fy ** 2).clamp_min(1.0 / max(h, w))
    field = torch.fft.irfft2(torch.fft.rfft2(noise) / f ** (beta / 2), s=(h, w))
    flat = field.flatten(1)
    lo, hi = flat.min(1)[0].view(b, 1, 1, 1), flat.max(1)[0].view(b, 1, 1, 1)
    return (field - lo) / (hi - lo + 1e-8)


# --------------------------------------------------------------------- optical
def apply_spectral_haze(rgb, tau0, alpha, airlight, field, spectral=True, return_rel=False):
    """rgb [B,3,H,W] raw DN; field [B,1,H,W] in [0,1]; tau0/alpha/airlight scalars.

    ``spectral=False`` (ablation): wavelength-independent transmission and airlight.
    ``return_rel``: also return the contrast transmittance dI/dJ = t, band mean [B,1,H,W].
    """
    lam = _lam(rgb) if spectral else torch.ones_like(_lam(rgb))
    tau = tau0 * (0.5 + field)
    t = torch.exp(-tau * lam ** (-alpha))
    a = airlight * lam ** (-0.3)  # slightly bluish airlight
    out = rgb * t + a * (1 - t)
    return (out, t.mean(1, keepdim=True)) if return_rel else out


def _density(canvas, coverage):
    b = canvas.shape[0]
    thr = torch.quantile(canvas.flatten(1), 1.0 - coverage, dim=1).view(b, 1, 1, 1)
    return ((canvas - thr) / (0.15 * (1 - thr) + 1e-6)).clamp(0, 1)


def apply_cloud_shadow(rgb, coverage, shift, k0, gamma, opacity, cloud_value,
                       generator=None, shadow_mode="cast", return_rel=False):
    """One cloud field per image; ``shift`` = (dy, dx) pixels of the cast shadow.

    ``shadow_mode`` (ablations): 'cast' = the cloud field shifted (default), 'coloc' = the
    shadow directly under its cloud (same cloud field, no offset), 'none' = no shadow,
    'indep' = a shadow field of the same coverage drawn independently of the cloud (dark
    patches without the cloud-shadow geometry).  ``return_rel``: also return the contrast transmittance
    dI/dJ = (1 - opacity cloud)(1 - shadow (1 - k)), band mean [B,1,H,W].
    """
    b, _, h, w = rgb.shape
    dy, dx = int(round(shift[0])), int(round(shift[1]))
    pad = max(abs(dy), abs(dx))
    canvas = fractal_field(b, h + 2 * pad, w + 2 * pad, rgb.device, beta=3.0,
                           generator=generator)
    density = _density(canvas, coverage)
    cloud = density[:, :, pad:pad + h, pad:pad + w]
    if shadow_mode == "cast":
        shadow = density[:, :, pad - dy:pad - dy + h, pad - dx:pad - dx + w]
    elif shadow_mode == "coloc":  # same cloud field, shadow directly beneath it
        shadow = cloud
    elif shadow_mode == "none":
        shadow = torch.zeros_like(cloud)
    elif shadow_mode == "indep":
        shadow = _density(fractal_field(b, h, w, rgb.device, beta=3.0, generator=generator),
                          coverage)
    else:
        raise ValueError(shadow_mode)
    k = (k0 * _lam(rgb) ** (-gamma)).clamp(max=1.0)
    shade = 1 - shadow * (1 - k)
    out = rgb * shade
    texture = 0.9 + 0.1 * canvas[:, :, pad:pad + h, pad:pad + w]
    out = out * (1 - opacity * cloud) + cloud_value * texture * opacity * cloud
    if return_rel:
        return out, cloud, (1 - opacity * cloud) * shade.mean(1, keepdim=True)
    return out, cloud


# ------------------------------------------------------------------------- SAR
def _to_intensity(sar, domain):
    return sar.clamp_min(0) ** 2 if domain == "amp" else 10 ** (sar / 10)


def _from_intensity(power, domain):
    return power.sqrt() if domain == "amp" else 10 * torch.log10(power.clamp_min(1e-10))


def apply_speckle(sar, looks, domain, generator=None):
    shape = torch.full(tuple(sar.shape), float(looks))
    gamma = torch._standard_gamma(shape, generator=generator).div(looks).to(sar.device)
    if domain == "amp":
        return sar * gamma
    return sar + 10 * torch.log10(gamma.clamp_min(1e-6))


def apply_calib(sar, offset_db, ramp_db, domain, axis=-1):
    n = sar.shape[axis]
    pos = torch.linspace(-0.5, 0.5, n, device=sar.device, dtype=sar.dtype)
    view = [1] * sar.ndim
    view[axis] = n
    delta = offset_db + ramp_db * pos.view(view)
    return sar * 10 ** (delta / 20) if domain == "amp" else sar + delta


def apply_noisefloor(sar, level, domain, axis=-2, generator=None):
    """``level``: noise amplitude in DN ('amp') or NESZ in dB ('db')."""
    n = sar.shape[axis]
    pos = torch.arange(n, device=sar.device, dtype=sar.dtype)
    view = [1] * sar.ndim
    view[axis] = n
    period = n / 3.0
    scallop = 1 + 0.3 * torch.sin(2 * math.pi * pos / period + 6.283 * _rand(generator))
    noise_power = (level ** 2 if domain == "amp" else 10 ** (level / 10)) * scallop.view(view)
    expo = -torch.log(torch.rand(sar.shape, generator=generator).clamp_min(1e-6)).to(sar.device)
    return _from_intensity(_to_intensity(sar, domain) + noise_power * expo, domain)


def apply_rfi(sar, fraction, level, domain, axis=-2, generator=None):
    """Bright streaks on a ``fraction`` of the lines along ``axis``; ``level`` as noisefloor."""
    if axis == -1:
        return apply_rfi(sar.transpose(-1, -2), fraction, level, domain, -2,
                         generator).transpose(-1, -2)
    b, c, h, w = sar.shape
    lines = torch.rand((b, h), generator=generator) < fraction
    profile = torch.zeros((b, 1, h, w))
    for i, r in lines.nonzero().tolist():
        start = int(_rand(generator) * w * 0.7)
        length = int(w * (0.3 + 0.7 * _rand(generator)))
        width = 1 + int(_rand(generator) * 3)
        profile[i, 0, r:r + width, start:start + length] = 0.5 + 0.5 * _rand(generator)
    power = (level ** 2 if domain == "amp" else 10 ** (level / 10)) * profile.to(sar.device)
    return _from_intensity(_to_intensity(sar, domain) + power, domain)


# -------------------------------------------------------------------- geometry
def apply_coreg(sar, dy, dx):
    """Shift ``sar`` [B,C,H,W] by (dy, dx) pixels (bilinear, border padding)."""
    b, _, h, w = sar.shape
    ys = torch.linspace(-1, 1, h, device=sar.device) - 2 * dy / max(h - 1, 1)
    xs = torch.linspace(-1, 1, w, device=sar.device) - 2 * dx / max(w - 1, 1)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gx, gy], -1).unsqueeze(0).expand(b, -1, -1, -1).to(sar.dtype)
    return F.grid_sample(sar, grid, mode="bilinear", padding_mode="border", align_corners=True)


# -------------------------------------------------------------- training draws
def _raw(x, mean, std):
    mean, std = mean.view(1, -1, 1, 1), std.view(1, -1, 1, 1)
    return x.float() * std + mean, mean, std


ABLATIONS = ("noshadow", "coloc", "indepshadow", "grey")


def _coverage(u, mode):
    """Cloud cover from one uniform draw u: 'uniform' U(0.05, 0.6) (original recipe) or
    'ushape' 0.02 + 0.98 * arcsine(u) -- the U-shaped (Beta(1/2, 1/2)) cloud-fraction
    distribution of real scenes (mostly near-clear or overcast), reaching full cover."""
    if mode == "uniform":
        return 0.05 + 0.55 * u
    if mode == "ushape":
        return 0.02 + 0.98 * math.sin(0.5 * math.pi * u) ** 2
    raise ValueError(f"unknown coverage mode {mode!r}")


def random_optical(rgb, mean, std, prob, unit=1.0, max_dn=255.0, gsd=5.0,
                   types=OPTICAL_TYPES, ablate=(), return_rel=False, coverage_mode="uniform"):
    """Degrade each image's optical input with probability ``prob`` (one family each).

    ``ablate`` (component ablations, PHYS_PLAN.md section 5): 'noshadow' drops the cast
    shadow, 'coloc' puts it under its cloud (zero offset), 'indepshadow' draws it
    independently of the cloud, 'grey' removes all wavelength dependence (haze alpha = 0,
    flat airlight, shadow gamma = 0).  Each image's parameter draws are the same in every
    variant ('indepshadow' additionally draws its shadow field).
    ``return_rel``: also return the per-pixel optical reliability target [B,1,H,W] (contrast
    transmittance, 1 for untouched images).
    """
    bad = set(ablate) - set(ABLATIONS)
    if bad:
        raise ValueError(f"unknown ablation(s) {sorted(bad)}")
    grey = "grey" in ablate
    shadow_mode = "none" if "noshadow" in ablate else "indep" if "indepshadow" in ablate \
        else "coloc" if "coloc" in ablate else "cast"
    chosen = torch.rand(rgb.shape[0]) < prob
    rel = torch.ones((rgb.shape[0], 1) + tuple(rgb.shape[2:]), device=rgb.device) \
        if return_rel else None
    if not bool(chosen.any()):
        return (rgb, rel) if return_rel else rgb
    raw, mean, std = _raw(rgb, mean, std)
    out = []
    for i in range(rgb.shape[0]):
        img = raw[i:i + 1]
        if chosen[i]:
            kind = types[int(torch.randint(len(types), (1,)))]
            if kind == "spectral_haze":
                field = fractal_field(1, img.shape[2], img.shape[3], img.device, beta=3.0)
                tau0, alpha, air = 0.1 + 1.4 * _rand(), 2.0 * _rand(), unit * (190 + 60 * _rand())
                img, r = apply_spectral_haze(img, tau0, 0.0 if grey else alpha, air, field,
                                             spectral=not grey, return_rel=True)
            elif kind == "cloud_shadow":
                height = 500 + 2500 * _rand()                     # cloud height (m)
                zenith = math.radians(20 + 40 * _rand())          # solar zenith
                azimuth = 2 * math.pi * _rand()
                d = min(height * math.tan(zenith) / gsd, 0.75 * img.shape[2])
                coverage, k0, gamma = _coverage(_rand(), coverage_mode), 0.15 + 0.35 * _rand(), 1 + 2 * _rand()
                opacity, value = 0.7 + 0.3 * _rand(), unit * (215 + 35 * _rand())
                img, _, r = apply_cloud_shadow(
                    img, coverage, (d * math.sin(azimuth), d * math.cos(azimuth)),
                    k0, 0.0 if grey else gamma, opacity, value,
                    shadow_mode=shadow_mode, return_rel=True)
            else:
                raise ValueError(kind)
            img = img.clamp(0, max_dn)
            if return_rel:
                rel[i:i + 1] = r
        out.append(img)
    out = ((torch.cat(out, 0) - mean) / std).to(rgb.dtype)
    return (out, rel) if return_rel else out


def random_sar(sar, mean, std, prob, domain="amp", max_dn=255.0, types=SAR_TYPES):
    """Degrade each image's SAR input with probability ``prob`` (one family each)."""
    chosen = torch.rand(sar.shape[0]) < prob
    if not bool(chosen.any()):
        return sar
    raw, mean, std = _raw(sar, mean, std)
    out = []
    for i in range(sar.shape[0]):
        img = raw[i:i + 1]
        if chosen[i]:
            kind = types[int(torch.randint(len(types), (1,)))]
            axis = -1 if _rand() < 0.5 else -2   # range direction after flips / rotations
            amp = domain == "amp"
            if kind == "speckle":
                img = apply_speckle(img, 1.0 + 15.0 * _rand(), domain)
            elif kind == "calib":
                img = apply_calib(img, 6 * _rand() - 3, 4 * _rand() - 2, domain, axis)
            elif kind == "noisefloor":
                level = 10 + 50 * _rand() if amp else -28 + 14 * _rand()
                img = apply_noisefloor(img, level, domain, -3 - axis)
            elif kind == "rfi":
                level = 150 + 100 * _rand() if amp else -5 + 10 * _rand()
                img = apply_rfi(img, 0.005 + 0.095 * _rand(), level, domain, axis)
            else:
                raise ValueError(kind)
            if amp:
                img = img.clamp(0, max_dn)
        out.append(img)
    return ((torch.cat(out, 0) - mean) / std).to(sar.dtype)


def random_coreg(sar, prob, max_shift=12.0):
    """Shift each image's (normalised) SAR input with probability ``prob``."""
    chosen = (torch.rand(sar.shape[0]) < prob).tolist()
    if not any(chosen):
        return sar
    out = []
    for i, c in enumerate(chosen):
        img = sar[i:i + 1]
        if c:
            r, a = max_shift * _rand(), 2 * math.pi * _rand()
            img = apply_coreg(img, r * math.sin(a), r * math.cos(a))
        out.append(img)
    return torch.cat(out, 0)


__all__ = ["OPTICAL_TYPES", "SAR_TYPES", "ABLATIONS", "fractal_field", "apply_spectral_haze",
           "apply_cloud_shadow", "apply_speckle", "apply_calib", "apply_noisefloor",
           "apply_rfi", "apply_coreg", "random_optical", "random_sar", "random_coreg"]
