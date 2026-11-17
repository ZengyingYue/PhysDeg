# Covariance-deviation beta_obs

This module implements per-image, per-modality observed-input reliability. It
does not read MetaRS's temporary `var_matrix`. Clean-domain covariance mean,
variance, masks, thresholds, and scales are registered buffers and are saved in
the calibration artifact/checkpoint.

The implementation intentionally uses spatial covariance independently for
every image. Results therefore do not depend on evaluation batch composition.
See `code/experiments/covariance_beta_obs/README.md` for the validation protocol.

