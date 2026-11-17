"""Unit tests for the revision components (run: python -m pytest experiments/revision)."""
from types import SimpleNamespace

import torch

from module.baseline.CoDER import CoDER
from module.baseline.moar_fusion import MoARectifyModule
from module.evidential import dempster_combine
from module.reliability.covariance_deviation import CovarianceDeviationReliability

K = 7


def combiner(mode):
    stub = SimpleNamespace(recovered_combination=mode,
                           _combine_evidence=lambda a, b, k: dempster_combine(a, b, k))
    return lambda a_obs, a_rec, beta: CoDER._combine_recovered(stub, a_obs, a_rec, beta, K)


def uncertainty(alpha):
    return K / alpha.sum(dim=1)


def test_average_is_idempotent_for_any_discount():
    e = torch.rand(2, K, 4, 4) * 20
    for beta in (0.05, 0.3, 1.0):
        fused = combiner("average")(e + 1, beta * e + 1, torch.tensor(beta))
        assert torch.allclose(fused, e + 1, atol=1e-5)


def test_average_never_inflates_evidence():
    torch.manual_seed(0)
    e_obs, e_rec = torch.rand(8, K, 4, 4) * 5, torch.rand(8, K, 4, 4) * 50
    beta = torch.rand(8, 1, 1, 1)
    fused = combiner("average")(e_obs + 1, beta * e_rec + 1, beta)
    assert bool((uncertainty(fused) >= uncertainty(e_obs + 1) - 1e-6).all())


def test_dempster_double_counts_a_duplicate():
    e = torch.rand(2, K, 4, 4) * 20
    fused = combiner("dempster")(e + 1, e + 1, torch.tensor(1.0))
    assert bool((uncertainty(fused) < uncertainty(e + 1)).all())


def test_average_with_zero_discount_is_observation_only():
    e_obs, e_rec = torch.rand(2, K, 4, 4) * 5, torch.rand(2, K, 4, 4) * 5
    fused = combiner("average")(e_obs + 1, 0 * e_rec + 1, torch.tensor(0.0))
    assert torch.allclose(fused, e_obs + 1, atol=1e-6)


def _router_grad(mode):
    torch.manual_seed(0)
    module = MoARectifyModule(32, num_experts=3, top_k=1, gate_mode=mode, null_expert=True)
    module.expert_bias[3] = -10.0  # keep the null option out of this test
    rgb, sar, weight = (torch.randn(4, 32, 8, 8) for _ in range(3))
    out_rgb, out_sar = module(rgb, sar)
    ((out_rgb * weight).sum() + (out_sar * weight).sum()).backward()
    return float(module.keys.grad.norm())


def test_switch_gate_trains_router_and_legacy_does_not():
    assert _router_grad("renorm") == 0.0
    assert _router_grad("switch") > 0.0


def test_diagonal_term_detects_variance_only_shift():
    torch.manual_seed(0)
    module = CovarianceDeviationReliability([8], diag_weight=0.25)
    clean = [torch.randn(64, 8, 6, 6)]
    module.update(clean, "rgb")
    module.update(clean, "sar")
    module.finalize_statistics()
    for modality in ("rgb", "sar"):
        module.set_score_calibration(module.deviation_scores(clean, modality), modality)
        module.set_diag_calibration(module.diagonal_scores(clean, modality), modality)
    noisy = [torch.randn(16, 8, 6, 6) * 3.0]  # same (zero) correlations, larger variance
    assert float(module(noisy, "rgb").mean()) < float(module(clean[:1] * 1, "rgb").mean())


def test_no_recovered_stream_returns_observation():
    stub = SimpleNamespace(recovered_combination="average", zero_recovered_evidence=True,
                           _combine_evidence=lambda a, b, k: dempster_combine(a, b, k))
    e = torch.rand(2, K, 4, 4) * 5
    fused = CoDER._combine_recovered(stub, e + 1, torch.ones_like(e), torch.tensor(0.3), K)
    assert torch.equal(fused, e + 1)


def test_duplicate_is_idempotent_with_degraded_observation():
    # beta_obs < 1 at test time: alpha_obs - 1 = b*e, alpha_rec - 1 = b*g*e.
    e = torch.rand(2, K, 4, 4) * 20
    for b, g in ((0.4, 0.3), (0.9, 1.0), (0.2, 0.05)):
        fused = combiner("average")(b * e + 1, b * g * e + 1, torch.tensor(b * g))
        assert torch.allclose(fused, b * e + 1, atol=1e-5)
