# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lookahead verification of the dual-key (TextSeal) and Gaussian copula
Gumbel-max watermarks."""

import math

import pytest
import torch

from tests.watermarking.test_lookahead import _blocks
from tests.watermarking.test_lookahead_ppl import _paths
from vllm.config.watermarking import WatermarkConfig, derive_watermark_key
from vllm.v1.watermarking import (
    GaussianCopulaWatermarker,
    LookaheadDualKeyGumbelWatermarker,
    LookaheadGaussianCopulaWatermarker,
    LookaheadGumbelWatermarker,
    SupportsSpeculativeDecoding,
    SupportsVerificationLogits,
    create_watermarker,
)
from vllm.v1.watermarking.spec_decode import (
    create_speculative_draft_watermarker,
    speculative_target_watermark_key,
)

VARIANTS = {
    "lookahead_dual_key_gumbel": LookaheadDualKeyGumbelWatermarker,
    "lookahead_gaussian_copula": LookaheadGaussianCopulaWatermarker,
}


def _variant(algorithm: str, **kwargs) -> LookaheadGumbelWatermarker:
    if algorithm == "lookahead_dual_key_gumbel":
        kwargs.setdefault("alpha", 0.5)
    return VARIANTS[algorithm](**kwargs)


def _chosen(watermarker: LookaheadGumbelWatermarker, rows: dict) -> torch.Tensor:
    children = watermarker.children(rows["draft_tokens"], rows["cu_num_logits"])
    return watermarker.lookahead_tokens(
        rows["logits"],
        rows["contexts"],
        children,
        rows["cu_num_logits"],
        rows["skip_mask"],
    )


@pytest.mark.parametrize("algorithm", VARIANTS)
def test_config_selects_the_lookahead_variant(algorithm: str):
    config = WatermarkConfig(
        algorithm=algorithm,
        key=42,
        context_width=2,
        alpha=0.2,
        correlation=0.9,
        aggregator="lin",
        aggregator_lambda=0.5,
    )
    watermarker = create_watermarker(config)

    assert isinstance(watermarker, VARIANTS[algorithm])
    assert isinstance(watermarker, SupportsVerificationLogits)
    # Drafts stay unwatermarked, so the keys are not exposed as draft and
    # target watermarkers.
    assert not isinstance(watermarker, SupportsSpeculativeDecoding)
    assert (watermarker.aggregator, watermarker.aggregator_lambda) == ("lin", 0.5)
    if algorithm == "lookahead_dual_key_gumbel":
        assert watermarker.alpha == 0.2
    else:
        assert watermarker.correlation == 0.9
    assert config.supports_speculative_decoding
    assert (
        create_speculative_draft_watermarker(
            watermarker, 4, torch.device("cpu"), allow_target_only=False
        )
        is None
    )
    assert speculative_target_watermark_key(config) is None
    with pytest.raises(ValueError):
        WatermarkConfig(algorithm=algorithm, key=1, aggregator="optpath")


def test_dual_key_at_alpha_zero_or_one_is_lookahead_gumbel_under_that_key():
    rows = _blocks(64, 5, 97, 3, seed=1)
    for alpha, domain in ((0.0, b"key_a"), (1.0, b"key_b")):
        dual = LookaheadDualKeyGumbelWatermarker(
            key=7, context_width=3, alpha=alpha, aggregator="tsum"
        )
        single = LookaheadGumbelWatermarker(
            key=derive_watermark_key(7, domain), context_width=3, aggregator="tsum"
        )
        assert torch.equal(_chosen(dual, rows), _chosen(single, rows))


def test_copula_at_correlation_one_is_lookahead_gumbel():
    """Fully keyed noise is the Gumbel of the keyed uniform, up to rounding."""
    rows = _blocks(64, 5, 97, 3, seed=2)
    copula = LookaheadGaussianCopulaWatermarker(
        key=7, context_width=3, correlation=1.0, aggregator="tsum"
    )
    gumbel = LookaheadGumbelWatermarker(key=7, context_width=3, aggregator="tsum")

    agreement = (_chosen(copula, rows) == _chosen(gumbel, rows)).double().mean()

    assert agreement.item() > 0.99


def test_copula_samples_like_gaussian_copula_without_speculation():
    rows = _blocks(32, 1, 97, 3, seed=3)
    expected = GaussianCopulaWatermarker(key=9, context_width=3)._sample_watermarked(
        rows["logits"], rows["contexts"]
    )
    output = LookaheadGaussianCopulaWatermarker(
        key=9, context_width=3
    )._sample_watermarked(rows["logits"], rows["contexts"])

    assert torch.equal(output.token_ids, expected.token_ids)


@pytest.mark.parametrize("algorithm", VARIANTS)
@pytest.mark.parametrize("aggregator", ["tsum", "max", "pi2"])
def test_every_drafted_row_stays_an_exact_sample(algorithm: str, aggregator: str):
    vocab_size, num_paths, draft = 6, 6000, 1
    probabilities = torch.tensor([0.1, 0.35, 0.05, 0.3, 0.12, 0.08])
    logits = torch.stack(
        [
            probabilities.log(),
            3 * torch.randn(vocab_size, generator=torch.Generator().manual_seed(1)),
        ]
    )
    rows = _paths(num_paths, logits, [draft], seed=2)
    watermarker = _variant(algorithm, key=11, context_width=2, aggregator=aggregator)

    chosen = _chosen(watermarker, rows)

    frequencies = torch.bincount(chosen[0::2], minlength=vocab_size) / num_paths
    assert torch.allclose(frequencies, probabilities, atol=0.02), frequencies


@pytest.mark.parametrize("algorithm", VARIANTS)
def test_the_emitted_path_has_the_model_law(algorithm: str):
    """The joint law of the first rejection and the token emitted there is the
    product of the rows' distributions."""
    vocab_size, depth, num_paths = 5, 3, 8000
    generator = torch.Generator().manual_seed(3)
    logits = 1.5 * torch.randn(depth, vocab_size, generator=generator)
    drafts = [int(logits[0].argmax()), int(logits[1].argmax())]
    rows = _paths(num_paths, logits, drafts, seed=4)
    watermarker = _variant(algorithm, key=13, context_width=depth)

    chosen = _chosen(watermarker, rows).view(num_paths, depth)

    laws = logits.double().softmax(-1)
    expected = torch.zeros(depth, vocab_size, dtype=torch.float64)
    observed = torch.zeros(depth, vocab_size, dtype=torch.float64)
    reach = float(num_paths)
    alive = torch.ones(num_paths, dtype=torch.bool)
    for i in range(depth):
        row_law = laws[i].clone()
        if i < depth - 1:
            row_law[drafts[i]] = 0.0  # emitted here only when rejected
        expected[i] = reach * row_law
        emitted = alive & ((chosen[:, i] != drafts[i]) if i < depth - 1 else True)
        observed[i] = torch.bincount(chosen[emitted, i], minlength=vocab_size).double()
        if i < depth - 1:
            reach *= laws[i, drafts[i]].item()
            alive = alive & (chosen[:, i] == drafts[i])
    keep = expected >= 5
    chi2 = ((observed[keep] - expected[keep]) ** 2 / expected[keep]).sum().item()
    dof = int(keep.sum()) - 1
    # chi-square upper tail via the Wilson-Hilferty normal approximation
    z = ((chi2 / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    assert z < 3.5, (chi2, dof, observed, expected)


def _detection_scores(
    watermarker: LookaheadGumbelWatermarker,
    contexts: torch.Tensor,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """Each token's detection score, Exp(1) without the watermark."""

    def score(prf) -> torch.Tensor:
        uniforms = prf.uniform(contexts, tokens.unsqueeze(-1)).squeeze(-1)
        return -torch.log1p(-uniforms.double())

    if isinstance(watermarker, LookaheadDualKeyGumbelWatermarker):
        alpha = watermarker.alpha
        return (1 - alpha) * score(watermarker.prf) + alpha * score(
            watermarker.key_b_prf
        )
    return score(watermarker.prf)


@pytest.mark.parametrize("algorithm", VARIANTS)
def test_lookahead_raises_the_emitted_score_at_equal_acceptance(algorithm: str):
    vocab_size, depth, num_paths = 64, 6, 4000
    generator = torch.Generator().manual_seed(5)
    logits = 3.0 * torch.randn(depth, vocab_size, generator=generator)
    drafts = [int(logits[i].argmax()) for i in range(depth - 1)]
    rows = _paths(num_paths, logits, drafts, seed=6)
    results = {}
    for aggregator in ("pi1", "tsum"):
        watermarker = _variant(
            algorithm, key=17, context_width=depth, aggregator=aggregator
        )
        chosen = _chosen(watermarker, rows).view(num_paths, depth)
        scores = _detection_scores(watermarker, rows["contexts"], chosen.view(-1)).view(
            num_paths, depth
        )
        rejected = chosen != torch.tensor(drafts + [-1])
        first = torch.where(
            rejected.any(1), rejected.to(torch.int8).argmax(1), depth - 1
        )
        emitted = torch.arange(depth).unsqueeze(0) <= first.unsqueeze(1)
        results[aggregator] = (
            (scores * emitted).sum().item() / emitted.sum().item() - 1.0,
            first.double().mean().item(),
        )
    # Acceptance is the same: every row keeps its law.
    assert abs(results["tsum"][1] - results["pi1"][1]) < 0.06, results
    assert results["tsum"][0] > 1.05 * results["pi1"][0], results


def test_copula_verifies_partial_contexts_without_the_watermark():
    rows = _blocks(16, 4, 61, 2, seed=7)
    rows["skip_mask"][:] = False
    rows["contexts"][0, 0] = -1
    watermarker = LookaheadGaussianCopulaWatermarker(key=5, context_width=2)

    output = watermarker.verification_logits(**rows)

    assert torch.equal(output[0], rows["logits"][0])
    assert (output[1:] == 0).sum(-1).eq(1).all()
    assert torch.isneginf(output[1:]).sum(-1).eq(61 - 1).all()
