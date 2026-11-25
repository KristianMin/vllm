# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from tests.watermarking.test_lookahead import _blocks
from vllm.config.watermarking import LOOKAHEAD_PPL_AGGREGATORS, WatermarkConfig
from vllm.v1.watermarking import (
    LookaheadPPLSoftWatermarker,
    PPLSoftWatermarker,
    SupportsVerificationLogits,
    create_watermarker,
)
from vllm.v1.watermarking.lookahead_ppl import aggregate_uniforms
from vllm.v1.watermarking.spec_decode import (
    create_speculative_draft_watermarker,
    speculative_target_watermark_key,
)


def test_config_selects_lookahead_ppl_soft():
    config = WatermarkConfig(
        algorithm="lookahead_ppl_soft",
        key=42,
        context_width=2,
        epsilon=0.1,
        aggregator="gauss",
        aggregator_lambda=0.5,
    )
    watermarker = create_watermarker(config)

    assert isinstance(watermarker, LookaheadPPLSoftWatermarker)
    assert isinstance(watermarker, SupportsVerificationLogits)
    assert (watermarker.aggregator, watermarker.aggregator_lambda) == ("gauss", 0.5)
    assert watermarker.epsilon == 0.1
    # Natively speculative: drafts stay unwatermarked, recovery is not keyed.
    assert config.supports_speculative_decoding
    assert (
        create_speculative_draft_watermarker(
            watermarker, 4, torch.device("cpu"), allow_target_only=False
        )
        is None
    )
    assert speculative_target_watermark_key(config) is None
    with pytest.raises(ValueError):
        WatermarkConfig(algorithm="lookahead_ppl_soft", key=1, aggregator="lin")
    with pytest.raises(ValueError):
        WatermarkConfig(algorithm="lookahead_gumbel", key=1, aggregator="optpath")


@pytest.mark.parametrize("aggregator", ["pi2", "tsum", "max", "gauss"])
def test_aggregators_keep_two_uniforms_uniform(aggregator: str):
    generator = torch.Generator().manual_seed(0)
    u, v = torch.rand(2, 400_000, generator=generator, dtype=torch.float64)
    values = aggregate_uniforms(aggregator, 0.5, u, v).sort().values
    empirical = (
        torch.arange(1, values.numel() + 1, dtype=torch.float64) / values.numel()
    )
    assert (values - empirical).abs().max().item() < 0.005


def test_pi1_is_plain_ppl_soft_and_bonus_rows_never_look_ahead():
    rows = _blocks(64, 5, 97, 3, seed=1)
    plain = (
        PPLSoftWatermarker(key=7, context_width=3)
        ._sample_watermarked(rows["logits"], rows["contexts"])
        .token_ids
    )
    bonus = rows["cu_num_logits"][1:].long() - 1
    skip = rows["skip_mask"] | (rows["contexts"] < 0).any(-1)

    for aggregator in LOOKAHEAD_PPL_AGGREGATORS:
        watermarker = LookaheadPPLSoftWatermarker(
            key=7, context_width=3, aggregator=aggregator
        )
        chosen = watermarker.lookahead_tokens(
            rows["logits"],
            rows["contexts"],
            rows["draft_tokens"],
            rows["cu_num_logits"],
            skip,
        )
        assert torch.equal(chosen[bonus], plain[bonus])
        if aggregator == "pi1":
            assert torch.equal(chosen, plain)


def exact_laws(
    watermarker: LookaheadPPLSoftWatermarker,
    logits: torch.Tensor,
    contexts: torch.Tensor,
) -> torch.Tensor:
    """The ``ppl_soft`` law of every row over the keys, from its exact race
    law: [rows, vocab] float64."""
    order, valid, offsets = watermarker.race_terms(logits, contexts)
    laws = torch.zeros(logits.shape, dtype=torch.float64)
    for j in range(order.shape[1]):
        winner = torch.full((order.shape[0],), j)
        mass = watermarker.winner_joint(offsets, valid, winner).sum(-1)
        laws.scatter_add_(1, order[:, j : j + 1], (mass * valid[:, j]).unsqueeze(1))
    return laws


def _paths(num_paths: int, logits: torch.Tensor, drafts: list[int], seed: int):
    """Verification rows of ``num_paths`` requests along one drafted path:
    fixed per-row logits, fresh keys per request. The context is as wide as the
    path is deep, so every row's context keeps a random token and every row
    of every request gets a fresh key (with a narrower context the deepest rows
    would share one key across requests)."""
    depth = logits.shape[0]
    width = depth
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.empty(num_paths, width + depth - 1, dtype=torch.long)
    tokens[:, :width] = torch.randint(0, 2**30, (num_paths, width), generator=generator)
    if depth > 1:
        tokens[:, width:] = torch.tensor(drafts)
    contexts = torch.stack([tokens[:, i : i + width] for i in range(depth)], 1)
    return {
        "logits": logits.repeat(num_paths, 1),
        "contexts": contexts.reshape(-1, width),
        "draft_tokens": torch.tensor(drafts + [0]).repeat(num_paths).int(),
        "cu_num_logits": torch.arange(0, depth * num_paths + 1, depth).int(),
        "skip_mask": torch.zeros(depth * num_paths, dtype=torch.bool),
    }


@pytest.mark.parametrize("aggregator", ["tsum", "max", "gauss", "pi2", "optpath"])
def test_every_drafted_row_stays_an_exact_sample(aggregator: str):
    """A drafted row picks each token with its ``ppl_soft`` probability, whatever
    the row below sends up."""
    vocab_size, num_paths, draft = 6, 6000, 1
    probabilities = torch.tensor([0.1, 0.35, 0.05, 0.3, 0.12, 0.08])
    logits = torch.stack(
        [
            probabilities.log(),
            3 * torch.randn(vocab_size, generator=torch.Generator().manual_seed(1)),
        ]
    )
    rows = _paths(num_paths, logits, [draft], seed=2)
    watermarker = LookaheadPPLSoftWatermarker(
        key=11, context_width=2, aggregator=aggregator
    )  # depth 2: contexts of width 2

    chosen = watermarker.lookahead_tokens(**rows)

    law = exact_laws(watermarker, rows["logits"][0::2], rows["contexts"][0::2]).mean(0)
    frequencies = torch.bincount(chosen[0::2], minlength=vocab_size) / num_paths
    assert torch.allclose(frequencies.double(), law, atol=0.02), (frequencies, law)


@pytest.mark.parametrize("aggregator", ["tsum", "optpath"])
def test_the_emitted_path_has_the_ppl_soft_law(aggregator: str):
    """The joint law of the first rejection and the token emitted there is the
    product of the rows' ``ppl_soft`` laws (the rows' winners are independent)."""
    vocab_size, depth, num_paths = 5, 3, 8000
    generator = torch.Generator().manual_seed(3)
    logits = 1.5 * torch.randn(depth, vocab_size, generator=generator)
    drafts = [int(logits[0].argmax()), int(logits[1].argmax())]
    rows = _paths(num_paths, logits, drafts, seed=4)
    watermarker = LookaheadPPLSoftWatermarker(
        key=13, context_width=depth, aggregator=aggregator
    )

    chosen = watermarker.lookahead_tokens(**rows).view(num_paths, depth)
    laws = exact_laws(watermarker, rows["logits"], rows["contexts"]).view(
        num_paths, depth, vocab_size
    )

    expected = torch.zeros(depth, vocab_size, dtype=torch.float64)
    observed = torch.zeros(depth, vocab_size, dtype=torch.float64)
    reach = torch.ones(num_paths, dtype=torch.float64)
    alive = torch.ones(num_paths, dtype=torch.bool)
    for i in range(depth):
        row_law = laws[:, i].clone()
        if i < depth - 1:
            row_law[:, drafts[i]] = 0.0  # emitted here only when rejected
        expected[i] = (reach.unsqueeze(1) * row_law).sum(0)
        emitted = alive & ((chosen[:, i] != drafts[i]) if i < depth - 1 else True)
        observed[i] = torch.bincount(chosen[emitted, i], minlength=vocab_size).double()
        if i < depth - 1:
            reach = reach * laws[:, i, drafts[i]]
            alive = alive & (chosen[:, i] == drafts[i])
    keep = expected >= 5
    chi2 = ((observed[keep] - expected[keep]) ** 2 / expected[keep]).sum().item()
    dof = int(keep.sum()) - 1
    # chi-square upper tail via the Wilson-Hilferty normal approximation
    z = ((chi2 / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    assert z < 3.5, (chi2, dof, observed, expected)


def test_lookahead_raises_the_emitted_score_at_equal_acceptance():
    vocab_size, depth, num_paths = 64, 6, 4000
    generator = torch.Generator().manual_seed(5)
    logits = 6.0 * torch.randn(depth, vocab_size, generator=generator)
    drafts = [int(logits[i].argmax()) for i in range(depth - 1)]
    rows = _paths(num_paths, logits, drafts, seed=6)
    results = {}
    for aggregator in ("pi1", "tsum", "optpath"):
        watermarker = LookaheadPPLSoftWatermarker(
            key=17, context_width=depth, aggregator=aggregator
        )
        chosen = watermarker.lookahead_tokens(**rows).view(num_paths, depth)
        scores = watermarker.scores(rows["contexts"], vocab_size).view(
            num_paths, depth, vocab_size
        )
        own = scores.gather(2, chosen.unsqueeze(-1)).squeeze(-1).double()
        drafted = torch.tensor(drafts + [-1])
        rejected = chosen != drafted
        first = torch.where(
            rejected.any(1), rejected.to(torch.int8).argmax(1), depth - 1
        )
        emitted = torch.arange(depth).unsqueeze(0) <= first.unsqueeze(1)
        results[aggregator] = (
            (own * emitted).sum().item() / emitted.sum().item(),
            first.double().mean().item(),
        )
    excess = {name: score - 15.0 for name, (score, _) in results.items()}
    # Acceptance is the same for every aggregator: the rows keep their law.
    assert abs(results["tsum"][1] - results["pi1"][1]) < 0.06, results
    assert abs(results["optpath"][1] - results["pi1"][1]) < 0.06, results
    assert excess["tsum"] > 1.03 * excess["pi1"], excess
    assert excess["optpath"] > 1.03 * excess["pi1"], excess


def test_verification_logits_are_point_masses_outside_the_skip_mask():
    rows = _blocks(16, 4, 61, 2, seed=3)
    rows["contexts"][0, 0] = -1  # a row without a full context keeps its logits
    watermarker = LookaheadPPLSoftWatermarker(key=5, context_width=2)
    skip = rows["skip_mask"] | (rows["contexts"] < 0).any(-1)
    chosen = watermarker.lookahead_tokens(
        rows["logits"],
        rows["contexts"],
        rows["draft_tokens"],
        rows["cu_num_logits"],
        skip,
    )

    output = watermarker.verification_logits(**rows)

    assert bool(skip[0])
    assert torch.equal(output[skip], rows["logits"][skip])
    assert torch.equal(output[~skip].argmax(-1), chosen[~skip])
    assert (output[~skip] == 0).sum(-1).eq(1).all()
    assert torch.isneginf(output[~skip]).sum(-1).eq(61 - 1).all()
