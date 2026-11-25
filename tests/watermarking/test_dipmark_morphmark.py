# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: D200, D212, E501, SIM300
# (The TextSeal reference code below is copied verbatim, with `fmt: off`.)

"""DiPMark and MorphMark, ported from Meta's TextSeal: reweighting math,
sampling, detection, configuration, speculative decoding, and equivalence with
TextSeal's own functions on identical inputs."""

import itertools
import math
from types import SimpleNamespace

import pytest
import torch

import vllm.config.vllm as vllm_config_module
from vllm.config import VllmConfig
from vllm.config.watermarking import WatermarkConfig
from vllm.platforms import current_platform
from vllm.v1.watermarking import (
    DiPMarkWatermarkDetector,
    DiPMarkWatermarker,
    MorphMarkWatermarkDetector,
    MorphMarkWatermarker,
    RedGreenWatermarkDetector,
    create_watermarker,
)
from vllm.v1.watermarking import dipmark_morphmark as dipmark_morphmark_module
from vllm.v1.watermarking.dipmark_morphmark import (
    MORPHMARK_EPS,
    dipmark_permutation,
    dipmark_reweight,
    morphmark_logits,
    morphmark_strength,
)
from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler
from vllm.v1.watermarking.red_green import _binomial_survival
from vllm.v1.watermarking.spec_decode import (
    create_speculative_draft_watermarker,
    speculative_target_watermark_key,
)
from vllm.v1.watermarking.watermarker import (
    RandomSampler,
    SupportsSpeculativeDecoding,
    SupportsVerificationLogits,
)

ALGORITHMS = ("dipmark", "morphmark")
requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)


def _argmax_sampler(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


def _multinomial_sampler(seed: int):
    generator = torch.Generator().manual_seed(seed)

    def sample(logits: torch.Tensor) -> torch.Tensor:
        return torch.multinomial(logits.softmax(-1), 1, generator=generator).squeeze(-1)

    return sample


def _watermarker(algorithm: str, **kwargs):
    cls = DiPMarkWatermarker if algorithm == "dipmark" else MorphMarkWatermarker
    return cls(key=kwargs.pop("key", 42), **kwargs)


def _random_contexts(num_rows: int, width: int, vocab_size: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(-1, vocab_size, (num_rows, width), generator=generator)


# ---------------------------------------------------------------------------
# Configuration and factory


def test_config_selects_dipmark():
    config = WatermarkConfig(
        algorithm="dipmark", key=42, context_width=2, dipmark_alpha=0.3, gamma=0.25
    )
    watermarker = create_watermarker(config)

    assert isinstance(watermarker, DiPMarkWatermarker)
    assert watermarker.context_width == 2
    assert (watermarker.alpha, watermarker.gamma) == (0.3, 0.25)
    assert watermarker.prf.key == 42
    assert not config.supports_speculative_decoding


def test_config_selects_morphmark():
    config = WatermarkConfig(
        algorithm="morphmark",
        key=42,
        context_width=3,
        morphmark_k=2.0,
        morphmark_p0=0.1,
        gamma=0.4,
    )
    watermarker = create_watermarker(config)

    assert isinstance(watermarker, MorphMarkWatermarker)
    assert watermarker.context_width == 3
    assert (watermarker.k, watermarker.p_0, watermarker.gamma) == (2.0, 0.1, 0.4)
    assert not config.supports_speculative_decoding


def test_defaults_are_textseals():
    """TextSeal's config.py: alpha 0.2, k_morphmark 1.30, p_0 0.15, gamma 0.5."""
    dipmark = create_watermarker(WatermarkConfig(algorithm="dipmark", key=1))
    morphmark = create_watermarker(WatermarkConfig(algorithm="morphmark", key=1))

    assert (dipmark.alpha, dipmark.gamma, dipmark.context_width) == (0.2, 0.5, 4)
    assert (morphmark.k, morphmark.p_0, morphmark.gamma) == (1.3, 0.15, 0.5)
    # The detectors default to the same green-list fraction.
    assert DiPMarkWatermarkDetector(key=1).gamma == 0.5
    assert MorphMarkWatermarkDetector(key=1).gamma == 0.5


@pytest.mark.parametrize(
    ("algorithm", "overrides"),
    [
        ("dipmark", {"dipmark_alpha": 0.0}),
        ("dipmark", {"dipmark_alpha": -0.1}),
        ("dipmark", {"dipmark_alpha": 0.51}),
        ("dipmark", {"gamma": 0.0}),
        ("dipmark", {"gamma": 1.0}),
        ("morphmark", {"morphmark_k": 0.0}),
        ("morphmark", {"morphmark_k": -1.0}),
        ("morphmark", {"morphmark_p0": -0.01}),
        ("morphmark", {"morphmark_p0": 1.0}),
        ("morphmark", {"gamma": 1.0}),
    ],
)
def test_config_rejects_invalid_parameters(algorithm: str, overrides: dict):
    with pytest.raises(ValueError):
        WatermarkConfig(algorithm=algorithm, key=42, **overrides)


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (DiPMarkWatermarker, {"alpha": 0.0}),
        (DiPMarkWatermarker, {"alpha": 0.6}),
        (DiPMarkWatermarker, {"gamma": 1.0}),
        (DiPMarkWatermarker, {"context_width": 0}),
        (MorphMarkWatermarker, {"k": 0.0}),
        (MorphMarkWatermarker, {"p_0": 1.0}),
        (MorphMarkWatermarker, {"p_0": -0.5}),
        (MorphMarkWatermarker, {"gamma": 0.0}),
    ],
)
def test_constructors_reject_invalid_parameters(cls, kwargs: dict):
    with pytest.raises(ValueError):
        cls(key=42, **kwargs)


def test_config_accepts_the_parameter_bounds():
    WatermarkConfig(algorithm="dipmark", key=42, dipmark_alpha=0.5)
    WatermarkConfig(algorithm="morphmark", key=42, morphmark_p0=0.0)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_watermarker_contract(algorithm: str):
    watermarker = create_watermarker(
        WatermarkConfig(algorithm=algorithm, key=42, context_width=4)
    )
    logits = torch.zeros(2, 128)
    contexts = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]])

    first = watermarker.sample(logits, contexts, _argmax_sampler)
    second = watermarker.sample(logits, contexts, _argmax_sampler)

    assert first.token_ids.shape == (2,)
    assert first.logits.shape == logits.shape
    assert torch.equal(first.token_ids, second.token_ids)
    assert not watermarker.stateful


# ---------------------------------------------------------------------------
# DiPMark reweighting


def test_dipmark_reweight_matches_the_closed_form():
    probs = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float64)
    identity = torch.arange(4).unsqueeze(0)

    # F = (.1, .3, .6, 1); max(F - .25, 0) + max(F - .75, 0) = (0, .05, .35, 1).
    reweighted = dipmark_reweight(probs, identity, alpha=0.25)

    torch.testing.assert_close(
        reweighted, torch.tensor([[0.0, 0.05, 0.30, 0.65]], dtype=torch.float64)
    )
    # The permutation only reorders the cumulative distribution.
    reverse = identity.flip(-1)
    # Reverse order: F = (.4, .7, .9, 1) over tokens 3, 2, 1, 0.
    torch.testing.assert_close(
        dipmark_reweight(probs, reverse, alpha=0.25),
        torch.tensor([[0.20, 0.35, 0.30, 0.15]], dtype=torch.float64),
    )


@pytest.mark.parametrize("alpha", [0.05, 0.2, 0.37, 0.5])
def test_dipmark_is_unbiased_over_permutations(alpha: float):
    """Averaged over every permutation of a small vocabulary, the reweighted
    law is the model's law (DiPMark's unbiasedness), exactly."""
    probs = torch.tensor([0.05, 0.0, 0.4, 0.25, 0.3], dtype=torch.float64)
    permutations = torch.tensor(list(itertools.permutations(range(5))))

    reweighted = dipmark_reweight(
        probs.expand(len(permutations), -1), permutations, alpha
    )

    assert torch.all(reweighted >= 0)
    torch.testing.assert_close(
        reweighted.sum(-1), torch.ones(len(permutations), dtype=torch.float64)
    )
    torch.testing.assert_close(reweighted.mean(0), probs, atol=1e-13, rtol=0)
    # A zero-probability token stays at zero under every permutation.
    assert torch.all(reweighted[:, 1] == 0)


def test_dipmark_averages_a_permutation_and_its_reverse_to_the_law():
    generator = torch.Generator().manual_seed(0)
    probs = torch.softmax(3 * torch.randn(6, 300, generator=generator), -1).double()
    probs = probs / probs.sum(-1, keepdim=True)  # a law in float64 too
    permutations = torch.stack(
        [torch.randperm(300, generator=generator) for _ in range(6)]
    )

    forward = dipmark_reweight(probs, permutations, 0.3)
    backward = dipmark_reweight(probs, permutations.flip(-1), 0.3)

    torch.testing.assert_close((forward + backward) / 2, probs, atol=1e-13, rtol=0)


def test_dipmark_is_unbiased_over_keys():
    """Over contexts, the keyed permutations average the reweighted law back
    to the model's law (Monte Carlo with the Philox PRF)."""
    num_rows = 40_000
    logits = torch.tensor([[1.5, 0.2, -0.3, 0.8, -1.0, 0.0]]).expand(num_rows, -1)
    contexts = _random_contexts(num_rows, 2, 50_000, seed=1)
    watermarker = DiPMarkWatermarker(key=7, context_width=2, alpha=0.4)

    reweighted = watermarker.watermark_logits(logits, contexts).double().softmax(-1)
    mean = reweighted.mean(0)
    standard_error = reweighted.std(0) / math.sqrt(num_rows)
    probs = logits[0].double().softmax(-1)

    assert torch.all((mean - probs).abs() <= 5 * standard_error + 1e-6), (
        mean,
        probs,
        standard_error,
    )
    # The single rows are far from the law: the watermark is there.
    assert (reweighted - probs).abs().sum(-1).mean() > 0.3


def test_dipmark_permutation_puts_the_green_tokens_last():
    watermarker = DiPMarkWatermarker(key=3, context_width=2, gamma=0.3)
    contexts = _random_contexts(5, 2, 1000, seed=2)
    uniforms = watermarker.uniforms(contexts, 1000)
    permutation = dipmark_permutation(uniforms)
    green = watermarker.green_list(contexts, 1000)

    ordered_green = green.gather(-1, permutation)
    num_green = green.sum(-1, keepdim=True)
    positions = torch.arange(1000).expand(5, -1)
    # Red tokens first, then every green token.
    assert torch.equal(ordered_green, positions >= 1000 - num_green)
    # Descending PRF values, ties by token id.
    ordered = uniforms.gather(-1, permutation)
    assert torch.all(ordered[:, :-1] >= ordered[:, 1:])


def test_dipmark_permutation_breaks_ties_by_token_id():
    uniforms = torch.tensor([[0.5, 0.9, 0.5, 0.1, 0.9]])

    assert dipmark_permutation(uniforms).tolist() == [[1, 4, 0, 2, 3]]


@pytest.mark.parametrize("alpha", [0.1, 0.2, 0.5])
def test_dipmark_moves_alpha_onto_the_green_tokens_of_a_flat_law(alpha: float):
    """On a flat law, the green tokens (the last g = #green / V of the order)
    get mass 1 - max(1 - g - alpha, 0) - max(alpha - g, 0) after the
    reweighting (g + alpha when alpha <= g <= 1 - alpha), instead of g."""
    vocab_size = 1000
    watermarker = DiPMarkWatermarker(key=11, context_width=3, alpha=alpha)
    contexts = _random_contexts(16, 3, 50_000, seed=3)
    reweighted = watermarker.watermark_logits(
        torch.zeros(16, vocab_size), contexts
    ).softmax(-1)
    green = watermarker.green_list(contexts, vocab_size)
    green_share = green.double().mean(-1)

    green_mass = (reweighted.double() * green).sum(-1)

    expected = (
        1 - (1 - green_share - alpha).clamp_min(0) - (alpha - green_share).clamp_min(0)
    )
    torch.testing.assert_close(green_mass, expected, atol=1e-5, rtol=0)
    assert torch.all(green_mass > green_share + alpha / 2)


# ---------------------------------------------------------------------------
# MorphMark reweighting


def _textseal_morphmark_formula(probs, green, k, p_0):
    """TextSeal's MorphMark update (generator.py, MorphMarkGenerator.sample_next)
    in float64, for P_G > 0."""
    eps = 1e-6
    green_mass = (probs * green).sum(-1, keepdim=True)
    strength = torch.where(green_mass > p_0, k * green_mass, eps).clamp(max=1 - eps)
    factor = torch.where(
        green,
        1.0 + strength * (1.0 - green_mass) / green_mass,
        1.0 - strength,
    )
    updated = probs * factor
    return updated / updated.sum(-1, keepdim=True)


def test_morphmark_strength_is_linear_with_a_floor_and_a_ceiling():
    green_mass = torch.tensor([0.0, 0.1, 0.15, 0.2, 0.5, 0.75, 0.9, 1.0])

    strength = morphmark_strength(green_mass, k=1.3, p_0=0.15)

    expected = torch.tensor(
        [
            MORPHMARK_EPS,
            MORPHMARK_EPS,
            MORPHMARK_EPS,  # P_G == p_0 is not above it
            0.26,
            0.65,
            0.975,
            1 - MORPHMARK_EPS,
            1 - MORPHMARK_EPS,
        ]
    )
    torch.testing.assert_close(strength, expected)


@pytest.mark.parametrize(("k", "p_0"), [(1.3, 0.15), (0.5, 0.0), (3.0, 0.4)])
def test_morphmark_matches_the_textseal_formula(k: float, p_0: float):
    generator = torch.Generator().manual_seed(4)
    logits = 2 * torch.randn(64, 200, generator=generator)
    green = torch.rand(64, 200, generator=generator) < 0.5
    # Rows from a low to a high green mass.
    logits += torch.linspace(-6, 6, 64).unsqueeze(-1) * green

    reweighted = morphmark_logits(logits, green, k, p_0).double().softmax(-1)
    expected = _textseal_morphmark_formula(logits.double().softmax(-1), green, k, p_0)

    torch.testing.assert_close(reweighted, expected, atol=1e-6, rtol=1e-5)


def test_morphmark_moves_r_of_the_red_mass_and_keeps_proportions():
    generator = torch.Generator().manual_seed(5)
    logits = torch.randn(8, 50, generator=generator, dtype=torch.float64).float()
    green = torch.rand(8, 50, generator=generator) < 0.4
    probs = logits.double().softmax(-1)
    green_mass = (probs * green).sum(-1)

    reweighted = morphmark_logits(logits, green, 1.3, 0.15).double().softmax(-1)

    strength = morphmark_strength(green_mass, 1.3, 0.15)
    new_green_mass = (reweighted * green).sum(-1)
    torch.testing.assert_close(
        new_green_mass, green_mass + strength * (1 - green_mass), atol=1e-6, rtol=0
    )
    # Within each group, the law's proportions are kept.
    ratio = reweighted / probs
    for row in range(8):
        for group in (green[row], ~green[row]):
            values = ratio[row][group]
            torch.testing.assert_close(
                values, values[0].expand_as(values), atol=0, rtol=1e-5
            )


def test_morphmark_leaves_a_low_green_mass_almost_unchanged():
    """P_G <= p_0: strength 1e-6, so the law moves by at most 1e-6 in total
    variation."""
    probs = torch.full((1, 100), 1.0 / 100)
    green = torch.zeros(1, 100, dtype=torch.bool)
    green[0, :10] = True  # P_G = 0.1 <= 0.15

    reweighted = morphmark_logits(probs.log(), green, 1.3, 0.15).double().softmax(-1)

    total_variation = 0.5 * (reweighted - probs.double()).abs().sum()
    assert total_variation <= 2e-6


def test_morphmark_keeps_single_colored_supports():
    logits = torch.randn(3, 16)
    green = torch.zeros(3, 16, dtype=torch.bool)
    green[0] = True  # all green
    green[2, :8] = True
    logits[2, :8] = -torch.inf  # the green tokens are filtered out: all red

    output = morphmark_logits(logits, green, 1.3, 0.15)

    assert torch.equal(output[0], logits[0])
    assert torch.equal(output[1], logits[1])
    assert torch.equal(output[2], logits[2])


@pytest.mark.parametrize("green_logit", [-1e4, -1e30])
def test_morphmark_handles_a_vanishing_green_mass(green_logit: float):
    """Green logits far below the red ones: P_G underflows to 0, the green
    tokens get the floor's 1e-6 of the mass, and nothing overflows."""
    logits = torch.zeros(1, 8)
    green = torch.zeros(1, 8, dtype=torch.bool)
    green[0, :2] = True
    logits[0, :2] = green_logit

    reweighted = morphmark_logits(logits, green, 1.3, 0.15)

    assert torch.isfinite(reweighted).all()
    probs = reweighted.double().softmax(-1)
    # At -1e30 float32 cannot tell the two green tokens' share of their group
    # (log 2 is below its resolution), so the mass is only of the right order.
    tolerance = 1e-2 if green_logit > -1e5 else 1.0
    assert probs[0, :2].sum().item() == pytest.approx(MORPHMARK_EPS, rel=tolerance)


# ---------------------------------------------------------------------------
# Watermarker behavior


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_watermark_logits_skip_rows_and_keep_filtered_tokens(algorithm, dtype):
    watermarker = _watermarker(algorithm, context_width=2)
    logits = (2 * torch.randn(3, 64)).to(dtype)
    logits[:, :4] = -torch.inf
    contexts = torch.tensor([[1, 2], [3, 4], [5, 6]])
    skip_mask = torch.tensor([False, True, False])

    output = watermarker.watermark_logits(logits, contexts, skip_mask)
    unskipped = watermarker.watermark_logits(logits, contexts)

    assert output.dtype == torch.float32
    assert torch.equal(output[1], logits[1].float())
    assert torch.equal(output[[0, 2]], unskipped[[0, 2]])
    assert not torch.equal(output[0], logits[0].float())
    # Filtered tokens stay filtered and the rows stay distributions.
    assert torch.isneginf(output[:, :4]).all()
    assert not torch.isnan(output).any()


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_watermark_logits_in_chunks_match_one_chunk(algorithm, monkeypatch):
    watermarker = _watermarker(algorithm, context_width=2)
    logits = 2 * torch.randn(10, 32)
    contexts = _random_contexts(10, 2, 1000, seed=6)
    whole = watermarker.watermark_logits(logits, contexts)

    monkeypatch.setattr(dipmark_morphmark_module, "_MAX_CHUNK_ELEMENTS", 96)
    chunked = watermarker.watermark_logits(logits, contexts)

    torch.testing.assert_close(chunked, whole, atol=1e-6, rtol=0)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_sample_draws_from_the_watermarked_logits(algorithm):
    watermarker = _watermarker(algorithm, context_width=1)
    logits = torch.randn(4, 32)
    contexts = torch.tensor([[7], [8], [9], [10]])
    skip_mask = torch.tensor([False, True, False, False])

    sample = watermarker.sample(logits, contexts, _argmax_sampler, skip_mask)

    expected = watermarker.watermark_logits(logits, contexts, skip_mask).argmax(-1)
    assert torch.equal(sample.token_ids, expected)
    # Processed logprobs see the logits before the watermark.
    assert sample.logits is logits


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_sample_requires_a_random_sampler(algorithm):
    with pytest.raises(ValueError, match="random sampler"):
        _watermarker(algorithm).sample(torch.zeros(1, 8), torch.zeros(1, 4))


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_sampled_tokens_follow_the_watermarked_law(algorithm):
    """Many draws at one context follow the reweighted law (chi-squared)."""
    num_draws, vocab_size = 20_000, 8
    watermarker = _watermarker(algorithm, context_width=2)
    logits = torch.tensor([[1.0, 0.5, 0.0, -0.5, 0.3, 0.8, -1.0, 0.1]])
    contexts = torch.tensor([[3, 5]])
    law = watermarker.watermark_logits(logits, contexts).double().softmax(-1)[0]

    sample = watermarker.sample(
        logits.expand(num_draws, -1),
        contexts.expand(num_draws, -1),
        _multinomial_sampler(0),
    )
    counts = torch.bincount(sample.token_ids, minlength=vocab_size).double()

    support = law > 0
    expected = num_draws * law[support]
    statistic = ((counts[support] - expected) ** 2 / expected).sum().item()
    # chi-squared with < 8 degrees of freedom: P(X > 30) < 1e-4.
    assert statistic < 30
    assert counts[~support].sum() == 0


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_sampling_favors_the_green_list(algorithm):
    """Over contexts, the sampled tokens are green more often than gamma."""
    num_rows, vocab_size = 4000, 500
    watermarker = _watermarker(algorithm, context_width=2)
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(num_rows, vocab_size, generator=generator)
    contexts = _random_contexts(num_rows, 2, 50_000, seed=8)

    sample = watermarker.sample(logits, contexts, _multinomial_sampler(1))
    green = watermarker.green_list(contexts, vocab_size)
    green_rate = green[torch.arange(num_rows), sample.token_ids].double().mean()

    # Expected green probability under the watermark, from the law itself.
    law = watermarker.watermark_logits(logits, contexts).double().softmax(-1)
    expected = (law * green).sum(-1).mean()
    assert green_rate.item() == pytest.approx(expected.item(), abs=0.03)
    assert expected > 0.6


# ---------------------------------------------------------------------------
# Detection


def _generate(watermarker, num_tokens: int, vocab_size: int, seed: int):
    """Sample a sequence from a toy model (logits keyed on the previous
    token) under the watermark, building each context as the GPU sampler
    does (-1 before the first generated tokens)."""
    generator = torch.Generator().manual_seed(seed)
    table = torch.randn(64, vocab_size, generator=generator)
    width = watermarker.context_width
    token_ids: list[int] = []
    for _ in range(num_tokens):
        context = ([-1] * width + token_ids)[len(token_ids) :]
        previous = token_ids[-1] if token_ids else 0
        logits = watermarker.watermark_logits(
            table[previous % 64].unsqueeze(0), torch.tensor([context])
        )
        token_ids.append(
            int(torch.multinomial(logits.softmax(-1), 1, generator=generator))
        )
    return token_ids


@pytest.mark.parametrize(
    ("algorithm", "detector_cls"),
    [("dipmark", DiPMarkWatermarkDetector), ("morphmark", MorphMarkWatermarkDetector)],
)
def test_detector_is_the_red_green_test_on_the_generation_green_list(
    algorithm, detector_cls
):
    watermarker = _watermarker(algorithm, context_width=2, gamma=0.4)
    detector = detector_cls(key=42, context_width=2, gamma=0.4)
    contexts = _random_contexts(200, 2, 1000, seed=9)
    targets = torch.randint(0, 1000, (200,), generator=torch.Generator().manual_seed(9))

    scores = detector._score_tokens(contexts, targets)
    green = watermarker.green_list(contexts, 1000)[torch.arange(200), targets]

    assert isinstance(detector, RedGreenWatermarkDetector)
    assert torch.equal(scores, green.double())
    token_ids = targets.tolist()
    expected = RedGreenWatermarkDetector(key=42, context_width=2, gamma=0.4).detect(
        token_ids
    )
    assert detector.detect(token_ids) == expected


@pytest.mark.parametrize(
    "detector_cls", [DiPMarkWatermarkDetector, MorphMarkWatermarkDetector]
)
def test_detection_null_p_values_are_uniform(detector_cls):
    """Without the watermark, P(p <= a) <= a: the exact binomial tail is
    valid (conservative only by the binomial's discreteness)."""
    num_texts, num_tokens, vocab_size = 2000, 100, 1000
    detector = detector_cls(key=42, context_width=2)
    generator = torch.Generator().manual_seed(10)
    # Zipf-like unwatermarked text, with repeated tokens and contexts.
    law = 1.0 / torch.arange(1, vocab_size + 1).double()
    texts = torch.multinomial(
        law / law.sum(), num_texts * num_tokens, replacement=True, generator=generator
    )

    p_values = torch.tensor(
        [
            detector.detect(text.tolist()).p_value
            for text in texts.view(num_texts, num_tokens)
        ]
    )

    for level in (0.01, 0.05, 0.1, 0.25, 0.5):
        rate = (p_values <= level).double().mean().item()
        margin = 3 * math.sqrt(level * (1 - level) / num_texts)
        assert rate <= level + margin, (level, rate)
        # Not degenerate: close to the level, below it by the discreteness.
        assert rate >= level / 2 - margin, (level, rate)


@pytest.mark.parametrize(
    ("algorithm", "detector_cls", "kwargs"),
    [
        ("dipmark", DiPMarkWatermarkDetector, {}),
        ("dipmark", DiPMarkWatermarkDetector, {"alpha": 0.5}),
        ("morphmark", MorphMarkWatermarkDetector, {}),
    ],
)
def test_watermarked_text_is_detected(algorithm, detector_cls, kwargs):
    watermarker = _watermarker(algorithm, context_width=2, **kwargs)
    token_ids = _generate(watermarker, 300, 1000, seed=11)

    watermarked = detector_cls(key=42, context_width=2).detect(token_ids)
    other_key = detector_cls(key=43, context_width=2).detect(token_ids)
    unwatermarked = detector_cls(key=42, context_width=2).detect(
        _generate(_watermarker(algorithm, key=99, context_width=2), 300, 1000, seed=11)
    )

    assert watermarked.is_watermarked
    assert watermarked.p_value < 1e-5
    assert watermarked.p_value == pytest.approx(
        _binomial_survival(round(watermarked.score), watermarked.num_scored_tokens, 0.5)
    )
    assert not other_key.is_watermarked
    assert not unwatermarked.is_watermarked


# ---------------------------------------------------------------------------
# Speculative decoding: target-only, verified on the watermarked target law


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_target_only_speculative_decoding_keys_only_the_target(algorithm):
    config = WatermarkConfig(
        algorithm=algorithm, key=42, allow_target_only_watermarking=True
    )
    watermarker = create_watermarker(config)

    draft = create_speculative_draft_watermarker(
        watermarker,
        max_num_reqs=2,
        device=torch.device("cpu"),
        allow_target_only=True,
    )

    assert draft is None
    assert isinstance(watermarker, SupportsVerificationLogits)
    assert not isinstance(watermarker, SupportsSpeculativeDecoding)
    # Verification samples from the watermarked target logits: no keyed recovery.
    assert speculative_target_watermark_key(config) is None


def _vllm_config(watermark_config: WatermarkConfig) -> VllmConfig:
    config = object.__new__(VllmConfig)
    config.watermark_config = watermark_config
    config.speculative_config = SimpleNamespace(
        method="mtp",
        draft_sample_method="probabilistic",
        rejection_sample_method="standard",
        parallel_drafting=False,
    )
    return config


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_speculative_decoding_requires_target_only(algorithm):
    config = _vllm_config(WatermarkConfig(algorithm=algorithm, key=42))

    with pytest.raises(ValueError, match="allow_target_only_watermarking=true"):
        config._check_supports_watermarking()


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_target_only_is_not_reported_as_diluted(algorithm, monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(
        vllm_config_module.logger,
        "warning_once",
        lambda message, *args, **kwargs: warnings.append(message),
    )
    config = _vllm_config(
        WatermarkConfig(
            algorithm=algorithm, key=42, allow_target_only_watermarking=True
        )
    )

    assert config._check_supports_watermarking()
    assert not any("Target-only watermarking leaves" in w for w in warnings)


# Request 0 verifies drafts [9, 7] and then its bonus row; request 1 opted out.
_VERIFY_ROWS = {
    "expanded_idx_mapping": torch.tensor([0, 0, 0, 1]),
    "expanded_local_pos": torch.tensor([0, 1, 2, 0]),
    "draft_sampled": torch.tensor([8, 9, 7, 6]),
    "cu_num_logits": torch.tensor([0, 3, 4]),
}
_VERIFY_CONTEXTS = torch.tensor([[7, 8], [8, 9], [9, 7], [5, 6]])


def _verification_sampler(watermarker) -> GPUWatermarkSampler:
    sampler = object.__new__(GPUWatermarkSampler)
    sampler.watermarker = watermarker
    sampler.deduplicate_contexts = "single_turn"
    sampler.deduplicate_contexts_max_history = None
    sampler.num_speculative_tokens = 2
    sampler.use_fp64_gumbel = False
    sampler.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(
            gpu=torch.tensor([[100, 7, 8, 0, 0], [100, 101, 5, 6, 0]])
        ),
        prompt_len=SimpleNamespace(gpu=torch.tensor([1, 2])),
        total_len=SimpleNamespace(gpu=torch.tensor([3, 4])),
    )
    sampler.watermarking = SimpleNamespace(gpu=torch.tensor([True, False]))
    sampler.sampling_states = SimpleNamespace(
        temperature=SimpleNamespace(gpu=torch.tensor([1.0, 1.0])),
        seeds=SimpleNamespace(gpu=torch.zeros(2, dtype=torch.int64)),
    )
    return sampler


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_gpu_sampler_verifies_against_the_watermarked_logits(algorithm):
    """Each verification row (contexts ending with the drafts before it) is
    reweighted; the opted-out request keeps its logits."""
    watermarker = _watermarker(algorithm, context_width=2)
    logits = torch.randn(4, 16)

    output = _verification_sampler(watermarker).watermark_verification_logits(
        logits, **_VERIFY_ROWS
    )

    expected = watermarker.watermark_logits(
        logits, _VERIFY_CONTEXTS, torch.tensor([False, False, False, True])
    )
    assert torch.equal(output, expected)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_rejection_sampling_against_the_verification_law_emits_it(algorithm):
    """Standard speculative sampling of a draft against the verification law
    q (accept with min(1, q/p), else draw from (q - p)+) emits tokens that
    follow q exactly: the reweighted law, so no dilution."""
    watermarker = _watermarker(algorithm, context_width=2)
    generator = torch.Generator().manual_seed(12)
    logits = torch.randn(1, 12, generator=generator)
    contexts = torch.tensor([[4, 2]])
    target = (
        watermarker.verification_logits(
            logits,
            contexts,
            torch.zeros(1),
            torch.tensor([0, 1]),
            torch.tensor([False]),
        )
        .double()
        .softmax(-1)[0]
    )
    draft = torch.randn(12, generator=generator).double().softmax(-1)

    # Exact output law of one speculative step.
    accept = torch.minimum(torch.ones_like(draft), target / draft)
    residual = (target - draft).clamp_min(0)
    rejected = 1 - (draft * accept).sum()
    output = draft * accept + rejected * residual / residual.sum()

    torch.testing.assert_close(output, target, atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# Equivalence with Meta's TextSeal on identical inputs.
#
# The classes below hold verbatim copies (up to trailing whitespace) of
# TextSeal's methods, from textseal/watermarking/generator.py of
# https://github.com/facebookresearch/textseal (Apache-2.0, Copyright (c) Meta
# Platforms, Inc. and affiliates):
#   _TextSealMorphMark.sample_next  = MorphMarkGenerator.sample_next, lines 459-528
#   _TextSealDipmark.dip_reweight_probs = DipmarkGenerator.dip_reweight_probs, lines 542-572
#   _TextSealDipmark.sample_next    = DipmarkGenerator.sample_next, lines 574-619
# Importing generator.py pulls in transformers models and TextSeal's own PRF,
# so the methods are copied. TextSeal's keyed hash is replaced by the native
# PRF through `score_all_next_tokens` below, which feeds the methods vLLM's
# Philox uniforms with TextSeal's binary / uniform switch.


def score_all_next_tokens(wm_windows, wm_args, vocab_size):
    """Stands in for TextSeal's core.score_all_next_tokens (core.py lines
    563-580) and score_listed_tokens (lines 521-560): the scores of every
    token, from the native PRF uniforms in `wm_args.uniforms` instead of
    TextSeal's 13-bit hash. method "binary" (greenlist, MorphMark and every
    detector of the three) is prf_binary's `(u < gamma)`, as float; "uniform"
    (DiPMark generation) is `u`."""
    uniforms = wm_args.uniforms[:, :vocab_size]
    method = wm_args.method.lower()
    if method.startswith("bin"):  # binary
        return (uniforms < wm_args.gamma).to(torch.int8).float()
    return uniforms.float()


class _TextSealMorphMark:
    def __init__(self, uniforms, k_morphmark=1.30, p_0=0.15, gamma=0.5):
        self.k_morphmark = k_morphmark
        self.p_0 = p_0
        self.gamma = gamma
        # build_generator: MorphMark samples with the binary method.
        self.wm_args = SimpleNamespace(method="binary", gamma=gamma, uniforms=uniforms)

    # fmt: off
    def sample_next(
        self,
        logits: torch.FloatTensor,
        ngram_tokens: torch.LongTensor,
        temperature: float = 0.8,
        top_p: float = 0.95,
        return_probs: bool = False,
    ) -> torch.LongTensor:
        """
        MorphMark sampling with adaptive watermark strength based on green mass.
        """
        eps = 1e-6
        bsz, vocab_size = logits.shape

        if temperature == 0.0:
            next_token = torch.argmax(logits, dim=-1)
            if return_probs:
                raise NotImplementedError("return_probs=True not implemented for argmax sampling")
            return next_token.reshape(-1)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)

        # Identify green list and calculate P_G
        green_mask = score_all_next_tokens(ngram_tokens, self.wm_args, vocab_size).bool()

        P_G = (probs * green_mask.float()).sum(dim=-1, keepdim=True) # b 1
        # P_G = torch.clamp(P_G, min=1e-6, max=1.0 - 1e-6)

        # Calculate adaptive strength r
        r = self.k_morphmark * P_G
        # r = torch.exp(self.k_morphmark * P_G) - 1.0
        r = torch.where(P_G > self.p_0, r, torch.tensor(eps, device=probs.device))
        r = torch.clamp(r, max=1-eps)

        # Update probability vector
        green_factor = 1.0 + (r * (1.0 - P_G) / P_G)
        red_factor = 1.0 - r

        green_factor = green_factor.expand_as(probs)
        red_factor = red_factor.expand_as(probs)

        modifiers = torch.where(green_mask, green_factor, red_factor)
        wm_probs = probs * modifiers
        wm_probs = wm_probs / wm_probs.sum(dim=-1, keepdim=True)

        # Sample with top-p
        if top_p < 1.0:
            probs_sort, probs_idx = torch.sort(wm_probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)

            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0

            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

            next_token = torch.multinomial(probs_sort, num_samples=1)
            next_token = torch.gather(probs_idx, -1, next_token)

            if return_probs:
                # Return the full probability distribution in vocab space
                sampled_probs = torch.zeros_like(wm_probs, dtype=probs_sort.dtype).scatter_(1, probs_idx, probs_sort)
                next_token = next_token.reshape(-1)
                return next_token, sampled_probs
        else:
            next_token = torch.multinomial(wm_probs, num_samples=1)
            if return_probs:
                next_token = next_token.reshape(-1)
                return next_token, wm_probs

        return next_token.reshape(-1)
    # fmt: on


class _TextSealDipmark:
    def __init__(self, uniforms, alpha=0.2, gamma=0.5):
        self.alpha = alpha
        # build_generator: DiPMark samples with the uniform method.
        self.wm_args = SimpleNamespace(method="uniform", gamma=gamma, uniforms=uniforms)

    # fmt: off
    def dip_reweight_probs(self, probs: torch.FloatTensor, perm) -> torch.FloatTensor:
        """
        Apply DiP-style reweighting to probs given a permutation.
        Some credits to https://github.com/yihwu/DiPmark/blob/main/dipmark/dipmark.py
        """
        # Shuffle probs according to code
        s_probs = torch.gather(probs, -1, perm)
        # Compute cumulative probability
        s_cumsum = torch.cumsum(s_probs, dim=-1)
        # Find α and (1−α) boundaries
        boundary_1 = torch.argmax((s_cumsum > self.alpha).int(), dim=-1, keepdim=True) # bsz x 1
        boundary_2 = torch.argmax((s_cumsum > (1 - self.alpha)).int(), dim=-1, keepdim=True) # bsz x 1
        # Compute partial membership of boundary to the different sides
        def boundary_fraction(boundary, cutoff):
            p_boundary = s_probs.gather(-1, boundary)
            portion = (s_cumsum.gather(-1, boundary) - cutoff) / (p_boundary + 1e-12)
            return torch.clamp(portion, 0, 1)
        portion_in_right_1 = boundary_fraction(boundary_1, self.alpha)
        portion_in_right_2 = boundary_fraction(boundary_2, 1 - self.alpha)
        # Construct smooth indicator masks
        s_mask_1 = (s_cumsum > self.alpha).type_as(probs)
        s_mask_1.scatter_(-1, boundary_1, portion_in_right_1)
        s_mask_2 = (s_cumsum > (1 - self.alpha)).type_as(probs)
        s_mask_2.scatter_(-1, boundary_2, portion_in_right_2)
        # Combine both sides symmetrically, starts with 0 until α, then 0.5 until 1-α, then 1
        s_mask = 0.5 * (s_mask_1 + s_mask_2) # bsz x vocab_size
        # print("Dipmark reweighting mask stats: "(s_mask==0).sum().item(),(s_mask==0.5).sum().item(),(s_mask==1).sum().item())
        s_shift_probs = s_probs * s_mask
        s_shift_probs = s_shift_probs / (s_shift_probs.sum(dim=-1, keepdim=True) + 1e-12)
        shift_probs = torch.gather(s_shift_probs, -1, torch.argsort(perm, dim=-1))
        return shift_probs

    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.LongTensor, # (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
        return_probs: bool = False,
    ) -> torch.LongTensor:
        """
        From ngram tokens, select the next token based on the following:
        - hash the ngram tokens and the secret key
        - use the hash to generate green/red tokens
        - reweight probabilities depending on cumulative distribution
        """
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

            bsz, vocab_size = ngram_tokens.shape[0], logits.shape[-1]
            # Get uniform random scores for all tokens
            uniform_scores = score_all_next_tokens(ngram_tokens, self.wm_args, vocab_size)  # (bsz, vocab_size)
            # Reorder uniform scores according to probability sorting to align the scores to the good tokens
            rs_sorted = uniform_scores[
                torch.arange(bsz, device=probs_idx.device).unsqueeze(1), probs_idx
            ]  # reorder according to prob sorting
            # Permutation for reweighting, descending order: high rs first (r < γ means green)
            perm = torch.argsort(rs_sorted, dim=-1, descending=True) # (bsz, vocab_size)
            # Reweight logits according to DiP
            probs_sort = self.dip_reweight_probs(probs_sort, perm)
            # Select
            next_token = torch.multinomial(probs_sort, num_samples=1) # next token (int), in space ordered by probs
            next_token = torch.gather(probs_idx, -1, next_token) # next token (int), in vocab space
            if return_probs:
                sampled_probs = torch.zeros_like(probs, dtype=probs_sort.dtype).scatter_(1, probs_idx, probs_sort)
                next_token = next_token.reshape(-1)
                return next_token, sampled_probs
        else:
            next_token = torch.argmax(logits, dim=-1)
            if return_probs:
                raise NotImplementedError("return_probs=True not implemented for argmax sampling")
        next_token = next_token.reshape(-1)
        return next_token
    # fmt: on


def _textseal_top_p_logits(logits, temperature, top_p):
    """The temperature and top-p law TextSeal's DiPMark reweights
    (DipmarkGenerator.sample_next, before dip_reweight_probs), as the logits
    vLLM's watermark receives."""
    probs = torch.softmax(logits / temperature, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > top_p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    return torch.zeros_like(probs).scatter_(1, probs_idx, probs_sort).log()


def _rows_without_ties(uniforms: torch.Tensor) -> torch.Tensor:
    ordered = uniforms.sort(-1).values
    return ~(ordered[:, 1:] == ordered[:, :-1]).any(-1)


@pytest.mark.parametrize(("temperature", "top_p"), [(1.0, 1.0), (0.7, 0.9)])
@pytest.mark.parametrize("alpha", [0.2, 0.45])
def test_dipmark_matches_textseal(temperature: float, top_p: float, alpha: float):
    torch.manual_seed(0)  # TextSeal's multinomial draw
    num_rows, vocab_size = 12, 512
    generator = torch.Generator().manual_seed(13)
    logits = 3 * torch.randn(num_rows, vocab_size, generator=generator)
    contexts = _random_contexts(num_rows, 4, 50_000, seed=14)
    watermarker = DiPMarkWatermarker(key=5, context_width=4, alpha=alpha)
    uniforms = watermarker.uniforms(contexts, vocab_size)
    # TextSeal's argsort is not stable: compare rows without tied uniforms.
    rows = _rows_without_ties(uniforms)
    assert rows.sum() >= num_rows - 2
    logits, contexts, uniforms = logits[rows], contexts[rows], uniforms[rows]

    textseal = _TextSealDipmark(uniforms, alpha=alpha)
    _, textseal_law = textseal.sample_next(
        logits, contexts, temperature, top_p, return_probs=True
    )
    native_law = watermarker.watermark_logits(
        _textseal_top_p_logits(logits, temperature, top_p), contexts
    ).softmax(-1)

    torch.testing.assert_close(native_law, textseal_law, atol=1e-5, rtol=1e-4)
    # The same permutation: TextSeal's order (in its probability-sorted index
    # space, mapped back to token ids) is the native order.
    probs = torch.softmax(logits / temperature, dim=-1)
    probs_idx = torch.sort(probs, dim=-1, descending=True).indices
    rs_sorted = uniforms[torch.arange(len(uniforms)).unsqueeze(1), probs_idx]
    textseal_perm = torch.argsort(rs_sorted, dim=-1, descending=True)
    assert torch.equal(
        probs_idx.gather(-1, textseal_perm), dipmark_permutation(uniforms)
    )
    # The same green list: TextSeal's detector scores DiPMark with the binary
    # method, which the native detector reproduces.
    textseal_green = score_all_next_tokens(
        contexts,
        SimpleNamespace(method="binary", gamma=0.5, uniforms=uniforms),
        vocab_size,
    ).bool()
    assert torch.equal(textseal_green, watermarker.green_list(contexts, vocab_size))


def test_dipmark_reweight_matches_textseal_on_any_permutation():
    generator = torch.Generator().manual_seed(15)
    probs = torch.softmax(2 * torch.randn(16, 300, generator=generator), -1)
    probs[:, :20] = 0  # a truncated support
    probs = probs / probs.sum(-1, keepdim=True)
    permutations = torch.stack(
        [torch.randperm(300, generator=generator) for _ in range(16)]
    )
    for alpha in (0.1, 0.2, 0.5):
        textseal = _TextSealDipmark(None, alpha=alpha)

        torch.testing.assert_close(
            dipmark_reweight(probs, permutations, alpha).float(),
            textseal.dip_reweight_probs(probs, permutations),
            atol=1e-5,
            rtol=1e-4,
        )


@pytest.mark.parametrize("temperature", [1.0, 0.8])
@pytest.mark.parametrize(("k", "p_0"), [(1.3, 0.15), (2.0, 0.3)])
def test_morphmark_matches_textseal(temperature: float, k: float, p_0: float):
    """Without truncation (top_p = 1), TextSeal's MorphMark law and the native
    one are the same law, from the same green list."""
    torch.manual_seed(0)  # TextSeal's multinomial draw
    num_rows, vocab_size = 24, 512
    generator = torch.Generator().manual_seed(16)
    contexts = _random_contexts(num_rows, 4, 50_000, seed=17)
    watermarker = MorphMarkWatermarker(key=6, context_width=4, k=k, p_0=p_0)
    uniforms = watermarker.uniforms(contexts, vocab_size)
    green = watermarker.green_list(contexts, vocab_size)
    logits = 2 * torch.randn(num_rows, vocab_size, generator=generator)
    # Rows from a low green mass (below p_0) to a high one (r capped).
    logits += torch.linspace(-5, 5, num_rows).unsqueeze(-1) * green

    textseal = _TextSealMorphMark(uniforms, k_morphmark=k, p_0=p_0)
    _, textseal_law = textseal.sample_next(
        logits, contexts, temperature, 1.0, return_probs=True
    )
    native_law = watermarker.watermark_logits(logits / temperature, contexts).softmax(
        -1
    )

    torch.testing.assert_close(native_law, textseal_law, atol=1e-5, rtol=1e-4)
    textseal_green = score_all_next_tokens(contexts, textseal.wm_args, vocab_size)
    assert torch.equal(textseal_green.bool(), green)
    green_mass = (torch.softmax(logits / temperature, -1) * green).sum(-1)
    assert (green_mass <= p_0).any() and (k * green_mass >= 1).any()


def test_morphmark_textseal_applies_top_p_after_the_reweighting():
    """The documented difference: TextSeal reweights the whole temperature law
    and then applies top-p; vLLM reweights the law top-p already truncated.
    TextSeal's law is the top-p truncation of the native law of the
    untruncated logits."""
    torch.manual_seed(0)
    num_rows, vocab_size, top_p = 8, 256, 0.8
    generator = torch.Generator().manual_seed(18)
    logits = 2 * torch.randn(num_rows, vocab_size, generator=generator)
    contexts = _random_contexts(num_rows, 4, 50_000, seed=19)
    watermarker = MorphMarkWatermarker(key=8, context_width=4)
    uniforms = watermarker.uniforms(contexts, vocab_size)

    _, textseal_law = _TextSealMorphMark(uniforms).sample_next(
        logits, contexts, 1.0, top_p, return_probs=True
    )
    reweight_then_top_p = _textseal_top_p_logits(
        watermarker.watermark_logits(logits, contexts), 1.0, top_p
    ).softmax(-1)
    top_p_then_reweight = watermarker.watermark_logits(
        _textseal_top_p_logits(logits, 1.0, top_p), contexts
    ).softmax(-1)

    torch.testing.assert_close(reweight_then_top_p, textseal_law, atol=1e-5, rtol=1e-4)
    assert (top_p_then_reweight - textseal_law).abs().max() > 1e-3


# ---------------------------------------------------------------------------
# CUDA


@requires_cuda
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_watermark_logits_match_the_cpu(algorithm: str, dtype: torch.dtype):
    watermarker = _watermarker(algorithm, key=2**40 + 7, context_width=4)
    num_rows, vocab_size = 40, 50_257
    generator = torch.Generator().manual_seed(20)
    logits = (3 * torch.randn(num_rows, vocab_size, generator=generator)).to(dtype)
    logits[:, :10] = -torch.inf
    contexts = _random_contexts(num_rows, 4, vocab_size, seed=21)
    skip_mask = torch.rand(num_rows, generator=generator) < 0.25

    cpu = watermarker.watermark_logits(logits, contexts, skip_mask)
    cuda = watermarker.watermark_logits(
        logits.cuda(), contexts.cuda(), skip_mask.cuda()
    ).cpu()

    assert torch.equal(
        watermarker.uniforms(contexts.cuda(), vocab_size).cpu(),
        watermarker.uniforms(contexts, vocab_size),
    )
    assert torch.equal(torch.isneginf(cuda), torch.isneginf(cpu))
    finite = torch.isfinite(cpu)
    # DiPMark's boundary token keeps F - alpha of its mass: its log-mass
    # amplifies the rounding of the cumulative sums, summed in another order
    # on CUDA.
    torch.testing.assert_close(cuda[finite], cpu[finite], atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(
        cuda.double().softmax(-1), cpu.double().softmax(-1), atol=1e-6, rtol=1e-4
    )


@requires_cuda
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("use_fp64", [False, True])
def test_cuda_sample_draws_with_the_random_sampler(algorithm: str, use_fp64: bool):
    watermarker = _watermarker(algorithm, key=2**40 + 7, context_width=4)
    generator = torch.Generator(device="cuda").manual_seed(0)
    num_rows, vocab_size = 64, 50_257
    logits = 3 * torch.randn(num_rows, vocab_size, device="cuda", generator=generator)
    logits[:, :10] = -torch.inf
    contexts = torch.randint(
        -1, vocab_size, (num_rows, 4), device="cuda", generator=generator
    )
    skip_mask = torch.rand(num_rows, device="cuda", generator=generator) < 0.25
    random_sampler = RandomSampler(
        expanded_idx_mapping=torch.randint(
            0, 4, (num_rows,), device="cuda", generator=generator
        ),
        temperatures=torch.tensor([1.0, 0.7, 0.0, 1.3], device="cuda"),
        seeds=torch.randint(0, 2**62, (4,), device="cuda", generator=generator),
        positions=torch.randint(
            0, 4096, (num_rows,), device="cuda", generator=generator
        ),
        use_fp64=use_fp64,
    )

    sample = watermarker.sample(logits, contexts, random_sampler, skip_mask)
    watermarked = watermarker.watermark_logits(logits, contexts, skip_mask)

    assert torch.equal(sample.token_ids, random_sampler(watermarked))
    assert sample.logits is logits


@requires_cuda
@pytest.mark.parametrize(
    ("watermark_config", "detector_cls"),
    [
        (
            {"algorithm": "dipmark", "key": 42, "dipmark_alpha": 0.5},
            DiPMarkWatermarkDetector,
        ),
        ({"algorithm": "morphmark", "key": 42}, MorphMarkWatermarkDetector),
    ],
    ids=ALGORITHMS,
)
def test_llm_generated_sequence_is_watermarked(request, watermark_config, detector_cls):
    """Engine-level generation is detected; the opted-out request is not."""
    from vllm import SamplingParams

    try:
        vllm_runner = request.getfixturevalue("vllm_runner")
    except pytest.FixtureLookupError:
        pytest.skip("needs the vllm_runner fixture of tests/conftest.py")
    runner = vllm_runner(
        "facebook/opt-125m",
        watermark_config=watermark_config,
        seed=0,
        enforce_eager=True,
        max_model_len=512,
        gpu_memory_utilization=0.2,
    )
    prompt = "Tell me a story about an explorer who discovers a mysterious island."
    with runner:
        watermarked, unwatermarked = (
            runner.llm.generate(
                prompt,
                SamplingParams(
                    temperature=1.0, top_p=1.0, max_tokens=256, watermarking=flag
                ),
            )[0]
            .outputs[0]
            .token_ids
            for flag in (True, False)
        )
    detector = detector_cls(key=42)

    assert detector.detect(list(watermarked)).p_value < 1e-3
    assert detector.detect(list(unwatermarked)).p_value > 1e-3
