# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy
import pytest
import torch
from scipy import integrate, stats

from vllm.platforms import current_platform
from vllm.v1.watermarking import (
    GaussianCopulaWatermarkDetector,
    GaussianCopulaWatermarker,
)
from vllm.v1.watermarking.gaussian_copula import gumbels
from vllm.v1.watermarking.watermarker import RandomSampler

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)


def _argmax_sampler(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


def test_tokens_without_a_full_context_are_not_watermarked():
    watermarker = GaussianCopulaWatermarker(key=42, context_width=2)
    logits = 2 * torch.randn(3, 64, generator=torch.Generator().manual_seed(0))
    contexts = torch.tensor([[-1, -1], [-1, 5], [4, 5]])

    sample = watermarker.sample(logits, contexts, _argmax_sampler)

    assert torch.equal(sample.token_ids[:2], logits[:2].argmax(dim=-1))
    assert sample.logits is logits


def test_detector_scores_each_distinct_pair_once():
    """As the reference, every (context, token) pair with a full context counts
    once."""
    detector = GaussianCopulaWatermarkDetector(key=42, context_width=1)

    contexts, targets = detector._prepare_inputs([3, 5, 3, 5, 3, 7])

    assert contexts.tolist() == [[3], [5], [3]]
    assert targets.tolist() == [5, 3, 7]


def test_detector_reports_the_uncorrected_gamma_tail():
    detector = GaussianCopulaWatermarkDetector(key=42, context_width=2)
    result = detector.detect(list(range(30)))

    assert result.p_value == pytest.approx(
        stats.gamma.sf(result.score, result.num_scored_tokens), rel=1e-10
    )


@pytest.mark.parametrize("correlation", [0.3, 0.98551, 1.0])
def test_gumbels_are_standard_gumbel(correlation: float):
    """Every z is standard normal, so the noise is Gumbel(0, 1) and the
    sampled token follows the model's distribution."""
    generator = torch.Generator().manual_seed(0)
    u = torch.rand(100_000, generator=generator, dtype=torch.float64)
    noise = torch.randn(100_000, generator=generator, dtype=torch.float64)

    noise = gumbels(u.clamp(1e-12, 1 - 1e-12), noise, correlation)

    assert stats.kstest(noise.numpy(), stats.gumbel_r.cdf).pvalue > 1e-3


def _distinct_contexts(num_rows: int) -> torch.Tensor:
    return torch.stack(
        [torch.arange(num_rows) // 128 + 10, torch.arange(num_rows) % 128 + 10],
        dim=-1,
    )


_PROBS = torch.tensor([0.4, 0.25, 0.15, 0.1, 0.06, 0.04])


@pytest.mark.parametrize("candidate_limit", [3, 4096])
def test_sample_follows_the_model_distribution(monkeypatch, candidate_limit):
    monkeypatch.setattr(
        "vllm.v1.watermarking.gaussian_copula.NUM_CANDIDATES", candidate_limit
    )
    num_rows = 20_000
    watermarker = GaussianCopulaWatermarker(key=42, context_width=2)

    sample = watermarker.sample(
        _PROBS.log().expand(num_rows, -1), _distinct_contexts(num_rows)
    )

    counts = torch.bincount(sample.token_ids, minlength=len(_PROBS))
    assert stats.chisquare(counts.numpy(), num_rows * _PROBS.numpy()).pvalue > 1e-3


def _generate(
    watermarker: GaussianCopulaWatermarker, seed: int, watermarked: bool = True
) -> list[int]:
    """Generate as GPUWatermarkSampler does: repeated contexts and contexts
    reaching before the output get plain sampling."""
    generator = torch.Generator().manual_seed(seed)

    def plain(logits: torch.Tensor) -> torch.Tensor:
        return torch.multinomial(logits.softmax(-1), 1, generator=generator)[:, 0]

    tokens: list[int] = []
    for _ in range(200):
        logits = 2 * torch.randn(1, 1000, generator=generator)
        if not watermarked:
            tokens.append(int(plain(logits)))
            continue
        context = ([-1, -1] + tokens)[-2:]
        padded = [-1, -1] + tokens
        earlier = {tuple(padded[i : i + 2]) for i in range(len(tokens))}
        skip = torch.tensor([tuple(context) in earlier])
        sample = watermarker.sample(logits, torch.tensor([context]), plain, skip)
        tokens.append(int(sample.token_ids))
    return tokens


def test_generated_text_is_detected():
    watermarker = GaussianCopulaWatermarker(key=42, context_width=2)
    watermarked = _generate(watermarker, seed=1)
    unwatermarked = _generate(watermarker, seed=2, watermarked=False)

    detector = GaussianCopulaWatermarkDetector(key=42, context_width=2)
    detection = detector.detect(watermarked)

    assert detection.is_watermarked
    assert detection.p_value < 1e-6
    assert not detector.detect(unwatermarked).is_watermarked
    assert (
        not GaussianCopulaWatermarkDetector(key=43, context_width=2)
        .detect(watermarked)
        .is_watermarked
    )


def _random_sampler(num_rows: int) -> RandomSampler:
    return RandomSampler(
        expanded_idx_mapping=torch.arange(num_rows, dtype=torch.int32, device="cuda"),
        temperatures=torch.ones(num_rows, device="cuda"),
        seeds=torch.arange(num_rows, dtype=torch.int64, device="cuda") + 1,
        positions=torch.full((num_rows,), 5, dtype=torch.int64, device="cuda"),
    )


def _kernel_inputs(num_rows: int = 32, vocab_size: int = 5_000):
    generator = torch.Generator().manual_seed(3)
    logits = 3 * torch.randn(num_rows, vocab_size, generator=generator)
    contexts = torch.randint(0, vocab_size, (num_rows, 2), generator=generator)
    return logits, contexts


@requires_cuda
def test_kernel_with_full_correlation_matches_the_reference():
    """With correlation 1 the noise is fully keyed."""
    watermarker = GaussianCopulaWatermarker(key=42, context_width=2, correlation=1.0)
    logits, contexts = _kernel_inputs(vocab_size=1000)

    sample = watermarker.sample(logits.cuda(), contexts.cuda(), _random_sampler(32))

    u = watermarker.prf.uniform(contexts, torch.arange(logits.shape[-1])).double()
    noise = gumbels(u, torch.zeros_like(u), 1.0)
    assert torch.equal(
        sample.token_ids.cpu(), torch.argmax(logits.double() + noise, dim=-1)
    )


@requires_cuda
def test_kernel_draws_the_plain_sample_without_watermark():
    """The fresh noise is the request's own sampling noise: rows without a full
    context, and every row at a vanishing correlation, get the token the plain
    sampler draws."""
    logits, contexts = _kernel_inputs(vocab_size=1000)
    logits, contexts = logits.cuda(), contexts.cuda()
    contexts[-4:, 0] = -1
    random_sampler = _random_sampler(32)
    plain = random_sampler(logits)

    keyed = GaussianCopulaWatermarker(key=42, context_width=2).sample(
        logits, contexts, random_sampler
    )
    weak = GaussianCopulaWatermarker(key=42, context_width=2, correlation=1e-6).sample(
        logits, contexts, random_sampler
    )

    assert torch.equal(keyed.token_ids[-4:], plain[-4:])
    assert torch.equal(weak.token_ids, plain)


@requires_cuda
@pytest.mark.parametrize("candidate_limit", [3, 4096])
def test_kernel_samples_follow_the_model_distribution(monkeypatch, candidate_limit):
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.sample.gaussian_copula_watermark.NUM_CANDIDATES",
        candidate_limit,
    )
    num_rows = 20_000
    watermarker = GaussianCopulaWatermarker(key=42, context_width=2)

    sample = watermarker.sample(
        _PROBS.log().expand(num_rows, -1).cuda(),
        _distinct_contexts(num_rows).cuda(),
        _random_sampler(num_rows),
    )

    counts = torch.bincount(sample.token_ids.cpu(), minlength=len(_PROBS))
    assert stats.chisquare(counts.numpy(), num_rows * _PROBS.numpy()).pvalue > 1e-3


@requires_cuda
@pytest.mark.parametrize("use_fp64", [False, True])
def test_capped_kernel_keeps_plain_tail_and_keys_only_head_tokens(use_fp64):
    watermarker = GaussianCopulaWatermarker(key=42, context_width=2, correlation=1.0)
    logits, contexts = _kernel_inputs(num_rows=128, vocab_size=8192)
    # A substantial tail makes this check independent of rare random events.
    logits = logits / 6
    sampler = _random_sampler(128)
    if use_fp64:
        from dataclasses import replace

        sampler = replace(sampler, use_fp64=True)
    plain = sampler(logits.cuda()).cpu()
    head_logits, head_ids = logits.topk(4096, dim=-1)
    in_head = (head_ids == plain.unsqueeze(-1)).any(dim=-1)
    assert in_head.any() and (~in_head).any()
    uniforms = watermarker.prf.uniform(contexts, head_ids).double()
    noise = gumbels(uniforms, torch.zeros_like(uniforms), 1.0)
    winners = head_ids.gather(-1, (head_logits + noise).argmax(-1, keepdim=True))[:, 0]
    skip = torch.arange(128, device="cuda") % 7 == 0

    sample = watermarker.sample(logits.cuda(), contexts.cuda(), sampler, skip)

    expected = torch.where(in_head & ~skip.cpu(), winners, plain)
    assert torch.equal(sample.token_ids.cpu(), expected)


def test_surprisal_weighting_without_surprisal_is_the_plain_test():
    detector = GaussianCopulaWatermarkDetector(key=42, context_width=2)
    token_ids = [3, 5, 3, 5, 3, 7] + list(range(40))

    plain = detector.detect(token_ids)
    weighted = detector.detect(token_ids, logprobs=[0.0] * len(token_ids))

    assert weighted.num_scored_tokens == plain.num_scored_tokens
    assert weighted.score == pytest.approx(plain.score, rel=1e-12)
    # The plain test sums the integer-shape Gamma series, this one gammaincc.
    assert weighted.p_value == pytest.approx(plain.p_value, rel=1e-7)


def test_surprisal_weights_follow_automarks_levels():
    detector = GaussianCopulaWatermarkDetector(key=42, context_width=1)
    token_ids = [1, 2, 3, 4, 5]
    # Levels round(4 * min(surprisal, 2.5)): None and positive log-probs weigh 1.
    logprobs = [-9.0, -0.3, None, -7.0, 0.5]

    weighted = detector.detect(token_ids, logprobs)
    contexts, targets = detector._prepare_inputs(token_ids)
    scores = detector._score_tokens(contexts, targets)

    weights = torch.tensor([1.25, 1.0, 3.5, 1.0], dtype=torch.float64)
    assert weighted.score == pytest.approx((weights * scores).sum().item())


def test_surprisal_weighted_p_values_are_uniform_without_watermark():
    detector = GaussianCopulaWatermarkDetector(key=42, context_width=2)
    generator = torch.Generator().manual_seed(0)
    p_values = []
    for _ in range(400):
        token_ids = torch.randint(0, 50_000, (120,), generator=generator).tolist()
        logprobs = (-4 * torch.rand(120, generator=generator)).tolist()
        p_values.append(detector.detect(token_ids, logprobs).p_value)

    assert stats.kstest(p_values, "uniform").pvalue > 1e-3


def test_surprisal_weighting_detects_generated_text():
    watermarker = GaussianCopulaWatermarker(key=42, context_width=2)
    watermarked = _generate(watermarker, seed=1)
    unwatermarked = _generate(watermarker, seed=2, watermarked=False)
    detector = GaussianCopulaWatermarkDetector(key=42, context_width=2)

    assert detector.detect(watermarked, [-1.0] * len(watermarked)).p_value < 1e-6
    assert not detector.detect(
        unwatermarked, [-1.0] * len(unwatermarked)
    ).is_watermarked


def test_logprobs_need_one_value_per_token():
    detector = GaussianCopulaWatermarkDetector(key=42, context_width=2)

    with pytest.raises(ValueError, match="one value per token"):
        detector.detect([1, 2, 3], [-1.0, -1.0])


class _PieceTokenizer:
    def __init__(self, pieces: list[str]) -> None:
        self.pieces = pieces

    def __len__(self) -> int:
        return len(self.pieces)

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [self.pieces[token_id] for token_id in token_ids]


# 0 "▁The", 1 "▁cat", 2 "s", 3 ".", 4 "\n", 5 "▁sat", 6 "(", 7 "on", 8 "<0x0A>"
_SENTENCEPIECE = ["▁The", "▁cat", "s", ".", "\n", "▁sat", "(", "on", "<0x0A>"]
# The same vocabulary as byte-level BPE pieces: "Ġ" is a space, "Ċ" a newline.
_BYTE_LEVEL = ["ĠThe", "Ġcat", "s", ".", "Ċ", "Ġsat", "(", "on", "Ċ"]


@pytest.mark.parametrize("pieces", [_SENTENCEPIECE, _BYTE_LEVEL])
def test_word_start_weights_mark_word_and_sentence_starts(pieces):
    detector = GaussianCopulaWatermarkDetector(
        key=42,
        context_width=1,
        tokenizer=_PieceTokenizer(pieces),
        word_start_weight=3.0,
    )
    token_ids = [0, 1, 2, 3, 0, 4, 5, 6, 7, 8, 1, 2]

    levels = detector.word_starts.levels(token_ids, range(len(token_ids)))

    # Weight 3 (level 8) for the word starts " cat" and "on" (after "(") and
    # " The" after the sentence end; 1 for the first token, tokens inside a
    # word or glued to it ("s", ".", "\n", "("), and tokens after a newline.
    assert levels.tolist() == [0, 8, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0]


def test_word_start_weighting_matches_the_two_gamma_tail():
    pieces = ["▁w" + str(i) if i % 3 == 0 else "x" + str(i) for i in range(300)]
    detector = GaussianCopulaWatermarkDetector(
        key=42,
        context_width=2,
        tokenizer=_PieceTokenizer(pieces),
        word_start_weight=3.0,
    )
    token_ids = torch.randint(
        0, 300, (200,), generator=torch.Generator().manual_seed(1)
    ).tolist()

    result = detector.detect(token_ids)

    positions = detector._scored_positions(token_ids)
    contexts, targets = detector._pairs(token_ids, positions)
    scores = detector._score_tokens(contexts, targets).numpy()
    heavy = detector.word_starts.levels(token_ids, positions).numpy() == 8
    total = 3 * scores[heavy].sum() + scores[~heavy].sum()
    # P(3 * Gamma(n_A) + Gamma(n_B) >= total), by numerical integration.
    x = numpy.linspace(0, total / 3, 200_001)[1:]
    density = stats.gamma.pdf(x, heavy.sum()) * stats.gamma.sf(
        total - 3 * x, (~heavy).sum()
    )
    exact = stats.gamma.sf(total / 3, heavy.sum()) + integrate.trapezoid(density, x)
    assert result.score == pytest.approx(total)
    assert result.p_value == pytest.approx(exact, rel=1e-2)


def test_word_start_weighting_needs_the_tokenizer():
    with pytest.raises(ValueError, match="tokenizer"):
        GaussianCopulaWatermarkDetector(key=42, word_start_weight=3.0)
