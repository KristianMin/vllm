# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import math

import numpy as np
import pytest
import torch
from scipy import stats

from vllm.config.watermarking import WatermarkConfig, derive_watermark_key
from vllm.platforms import current_platform
from vllm.v1.watermarking import (
    GaussianCopulaWatermarker,
    TextSeal2WatermarkDetector,
    TextSeal2Watermarker,
    create_watermarker,
)
from vllm.v1.watermarking.localized import entropy_weights, weighted_gamma_p
from vllm.v1.watermarking.textseal2 import (
    DEFAULT_CORRELATION,
    race_gumbels,
    virtual_normals,
)
from vllm.v1.watermarking.watermarker import RandomSampler

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)

_PROBS = torch.tensor([0.4, 0.25, 0.15, 0.1, 0.06, 0.04])


def _argmax_sampler(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


def _distinct_contexts(num_rows: int) -> torch.Tensor:
    return torch.stack(
        [torch.arange(num_rows) // 128 + 10, torch.arange(num_rows) % 128 + 10],
        dim=-1,
    )


def test_keys_come_from_the_master_key():
    watermarker = TextSeal2Watermarker(key=42)
    detector = TextSeal2WatermarkDetector(key=42)

    assert watermarker.prf.key == derive_watermark_key(42, b"key_a")
    assert watermarker.key_b_prf.key == derive_watermark_key(42, b"key_b")
    assert detector.public_key == watermarker.prf.key
    assert [prf.key for prf in detector.prfs] == [
        watermarker.prf.key,
        watermarker.key_b_prf.key,
    ]


def test_virtual_key_is_standard_normal_from_independent_keys():
    watermarker = TextSeal2Watermarker(key=42, context_width=2)
    contexts = _distinct_contexts(4000)
    tokens = torch.arange(25)

    virtual = watermarker.virtual_key(contexts, tokens).flatten()
    key_a = torch.special.ndtri(
        watermarker.prf.uniform(contexts, tokens).double()
    ).flatten()
    key_b = torch.special.ndtri(
        watermarker.key_b_prf.uniform(contexts, tokens).double()
    ).flatten()

    assert stats.kstest(virtual.numpy(), stats.norm.cdf).pvalue > 1e-3
    assert torch.allclose(virtual, (key_a + key_b) / math.sqrt(2))
    assert abs(torch.corrcoef(torch.stack([key_a, key_b]))[0, 1].item()) < 0.01


@pytest.mark.parametrize("correlation", [0.3, DEFAULT_CORRELATION, 1.0])
def test_race_gumbels_are_standard_gumbel(correlation: float):
    generator = torch.Generator().manual_seed(0)
    uniforms = [
        torch.rand(100_000, generator=generator, dtype=torch.float64).clamp(1e-12, 1)
        for _ in range(2)
    ]
    noise = torch.randn(100_000, generator=generator, dtype=torch.float64)

    gumbels = race_gumbels(virtual_normals(uniforms), noise, correlation)

    assert stats.kstest(gumbels.numpy(), stats.gumbel_r.cdf).pvalue > 1e-3


def test_one_key_is_the_gaussian_copula():
    """With a single key, the race noise is the Gaussian copula's."""
    generator = torch.Generator().manual_seed(1)
    u = torch.rand(1000, generator=generator, dtype=torch.float64)
    noise = torch.randn(1000, generator=generator, dtype=torch.float64)
    from vllm.v1.watermarking.gaussian_copula import gumbels

    assert torch.allclose(
        race_gumbels(virtual_normals([u]), noise, 0.7), gumbels(u, noise, 0.7)
    )


def test_tokens_without_a_full_context_are_not_watermarked():
    watermarker = TextSeal2Watermarker(key=42, context_width=2)
    logits = 2 * torch.randn(3, 64, generator=torch.Generator().manual_seed(0))
    contexts = torch.tensor([[-1, -1], [-1, 5], [4, 5]])

    sample = watermarker.sample(logits, contexts, _argmax_sampler)

    assert torch.equal(sample.token_ids[:2], logits[:2].argmax(dim=-1))
    assert sample.logits is logits


def test_full_correlation_ignores_the_fresh_noise():
    logits = 2 * torch.randn(64, 500, generator=torch.Generator().manual_seed(2))
    contexts = _distinct_contexts(64)
    first = TextSeal2Watermarker(key=3, context_width=2, correlation=1.0)
    second = TextSeal2Watermarker(key=3, context_width=2, correlation=1.0)
    second.fresh_noise(torch.Size([7]), logits.device)  # advance its generator

    assert torch.equal(
        first.sample(logits, contexts).token_ids,
        second.sample(logits, contexts).token_ids,
    )


def test_sample_follows_the_model_distribution():
    num_rows = 20_000
    watermarker = TextSeal2Watermarker(key=42, context_width=2)

    sample = watermarker.sample(
        _PROBS.log().expand(num_rows, -1), _distinct_contexts(num_rows)
    )

    counts = torch.bincount(sample.token_ids, minlength=len(_PROBS))
    assert stats.chisquare(counts.numpy(), num_rows * _PROBS.numpy()).pvalue > 1e-3


def _generate(
    watermarker: TextSeal2Watermarker | None, seed: int, length: int = 200
) -> list[int]:
    """Generate as GPUWatermarkSampler does: repeated contexts and contexts
    reaching before the output get plain sampling."""
    generator = torch.Generator().manual_seed(seed)

    def plain(logits: torch.Tensor) -> torch.Tensor:
        return torch.multinomial(logits.softmax(-1), 1, generator=generator)[:, 0]

    tokens: list[int] = []
    for _ in range(length):
        logits = 2 * torch.randn(1, 1000, generator=generator)
        if watermarker is None:
            tokens.append(int(plain(logits)))
            continue
        padded = [-1, -1] + tokens
        context = padded[-2:]
        earlier = {tuple(padded[i : i + 2]) for i in range(len(tokens))}
        skip = torch.tensor([tuple(context) in earlier])
        sample = watermarker.sample(logits, torch.tensor([context]), plain, skip)
        tokens.append(int(sample.token_ids))
    return tokens


def test_generated_text_is_detected_privately_and_publicly():
    watermarked = _generate(TextSeal2Watermarker(key=42, context_width=2), seed=1)
    unwatermarked = _generate(None, seed=2)
    detector = TextSeal2WatermarkDetector(key=42, context_width=2)

    private = detector.detect(watermarked)
    public = detector.detect_public(watermarked)

    assert private.is_watermarked and private.p_value < 1e-20
    assert public.is_watermarked and public.p_value < 1e-6
    # The public key carries 1 / sqrt(2) of the virtual key's signal.
    assert private.p_value < public.p_value
    from_public_key = TextSeal2WatermarkDetector(
        detector.public_key, context_width=2, public=True
    ).detect(watermarked)
    assert from_public_key == public
    assert not detector.detect(unwatermarked).is_watermarked
    assert not detector.detect_public(unwatermarked).is_watermarked
    other = TextSeal2WatermarkDetector(key=43, context_width=2)
    assert not other.detect(watermarked).is_watermarked


def test_null_p_values_are_uniform():
    detector = TextSeal2WatermarkDetector(key=5, context_width=2)
    generator = torch.Generator().manual_seed(3)
    p_values = [
        detector.detect(torch.randint(0, 5000, (60,), generator=generator).tolist())
        for _ in range(300)
    ]

    for values in ([d.p_value for d in p_values],):
        assert stats.kstest(values, "uniform").pvalue > 1e-3


def test_detector_scores_each_distinct_pair_once():
    detector = TextSeal2WatermarkDetector(key=42, context_width=1)

    contexts, targets = detector._prepare_inputs([3, 5, 3, 5, 3, 7])

    assert contexts.tolist() == [[3], [5], [3]]
    assert targets.tolist() == [5, 3, 7]


def test_entropy_weights_the_scored_pairs():
    tokens = _generate(TextSeal2Watermarker(key=42, context_width=2), seed=4)
    entropies = torch.rand(len(tokens), generator=torch.Generator().manual_seed(5))
    detector = TextSeal2WatermarkDetector(key=42, context_width=2)

    weighted = detector.detect(tokens, entropies.tolist())

    contexts, targets, positions = detector._pairs(tokens)
    scores = detector._score_tokens(contexts, targets).numpy()
    weights = entropy_weights(entropies.double().numpy()[positions])
    assert weighted.num_scored_tokens == len(targets)
    assert weighted.p_value == weighted_gamma_p(scores, weights, 1.0)
    assert weighted.score == pytest.approx(float(np.sum(weights * scores)))
    with pytest.raises(ValueError, match="one value per token"):
        detector.detect(tokens, entropies[:-1].tolist())


def test_localized_detection_finds_the_watermarked_span():
    head = _generate(None, seed=6, length=150)
    span = _generate(TextSeal2Watermarker(key=42, context_width=2), seed=7)
    tokens = head + span + _generate(None, seed=8, length=150)
    detector = TextSeal2WatermarkDetector(key=42, context_width=2)

    result = detector.detect_localized(tokens, min_length=32)

    assert result.is_watermarked
    assert len(result.token_labels) == len(tokens)
    if result.regions:
        start, end, _ = result.regions[0]
        assert start < len(head) + len(span) and end > len(head)
    labelled = np.flatnonzero(result.token_labels)
    inside = (labelled >= len(head)) & (labelled < len(head) + len(span))
    assert inside.mean() > 0.8


def test_config_defaults_and_factory():
    textseal2 = create_watermarker(WatermarkConfig(algorithm="textseal2", key=1))
    copula = create_watermarker(WatermarkConfig(algorithm="gaussian_copula", key=1))
    tuned = create_watermarker(
        WatermarkConfig(algorithm="textseal2", key=1, correlation=0.5, context_width=3)
    )

    assert isinstance(textseal2, TextSeal2Watermarker)
    assert textseal2.correlation == DEFAULT_CORRELATION == 0.9
    assert isinstance(copula, GaussianCopulaWatermarker)
    assert copula.correlation == 0.98551
    assert (tuned.correlation, tuned.context_width) == (0.5, 3)
    with pytest.raises(ValueError):
        TextSeal2Watermarker(key=1, correlation=0.0)


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
    watermarker = TextSeal2Watermarker(key=42, context_width=2, correlation=1.0)
    logits, contexts = _kernel_inputs()

    sample = watermarker.sample(logits.cuda(), contexts.cuda(), _random_sampler(32))

    virtual = watermarker.virtual_key(contexts, torch.arange(logits.shape[-1]))
    noise = race_gumbels(virtual, torch.zeros_like(virtual), 1.0)
    assert torch.equal(
        sample.token_ids.cpu(), torch.argmax(logits.double() + noise, dim=-1)
    )


@requires_cuda
def test_kernel_draws_the_plain_sample_without_watermark():
    """The fresh noise is the request's own sampling noise: rows without a full
    context, and every row at a vanishing correlation, get the token the plain
    sampler draws."""
    logits, contexts = _kernel_inputs()
    logits, contexts = logits.cuda(), contexts.cuda()
    contexts[-4:, 0] = -1
    random_sampler = _random_sampler(32)
    plain = random_sampler(logits)

    keyed = TextSeal2Watermarker(key=42, context_width=2).sample(
        logits, contexts, random_sampler
    )
    weak = TextSeal2Watermarker(key=42, context_width=2, correlation=1e-6).sample(
        logits, contexts, random_sampler
    )

    assert torch.equal(keyed.token_ids[-4:], plain[-4:])
    assert torch.equal(weak.token_ids, plain)


@requires_cuda
@pytest.mark.parametrize("use_fp64", [False, True])
def test_kernel_samples_follow_the_model_distribution(use_fp64: bool):
    num_rows = 20_000
    watermarker = TextSeal2Watermarker(key=42, context_width=2)
    random_sampler = RandomSampler(
        expanded_idx_mapping=torch.arange(num_rows, dtype=torch.int32, device="cuda"),
        temperatures=torch.ones(num_rows, device="cuda"),
        seeds=torch.arange(num_rows, dtype=torch.int64, device="cuda") + 1,
        positions=torch.full((num_rows,), 5, dtype=torch.int64, device="cuda"),
        use_fp64=use_fp64,
    )

    sample = watermarker.sample(
        _PROBS.log().expand(num_rows, -1).cuda(),
        _distinct_contexts(num_rows).cuda(),
        random_sampler,
    )

    counts = torch.bincount(sample.token_ids.cpu(), minlength=len(_PROBS))
    assert stats.chisquare(counts.numpy(), num_rows * _PROBS.numpy()).pvalue > 1e-3


# ---- the speculative rule ------------------------------------------------------


def _paper_law(P, Q, drafts, law):
    """One request's outcome law straight from the paper's equations (Eq. 10-11
    for block verification, Remark 2 for token-level), in plain Python."""
    K, V = len(drafts), len(P[0])
    if law == "block":
        c = [1.0]
        for i in range(K):
            c.append(min(1.0, c[-1] * P[i][drafts[i]] / Q[i][drafts[i]]))
        u = [[max(c[i] * P[i][v] - Q[i][v], 0.0) for v in range(V)] for i in range(K)]
        z = [sum(row) for row in u]
        h = [1.0] + [
            z[i] / (z[i] + 1 - c[i]) if z[i] + 1 - c[i] > 0 else 1.0
            for i in range(1, K)
        ]
        h.append(c[K])
        stop = [
            h[i] * math.prod(1 - h[j] for j in range(i + 1, K + 1))
            for i in range(K + 1)
        ]
        r = [[x / z[i] if z[i] > 0 else 0.0 for x in u[i]] for i in range(K)]
    else:
        a = [min(1.0, P[i][drafts[i]] / Q[i][drafts[i]]) for i in range(K)]
        stop = [math.prod(a[:i]) * (1 - a[i]) for i in range(K)] + [math.prod(a)]
        r = []
        for i in range(K):
            excess = [max(P[i][v] - Q[i][v], 0.0) for v in range(V)]
            r.append([x / sum(excess) if sum(excess) > 0 else 0.0 for x in excess])
    r.append(list(P[K]))
    pi = [[stop[i] * r[i][v] for v in range(V)] for i in range(K + 1)]
    for i in range(K):
        pi[i][drafts[i]] = 0.0
    return pi, stop


@pytest.mark.parametrize("law", ["block", "token"])
def test_outcome_laws_follow_the_paper(law: str):
    from vllm.v1.watermarking import textseal2_verification as verification

    generator = torch.Generator().manual_seed(0)
    for _ in range(20):
        K = int(torch.randint(1, 4, (1,), generator=generator))
        P = torch.softmax(2 * torch.randn(K + 1, 6, generator=generator).double(), -1)
        Q = torch.softmax(2 * torch.randn(K, 6, generator=generator).double(), -1)
        drafts = torch.multinomial(Q, 1, generator=generator)[:, 0]
        expected_pi, expected_stop = _paper_law(
            P.tolist(), Q.tolist(), drafts.tolist(), law
        )

        pi, stop = verification.outcome_law(
            P.unsqueeze(0),
            torch.cat([Q, torch.zeros(1, 6, dtype=torch.float64)]).unsqueeze(0),
            torch.cat([drafts, torch.zeros(1, dtype=torch.long)]).unsqueeze(0),
            torch.tensor([K]),
            law,
        )

        assert torch.allclose(pi[0], torch.tensor(expected_pi, dtype=torch.float64))
        assert torch.allclose(stop[0], torch.tensor(expected_stop, dtype=torch.float64))
        assert pi.sum().item() == pytest.approx(1.0)


@pytest.mark.parametrize("name", ["max", "expsum", "tsum"])
def test_pass_aggregates_are_exponential_and_increasing(name: str):
    from vllm.v1.watermarking.textseal2_verification import pass_aggregate

    generator = torch.Generator().manual_seed(1)
    a, b = torch.empty(2, 100_000, dtype=torch.float64).exponential_(
        generator=generator
    )

    psi = pass_aggregate(a, b, name)

    assert stats.kstest(psi.numpy(), stats.expon.cdf).pvalue > 1e-3
    assert (pass_aggregate(a * 1.1, b, name) >= psi).all()
    assert (pass_aggregate(a, b * 1.1, name) >= psi).all()
    assert torch.isinf(pass_aggregate(torch.tensor([torch.inf]), b[:1], name)).all()


def test_residual_noise_identifies_the_round():
    from vllm.v1.watermarking.textseal2 import residual_noise

    seeds = torch.tensor([7, 7, 7, 8])
    positions = torch.tensor([10, 10, 11, 10])
    steps = torch.tensor([1, 1, 1, 1])

    noise = residual_noise(seeds, positions, steps, 5000)
    other_step = residual_noise(seeds, positions, steps + 1, 5000)

    assert torch.equal(noise[0], noise[1])
    assert not torch.equal(noise[0], noise[2]) and not torch.equal(noise[0], noise[3])
    assert not torch.equal(noise[0], other_step[0])
    assert stats.kstest(noise.flatten().numpy(), stats.norm.cdf).pvalue > 1e-3


class _DrafterSampler(RandomSampler):
    """The drafter's random sampler without its kernel (greedy rows only)."""

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1)


def _drafter_sampler(seeds, positions, step):
    n = seeds.shape[0]
    return _DrafterSampler(
        expanded_idx_mapping=torch.arange(n, dtype=torch.int32),
        temperatures=torch.ones(n),
        seeds=seeds,
        positions=positions,
        is_drafting=True,
        logits_cache=torch.empty(0),
        logits_cache_col=torch.full((n,), step),
        logits_cache_source=torch.empty(0),
    )


def _speculate(keyed: bool, law: str, num_sequences: int, seed: int = 0):
    """Generate two tokens per sequence with K = 2 drafts per round, as the
    engine does: keyed (or plain) drafts, residual verification, and
    rejection sampling, which emits the point masses up to the first
    rejected draft. The target and draft distributions depend on the position
    only; fresh prompts give every context its own keyed values."""
    from vllm.v1.watermarking.textseal2 import (
        TextSeal2DraftWatermarker,
        TextSeal2VerificationWatermarker,
    )

    width, K, length, vocab_size = 5, 2, 2, 5
    generator = torch.Generator().manual_seed(seed)
    shape = (length + K, vocab_size)
    p = torch.softmax(1.5 * torch.randn(shape, generator=generator).double(), -1)
    q = torch.softmax(p.log() + torch.randn(shape, generator=generator).double(), -1)
    drafter = TextSeal2DraftWatermarker(key=3, context_width=width)
    verifier = TextSeal2VerificationWatermarker(
        key=3, context_width=width, keyed_drafts=keyed, law=law
    )
    prompts = torch.randint(10, 2**30, (num_sequences, width), generator=generator)
    seeds = torch.randint(0, 2**62, (num_sequences,), generator=generator)
    texts = [prompt.tolist() for prompt in prompts]
    first_kept = None
    while True:
        active = [s for s in range(num_sequences) if len(texts[s]) < width + length]
        if not active:
            break
        t = torch.tensor([len(texts[s]) - width for s in active])
        committed = torch.tensor([len(texts[s]) for s in active])
        drafts = torch.zeros(len(active), K, dtype=torch.long)
        for j in range(K):
            contexts = torch.tensor(
                [
                    (texts[s] + drafts[b, :j].tolist())[-width:]
                    for b, s in enumerate(active)
                ]
            )
            if keyed:
                drafts[:, j] = drafter.sample(
                    q[t + j].log(),
                    contexts,
                    _drafter_sampler(seeds[active], committed - 1 + j, j),
                ).token_ids
            else:
                drafts[:, j] = torch.multinomial(q[t + j], 1, generator=generator)[:, 0]
        rows = [(b, s, j) for b, s in enumerate(active) for j in range(K + 1)]
        logits = torch.stack([p[t[b] + j].log() for b, _, j in rows])
        contexts = torch.tensor(
            [(texts[s] + drafts[b, :j].tolist())[-width:] for b, s, j in rows]
        )
        draft_tokens = torch.tensor(
            [int(drafts[b, j]) if j < K else 0 for b, _, j in rows]
        )
        draft_logits = torch.stack([q[t[b] + min(j, K - 1)].log() for b, _, j in rows])
        sampler = RandomSampler(
            expanded_idx_mapping=torch.arange(len(rows), dtype=torch.int32),
            temperatures=torch.ones(len(rows)),
            seeds=torch.tensor([int(seeds[s]) for _, s, _ in rows]),
            positions=torch.tensor([int(committed[b]) - 1 + j for b, _, j in rows]),
        )
        cu_num_logits = torch.arange(0, len(rows) + 1, K + 1, dtype=torch.int32)
        output = verifier.residual_verification_logits(
            logits,
            contexts,
            draft_tokens,
            cu_num_logits,
            torch.zeros(len(rows), dtype=torch.bool),
            draft_logits,
            sampler,
        )
        chosen = output.argmax(-1).view(len(active), K + 1)
        kept = []
        for b, s in enumerate(active):
            j = 0
            while j < K and chosen[b, j] == drafts[b, j]:
                texts[s].append(int(drafts[b, j]))
                j += 1
            texts[s].append(int(chosen[b, j]))
            kept.append(j)
        if first_kept is None:
            first_kept = sum(kept) / len(kept)
    paths = torch.tensor([text[width : width + length] for text in texts])
    return paths, first_kept, p, q


def _expected_first_kept(p, q, law, keyed, K=2):
    from vllm.v1.watermarking import textseal2_verification as verification

    total = 0.0
    for drafts in itertools.product(range(p.shape[1]), repeat=K):
        weight = math.prod(q[i, d].item() for i, d in enumerate(drafts))
        if keyed:
            Q = torch.cat([q[:K], torch.zeros(1, p.shape[1], dtype=q.dtype)])
        else:
            one_hot = torch.nn.functional.one_hot(torch.tensor(drafts), p.shape[1])
            Q = torch.cat([one_hot.double(), torch.zeros(1, p.shape[1], dtype=q.dtype)])
        _, stop = verification.outcome_law(
            p[: K + 1].double().unsqueeze(0),
            Q.double().unsqueeze(0),
            torch.tensor([list(drafts) + [0]]),
            torch.tensor([K]),
            law,
        )
        total += weight * (stop[0] * torch.arange(K + 1)).sum().item()
    return total


@pytest.mark.parametrize(
    "keyed, law", [(True, "block"), (True, "token"), (False, "block")]
)
def test_speculative_rule_is_exact_at_verification_acceptance(keyed: bool, law: str):
    num_sequences = 4000
    paths, first_kept, p, q = _speculate(keyed, law, num_sequences)

    expected = torch.outer(p[0], p[1]).flatten() * num_sequences
    counts = torch.bincount(
        paths[:, 0] * p.shape[1] + paths[:, 1], minlength=expected.numel()
    )
    assert stats.chisquare(counts.numpy(), expected.numpy()).pvalue > 1e-3
    # The acceptance is the verification law's: block (or token-level)
    # verification of keyed drafts, the target probability of plain ones.
    assert first_kept == pytest.approx(_expected_first_kept(p, q, law, keyed), abs=0.05)


def test_residual_verification_makes_point_masses_and_keeps_skipped_requests():
    from vllm.v1.watermarking.textseal2 import TextSeal2VerificationWatermarker

    generator = torch.Generator().manual_seed(4)
    logits = 2 * torch.randn(9, 40, generator=generator)
    contexts = torch.randint(0, 40, (9, 3), generator=generator)
    draft_tokens = torch.randint(0, 40, (9,), generator=generator)
    draft_tokens[4] = -1  # an invalid draft: its request keeps its logits
    cu_num_logits = torch.tensor([0, 3, 6, 7, 9], dtype=torch.int32)
    skip_mask = torch.zeros(9, dtype=torch.bool)
    skip_mask[6] = True  # every row of the third request is skipped
    contexts[7, 0] = -1  # a partial context races on the noise alone
    sampler = RandomSampler(
        expanded_idx_mapping=torch.arange(9, dtype=torch.int32),
        temperatures=torch.ones(9),
        seeds=torch.arange(9),
        positions=torch.arange(9),
    )
    verifier = TextSeal2VerificationWatermarker(key=3, context_width=3)

    output = verifier.residual_verification_logits(
        logits, contexts, draft_tokens, cu_num_logits, skip_mask, logits, sampler
    )

    raced = torch.tensor([1, 1, 1, 0, 0, 0, 0, 1, 1], dtype=torch.bool)
    assert torch.equal(output[~raced], logits[~raced])
    assert (torch.isfinite(output[raced]).sum(-1) == 1).all()
    for start, end in ((0, 3), (7, 9)):
        chosen = output[start:end].argmax(-1)
        drafts = draft_tokens[start : end - 1]
        # Point masses below the stopping row are the drafts.
        stop = next(
            (i for i in range(end - start - 1) if chosen[i] != drafts[i]),
            end - start - 1,
        )
        assert torch.equal(chosen[:stop], drafts[:stop])


def test_gpu_sampler_hands_the_draft_distributions_to_the_verifier():
    from types import SimpleNamespace

    from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler
    from vllm.v1.watermarking.textseal2 import TextSeal2VerificationWatermarker

    calls = {}

    class Recorder(TextSeal2VerificationWatermarker):
        def residual_verification_logits(self, *args):
            calls["args"] = args
            return args[0]

    temperatures = torch.tensor([0.5, 2.0, 0.0])
    sampler = SimpleNamespace(
        watermarker=Recorder(key=1, context_width=2),
        watermarking=SimpleNamespace(gpu=torch.ones(3, dtype=torch.bool)),
        sampling_states=SimpleNamespace(
            temperature=SimpleNamespace(gpu=temperatures),
            seeds=SimpleNamespace(gpu=torch.tensor([11, 22, 33])),
        ),
        deduplicate_contexts="none",
        use_fp64_gumbel=False,
        _get_contexts=lambda *args: torch.zeros(4, 2, dtype=torch.long),
    )
    draft_logits = torch.randn(3, 2, 7)
    expanded_idx_mapping = torch.tensor([1, 1, 1, 2], dtype=torch.int32)
    expanded_local_pos = torch.tensor([0, 1, 2, 0], dtype=torch.int32)
    pos = torch.tensor([40, 41, 42, 9])

    GPUWatermarkSampler.watermark_verification_logits(
        sampler,
        torch.randn(4, 7),
        expanded_idx_mapping,
        expanded_local_pos,
        torch.tensor([5, 6, 7, 8]),
        torch.tensor([0, 3, 4], dtype=torch.int32),
        draft_logits=draft_logits,
        pos=pos,
    )

    _, _, draft_tokens, _, skip_mask, row_draft_logits, random_sampler = calls["args"]
    assert draft_tokens.tolist() == [6, 7, 8, 5]
    # Each row's draft logits at its request and draft step, temperature-scaled
    # (the bonus row's step is clamped; greedy rows are not scaled).
    expected = torch.stack(
        [
            draft_logits[1, 0] / 2.0,
            draft_logits[1, 1] / 2.0,
            draft_logits[1, 1] / 2.0,
            draft_logits[2, 0],
        ]
    )
    assert torch.allclose(row_draft_logits, expected)
    assert torch.equal(random_sampler.positions, pos)
    assert torch.equal(random_sampler.seeds, torch.tensor([11, 22, 33]))
    assert skip_mask.tolist() == [False, False, False, True]


def test_config_selects_the_speculative_rule():
    from vllm.v1.watermarking import (
        SupportsResidualVerification,
        SupportsSpeculativeDecoding,
        SupportsVerificationLogits,
    )
    from vllm.v1.watermarking.spec_decode import (
        create_speculative_draft_watermarker,
        create_speculative_target_watermarker,
        speculative_target_watermark_key,
    )
    from vllm.v1.watermarking.textseal2 import (
        TextSeal2DraftWatermarker,
        TextSeal2VerificationWatermarker,
    )

    config = WatermarkConfig(
        algorithm="textseal2",
        key=1,
        verification_law="token",
        pass_aggregator="tsum",
        pass_source="child",
    )
    watermarker = create_watermarker(config)
    target = create_speculative_target_watermarker(watermarker)
    draft = create_speculative_draft_watermarker(
        watermarker, 4, torch.device("cpu"), allow_target_only=False
    )

    assert config.supports_speculative_decoding
    assert isinstance(watermarker, SupportsSpeculativeDecoding)
    assert isinstance(target, TextSeal2VerificationWatermarker)
    assert isinstance(target, SupportsVerificationLogits)
    assert isinstance(target, SupportsResidualVerification)
    assert (target.law, target.pass_aggregator, target.pass_source) == (
        "token",
        "tsum",
        "child",
    )
    assert draft is not None and isinstance(
        draft.watermarker, TextSeal2DraftWatermarker
    )
    assert speculative_target_watermark_key(config) is None

    plain = create_watermarker(
        WatermarkConfig(algorithm="textseal2", key=1, keyed_drafts=False)
    )
    assert not isinstance(plain, SupportsSpeculativeDecoding)
    assert plain.verifies_unwatermarked_drafts
    assert (
        create_speculative_draft_watermarker(
            plain, 4, torch.device("cpu"), allow_target_only=False
        )
        is None
    )


@requires_cuda
def test_draft_watermarker_writes_the_draft_logits_and_races_on_the_shared_noise():
    """Through the engine's DraftWatermarker: the drafts are the keyed races on
    the residual noise of (seed, position, step), and the draft logits land in
    the cache verification reads."""
    from vllm.v1.watermarking.spec_decode import create_speculative_draft_watermarker
    from vllm.v1.watermarking.textseal2 import race_clocks, residual_noise

    watermarker = create_watermarker(
        WatermarkConfig(algorithm="textseal2", key=5, context_width=2)
    )
    num_rows, vocab_size = 8, 3000
    drafter = create_speculative_draft_watermarker(
        watermarker, num_rows, torch.device("cuda"), allow_target_only=False
    )
    assert drafter is not None
    generator = torch.Generator().manual_seed(6)
    contexts = torch.randint(0, vocab_size, (num_rows, 2), generator=generator).cuda()
    drafter.prepare(
        contexts,
        torch.ones(num_rows, dtype=torch.bool, device="cuda"),
        torch.zeros(num_rows, 16, dtype=torch.int64, device="cuda"),
        torch.zeros(num_rows, dtype=torch.int32, device="cuda"),
        torch.full((num_rows,), 8, dtype=torch.int32, device="cuda"),
    )
    logits = 3 * torch.randn(num_rows, vocab_size, generator=generator).cuda()
    cache = torch.zeros(num_rows, 1, vocab_size, device="cuda")
    idx_mapping = torch.arange(num_rows, dtype=torch.int32, device="cuda")
    temperatures = torch.full((num_rows,), 0.7, device="cuda")
    seeds = torch.arange(num_rows, dtype=torch.int64, device="cuda") + 100
    positions = torch.full((num_rows,), 7, dtype=torch.int64, device="cuda")
    step = torch.tensor(0, dtype=torch.int64, device="cuda")

    drafts = drafter.sample(
        logits,
        idx_mapping,
        temperatures,
        seeds,
        positions,
        apply_temperature=True,
        is_drafting=True,
        logits_cache=cache,
        logits_cache_col=step,
    )

    assert torch.equal(cache[:, 0], logits)
    noise = residual_noise(seeds, positions, step.expand(num_rows), vocab_size)
    virtual = watermarker.draft_watermarker.virtual_key(
        contexts, torch.arange(vocab_size, device="cuda")
    )
    clocks = race_clocks(
        virtual,
        noise,
        watermarker.correlation,
        torch.ones(num_rows, dtype=torch.bool, device="cuda"),
    )
    expected = torch.argmax((logits / 0.7).double() - torch.log(clocks), dim=-1)
    assert torch.equal(drafts, expected)
