# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""block_verified_gumbel: keyed drafts, block (or token-level) verification on
the leftovers of the draft races, and the tau-pass lookahead."""

import itertools
import math
from types import SimpleNamespace

import pytest
import torch
from scipy import stats

from vllm.config.watermarking import WatermarkConfig
from vllm.platforms import current_platform
from vllm.v1.watermarking import (
    BlockVerifiedGumbelDraftWatermarker,
    BlockVerifiedGumbelVerificationWatermarker,
    BlockVerifiedGumbelWatermarker,
    GumbelWatermarkDetector,
    GumbelWatermarker,
    LookaheadGumbelWatermarker,
    SupportsDraftLaw,
    SupportsDraftLawVerification,
    SupportsSpeculativeDecoding,
    SupportsVerificationLogits,
    create_watermarker,
)
from vllm.v1.watermarking import textseal2_verification as verification
from vllm.v1.watermarking.block_verified_gumbel import draft_law_logits
from vllm.v1.watermarking.watermarker import RandomSampler

KEY = 11
AGGREGATORS = ("none", "identity", "max", "tsum", "expsum")
requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)


# ---- a vectorized engine on a toy model ---------------------------------------


def _toy_model(V: int, seed: int, sharpness: float = 1.5):
    """Prefix-dependent target and draft distributions: row ``prev`` is the
    law after token ``prev`` (row ``V``: the first generated token)."""
    generator = torch.Generator().manual_seed(seed)
    shape = (V + 1, V)
    p = torch.softmax(sharpness * torch.randn(shape, generator=generator).double(), -1)
    q = torch.softmax(p.log() + torch.randn(shape, generator=generator).double(), -1)
    return p, q


def _truncate(log_p: torch.Tensor, top_k: int | None) -> torch.Tensor:
    """Top-k as vLLM truncates, on log-probabilities (renormalized)."""
    if top_k is None:
        return log_p
    threshold = log_p.topk(top_k, dim=-1).values[:, -1:]
    log_p = log_p.masked_fill(log_p < threshold, float("-inf"))
    return log_p - log_p.logsumexp(-1, keepdim=True)


def _gather_contexts(ext: torch.Tensor, ends: torch.Tensor, width: int):
    index = ends.unsqueeze(1) - width + torch.arange(width)
    return ext.gather(1, index)


def _speculate(
    *,
    law: str = "block",
    aggregator: str = "none",
    num_sequences: int = 100_000,
    length: int = 3,
    K: int = 3,
    V: int = 5,
    seed: int = 0,
    top_k: int | None = None,
    top_p: float = 1.0,
    draft_top_k: int | None = None,
    variable_k: bool = False,
    draft_mask: float = 0.0,
    foreign: float = 0.0,
    verify_mask: float | None = None,
    pad: int = 0,
    target_slots: int = 1024,
    sharpness: float = 1.5,
    device: str = "cpu",
):
    """Generate ``length`` tokens per sequence the way the engine does:
    keyed drafts on the draft law, ``draft_law_verification_logits``, and
    rejection sampling, which emits the point masses up to the first rejected
    draft. Every sequence has its own prompt, so its contexts (and keyed
    lanes) are its own; ``draft_mask`` / ``verify_mask`` mark random rows as
    repeated contexts at the drafter / at verification (the same rows unless
    ``verify_mask`` is given); ``pad`` -1 tokens start every prompt (partial
    contexts); ``draft_top_k`` truncates the draft law alone; ``foreign`` is
    the share of drafts replaced by a token outside the draft law. Returns the
    generated paths and texts, the first round's kept drafts per sequence, the
    target and draft-law tables and the verifier."""
    width = length + K + 2
    p, q = _toy_model(V, seed, sharpness)
    log_p = _truncate(p.log(), top_k)
    num_rows_q = q.shape[0]
    q_law = draft_law_logits(
        q.log().float(),
        torch.full((num_rows_q,), draft_top_k or top_k or 2**30),
        torch.full((num_rows_q,), top_p),
        torch.zeros(num_rows_q),
    ).double()
    q_law = q_law - q_law.logsumexp(-1, keepdim=True)
    generator = torch.Generator().manual_seed(seed + 1)
    prompts = torch.randint(10, 2**30, (num_sequences, width), generator=generator)
    prompts[:, :pad] = -1
    seeds = torch.randint(0, 2**62, (num_sequences,), generator=generator)
    total = width + length + 2 * K + 2
    text = torch.zeros(num_sequences, total, dtype=torch.long)
    text[:, :width] = prompts
    n = torch.full((num_sequences,), width)
    drafter = BlockVerifiedGumbelDraftWatermarker(KEY, width)
    verifier = BlockVerifiedGumbelVerificationWatermarker(
        KEY, width, law=law, pass_aggregator=aggregator, target_slots=target_slots
    )
    first_kept = None
    while True:
        active = (n < width + length).nonzero().squeeze(1)
        if active.numel() == 0:
            break
        A = active.numel()
        n_a = n[active]
        Ks = (
            torch.randint(0, K + 1, (A,), generator=generator)
            if variable_k
            else torch.full((A,), K)
        )
        ext = text[active].clone()
        drafts = torch.zeros(A, K, dtype=torch.long)
        draft_masks = torch.rand(A, K, generator=generator) < draft_mask
        for j in range(K):
            ends = n_a + j
            contexts = _gather_contexts(ext, ends, width)
            prev = torch.where(
                ends - 1 >= width, ext.gather(1, (ends - 1)[:, None])[:, 0], V
            )
            law_logits = q_law[prev].float()
            keyed = drafter._sample_watermarked(
                law_logits.to(device), contexts.to(device)
            ).token_ids.cpu()
            noise = -torch.log(
                -torch.log(torch.rand(A, V, generator=generator).clamp_min(1e-30))
            )
            plain = torch.argmax(law_logits + noise, dim=-1)
            drafts[:, j] = torch.where(draft_masks[:, j], plain, keyed)
            # A token outside the draft law (its least likely one).
            outside = torch.rand(A, generator=generator) < foreign
            drafts[:, j] = torch.where(outside, law_logits.argmin(-1), drafts[:, j])
            ext.scatter_(1, ends[:, None], drafts[:, j : j + 1])

        sizes = Ks + 1
        request = torch.repeat_interleave(torch.arange(A), sizes)
        starts = torch.cumsum(sizes, 0) - sizes
        local = torch.arange(int(sizes.sum())) - starts[request]
        ends = n_a[request] + local
        contexts = _gather_contexts(ext[request], ends, width)
        prev_index = (ends - 1).clamp_min(0)
        prev = torch.where(
            ends - 1 >= width, ext[request].gather(1, prev_index[:, None])[:, 0], V
        )
        is_draft = local < Ks[request]
        draft_tokens = torch.where(
            is_draft, drafts[request, local.clamp_max(K - 1)] if K else 0, 0
        )
        if verify_mask is None:
            row_mask = (
                torch.where(
                    is_draft, draft_masks[request, local.clamp_max(K - 1)], False
                )
                if K
                else torch.zeros_like(is_draft)
            )
            bonus_mask = torch.rand(len(local), generator=generator) < draft_mask
            repeated = torch.where(is_draft, row_mask, bonus_mask)
        else:
            repeated = torch.rand(len(local), generator=generator) < verify_mask
        cu = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(sizes, 0)])
        sampler = RandomSampler(
            expanded_idx_mapping=request.to(torch.int32).to(device),
            temperatures=torch.ones(A, device=device),
            seeds=seeds[active].to(device),
            positions=(ends - 1).to(device),
        )
        output = verifier.draft_law_verification_logits(
            log_p[prev].float().to(device),
            contexts.to(device),
            draft_tokens.to(device),
            cu.to(torch.int32).to(device),
            torch.ones(len(local), dtype=torch.bool, device=device),
            repeated.to(device),
            q_law[prev].float().to(device),
            sampler,
        )
        chosen = output.argmax(-1).cpu()
        mismatch = ~is_draft | (chosen != draft_tokens)
        stop = torch.full((A,), K + 1).scatter_reduce(
            0, request, torch.where(mismatch, local, K + 1), reduce="amin"
        )
        emitted = local <= stop[request]
        positions = ends[emitted]
        rows = request[emitted]
        tokens = torch.where(local < stop[request], draft_tokens, chosen)[emitted]
        updated = text[active]
        updated[rows, positions] = tokens
        text[active] = updated
        n[active] = n_a + stop + 1
        if first_kept is None:
            first_kept = stop.double()
    return SimpleNamespace(
        paths=text[:, width : width + length],
        text=text[:, : width + length],
        width=width,
        kept=first_kept,
        log_p=log_p,
        q_law=q_law,
        verifier=verifier,
    )


def _path_law(log_p: torch.Tensor, V: int, length: int) -> torch.Tensor:
    law = torch.zeros(V**length, dtype=torch.float64)
    for index, path in enumerate(itertools.product(range(V), repeat=length)):
        prev, logp = V, 0.0
        for token in path:
            logp += log_p[prev, token].item()
            prev = token
        law[index] = math.exp(logp)
    return law


def _chisquare(counts: torch.Tensor, law: torch.Tensor, threshold: float, what: str):
    support = law > 0
    assert counts[~support].sum() == 0, f"{what}: outside the target's support"
    expected = law[support] / law[support].sum() * counts.sum()
    result = stats.chisquare(counts[support].numpy(), expected.numpy())
    assert result.pvalue > threshold, (what, result)


def _assert_target_law(paths, log_p, V, length, threshold=1e-4):
    """Chi-square tests of the generated paths against the target: the joint
    law, every position's marginal and every adjacent pair (the
    low-dimensional tests see biases of ~0.2 pp at 2e5 paths)."""
    law = _path_law(log_p, V, length).view((V,) * length)
    codes = torch.zeros(paths.shape[0], dtype=torch.long)
    for t in range(length):
        codes = codes * V + paths[:, t]
    counts = torch.bincount(codes, minlength=V**length).double()
    _chisquare(counts, law.flatten(), threshold, "joint")
    for t in range(length):
        others = tuple(u for u in range(length) if u != t)
        marginal = law.sum(others) if others else law
        _chisquare(
            torch.bincount(paths[:, t], minlength=V).double(),
            marginal,
            threshold,
            f"position {t}",
        )
    for t in range(length - 1):
        others = tuple(u for u in range(length) if u not in (t, t + 1))
        pair = (law.sum(others) if others else law).flatten()
        codes = paths[:, t] * V + paths[:, t + 1]
        _chisquare(
            torch.bincount(codes, minlength=V * V).double(),
            pair,
            threshold,
            f"pair {t}",
        )


def _expected_kept(log_p, q_law, V: int, K: int, law: str) -> float:
    """Mean drafts kept in a round from the start, averaged over drafts ~ q:
    the verification law's acceptance (closed form)."""
    total = 0.0
    for drafts in itertools.product(range(V), repeat=K):
        prev = [V] + list(drafts)
        weight = math.prod(q_law[prev[i], d].exp().item() for i, d in enumerate(drafts))
        if weight == 0:
            continue
        P = torch.stack([log_p[prev[i]].exp() for i in range(K + 1)])
        Q = torch.cat(
            [
                torch.stack([q_law[prev[i]].exp() for i in range(K)]),
                torch.zeros(1, V, dtype=torch.float64),
            ]
        )
        _, stop = verification.outcome_law(
            P.unsqueeze(0),
            Q.unsqueeze(0),
            torch.tensor([list(drafts) + [0]]),
            torch.tensor([K]),
            law,
        )
        total += weight * (stop[0] * torch.arange(K + 1)).sum().item()
    return total


# ---- exactness and acceptance ---------------------------------------------------


@pytest.mark.parametrize("law", ["block", "token"])
@pytest.mark.parametrize("aggregator", AGGREGATORS)
def test_emitted_text_has_the_target_law_and_verification_acceptance(law, aggregator):
    """Monte Carlo over keys (every sequence has its own contexts): the first
    three tokens follow the target, prefix-dependent, and the drafts kept in
    the first round match the verification law's closed form, for every
    pass aggregator."""
    num_sequences, V, K = 200_000, 5, 3
    run = _speculate(
        law=law, aggregator=aggregator, num_sequences=num_sequences, V=V, K=K
    )
    _assert_target_law(run.paths, run.log_p, V, 3)
    expected = _expected_kept(run.log_p, run.q_law, V, K, law)
    kept = run.kept
    se = kept.std().item() / math.sqrt(num_sequences)
    assert abs(kept.mean().item() - expected) < 4 * se, (kept.mean(), expected, se)
    fallbacks, unkeyed, raced = run.verifier.fallback_stats()
    assert fallbacks == 0 and unkeyed == 0 and raced > 0


def test_block_accepts_more_than_token_and_shared():
    num_sequences, V, K = 60_000, 5, 3
    means = {}
    for law in ("block", "token", "shared"):
        run = _speculate(law=law, num_sequences=num_sequences, V=V, K=K, seed=3)
        _assert_target_law(run.paths, run.log_p, V, 3)
        means[law] = run.kept.mean().item()
    assert means["block"] > means["token"] > means["shared"], means


@pytest.mark.parametrize(
    "options",
    [
        {"top_k": 3},
        {"top_k": 4, "top_p": 0.8},
        {"variable_k": True},
        {"draft_mask": 0.3},
        {"draft_mask": 0.3, "verify_mask": 0.3},
        {"pad": 5},
        {"target_slots": 2, "draft_top_k": 2, "V": 8, "sharpness": 0.5},
    ],
    ids=[
        "top_k",
        "top_k_top_p",
        "variable_drafts",
        "repeated_contexts",
        "repeated_contexts_disagree",
        "partial_contexts",
        "rest_slot",
    ],
)
@pytest.mark.parametrize("aggregator", ["none", "tsum"])
def test_exact_under_truncation_masking_batching_and_partial_contexts(
    options, aggregator
):
    """The target's top-k (with the draft law truncated alike), requests with
    different numbers of drafts, repeated contexts on fresh lanes (also when
    the drafter and verification disagree about them), contexts that reach
    before the output, and the merged rest slot of a large support."""
    run = _speculate(aggregator=aggregator, num_sequences=100_000, seed=5, **options)
    _assert_target_law(run.paths, run.log_p, options.get("V", 5), 3)
    fallbacks, unkeyed, raced = run.verifier.fallback_stats()
    assert fallbacks == 0 and raced > 0
    if "verify_mask" not in options:
        assert unkeyed == 0


def test_drafts_that_lost_their_keyed_race_are_verified_on_fresh_lanes():
    """Repeated contexts at the drafter only (drafts sampled without the key,
    verified as keyed): the drafts that are not their keyed race's winner are
    verified on fresh lanes, and the text stays exact."""
    run = _speculate(num_sequences=100_000, draft_mask=0.3, verify_mask=0.0, seed=7)
    _assert_target_law(run.paths, run.log_p, 5, 3)
    fallbacks, unkeyed, raced = run.verifier.fallback_stats()
    assert fallbacks == 0 and unkeyed > 0


def test_drafts_outside_the_draft_law_use_the_shared_clocks():
    """Drafts the draft law cannot have produced send their request to the
    shared clocks, which is exact for any drafts."""
    run = _speculate(num_sequences=100_000, top_k=3, foreign=0.2, seed=8, K=2)
    _assert_target_law(run.paths, run.log_p, 5, 3)
    fallbacks, _, raced = run.verifier.fallback_stats()
    assert 0 < fallbacks < raced


@pytest.mark.parametrize("law", ["block", "token"])
@pytest.mark.parametrize(
    "aggregator, source",
    [(a, "own") for a in AGGREGATORS] + [("tsum", "child"), ("identity", "child")],
)
def test_fast_race_matches_the_reference(law, aggregator, source):
    """``residual_race_fast`` (vectorized law and chain) against
    ``textseal2_verification.residual_race`` on random rounds: ragged K
    (K = 0 included), drafts the target cannot emit, padding."""
    from vllm.v1.watermarking.block_verified_gumbel import (
        outcome_law_fast,
        residual_race_fast,
    )

    generator = torch.Generator().manual_seed(12)
    B, D, C = 3000, 6, 12
    K = torch.randint(0, D, (B,), generator=generator)
    depth = torch.arange(D).unsqueeze(0)
    P = torch.softmax(2 * torch.randn(B, D, C, generator=generator).double(), -1)
    P = P * (torch.rand(B, D, C, generator=generator) > 0.2)  # truncated supports
    P = P / P.sum(-1, keepdim=True).clamp_min(1e-300)
    Q = torch.softmax(2 * torch.randn(B, D, C, generator=generator).double(), -1)
    Q = Q * (depth < K.unsqueeze(1)).unsqueeze(2)
    E = torch.empty(B, D, C, dtype=torch.float64).exponential_(generator=generator)
    ratio = torch.where(Q > 0, E / torch.where(Q > 0, Q, 1.0), torch.inf)
    dpos = torch.where(depth < K.unsqueeze(1), ratio.argmin(-1), 0)
    pad = (depth > K.unsqueeze(1)).unsqueeze(2)
    P, E = P.masked_fill(pad, 0.0), E.masked_fill(pad, 1.0)

    pi, stop = verification.outcome_law(P, Q, dpos, K, law)
    pi_fast, stop_fast = outcome_law_fast(P, Q, dpos, K, law)
    assert torch.allclose(pi, pi_fast, rtol=1e-9, atol=1e-12)
    assert torch.allclose(stop, stop_fast, rtol=1e-9, atol=1e-12)
    expected = verification.residual_race(E, P, Q, dpos, K, law, aggregator, source)
    fast = residual_race_fast(E, P, Q, dpos, K, law, aggregator, source)
    assert torch.equal(expected[0], fast[0]) and torch.equal(expected[1], fast[1])


# ---- handoff edge cases ----------------------------------------------------------


def _round(verifier, logits, law, drafts, contexts, sizes, repeated=None, seeds=None):
    N = logits.shape[0]
    cu = torch.tensor([0] + torch.tensor(sizes).cumsum(0).tolist(), dtype=torch.int32)
    request = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
    sampler = RandomSampler(
        expanded_idx_mapping=request.to(torch.int32),
        temperatures=torch.ones(len(sizes)),
        seeds=torch.arange(len(sizes)) + 1 if seeds is None else seeds,
        positions=torch.arange(N),
    )
    return verifier.draft_law_verification_logits(
        logits,
        contexts,
        drafts,
        cu,
        torch.ones(N, dtype=torch.bool),
        torch.zeros(N, dtype=torch.bool) if repeated is None else repeated,
        law,
        sampler,
    ).argmax(-1)


def _distinct_contexts(N: int, width: int = 4, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(10, 2**30, (N, width), generator=generator)


@pytest.mark.parametrize("aggregator", AGGREGATORS)
def test_no_drafts_is_gumbel_max(aggregator):
    N, V = 500, 30
    logits = 2 * torch.randn(N, V, generator=torch.Generator().manual_seed(1))
    contexts = _distinct_contexts(N)
    verifier = BlockVerifiedGumbelVerificationWatermarker(
        KEY, 4, pass_aggregator=aggregator
    )
    chosen = _round(
        verifier, logits, logits, torch.zeros(N, dtype=torch.long), contexts, [1] * N
    )
    expected = GumbelWatermarker(KEY, 4)._sample_watermarked(logits, contexts).token_ids
    assert torch.equal(chosen, expected)


@pytest.mark.parametrize("law", ["block", "token"])
@pytest.mark.parametrize("aggregator", AGGREGATORS)
def test_target_equal_to_draft_keeps_every_draft(law, aggregator):
    B, K, V = 300, 4, 20
    N = B * (K + 1)
    logits = 2 * torch.randn(N, V, generator=torch.Generator().manual_seed(2))
    contexts = _distinct_contexts(N)
    drafts = (
        BlockVerifiedGumbelDraftWatermarker(KEY, 4)
        ._sample_watermarked(logits, contexts)
        .token_ids
    )
    verifier = BlockVerifiedGumbelVerificationWatermarker(
        KEY, 4, law=law, pass_aggregator=aggregator
    )
    chosen = _round(verifier, logits, logits, drafts, contexts, [K + 1] * B).view(
        B, K + 1
    )
    assert torch.equal(chosen[:, :K], drafts.view(B, K + 1)[:, :K])


@pytest.mark.parametrize("aggregator", AGGREGATORS)
def test_disjoint_supports_reject_at_once_with_a_correction_from_the_target(aggregator):
    """Disjoint supports of q and p: the first draft is rejected and the
    token emitted in its place follows p."""
    B, V = 40_000, 6
    p = torch.tensor([0.5, 0.3, 0.2, 0.0, 0.0, 0.0], dtype=torch.float64)
    q = torch.tensor([0.0, 0.0, 0.0, 0.6, 0.3, 0.1], dtype=torch.float64)
    logits = p.log().float().expand(2 * B, V).contiguous()
    law = q.log().float().expand(2 * B, V).contiguous()
    contexts = _distinct_contexts(2 * B, seed=3)
    drafts = (
        BlockVerifiedGumbelDraftWatermarker(KEY, 4)
        ._sample_watermarked(law, contexts)
        .token_ids
    )
    verifier = BlockVerifiedGumbelVerificationWatermarker(
        KEY, 4, pass_aggregator=aggregator
    )
    chosen = _round(verifier, logits, law, drafts, contexts, [2] * B).view(B, 2)
    assert (chosen[:, 0] != drafts.view(B, 2)[:, 0]).all()
    counts = torch.bincount(chosen[:, 0], minlength=V)[:3].double()
    assert stats.chisquare(counts.numpy(), (p[:3] * B).numpy()).pvalue > 1e-3


@pytest.mark.parametrize("aggregator", AGGREGATORS + ("shared",))
def test_handoff_example_acceptance(aggregator):
    """The handoff's example q = (0.6, 0.3, 0.1), p = (0.2, 0.4, 0.4), one
    draft: the rule keeps it with probability sum min(p, q) = 0.6 for every
    pass aggregator and emits p; shared-clock verification keeps it with
    probability 0.55."""
    B, V = 200_000, 3
    p = torch.tensor([0.2, 0.4, 0.4], dtype=torch.float64)
    q = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64)
    logits = p.log().float().expand(2 * B, V).contiguous()
    law = q.log().float().expand(2 * B, V).contiguous()
    contexts = _distinct_contexts(2 * B, seed=4)
    drafts = (
        BlockVerifiedGumbelDraftWatermarker(KEY, 4)
        ._sample_watermarked(law, contexts)
        .token_ids
    )
    shared = aggregator == "shared"
    verifier = BlockVerifiedGumbelVerificationWatermarker(
        KEY,
        4,
        law="shared" if shared else "block",
        pass_aggregator="none" if shared else aggregator,
    )
    chosen = _round(verifier, logits, law, drafts, contexts, [2] * B).view(B, 2)
    kept = chosen[:, 0] == drafts.view(B, 2)[:, 0]
    expected = 0.55 if shared else 0.6
    assert kept.double().mean().item() == pytest.approx(expected, abs=0.004)
    counts = torch.bincount(chosen[:, 0], minlength=V).double()
    assert stats.chisquare(counts.numpy(), (p * B).numpy()).pvalue > 1e-3


# ---- point-mass drafts: lookahead Gumbel-max ---------------------------------------


@pytest.mark.parametrize(
    "aggregator, lookahead",
    [
        ("identity", "pi1"),
        ("none", "pi2"),
        ("max", "max"),
        ("tsum", "tsum"),
        ("expsum", "expsum"),
    ],
)
def test_point_mass_drafts_are_lookahead_gumbel_max(aggregator, lookahead):
    """Drafts that do not depend on the key, verified as point masses, give
    lookahead Gumbel-max's rounds (same stopping row and emitted token)."""
    generator = torch.Generator().manual_seed(6)
    B, K, V = 2000, 4, 40
    sizes = torch.randint(1, K + 2, (B,), generator=generator)
    N = int(sizes.sum())
    cu = torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)]).to(torch.int32)
    logits = 4 * torch.randn(N, V, generator=generator)
    contexts = _distinct_contexts(N, width=6, seed=7)
    # Likely drafts, so that rounds go deep.
    drafts = torch.where(
        torch.rand(N, generator=generator) < 0.9,
        logits.argmax(-1),
        torch.randint(0, V, (N,), generator=generator),
    )
    skip = torch.zeros(N, dtype=torch.bool)
    ours = (
        BlockVerifiedGumbelVerificationWatermarker(KEY, 6, pass_aggregator=aggregator)
        .verification_logits(logits, contexts, drafts, cu, skip)
        .argmax(-1)
    )
    native = LookaheadGumbelWatermarker(KEY, 6, aggregator=lookahead).lookahead_tokens(
        logits, contexts, LookaheadGumbelWatermarker.children(drafts, cu), cu, skip
    )

    def outcome(chosen):
        request = torch.repeat_interleave(torch.arange(B), sizes)
        local = torch.arange(N) - (cu[:-1].long())[request]
        mismatch = (local == sizes[request] - 1) | (chosen != drafts)
        stop = torch.full((B,), N).scatter_reduce(
            0, request, torch.where(mismatch, local, N), reduce="amin"
        )
        return stop, chosen[cu[:-1].long() + stop]

    stop_ours, token_ours = outcome(ours)
    stop_native, token_native = outcome(native)
    agree = (stop_ours == stop_native) & (token_ours == token_native)
    # float32 (lookahead) against float64 (here): near ties may differ.
    assert agree.double().mean().item() > 0.998, agree.double().mean()
    assert stop_ours.double().mean() > 0.6


# ---- detection -----------------------------------------------------------------------


def test_detector_reads_the_lanes_the_rule_races_on():
    """The rule races on -log U and the Gumbel-max detector scores -log(1 - U)
    with the same key, PRF and contexts."""
    verifier = create_watermarker(
        WatermarkConfig(key=KEY, algorithm="block_verified_gumbel", context_width=3)
    ).target_watermarker
    detector = GumbelWatermarkDetector(KEY, context_width=3)
    contexts = torch.tensor([[-1, -1, 5], [5, 6, 7], [8, 9, 10]])
    tokens = torch.tensor([[1], [2], [3]])
    assert torch.equal(
        verifier.prf.uniform(contexts, tokens), detector.prf.uniform(contexts, tokens)
    )
    assert verifier.context_width == detector.context_width


def test_watermarked_text_is_detected_and_null_p_values_are_uniform():
    """The Gumbel-max detector's scores of the generated tokens, on the
    contexts they were generated at: watermarked with the key, Exp(1) (null)
    with another key."""
    run = _speculate(
        aggregator="tsum",
        num_sequences=300,
        length=40,
        K=3,
        V=40,
        seed=9,
        sharpness=1.0,
    )
    width = run.width
    positions = torch.arange(width, width + 40)
    contexts = run.text.unfold(1, width, 1)[:, positions - width]  # [S, 40, width]
    targets = run.text[:, positions]

    def p_values(key: int) -> list[float]:
        detector = GumbelWatermarkDetector(key, context_width=width)
        scores = detector._score_tokens(
            contexts.reshape(-1, width), targets.reshape(-1)
        )
        totals = scores.view(targets.shape).sum(-1)
        return [detector._get_p_value(t.item(), targets.shape[1]) for t in totals]

    watermarked = p_values(KEY)
    assert sorted(watermarked)[len(watermarked) // 2] < 1e-4
    assert stats.kstest(p_values(KEY + 1), "uniform").pvalue > 1e-3


# ---- configuration and plumbing ------------------------------------------------------


def test_config_and_factory():
    from vllm.v1.watermarking.spec_decode import (
        create_speculative_draft_watermarker,
        create_speculative_target_watermarker,
        speculative_target_watermark_key,
    )

    config = WatermarkConfig(
        key=1,
        algorithm="block_verified_gumbel",
        verification_law="token",
        pass_aggregator="identity",
        context_width=5,
    )
    assert config.supports_speculative_decoding
    assert not config.verifies_unwatermarked_drafts
    watermarker = create_watermarker(config)
    assert isinstance(watermarker, BlockVerifiedGumbelWatermarker)
    assert isinstance(watermarker, GumbelWatermarker)  # plain sampling: gumbel
    assert isinstance(watermarker, SupportsSpeculativeDecoding)
    target = create_speculative_target_watermarker(watermarker)
    assert isinstance(target, BlockVerifiedGumbelVerificationWatermarker)
    assert isinstance(target, SupportsVerificationLogits)
    assert isinstance(target, SupportsDraftLawVerification)
    assert (target.law, target.pass_aggregator, target.pass_source) == (
        "token",
        "identity",
        "own",
    )
    assert target.prf.key == GumbelWatermarker(1).prf.key  # gumbel's key, unchanged
    assert speculative_target_watermark_key(config) is None
    drafter = create_speculative_draft_watermarker(
        watermarker, 4, torch.device("cpu"), allow_target_only=False
    )
    assert drafter is not None and drafter.uses_draft_law
    assert isinstance(drafter.watermarker, BlockVerifiedGumbelDraftWatermarker)
    assert isinstance(drafter.watermarker, SupportsDraftLaw)
    assert drafter.watermarker.prf.key == target.prf.key  # one key
    assert drafter.watermarker.context_width == 5

    for law in ("block", "token", "shared"):
        for aggregator in AGGREGATORS:
            WatermarkConfig(
                key=1,
                algorithm="block_verified_gumbel",
                verification_law=law,
                pass_aggregator=aggregator,
            )
    with pytest.raises(ValueError):
        WatermarkConfig(key=1, algorithm="block_verified_gumbel", pass_aggregator="opt")
    with pytest.raises(ValueError):
        WatermarkConfig(key=1, algorithm="textseal2", verification_law="shared")
    with pytest.raises(ValueError):
        BlockVerifiedGumbelVerificationWatermarker(1, law="greedy")
    # The new aggregator is also TextSeal2's.
    WatermarkConfig(key=1, algorithm="textseal2", pass_aggregator="identity")


def test_vllm_refuses_greedy_drafts_for_keyed_drafting():
    """The rule races the drafts on the key, so vLLM's greedy drafting is
    refused for it (and still allowed for the lookahead watermarks)."""
    from vllm.config import VllmConfig

    def check(algorithm, method):
        stand_in = SimpleNamespace(
            watermark_config=WatermarkConfig(key=1, algorithm=algorithm),
            speculative_config=SimpleNamespace(
                draft_sample_method=method,
                rejection_sample_method="standard",
                parallel_drafting=False,
                method="draft_model",
            ),
        )
        return VllmConfig._check_supports_watermarking(stand_in)

    assert check("block_verified_gumbel", "probabilistic")
    with pytest.raises(ValueError, match="probabilistic"):
        check("block_verified_gumbel", "greedy")
    assert check("lookahead_gumbel", "greedy")


def test_draft_watermarker_races_watermarked_requests_on_the_draft_law(monkeypatch):
    """Through the engine's DraftWatermarker: a watermarked request drafts the
    keyed race on its draft law (temperature, then its top-k); opted-out and
    greedy requests get vLLM's temperature-only draft distribution."""
    from vllm.v1.watermarking import spec_decode
    from vllm.v1.watermarking.spec_decode import create_speculative_draft_watermarker

    seen = {}

    class Recording(RandomSampler):
        def __call__(self, logits):
            seen["logits"] = logits.clone()
            return logits.argmax(dim=-1)

    monkeypatch.setattr(spec_decode, "RandomSampler", Recording)
    watermarker = create_watermarker(
        WatermarkConfig(key=KEY, algorithm="block_verified_gumbel", context_width=3)
    )
    drafter = create_speculative_draft_watermarker(
        watermarker, 4, torch.device("cpu"), allow_target_only=False
    )
    assert drafter is not None
    generator = torch.Generator().manual_seed(13)
    V = 50
    contexts = torch.randint(0, 10**6, (3, 3), generator=generator)
    enabled = torch.tensor([True, False, True])
    drafter.prepare(
        contexts,
        enabled,
        torch.zeros(4, 8, dtype=torch.int64),
        torch.zeros(4, dtype=torch.int32),
        torch.full((4,), 4, dtype=torch.int32),
        top_k=torch.tensor([3, 3, 3], dtype=torch.int32),
        top_p=torch.ones(3),
        min_p=torch.zeros(3),
    )
    logits = 2 * torch.randn(3, V, generator=generator)
    temperature = torch.tensor([0.7, 0.7, 0.0])
    drafts = drafter.sample(
        logits,
        torch.arange(3, dtype=torch.int32),
        temperature,
        torch.arange(3),
        torch.arange(3),
        apply_temperature=True,
        is_drafting=True,
        logits_cache=torch.zeros(3, 1, V),
        logits_cache_col=torch.tensor(0),
    )
    law = draft_law_logits(
        logits / 0.7, torch.tensor([3] * 3), torch.ones(3), torch.zeros(3)
    )
    keyed = watermarker.draft_watermarker._sample_watermarked(law, contexts).token_ids
    assert drafts[0] == keyed[0]
    assert torch.isfinite(seen["logits"][0]).sum() == 3  # the law
    assert torch.isfinite(seen["logits"][1:]).all()  # temperature only
    assert torch.allclose(seen["logits"][1], logits[1] / 0.7)
    assert torch.equal(seen["logits"][2], logits[2])  # greedy: not scaled


def test_gpu_sampler_hands_the_draft_law_to_the_verifier():
    import functools

    from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler

    calls = {}

    class Recorder(BlockVerifiedGumbelVerificationWatermarker):
        def draft_law_verification_logits(self, *args):
            calls["args"] = args
            return args[0]

    temperatures = torch.tensor([0.5, 2.0, 0.0])
    repeated = torch.tensor([False, True, False, False])
    sampler = SimpleNamespace(
        watermarker=Recorder(key=1, context_width=2),
        watermarking=SimpleNamespace(gpu=torch.tensor([True, True, True])),
        sampling_states=SimpleNamespace(
            temperature=SimpleNamespace(gpu=temperatures),
            seeds=SimpleNamespace(gpu=torch.tensor([11, 22, 33])),
            top_k=SimpleNamespace(gpu=torch.tensor([7, 3, 7], dtype=torch.int32)),
            top_p=SimpleNamespace(gpu=torch.tensor([1.0, 0.9, 1.0])),
            min_p=SimpleNamespace(gpu=torch.tensor([0.0, 0.05, 0.0])),
        ),
        deduplicate_contexts="single_turn",
        use_fp64_gumbel=False,
        _get_contexts=lambda *args: torch.zeros(4, 2, dtype=torch.long),
        _get_repeated_contexts=lambda *args: repeated,
    )
    sampler._draft_law_verification_logits = functools.partial(
        GPUWatermarkSampler._draft_law_verification_logits, sampler
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

    (_, _, drafts, _, request_enabled, got_repeated, law, random_sampler) = calls[
        "args"
    ]
    assert drafts.tolist() == [6, 7, 8, 5]
    # Request 2 is greedy: its drafts are verified by vLLM.
    assert request_enabled.tolist() == [True, True, True, False]
    assert torch.equal(got_repeated, repeated)
    rows = torch.stack(
        [
            draft_logits[1, 0] / 2.0,
            draft_logits[1, 1] / 2.0,
            draft_logits[1, 1] / 2.0,
            draft_logits[2, 0],
        ]
    )
    expected = draft_law_logits(
        rows,
        torch.tensor([3, 3, 3, 7]),
        torch.tensor([0.9, 0.9, 0.9, 1.0]),
        torch.tensor([0.05, 0.05, 0.05, 0.0]),
    )
    assert torch.equal(law, expected)
    assert torch.equal(random_sampler.positions, pos)
    assert torch.equal(random_sampler.seeds, torch.tensor([11, 22, 33]))


def test_speculator_hands_the_requests_truncation_to_the_drafter():
    from vllm.v1.watermarking.spec_decode import DraftWatermarker
    from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

    drafter = DraftWatermarker(
        BlockVerifiedGumbelDraftWatermarker(KEY, 2),
        8,
        torch.device("cpu"),
        2,
        "none",
        None,
    )
    states = SimpleNamespace(
        top_k=SimpleNamespace(gpu=torch.arange(8, dtype=torch.int32) + 1),
        top_p=SimpleNamespace(gpu=torch.linspace(0.5, 1.0, 8)),
        min_p=SimpleNamespace(gpu=torch.linspace(0.0, 0.1, 8)),
    )
    sampler = SimpleNamespace(
        _get_contexts=lambda idx: torch.zeros(len(idx), 2, dtype=torch.long),
        watermarking=SimpleNamespace(gpu=torch.ones(8, dtype=torch.bool)),
        req_states=SimpleNamespace(
            all_token_ids=SimpleNamespace(gpu=torch.zeros(8, 4, dtype=torch.long)),
            prompt_len=SimpleNamespace(gpu=torch.zeros(8, dtype=torch.int32)),
            total_len=SimpleNamespace(gpu=torch.zeros(8, dtype=torch.int32)),
        ),
        sampling_states=states,
    )
    idx_mapping = torch.tensor([5, 2, 7])
    DraftModelSpeculator.prepare_watermarking(
        SimpleNamespace(draft_watermarker=drafter), sampler, idx_mapping
    )
    assert torch.equal(drafter.top_k[:3], states.top_k.gpu[idx_mapping])
    assert torch.equal(drafter.top_p[:3], states.top_p.gpu[idx_mapping])
    assert torch.equal(drafter.min_p[:3], states.min_p.gpu[idx_mapping])


def test_draft_law_follows_the_requests_truncation():
    generator = torch.Generator().manual_seed(14)
    logits = 2 * torch.randn(6, 300, generator=generator)
    logits[5, :] = 1.0  # ties everywhere
    top_k = torch.tensor([5, 2**30, 2**30, 40, 1, 2**30])
    top_p = torch.tensor([1.0, 0.8, 1.0, 0.9, 1.0, 1.0])
    min_p = torch.tensor([0.0, 0.0, 0.1, 0.05, 0.0, 0.0])
    law = draft_law_logits(logits, top_k, top_p, min_p)
    support = torch.isfinite(law)
    probs = torch.softmax(logits.double(), -1)
    order = probs.sort(-1, descending=True)
    assert support[0].sum() == 5 and support[0][logits[0].topk(5).indices].all()
    # top-p: the smallest prefix whose mass reaches 0.8
    needed = int((order.values[1].cumsum(0) < 0.8).sum()) + 1
    assert support[1].sum() == needed
    # min-p: p >= 0.1 * p_max
    assert torch.equal(support[2], probs[2] >= 0.1 * probs[2].max())
    assert support[3].sum() <= 40
    assert support[4].sum() == 1
    assert support[5].all()  # a tie at the threshold keeps the whole tie
    assert torch.equal(law[support], logits[support])
    # The same law from a batch of a different size.
    assert torch.equal(
        draft_law_logits(logits[1:3], top_k[1:3], top_p[1:3], min_p[1:3]), law[1:3]
    )


# ---- CUDA ------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("aggregator", ["none", "tsum"])
def test_cuda_rounds_are_exact(aggregator):
    """The Monte Carlo on CUDA: drafts from the Philox Gumbel-max kernel,
    verification with the compiled PRF (and the kernel for the rest slot)."""
    for options in (
        {},
        {"draft_mask": 0.3},
        {"target_slots": 2, "draft_top_k": 2, "V": 8, "sharpness": 0.5},
    ):
        run = _speculate(
            aggregator=aggregator,
            num_sequences=100_000,
            seed=21,
            device="cuda",
            **options,
        )
        _assert_target_law(run.paths, run.log_p, options.get("V", 5), 3)
        fallbacks, unkeyed, raced = run.verifier.fallback_stats()
        assert fallbacks == 0 and raced > 0
        assert unkeyed <= 1e-4 * raced * 4  # float32 drafting near ties only


@requires_cuda
def test_cuda_matches_cpu():
    """Draft law, drafts, lanes and verified rounds agree between CUDA and the
    CPU reference on the same inputs."""
    generator = torch.Generator().manual_seed(15)
    B, K, V = 400, 4, 3000
    sizes = torch.randint(1, K + 2, (B,), generator=generator)
    N = int(sizes.sum())
    cu = torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)]).to(torch.int32)
    request = torch.repeat_interleave(torch.arange(B), sizes)
    logits = 3 * torch.randn(N, V, generator=generator)
    draft = logits + torch.randn(N, V, generator=generator)
    top_k = torch.where(torch.rand(N, generator=generator) < 0.5, 20, 2**30)
    top_p = torch.where(torch.rand(N, generator=generator) < 0.5, 0.9, 1.0)
    target = draft_law_logits(logits, top_k, top_p, torch.zeros(N))
    law_cpu = draft_law_logits(draft, top_k, top_p, torch.zeros(N))
    law_cuda = draft_law_logits(
        draft.cuda(), top_k.cuda(), top_p.cuda(), torch.zeros(N).cuda()
    )
    assert torch.equal(torch.isfinite(law_cpu), torch.isfinite(law_cuda.cpu()))
    contexts = torch.randint(10, 2**30, (N, 5), generator=generator)
    drafter = BlockVerifiedGumbelDraftWatermarker(KEY, 5)
    drafts_cpu = drafter._sample_watermarked(law_cpu, contexts).token_ids
    drafts_cuda = drafter._sample_watermarked(law_cpu.cuda(), contexts.cuda()).token_ids
    assert (drafts_cpu == drafts_cuda.cpu()).double().mean() > 0.999
    repeated = torch.rand(N, generator=generator) < 0.1

    def verify(device, target_slots):
        verifier = BlockVerifiedGumbelVerificationWatermarker(
            KEY, 5, pass_aggregator="tsum", target_slots=target_slots
        )
        sampler = RandomSampler(
            expanded_idx_mapping=request.to(torch.int32).to(device),
            temperatures=torch.ones(B, device=device),
            seeds=torch.arange(B, device=device),
            positions=torch.arange(N, device=device),
        )
        out = verifier.draft_law_verification_logits(
            target.to(device),
            contexts.to(device),
            drafts_cpu.to(device),
            cu.to(device),
            torch.ones(N, dtype=torch.bool, device=device),
            repeated.to(device),
            law_cpu.to(device),
            sampler,
        )
        return out.argmax(-1).cpu()

    for target_slots in (1024, 4):  # 4: the rest slot (Philox kernel on CUDA)
        agree = (
            (verify("cpu", target_slots) == verify("cuda", target_slots))
            .double()
            .mean()
        )
        assert agree > 0.999, (target_slots, agree)


@requires_cuda
@pytest.mark.parametrize("aggregator", ["none", "identity", "tsum"])
def test_cuda_compiled_race_matches_eager(aggregator):
    """The opt-in torch.compile path (VLLM_BV_GUMBEL_COMPILE=1) on CUDA."""
    from vllm.v1.watermarking.block_verified_gumbel import _race, residual_race_fast

    generator = torch.Generator().manual_seed(16)
    for B, D, C in ((64, 4, 40), (37, 6, 1100), (5, 1, 7)):
        K = torch.randint(0, D, (B,), generator=generator)
        depth = torch.arange(D).unsqueeze(0)
        P = torch.softmax(2 * torch.randn(B, D, C, generator=generator).double(), -1)
        Q = torch.softmax(2 * torch.randn(B, D, C, generator=generator).double(), -1)
        Q = Q * (depth < K.unsqueeze(1)).unsqueeze(2)
        E = torch.empty(B, D, C, dtype=torch.float64).exponential_(generator=generator)
        ratio = torch.where(Q > 0, E / torch.where(Q > 0, Q, 1.0), torch.inf)
        dpos = torch.where(depth < K.unsqueeze(1), ratio.argmin(-1), 0)
        inputs = [x.cuda() for x in (E, P, Q, dpos, K)]
        compiled = _race(torch.device("cuda"), True)
        assert compiled is not residual_race_fast
        expected = residual_race_fast(*inputs, "block", aggregator, "own")
        got = compiled(*inputs, "block", aggregator, "own")
        assert torch.equal(expected[0], got[0]) and torch.equal(expected[1], got[1])
