# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The speculative-watermarking family on block_verified_gumbel: the
point-mass verification law (kpr), sampled drafts on the raw lanes (sp, st,
sb), the unwatermarked baselines (nowm), plain block verification with a
second key (kb.plain) and its fused detector, and strict repeated-context
masking. The Monte Carlo tests run the toy engine of ``spec_family_engine``."""

import math
import zlib
from types import SimpleNamespace

import pytest
import torch
from scipy import stats

import tests.watermarking.spec_family_engine as engine
from vllm.config.watermarking import WatermarkConfig, derive_watermark_key
from vllm.platforms import current_platform
from vllm.v1.watermarking import (
    BlockVerifiedGumbelDraftWatermarker,
    BlockVerifiedGumbelVerificationWatermarker,
    FusedTwoKeyGumbelWatermarkDetector,
    GumbelWatermarkDetector,
    LookaheadGumbelWatermarker,
    SampledDraftWatermarker,
    create_watermarker,
)
from vllm.v1.watermarking import textseal2_verification as verification
from vllm.v1.watermarking.block_verified_gumbel import (
    KEY_B_DOMAIN,
    draft_law_logits,
    fused_gamma_sf,
    measured_key_share,
    outcome_law_fast,
    plain_block_decision,
    residual_race_fast,
)
from vllm.v1.watermarking.evaluated_contexts import EvaluatedContexts, context_hashes
from vllm.v1.watermarking.prfs import create_prf
from vllm.v1.watermarking.watermarker import RandomSampler

KEY = engine.KEY
PASSES = ("identity", "tsum", "max", "expsum", "none")
THRESHOLD = 1e-4
requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)


def _random_block(seed: int, B: int = 64, D: int = 4, C: int = 6):
    generator = torch.Generator().manual_seed(seed)
    P = torch.softmax(2 * torch.randn(B, D, C, generator=generator).double(), -1)
    Q = torch.softmax(2 * torch.randn(B, D, C, generator=generator).double(), -1)
    P[: B // 4, :, -1] = 0.0  # drafts the target cannot emit
    Q[B // 4 : B // 2, :, 0] = 0.0  # tokens only the target has
    P = P / P.sum(-1, keepdim=True)
    Q = Q / Q.sum(-1, keepdim=True)
    K = torch.randint(0, D, (B,), generator=generator)
    dpos = torch.multinomial(Q.reshape(-1, C), 1, generator=generator).view(B, D)
    E = torch.empty(B, D, C, dtype=torch.float64).exponential_(generator=generator)
    return E, P, Q, dpos, K


# ---- configuration -----------------------------------------------------------


def test_config_and_factory():
    def make(**kwargs):
        return create_watermarker(
            WatermarkConfig(key=KEY, algorithm="block_verified_gumbel", **kwargs)
        )

    w = make(verification_law="point", pass_aggregator="tsum")
    assert w.target_watermarker.law == "point"
    assert isinstance(w.draft_watermarker, BlockVerifiedGumbelDraftWatermarker)
    assert not w.draft_watermarker.samples_draft_rows
    w = make(keyed_drafts=False, verification_law="block", pass_aggregator="tsum")
    assert isinstance(w.draft_watermarker, SampledDraftWatermarker)
    assert not w.target_watermarker.keyed_drafts
    w = make(keyed_drafts=False, fresh_lanes=True, verification_law="token")
    assert w.target_watermarker.fresh_lanes and w.fresh_lanes
    w = make(residual_race=False)
    assert w.target_watermarker.key_b_prf is not None
    w = make(deduplicate_contexts="strict")
    assert w.draft_watermarker.samples_draft_rows  # fresh noise on repeated rows
    # The defaults are the old rule.
    w = make()
    t = w.target_watermarker
    assert (t.keyed_drafts, t.fresh_lanes, t.residual_race) == (True, False, True)
    assert not w.draft_watermarker.samples_draft_rows
    for bad in (
        dict(algorithm="textseal2", verification_law="point"),
        dict(algorithm="block_verified_gumbel", residual_race=False, pass_aggregator="max"),
        dict(algorithm="block_verified_gumbel", residual_race=False, keyed_drafts=False),
        dict(algorithm="block_verified_gumbel", residual_race=False, verification_law="point"),
        dict(algorithm="block_verified_gumbel", fresh_lanes=True),
        dict(
            algorithm="block_verified_gumbel",
            keyed_drafts=False,
            fresh_lanes=True,
            pass_aggregator="tsum",
        ),
        dict(algorithm="block_verified_gumbel", keyed_drafts=False, verification_law="shared"),
        dict(algorithm="gumbel", residual_race=False),
        dict(algorithm="gumbel", deduplicate_contexts="strict"),
        dict(algorithm="textseal2", deduplicate_contexts="strict"),
    ):
        with pytest.raises(ValueError):
            WatermarkConfig(key=KEY, **bad)
    WatermarkConfig(key=KEY, algorithm="lookahead_gumbel", deduplicate_contexts="strict")


# ---- the laws and the races ---------------------------------------------------


def test_point_law_is_point_mass_verification():
    E, P, Q, dpos, K = _random_block(1)
    pi, stop = verification.point_law(P, Q, dpos, K)
    pi_fast, stop_fast = outcome_law_fast(P, Q, dpos, K, "point")
    assert torch.allclose(pi, pi_fast, atol=1e-12)
    assert torch.allclose(stop, stop_fast, atol=1e-12)
    assert torch.allclose(stop.sum(1), torch.ones(len(K), dtype=torch.float64))
    assert torch.allclose(pi.sum(2), stop, atol=1e-12)
    B, D, C = P.shape
    for b in range(B):
        reach = 1.0
        for i in range(int(K[b]) + 1):
            p = P[b, i]
            if i < K[b]:
                d = int(dpos[b, i])
                expected = reach * p.clone()
                expected[d] = 0.0
                assert torch.allclose(pi[b, i], expected, atol=1e-12)
                reach *= float(p[d])
            else:
                assert torch.allclose(pi[b, i], reach * p, atol=1e-12)
    # = token_law with point-mass drafts
    point_masses = torch.zeros_like(Q).scatter_(2, dpos.unsqueeze(2), 1.0)
    pi_token, stop_token = verification.token_law(P, point_masses, dpos, K)
    assert torch.allclose(pi, pi_token, atol=1e-12)
    assert torch.allclose(stop, stop_token, atol=1e-12)


@pytest.mark.parametrize("law", ["block", "token", "point"])
@pytest.mark.parametrize("aggregator", PASSES)
@pytest.mark.parametrize("source", ["own", "child"])
@pytest.mark.parametrize("raw", [False, True])
def test_fast_race_matches_the_reference(law, aggregator, source, raw):
    for seed in range(4):
        E, P, Q, dpos, K = _random_block(10 + seed)
        expected = verification.residual_race(
            E,
            P,
            Q,
            dpos,
            K,
            law,
            aggregator,
            source,
            use_leftovers=not raw,
            pass_time="raw" if raw else "draft",
        )
        got = residual_race_fast(E, P, Q, dpos, K, law, aggregator, source, raw)
        assert torch.equal(got[0], expected[0])
        assert torch.equal(got[1], expected[1])


def test_default_residual_race_is_unchanged():
    for seed in range(3):
        E, P, Q, dpos, K = _random_block(20 + seed)
        for law in ("block", "token"):
            for aggregator in PASSES:
                old = verification.residual_race(E, P, Q, dpos, K, law, aggregator)
                new = verification.residual_race(
                    E, P, Q, dpos, K, law, aggregator, "own", True, "draft"
                )
                assert torch.equal(old[0], new[0]) and torch.equal(old[1], new[1])


def test_point_law_with_identity_is_the_targets_own_race():
    """kpr.pi1: row by row, the target's Gumbel-max race on (tau_i at d_i,
    the leftovers elsewhere); the round stops at the first row whose pick is
    not its draft (the reference's characterization)."""
    E, P, Q, dpos, K = _random_block(3, B=512)
    stopping, column = verification.residual_race(E, P, Q, dpos, K, "point", "identity")
    tau = verification.draft_times(E, Q, dpos, K)
    lanes = (E - tau.unsqueeze(2) * Q).clamp_min(0.0)
    lanes = lanes.scatter(2, dpos.unsqueeze(2), tau.unsqueeze(2))
    B, D, C = P.shape
    for b in range(B):
        for i in range(int(K[b]) + 1):
            row = lanes[b, i] if i < K[b] else E[b, i]
            pick = int(verification.race(row, P[b, i]))
            if i == K[b] or pick != int(dpos[b, i]):
                assert int(stopping[b]) == i and int(column[b]) == pick
                break


def test_plain_decision():
    """Plain block verification: the stopping row is the fresh draw's race on
    the stopping law, then key A's race on p at the bonus row or key B's on
    the residual law at a rejected row."""
    E, P, Q, dpos, K = _random_block(4, B=256)
    generator = torch.Generator().manual_seed(5)
    E_B = torch.empty_like(E).exponential_(generator=generator)
    F = torch.empty(E.shape[:2], dtype=torch.float64).exponential_(generator=generator)
    stopping, column = plain_block_decision(E, E_B, F, P, Q, dpos, K, "block")
    pi, stop = verification.block_law(P, Q, dpos, K)
    assert torch.equal(stopping, verification.race(F, stop))
    for b in range(len(K)):
        i = int(stopping[b])
        lanes = E[b, i] if i == K[b] else E_B[b, i]
        assert int(column[b]) == int(verification.race(lanes, pi[b, i]))
        if i < K[b]:
            assert int(column[b]) != int(dpos[b, i])  # a rejected draft is not emitted


# ---- exactness and acceptance (Monte Carlo) --------------------------------------

EXACT_CASES = (
    [("kpr", a) for a in PASSES]
    + [("st", a) for a in PASSES]
    + [("sb", a) for a in PASSES]
    + [("sp", a) for a in PASSES]
    + [("nowm.sp", "none"), ("nowm.st", "none"), ("nowm.sb", "none")]
    + [("kb.plain", "none"), ("kt.plain", "none")]
    + [("gp", "identity"), ("gp", "tsum"), ("kb", "tsum"), ("kt", "identity")]
)


def _law_of(arm: str) -> str:
    spec = engine.ARMS[arm]
    return "point" if spec["drafts"] == "greedy" else spec["law"]


@pytest.mark.parametrize("arm,aggregator", EXACT_CASES)
@pytest.mark.parametrize("mask", ["random", "strict"])
def test_emitted_text_follows_the_target_and_acceptance_the_law(arm, aggregator, mask):
    """Joint, per-position and adjacent-pair laws of the emitted text against
    the target (support mismatches both ways and a near-deterministic row),
    and the first round's stopping row against the verification law averaged
    over the drafts: the same for every aggregator. ``random`` masks 30% of
    the rows (the masked-row treatment); ``strict`` runs with one-token
    contexts, so contexts recur and strict masking decides."""
    result = engine.run(
        arm,
        aggregator,
        num_sequences=150_000,
        length=3,
        K=2,
        V=4,
        seed=zlib.crc32(f"{arm}/{aggregator}/{mask}".encode()) % 1000,
        mask=mask,
        random_mask=0.3,
        width=6 if mask == "random" else 1,
        p_top_k=3,
        q_top_k=3,
    )
    p_values = engine.law_p_values(result)
    assert min(p_values.values()) > THRESHOLD, p_values
    stop_p, mean, expected = engine.stop_p_value(
        result, engine.ARMS[arm]["drafts"], _law_of(arm)
    )
    assert stop_p > THRESHOLD, (stop_p, mean, expected)


@pytest.mark.parametrize("arm", ["kpr", "sb", "st", "kb.plain"])
def test_variable_drafts_per_request(arm):
    result = engine.run(
        arm,
        "tsum" if arm != "kb.plain" else "none",
        num_sequences=100_000,
        length=3,
        K=3,
        V=4,
        seed=7,
        variable_k=True,
        mask="random",
        random_mask=0.2,
        width=7,
    )
    p_values = engine.law_p_values(result)
    assert min(p_values.values()) > THRESHOLD, p_values


def test_acceptance_identities():
    """kpr = sp = nowm.sp, kt (identity) = st = nowm.st, kb = kb.plain = sb =
    nowm.sb, gp = nowm.gp: the first round's stopping law is the verification
    law's (exact closed form) for every arm of a family."""
    families = {
        "point": ["kpr", "sp", "nowm.sp"],
        "token": ["kt", "st", "nowm.st", "kt.plain"],
        "block": ["kb", "sb", "nowm.sb", "kb.plain"],
    }
    for law, arms in families.items():
        means = []
        for arm in arms:
            aggregator = "none" if arm.startswith(("nowm", "kb.plain", "kt.plain")) else "tsum"
            result = engine.run(
                arm, aggregator, num_sequences=100_000, K=3, V=4, seed=11, length=2, width=6
            )
            stop_p, mean, expected = engine.stop_p_value(result, "keyed", law)
            assert stop_p > THRESHOLD, (arm, stop_p, mean, expected)
            means.append(mean)
        assert max(means) - min(means) < 0.03, (law, arms, means)


def test_sampled_point_mass_rule_is_lookahead_gumbel():
    """sp: block_verified_gumbel's point law on sampled drafts (raw lanes)
    decides like lookahead Gumbel-max on the same drafts, aggregator by
    aggregator (up to float32 near ties)."""
    generator = torch.Generator().manual_seed(9)
    R, K, V, width = 3000, 3, 6, 3
    sizes = torch.full((R,), K + 1)
    cu = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(sizes, 0)]).to(torch.int32)
    rows = int(sizes.sum())
    logits = 2 * torch.randn(rows, V, generator=generator)
    contexts = torch.randint(0, 10**6, (rows, width), generator=generator)
    draft_law = torch.randn(rows, V, generator=generator)
    drafts = torch.multinomial(torch.softmax(draft_law, -1), 1, generator=generator)[:, 0]
    for pass_aggregator, aggregator in engine.PASS_TO_LOOKAHEAD.items():
        sampled = BlockVerifiedGumbelVerificationWatermarker(
            KEY,
            width,
            law="point",
            pass_aggregator=pass_aggregator,
            keyed_drafts=False,
        )
        out = sampled.draft_law_verification_logits(
            logits,
            contexts,
            drafts,
            cu,
            torch.ones(rows, dtype=torch.bool),
            torch.zeros(rows, dtype=torch.bool),
            draft_law,
            RandomSampler(
                expanded_idx_mapping=torch.arange(R).repeat_interleave(K + 1),
                temperatures=torch.ones(R),
                seeds=torch.arange(R),
                positions=torch.arange(rows),
            ),
        ).argmax(-1)
        lookahead = LookaheadGumbelWatermarker(KEY, width, aggregator=aggregator)
        expected = lookahead.verification_logits(
            logits, contexts, drafts, cu, torch.zeros(rows, dtype=torch.bool)
        ).argmax(-1)

        def emitted(chosen):
            local = torch.arange(rows) % (K + 1)
            mismatch = (local == K) | (chosen != drafts)
            stop = torch.full((R,), K).scatter_reduce(
                0, torch.arange(R).repeat_interleave(K + 1),
                torch.where(mismatch, local, K), reduce="amin",
            )
            at_stop = chosen.view(R, K + 1).gather(1, stop.unsqueeze(1)).squeeze(1)
            return stop, at_stop

        a, b = emitted(out), emitted(expected)
        agree = ((a[0] == b[0]) & (a[1] == b[1])).double().mean()
        assert agree > 0.998, (pass_aggregator, float(agree))


# ---- plain block verification's detector --------------------------------------


def test_fused_gamma_survival_is_exact():
    generator = torch.Generator().manual_seed(2)
    for n, a in ((20, 0.7), (50, 0.93), (30, 0.5), (40, 0.0), (25, 1.0), (60, 0.31)):
        X = torch.empty(400_000, n, dtype=torch.float64).exponential_(generator=generator)
        Y = torch.empty(400_000, n, dtype=torch.float64).exponential_(generator=generator)
        S = (a * X + (1 - a) * Y).sum(1)
        for q in (0.5, 0.9, 0.99, 0.999):
            s = float(torch.quantile(S, q))
            assert abs(fused_gamma_sf(n, s, a) - (1 - q)) < 4 * math.sqrt(q * (1 - q) / 4e5) + 1e-4


@pytest.mark.parametrize("share", [0.5, 0.71, 0.93])
def test_fused_detector_null_p_values_are_uniform(share):
    detector = FusedTwoKeyGumbelWatermarkDetector(KEY, 4, key_share_a=share)
    plain = GumbelWatermarkDetector(KEY, 4)
    generator = torch.Generator().manual_seed(int(share * 100))
    p_values, p_key_a = [], []
    for _ in range(3000):
        tokens = torch.randint(0, 50_000, (120,), generator=generator).tolist()
        result = detector.detect(tokens)
        p_values.append(result.p_value)
        p_key_a.append(result.p_value_key_a)
        if len(p_values) <= 20:
            assert result.p_value_key_a == pytest.approx(plain.detect(tokens).p_value)
    assert stats.kstest(p_values, "uniform").pvalue > 1e-3
    assert stats.kstest(p_key_a, "uniform").pvalue > 1e-3
    fpr = sum(p <= 0.01 for p in p_values) / len(p_values)
    assert fpr < 0.02


def test_plain_text_is_detected_and_counts_the_keys():
    result = engine.run("kb.plain", "none", num_sequences=4000, length=6, K=3, V=8, width=10)
    n_a, n_b = (int(v) for v in result.key_counts)
    assert n_a > 0 and n_b > 0
    share = measured_key_share(n_a, n_b)
    key_b = create_prf("philox", derive_watermark_key(KEY, KEY_B_DOMAIN))
    assert key_b.key != create_prf("philox", KEY).key
    assert 0.0 < share < 1.0


# ---- strict masking -------------------------------------------------------------


def test_evaluated_contexts_commit_lookup_and_reset():
    store = EvaluatedContexts(3, 4, torch.device("cpu"))
    ctx = torch.tensor([[1, 2], [2, 3], [3, 4], [9, 9], [-1, 5]])
    req = torch.tensor([0, 0, 0, 2, -1])
    local = torch.tensor([0, 1, 2, 0, 0])
    assert not store.contains(ctx, req).any()
    store.commit(ctx, req, local, torch.tensor([True, True, True, False, True]))
    assert store.contains(ctx, req).tolist() == [True, True, True, False, False]
    # Another slot does not see slot 0's contexts.
    assert not store.contains(ctx[:3], torch.tensor([1, 1, 1])).any()
    # The ring keeps the last 4.
    store.commit(ctx[3:4], torch.tensor([0]), torch.tensor([0]), torch.tensor([True]))
    store.commit(ctx[1:2] + 100, torch.tensor([0]), torch.tensor([0]), torch.tensor([True]))
    assert store.contains(ctx[:4], torch.zeros(4, dtype=torch.long)).tolist() == [
        False,
        True,
        True,
        True,
    ]
    store.reset(0)
    assert not store.contains(ctx[:4], torch.zeros(4, dtype=torch.long)).any()
    hashes = context_hashes(torch.randint(-1, 10**5, (20_000, 4)))
    assert hashes.unique().numel() >= 19_990


def test_toy_masks_mirror_the_native_single_turn_rule():
    from vllm.v1.worker.gpu.sample.watermark import repeated_context_mask

    generator = torch.Generator().manual_seed(4)
    N, T, width = 300, 9, 2
    text = torch.randint(0, 3, (N, T), generator=generator)
    n = torch.randint(0, T, (N,), generator=generator)
    ends = n.clone()
    ctx = engine._contexts(text, ends, width)
    toy = engine._emitted_repeats(text, n, ctx, None, width)
    native = repeated_context_mask(
        text,
        torch.arange(N),
        torch.zeros(N, dtype=torch.int32),
        n.to(torch.int32),
        ctx,
        None,
    )
    assert torch.equal(toy, native)


def test_gpu_sampler_commits_and_finds_the_evaluated_contexts():
    import functools

    from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler

    contexts = torch.tensor([[1, 2], [2, 7], [7, 8], [5, 6]])
    seen = {}

    class Recorder(BlockVerifiedGumbelVerificationWatermarker):
        def draft_law_verification_logits(self, *args):
            seen["repeated"] = args[5]
            return args[0]

    store = EvaluatedContexts(4, 16, torch.device("cpu"))
    sampler = SimpleNamespace(
        watermarker=Recorder(key=1, context_width=2),
        watermarking=SimpleNamespace(gpu=torch.tensor([True, True, False, True])),
        sampling_states=SimpleNamespace(
            temperature=SimpleNamespace(gpu=torch.tensor([1.0, 1.0, 1.0, 1.0])),
            seeds=SimpleNamespace(gpu=torch.arange(4)),
            top_k=SimpleNamespace(gpu=torch.full((4,), 5, dtype=torch.int32)),
            top_p=SimpleNamespace(gpu=torch.ones(4)),
            min_p=SimpleNamespace(gpu=torch.zeros(4)),
        ),
        deduplicate_contexts="strict",
        evaluated_contexts=store,
        use_fp64_gumbel=False,
        _get_contexts=lambda *args: contexts,
        _get_repeated_contexts=lambda idx, ctx, local=None: store.contains(ctx, idx),
    )
    sampler._draft_law_verification_logits = functools.partial(
        GPUWatermarkSampler._draft_law_verification_logits, sampler
    )
    args = (
        torch.randn(4, 9),
        torch.tensor([1, 1, 1, 2], dtype=torch.int32),
        torch.tensor([0, 1, 2, 0], dtype=torch.int32),
        torch.tensor([5, 6, 7, 8]),
        torch.tensor([0, 3, 4], dtype=torch.int32),
    )
    kwargs = dict(draft_logits=torch.randn(4, 2, 9), pos=torch.tensor([3, 4, 5, 1]))
    GPUWatermarkSampler.watermark_verification_logits(sampler, *args, **kwargs)
    assert not seen["repeated"].any()
    # Request 1's three rows were committed; request 2 does not watermark.
    assert store.contains(contexts, torch.tensor([1, 1, 1, 2])).tolist() == [
        True,
        True,
        True,
        False,
    ]
    GPUWatermarkSampler.watermark_verification_logits(sampler, *args, **kwargs)
    assert seen["repeated"].tolist() == [True, True, True, False]


def test_draft_watermarker_masks_evaluated_contexts_and_draws_fresh_noise(monkeypatch):
    from vllm.v1.watermarking import spec_decode
    from vllm.v1.watermarking.spec_decode import create_speculative_draft_watermarker

    calls = {}

    class Recording(RandomSampler):
        def __call__(self, logits):
            calls["plain"] = calls.get("plain", 0) + 1
            return logits.argmax(dim=-1)

    monkeypatch.setattr(spec_decode, "RandomSampler", Recording)
    config = WatermarkConfig(
        key=KEY,
        algorithm="block_verified_gumbel",
        context_width=2,
        deduplicate_contexts="strict",
    )
    watermarker = create_watermarker(config)
    drafter = create_speculative_draft_watermarker(
        watermarker, 4, torch.device("cpu"), False, 2, "strict", None
    )
    store = EvaluatedContexts(4, 8, torch.device("cpu"))
    contexts = torch.tensor([[11, 12], [13, 14], [15, 16]])
    store.commit(contexts[:1], torch.tensor([0]), torch.tensor([0]), torch.tensor([True]))
    drafter.prepare(
        contexts,
        torch.tensor([True, True, False]),
        torch.full((4, 8), 3, dtype=torch.int64),
        torch.zeros(4, dtype=torch.int32),
        torch.zeros(4, dtype=torch.int32),
        top_k=torch.full((3,), 50, dtype=torch.int32),
        top_p=torch.ones(3),
        min_p=torch.zeros(3),
        evaluated_contexts=store,
    )
    V = 64
    logits = 3 * torch.randn(3, V, generator=torch.Generator().manual_seed(3))
    seeds = torch.tensor([101, 202, 303])
    tokens = drafter.sample(
        logits,
        torch.arange(3, dtype=torch.int32),
        torch.full((3,), 0.7),
        seeds,
        torch.tensor([40, 50, 60]),
        apply_temperature=True,
        is_drafting=True,
        logits_cache=torch.zeros(3, 2, V),
        logits_cache_col=torch.tensor(0),
    )
    law = draft_law_logits(logits / 0.7, torch.full((3,), 50), torch.ones(3), torch.zeros(3))
    keyed = watermarker.draft_watermarker._sample_watermarked(law, contexts).token_ids
    fresh_ctx = torch.stack(
        [seeds & 0xFFFFFFFF, seeds >> 32, torch.tensor([40, 50, 60]), torch.zeros(3, dtype=torch.long)], 1
    )
    from vllm.v1.watermarking.block_verified_gumbel import _DRAFT_NOISE_PRF, _gumbel_max

    fresh = _gumbel_max(_DRAFT_NOISE_PRF, law, fresh_ctx)
    assert tokens[0] == fresh[0]  # an evaluated context: fresh draft noise
    assert tokens[1] == keyed[1]  # keyed race
    assert calls["plain"] == 1  # the opted-out request draws vLLM's noise


def test_sampled_drafts_are_exact_samples_of_the_draft_law():
    drafter = SampledDraftWatermarker(3)
    generator = torch.Generator().manual_seed(8)
    V, R = 6, 200_000
    law = torch.log(torch.tensor([0.4, 0.25, 0.2, 0.1, 0.05, 0.0]))
    sampler = engine.ToySampler(
        expanded_idx_mapping=torch.arange(R),
        temperatures=torch.ones(R),
        seeds=torch.randint(0, 2**62, (R,), generator=generator),
        positions=torch.full((R,), 7),
        logits_cache_col=torch.full((R,), 2),
        generator=generator,
    )
    tokens = drafter.sample_draft_rows(
        law.expand(R, V).float(),
        torch.zeros(R, 3, dtype=torch.long),
        sampler,
        torch.zeros(R, dtype=torch.bool),
        torch.ones(R, dtype=torch.bool),
    )
    counts = torch.bincount(tokens, minlength=V).double()
    assert engine.chisquare_p(counts, law.exp().double()) > THRESHOLD
    # Fresh per (position, step): another step draws other noise.
    other = drafter.sample_draft_rows(
        law.expand(R, V).float(),
        torch.zeros(R, 3, dtype=torch.long),
        engine.ToySampler(**{**sampler.__dict__, "logits_cache_col": torch.full((R,), 3)}),
        torch.zeros(R, dtype=torch.bool),
        torch.ones(R, dtype=torch.bool),
    )
    assert (other != tokens).double().mean() > 0.3


def test_fresh_lanes_samples_with_vllm_without_speculation():
    watermarker = create_watermarker(
        WatermarkConfig(
            key=KEY,
            algorithm="block_verified_gumbel",
            keyed_drafts=False,
            fresh_lanes=True,
        )
    )
    logits = torch.randn(5, 30)

    class Fixed(RandomSampler):
        def __call__(self, logits):
            return torch.full((logits.shape[0],), 7)

    sampler = Fixed(torch.arange(5), torch.ones(5), torch.arange(5), torch.arange(5))
    for w in (watermarker, watermarker.target_watermarker):
        assert w.sample(logits, torch.zeros(5, 4, dtype=torch.long), sampler).token_ids.tolist() == [7] * 5


# ---- CUDA ---------------------------------------------------------------------------


@requires_cuda
def test_cuda_drafters_match_cpu():
    generator = torch.Generator().manual_seed(6)
    R, V = 512, 1000
    law = draft_law_logits(
        2 * torch.randn(R, V, generator=generator),
        torch.full((R,), 50),
        torch.ones(R),
        torch.zeros(R),
    )
    contexts = torch.randint(0, 10**6, (R, 4), generator=generator)
    seeds = torch.randint(0, 2**62, (R,), generator=generator)
    positions = torch.randint(0, 4000, (R,), generator=generator)
    steps = torch.randint(0, 5, (R,), generator=generator)
    repeated = torch.rand(R, generator=generator) < 0.3
    enabled = torch.ones(R, dtype=torch.bool)
    for drafter in (SampledDraftWatermarker(4), BlockVerifiedGumbelDraftWatermarker(KEY, 4, fresh_draft_noise=True)):
        out = {}
        for device in ("cpu", "cuda"):
            cache = torch.zeros(R, 5, V, device=device)
            sampler = engine.ToySampler(
                expanded_idx_mapping=torch.arange(R, device=device),
                temperatures=torch.ones(R, device=device),
                seeds=seeds.to(device),
                positions=positions.to(device),
                is_drafting=True,
                logits_cache=cache if device == "cuda" else None,
                logits_cache_col=steps.to(device),
                logits_cache_source=law.to(device) if device == "cuda" else None,
            )
            out[device] = drafter.sample_draft_rows(
                law.to(device), contexts.to(device), sampler, repeated.to(device), enabled.to(device)
            ).cpu()
            if device == "cuda":
                cached = cache[torch.arange(R), steps.cuda()].cpu()
                assert torch.equal(cached, law)
        assert (out["cpu"] == out["cuda"]).double().mean() > 0.999


@requires_cuda
def test_cuda_evaluated_contexts_match_cpu():
    generator = torch.Generator().manual_seed(7)
    ctx = torch.randint(-1, 10**6, (5000, 4), generator=generator)
    assert torch.equal(context_hashes(ctx), context_hashes(ctx.cuda()).cpu())
    # Requests' rows are contiguous with local positions 0, 1, ... (commit's contract).
    req = torch.randint(-1, 8, (40,), generator=torch.Generator().manual_seed(1)).sort().values
    local = torch.tensor([int((req[:i] == req[i]).sum()) for i in range(40)])
    for device in ("cpu", "cuda"):
        store = EvaluatedContexts(8, 64, torch.device(device))
        store.commit(ctx[:40].to(device), req.to(device), local.to(device), torch.ones(40, dtype=torch.bool, device=device))
        found = store.contains(ctx[:40].to(device), req.to(device)).cpu()
        assert found.tolist() == (req >= 0).tolist()


@requires_cuda
@pytest.mark.parametrize("arm", ["kpr", "st", "sb", "sp", "nowm.sb", "kb.plain"])
def test_cuda_rule_matches_cpu(arm):
    """The new rules decide the same on CUDA as on the CPU (float near ties
    aside): one verification step of random requests."""
    generator = torch.Generator().manual_seed(12)
    R, K, V, width = 400, 3, 300, 4
    sizes = torch.full((R,), K + 1)
    rows = int(sizes.sum())
    cu = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(sizes, 0)]).to(torch.int32)
    logits = draft_law_logits(
        2 * torch.randn(rows, V, generator=generator), torch.full((rows,), 50), torch.ones(rows), torch.zeros(rows)
    )
    law = draft_law_logits(
        logits + torch.randn(rows, V, generator=generator), torch.full((rows,), 50), torch.ones(rows), torch.zeros(rows)
    )
    drafts = torch.multinomial(torch.softmax(law, -1), 1, generator=generator)[:, 0]
    contexts = torch.randint(0, 10**6, (rows, width), generator=generator)
    repeated = torch.rand(rows, generator=generator) < 0.2
    # The compiled Philox PRF keeps one dynamo cache for the whole test process,
    # which the suite's many context widths and shapes fill: start this test
    # from an empty one, as an engine process does.
    torch._dynamo.reset()
    out = {}
    for device in ("cpu", "cuda"):
        verifier = engine.make_verifier(arm, "tsum", width)
        out[device] = verifier.draft_law_verification_logits(
            logits.to(device),
            contexts.to(device),
            drafts.to(device),
            cu.to(device),
            torch.ones(rows, dtype=torch.bool, device=device),
            repeated.to(device),
            law.to(device),
            RandomSampler(
                expanded_idx_mapping=torch.arange(R, device=device).repeat_interleave(K + 1),
                temperatures=torch.ones(R, device=device),
                seeds=torch.arange(R, device=device),
                positions=torch.arange(rows, device=device),
            ),
        ).argmax(-1).cpu()
    assert (out["cpu"] == out["cuda"]).double().mean() > 0.999
