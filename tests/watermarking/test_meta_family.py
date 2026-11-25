# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The TextSeal2 reference's speculative rules: TextSeal2's decision rules
(``spec_decision``, ``truncate_drafts``) and the two-key block baselines
(``two_key_block_verification``)."""

import itertools
import math
import zlib

import pytest
import torch

import spec_family_engine as engine
from vllm.config.watermarking import WatermarkConfig
from vllm.v1.watermarking import textseal2_verification as verification
from vllm.v1.watermarking.block_verified_gumbel import (
    FreshNoiseDraftWatermarker,
    SampledDraftWatermarker,
)
from vllm.v1.watermarking.factory import create_watermarker
from vllm.v1.watermarking.textseal2 import (
    TextSeal2DraftLawDraftWatermarker,
    TextSeal2DraftLawVerificationWatermarker,
    TextSeal2DraftWatermarker,
    TextSeal2SpeculativeWatermarker,
    TextSeal2VerificationWatermarker,
)
from vllm.v1.watermarking.two_key_block import (
    FusedTwoKeyWatermarkDetector,
    TwoKeyBlockDraftWatermarker,
    TwoKeyBlockVerificationWatermarker,
    TwoKeyBlockWatermarker,
    _binomial_mixture_sf,
)

DECISIONS = ("residual_j", "fresh_stop", "tau_stop", "agreement", "j_only", "raw_j")


# ---- configuration ----------------------------------------------------------


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"algorithm": "gumbel", "spec_decision": "fresh_stop"}, "only defined for textseal2"),
        ({"algorithm": "gumbel", "truncate_drafts": True}, "only defined for textseal2"),
        (
            {"algorithm": "textseal2", "spec_decision": "tau_stop", "verification_law": "token"},
            "token-level verification",
        ),
        (
            {"algorithm": "textseal2", "spec_decision": "raw_j", "verification_law": "token"},
            "token-level verification",
        ),
        (
            {"algorithm": "textseal2", "spec_decision": "fresh_stop", "pass_aggregator": "tsum"},
            "pass_aggregator",
        ),
        ({"algorithm": "textseal2", "spec_decision": "j_only"}, "keyed_drafts=false"),
        (
            {"algorithm": "textseal2", "spec_decision": "agreement", "keyed_drafts": False},
            "needs keyed drafts",
        ),
        (
            {"algorithm": "textseal2", "truncate_drafts": True, "keyed_drafts": False},
            "truncate_drafts",
        ),
        ({"algorithm": "gumbel", "two_key_block_verification": True}, "only defined for"),
        (
            {
                "algorithm": "synthid",
                "two_key_block_verification": True,
                "verification_law": "point",
            },
            "point",
        ),
        ({"algorithm": "synthid", "deduplicate_contexts": "strict"}, "strict"),
    ],
)
def test_config_rejects(fields, message):
    with pytest.raises(ValueError, match=message):
        WatermarkConfig(key=1, **fields)


@pytest.mark.parametrize(
    "fields",
    [
        {"spec_decision": d, "keyed_drafts": d != "j_only", "truncate_drafts": t}
        for d in DECISIONS
        for t in (False, True)
    ]
    + [
        {"spec_decision": d, "verification_law": "token"}
        for d in ("residual_j", "fresh_stop", "agreement")
    ]
    + [{"spec_decision": "residual_j", "pass_aggregator": "tsum", "truncate_drafts": True}]
    + [{"deduplicate_contexts": "strict"}],
)
def test_config_accepts_textseal2(fields):
    WatermarkConfig(key=1, algorithm="textseal2", **fields)


@pytest.mark.parametrize("algorithm", ["dual_key_gumbel", "gaussian_copula", "textgrain", "synthid"])
@pytest.mark.parametrize("law", ["block", "token"])
def test_config_accepts_two_key_block(algorithm, law):
    config = WatermarkConfig(
        key=1,
        algorithm=algorithm,
        two_key_block_verification=True,
        verification_law=law,
        deduplicate_contexts="strict",
    )
    assert config.supports_speculative_decoding
    assert not config.verifies_unwatermarked_drafts


# ---- factory ----------------------------------------------------------------


@pytest.mark.parametrize("decision", DECISIONS)
@pytest.mark.parametrize("truncate", [False, True])
def test_factory_textseal2_decisions(decision, truncate):
    config = WatermarkConfig(
        key=3,
        algorithm="textseal2",
        spec_decision=decision,
        keyed_drafts=decision != "j_only",
        truncate_drafts=truncate,
    )
    wm = create_watermarker(config)
    assert isinstance(wm, TextSeal2SpeculativeWatermarker)
    target = wm.target_watermarker
    assert target.spec_decision == decision
    assert isinstance(
        target,
        TextSeal2DraftLawVerificationWatermarker if truncate else TextSeal2VerificationWatermarker,
    )
    assert hasattr(target, "draft_law_verification_logits") == truncate
    drafter = wm.draft_watermarker
    if decision == "j_only":
        assert isinstance(drafter, SampledDraftWatermarker if truncate else FreshNoiseDraftWatermarker)
        assert hasattr(drafter, "draft_law_logits") == truncate
    else:
        assert isinstance(
            drafter, TextSeal2DraftLawDraftWatermarker if truncate else TextSeal2DraftWatermarker
        )
        assert hasattr(drafter, "draft_law_logits") == truncate


@pytest.mark.parametrize("algorithm", ["dual_key_gumbel", "gaussian_copula", "textgrain", "synthid"])
def test_factory_two_key_block(algorithm):
    config = WatermarkConfig(key=3, algorithm=algorithm, two_key_block_verification=True)
    wm = create_watermarker(config)
    assert isinstance(wm, TwoKeyBlockWatermarker)
    assert isinstance(wm.draft_watermarker, TwoKeyBlockDraftWatermarker)
    assert isinstance(wm.target_watermarker, TwoKeyBlockVerificationWatermarker)
    base = create_watermarker(WatermarkConfig(key=3, algorithm=algorithm))
    assert type(wm.base) is type(base)
    assert wm.context_width == base.context_width


def test_two_key_block_samples_like_its_base_without_speculation():
    config = WatermarkConfig(key=3, algorithm="gaussian_copula", two_key_block_verification=True)
    wm = create_watermarker(config)
    base = create_watermarker(WatermarkConfig(key=3, algorithm="gaussian_copula"))
    logits = torch.randn(64, 50, generator=torch.Generator().manual_seed(0))
    contexts = torch.randint(0, 50, (64, 4), generator=torch.Generator().manual_seed(1))
    # Fresh noise from the same seeded generator: the same tokens.
    a = wm.sample(logits, contexts).token_ids
    b = base.sample(logits, contexts).token_ids
    assert torch.equal(a, b)
    # The verifier samples with the same (fresh) base.
    fresh = create_watermarker(config)
    a = fresh.target_watermarker.sample(logits, contexts).token_ids
    b = create_watermarker(WatermarkConfig(key=3, algorithm="gaussian_copula")).sample(
        logits, contexts
    ).token_ids
    assert torch.equal(a, b)


# ---- one round: the law of (drafts, stopping row, token) --------------------


def _round_law_p(decision, law, n=2, V=4, trials=60_000, seed=0):
    """Chi-square p of one round's (d, I, y) against Q(d) pi(I, y) (the
    reference's ``check_block_rules``, vectorized over rounds)."""
    from scipy.stats import chi2

    gen = torch.Generator().manual_seed(seed)
    P = torch.softmax(torch.randn(n + 1, V, generator=gen, dtype=torch.float64) * 1.3, 1)
    Q = torch.softmax(torch.log(P[:n]) + torch.randn(n, V, generator=gen, dtype=torch.float64), 1)
    P[1, V - 1] = 0.0
    P[1] /= P[1].sum()
    Q[n - 1, 0] = 0.0
    Q[n - 1] /= Q[n - 1].sum()
    B = trials
    E = torch.empty(B, n + 1, V, dtype=torch.float64).exponential_(generator=gen)
    Qp = torch.cat([Q, torch.zeros(1, V, dtype=Q.dtype)])
    if decision == "j_only":
        d = torch.multinomial(Q, B, replacement=True, generator=gen).T
    else:
        d = torch.argmin(torch.where(Q > 0, E[:, :n] / Q.clamp(min=1e-300), torch.inf), 2)
    dp = torch.cat([d, torch.zeros(B, 1, dtype=d.dtype)], 1)
    K = torch.full((B,), n)
    u = torch.rand(B, n + 1, 2, generator=gen, dtype=torch.float64)
    F = torch.empty(B, n + 1, V, dtype=torch.float64).exponential_(generator=gen)
    keyed = torch.rand(B, n + 1, generator=gen) < 0.7
    I, y = verification.decide(
        decision, E, P.expand(B, -1, -1), Qp.expand(B, -1, -1), dp, K, law,
        keyed=keyed, u=u, F=F,
    )
    expected = {}
    for dd in itertools.product(range(V), repeat=n):
        dt = torch.tensor(list(dd) + [0])
        qd = float(torch.prod(Q[torch.arange(n), dt[:n]]))
        if qd == 0:
            continue
        pi, _ = verification.outcome_law(P[None], Qp[None], dt[None], torch.tensor([n]), law)
        for i, v in itertools.product(range(n + 1), range(V)):
            if float(pi[0, i, v]) > 0:
                expected[(dd, i, v)] = B * qd * float(pi[0, i, v])
    obs = {}
    for row in range(B):
        k = (tuple(d[row].tolist()), int(I[row]), int(y[row]))
        obs[k] = obs.get(k, 0) + 1
    impossible = sum(c for k, c in obs.items() if k not in expected)
    big = [k for k in expected if expected[k] >= 5]
    small = [k for k in expected if expected[k] < 5]
    o = [obs.get(k, 0) for k in big] + [sum(obs.get(k, 0) for k in small)]
    e = [expected[k] for k in big] + [sum(expected[k] for k in small)]
    stat = sum((a - b) ** 2 / b for a, b in zip(o, e) if b > 0)
    return float(chi2.sf(stat, len(o) - 1)), impossible


@pytest.mark.parametrize(
    "decision, law",
    [(d, "block") for d in DECISIONS if d != "raw_j"]
    + [(d, "token") for d in ("residual_j", "fresh_stop", "agreement")],
)
def test_one_round_follows_the_verification_law(decision, law):
    p, impossible = _round_law_p(decision, law, seed=zlib.crc32(f"{decision}{law}".encode()))
    assert impossible == 0
    assert p > 1e-4, p


def test_raw_j_is_biased():
    # The reference's negative control: keyed drafts raced on the raw clocks.
    p, _ = _round_law_p("raw_j", "block", trials=60_000, seed=5)
    assert p < 1e-6, p


def test_decide_validates():
    E = torch.ones(1, 2, 3, dtype=torch.float64)
    P = torch.full((1, 2, 3), 1 / 3, dtype=torch.float64)
    Q = P.clone()
    d = torch.zeros(1, 2, dtype=torch.long)
    K = torch.tensor([1])
    with pytest.raises(ValueError, match="token-level"):
        verification.decide("tau_stop", E, P, Q, d, K, "token")
    with pytest.raises(ValueError, match="fresh uniforms"):
        verification.decide("fresh_stop", E, P, Q, d, K, "block")
    with pytest.raises(ValueError, match="fresh clocks"):
        verification.decide("agreement", E, P, Q, d, K, "block", u=torch.rand(1, 2, 2))
    with pytest.raises(ValueError, match="pass_aggregator"):
        verification.decide("fresh_stop", E, P, Q, d, K, "block", pass_aggregator="tsum")


def test_draw_stop_lands_on_mass():
    stop = torch.tensor([[0.0, 0.3, 0.0, 0.7], [0.5, 0.5, 0.0, 0.0]], dtype=torch.float64)
    for u, want in [(0.0, (1, 0)), (0.29, (1, 0)), (0.31, (3, 0)), (0.999999, (3, 1)), (1.0, (3, 1))]:
        got = verification.draw_stop(stop, torch.tensor([u, u], dtype=torch.float64))
        assert tuple(got.tolist()) == want, (u, got)


def test_joint_rows_match_monte_carlo():
    gen = torch.Generator().manual_seed(3)
    V = 6
    p = torch.softmax(torch.randn(V, generator=gen, dtype=torch.float64), 0)
    q = torch.softmax(torch.randn(V, generator=gen, dtype=torch.float64), 0)
    p[2] = 0.0
    p /= p.sum()
    q[4] = 0.0
    q /= q.sum()
    E = torch.empty(400_000, V, dtype=torch.float64).exponential_(generator=gen)
    x = torch.argmin(torch.where(q > 0, E / q.clamp(min=1e-300), torch.inf), 1)
    xp = torch.argmin(torch.where(p > 0, E / p.clamp(min=1e-300), torch.inf), 1)
    emp = torch.zeros(V, V, dtype=torch.float64).index_put_(
        (x, xp), torch.ones(len(x), dtype=torch.float64), accumulate=True
    ) / len(x)
    rows, diag = verification.joint_rows(p.expand(V, -1), q.expand(V, -1), torch.arange(V))
    assert torch.allclose(rows, emp, atol=4e-3)
    assert torch.allclose(rows.diagonal(), diag)
    assert torch.allclose(rows.sum(1), q, atol=1e-9)
    assert torch.allclose(rows.sum(0), p, atol=1e-9)


# ---- the toy engine: multi-round exactness ----------------------------------


TS2_ARMS = [a for a in engine.ARMS if a.startswith("ts2.") and a != "ts2.residjraw"]
TWO_KEY_ARMS = [a for a in engine.ARMS if a.startswith("tk.")]


@pytest.mark.parametrize("arm", TS2_ARMS)
@pytest.mark.parametrize("mask, width", [("random", 6), ("strict", 1)])
def test_textseal2_decisions_are_exact_in_the_engine(arm, mask, width):
    seed = zlib.crc32(f"{arm}/{mask}".encode()) % 100000
    r = engine.run(arm, "none", num_sequences=60_000, length=3, K=2, V=4, mask=mask,
                   random_mask=0.3, width=width, p_top_k=3, q_top_k=3, seed=seed)
    p = engine.law_p_values(r)
    assert min(p.values()) > 1e-4, p
    stop_p, kept, kept_law = engine.stop_p_value(r, "keyed", engine.ARMS[arm]["law"])
    assert stop_p > 1e-4, (stop_p, kept, kept_law)


def test_textseal2_raw_j_is_biased_in_the_engine():
    r = engine.run("ts2.residjraw", "none", num_sequences=60_000, length=3, K=2, V=4,
                   mask="random", random_mask=0.3, width=6, p_top_k=3, q_top_k=3, seed=11)
    assert min(engine.law_p_values(r).values()) < 1e-6


@pytest.mark.parametrize("arm", TWO_KEY_ARMS)
def test_two_key_block_is_exact_on_average_over_keys(arm):
    # Every sequence salts its contexts, so it has keys of its own: the
    # engine averages over keys, where the baselines are exact.
    seed = zlib.crc32(arm.encode()) % 100000
    r = engine.run(arm, "none", num_sequences=40_000, length=3, K=2, V=4, mask="strict",
                   width=1, p_top_k=3, q_top_k=3, seed=seed)
    p = engine.law_p_values(r)
    assert min(p.values()) > 1e-4, p
    stop_p, kept, kept_law = engine.stop_p_value(r, "keyed", engine.ARMS[arm]["law"])
    assert stop_p > 1e-4, (stop_p, kept, kept_law)
    n_a, n_b = (int(v) for v in r.key_counts)
    assert n_a > 0 and n_b > 0


# ---- detection --------------------------------------------------------------


def test_binomial_mixture_sf_matches_enumeration():
    from scipy.stats import binom

    for n, a in [(6, 0.5), (7, 0.73), (5, 0.2), (4, 1.0), (4, 0.0)]:
        k = range(n + 1)
        values = {}
        for i, j in itertools.product(k, k):
            s = a * i + (1 - a) * j
            values[s] = values.get(s, 0.0) + binom.pmf(i, n, 0.5) * binom.pmf(j, n, 0.5)
        for s in sorted(values):
            want = sum(v for t, v in values.items() if t >= s - 1e-12)
            assert math.isclose(_binomial_mixture_sf(n, s, a), want, rel_tol=1e-9, abs_tol=1e-12)


@pytest.mark.parametrize("algorithm", ["dual_key_gumbel", "gaussian_copula", "textgrain", "synthid"])
def test_fused_detector_null_is_calibrated(algorithm):
    from scipy import stats

    detector = FusedTwoKeyWatermarkDetector(algorithm, key=99, key_share_a=0.7, depth=8)
    gen = torch.Generator().manual_seed(4)
    p_values = [
        detector.detect(torch.randint(0, 50_000, (60,), generator=gen).tolist()).p_value
        for _ in range(300)
    ]
    assert stats.kstest(p_values, "uniform").pvalue > 1e-3


# ---- CUDA: the same decisions as on the CPU ---------------------------------

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _random_round(B=48, n=4, V=40, seed=0, device="cpu"):
    gen = torch.Generator().manual_seed(seed)
    D = n + 1
    P = torch.softmax(torch.randn(B, D, V, generator=gen, dtype=torch.float64) * 2, -1)
    Q = torch.softmax(torch.randn(B, D, V, generator=gen, dtype=torch.float64) * 2, -1)
    P = torch.where(torch.rand(B, D, V, generator=gen) < 0.2, 0.0, P)
    P = P / P.sum(-1, keepdim=True)
    K = torch.randint(0, n + 1, (B,), generator=gen)
    is_draft = torch.arange(D).unsqueeze(0) < K.unsqueeze(1)
    Q = Q * is_draft.unsqueeze(-1)
    E = torch.empty(B, D, V, dtype=torch.float64).exponential_(generator=gen)
    d = torch.argmin(torch.where(Q > 0, E / Q.clamp(min=1e-300), torch.inf), -1)
    d = torch.where(is_draft, d, 0)
    u = torch.rand(B, D, 2, generator=gen, dtype=torch.float64)
    F = torch.empty(B, D, V, dtype=torch.float64).exponential_(generator=gen)
    keyed = torch.rand(B, D, generator=gen) < 0.7
    return [x.to(device) for x in (E, P, Q, d, K, u, F, keyed)]


@requires_cuda
@pytest.mark.parametrize(
    "decision, law",
    [(d, "block") for d in DECISIONS] + [(d, "token") for d in ("residual_j", "fresh_stop", "agreement")],
)
def test_decide_cuda_matches_cpu(decision, law):
    E, P, Q, d, K, u, F, keyed = _random_round()
    want = verification.decide(decision, E, P, Q, d, K, law, keyed=keyed, u=u, F=F)
    E, P, Q, d, K, u, F, keyed = _random_round(device="cuda")
    got = verification.decide(decision, E, P, Q, d, K, law, keyed=keyed, u=u, F=F)
    assert torch.equal(want[0], got[0].cpu()) and torch.equal(want[1], got[1].cpu())


def _verification_inputs(device, V=64, sizes=(3, 1, 6, 4), width=4, seed=1):
    from vllm.v1.watermarking.watermarker import RandomSampler

    gen = torch.Generator().manual_seed(seed)
    rows = sum(sizes)
    logits = torch.randn(rows, V, generator=gen) * 2
    draft_law = torch.randn(rows, V, generator=gen) * 2
    draft_law = torch.where(torch.rand(rows, V, generator=gen) < 0.3, float("-inf"), draft_law)
    contexts = torch.randint(0, V, (rows, width), generator=gen)
    contexts[0, 0] = -1  # a partial context
    drafts = torch.randint(0, V, (rows,), generator=gen)
    cu = torch.tensor([0] + list(itertools.accumulate(sizes)), dtype=torch.int32)
    request = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
    enabled = torch.tensor([True, True, False, True])[request]
    repeated = torch.rand(rows, generator=gen) < 0.2
    seeds = torch.randint(0, 2**62, (len(sizes),), generator=gen)
    positions = torch.arange(rows) + 7
    sampler = RandomSampler(
        expanded_idx_mapping=request.to(torch.int32).to(device),
        temperatures=torch.ones(len(sizes), device=device),
        seeds=seeds.to(device),
        positions=positions.to(device),
    )
    tensors = [x.to(device) for x in (logits, contexts, drafts, cu, enabled, repeated, draft_law)]
    return tensors, sampler


@requires_cuda
@pytest.mark.parametrize("decision", ["residual_j", "fresh_stop", "tau_stop", "agreement", "j_only"])
def test_textseal2_draft_law_verifier_cuda_matches_cpu(decision):
    verifier = TextSeal2DraftLawVerificationWatermarker(
        11, 4, correlation=0.9, keyed_drafts=decision != "j_only", spec_decision=decision
    )
    out = []
    for device in ("cpu", "cuda"):
        tensors, sampler = _verification_inputs(device)
        out.append(verifier.draft_law_verification_logits(*tensors, sampler).cpu())
    assert torch.equal(out[0].isfinite(), out[1].isfinite())
    raced = out[0].isfinite().sum(-1) == 1
    assert torch.equal(out[0].argmax(-1)[raced], out[1].argmax(-1)[raced])


@requires_cuda
@pytest.mark.parametrize("algorithm", ["dual_key_gumbel", "gaussian_copula", "synthid", "textgrain"])
def test_two_key_block_cuda_matches_cpu(algorithm):
    config = WatermarkConfig(key=7, algorithm=algorithm, two_key_block_verification=True,
                             alpha=0.5, num_blocks=4, num_columns=4)
    out = []
    for device in ("cpu", "cuda"):
        wm = create_watermarker(config)
        tensors, sampler = _verification_inputs(device)
        out.append(wm.target_watermarker.draft_law_verification_logits(*tensors, sampler).cpu())
    raced = out[0].isfinite().sum(-1) == 1
    assert torch.equal(raced, out[1].isfinite().sum(-1) == 1)
    agree = (out[0].argmax(-1)[raced] == out[1].argmax(-1)[raced]).double().mean()
    # textGrain's and SynthID's CUDA kernels round differently from torch.
    assert agree >= (1.0 if algorithm in ("dual_key_gumbel", "gaussian_copula") else 0.9), agree
