# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""A toy multi-round speculative-decoding engine for the speculative
watermarking family (``test_spec_family.py`` and the Monte Carlo report).

Every sequence decodes ``length`` tokens from prefix-dependent toy laws (the
target's and the draft's row after token ``prev``) in rounds of ``K`` drafts
and a bonus row, as the engine does: drafts (greedy, keyed or sampled), the
watermarker's verification logits, then rejection sampling, which emits the
point masses up to the first rejected draft and samples rows that keep their
logits as vLLM does (point-mass verification with fresh randomness).

Contexts are the last ``width`` generated tokens (-1 before the first), as the
native sampler builds them; the watermarker sees them prefixed by a salt of the
sequence, so the sequences race independent keyed lanes (each is its own
reply) while masking compares the unsalted contexts within a sequence:
``emitted`` is the native ``single_turn`` rule (a context of an earlier
generated position or of an earlier row of the round's path), ``strict``
adds the contexts of every row of the sequence's earlier rounds
(``EvaluatedContexts``), ``random`` marks random rows (``random_mask``).
"""

import itertools
import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch

from vllm.v1.watermarking import (
    BlockVerifiedGumbelDraftWatermarker,
    BlockVerifiedGumbelVerificationWatermarker,
    LookaheadGumbelWatermarker,
    SampledDraftWatermarker,
)
from vllm.v1.watermarking import textseal2_verification as verification
from vllm.v1.watermarking.block_verified_gumbel import draft_law_logits
from vllm.v1.watermarking.evaluated_contexts import EvaluatedContexts
from vllm.v1.watermarking.watermarker import RandomSampler

KEY = 23


@dataclass(frozen=True)
class ToySampler(RandomSampler):
    """vLLM's random sampler on the CPU: Gumbel-max on fresh noise."""

    generator: torch.Generator | None = field(default=None)

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        u = torch.rand(logits.shape, generator=self.generator, dtype=torch.float64)
        noise = -torch.log(-torch.log(u.clamp_min(1e-300)))
        return torch.argmax(logits.double() + noise, dim=-1)


# The arms: drafts and verifier (block_verified_gumbel switches, or lookahead).
ARMS = {
    "gp": dict(drafts="greedy", lookahead=True),
    "kp": dict(drafts="keyed", law="shared"),
    "kb": dict(drafts="keyed", law="block"),
    "kt": dict(drafts="keyed", law="token"),
    "kpr": dict(drafts="keyed", law="point"),
    "kb.plain": dict(drafts="keyed", law="block", residual_race=False),
    "kt.plain": dict(drafts="keyed", law="token", residual_race=False),
    "sp": dict(drafts="sampled", law="point"),
    "st": dict(drafts="sampled", law="token"),
    "sb": dict(drafts="sampled", law="block"),
    "nowm.sp": dict(drafts="sampled", law="point", fresh_lanes=True),
    "nowm.st": dict(drafts="sampled", law="token", fresh_lanes=True),
    "nowm.sb": dict(drafts="sampled", law="block", fresh_lanes=True),
}
# block_verified_gumbel pass aggregators and the lookahead's names for them.
PASS_TO_LOOKAHEAD = {
    "identity": "pi1",
    "none": "pi2",
    "tsum": "tsum",
    "max": "max",
    "expsum": "expsum",
}


def toy_laws(
    V: int,
    seed: int,
    sharpness: float = 1.5,
    p_top_k: int | None = None,
    q_top_k: int | None = None,
    near_deterministic: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log target and log draft-law tables [V + 1, V] (row ``prev``; row V
    starts the reply), p != q. ``p_top_k`` truncates the target (drafts the
    target cannot emit), ``q_top_k`` the draft law (tokens only the target
    has), ``near_deterministic`` makes the target's row after token 0 put
    0.999 on token 0."""
    generator = torch.Generator().manual_seed(seed)
    shape = (V + 1, V)
    p = torch.softmax(sharpness * torch.randn(shape, generator=generator).double(), -1)
    if near_deterministic:
        row = torch.full((V,), math.log(1e-3 / (V - 1)), dtype=torch.float64)
        row[0] = math.log(0.999)
        p[0] = torch.softmax(row, -1)
    q = torch.softmax(p.log() + torch.randn(shape, generator=generator).double(), -1)
    log_p = p.log()
    if p_top_k is not None:
        threshold = log_p.topk(p_top_k, dim=-1).values[:, -1:]
        log_p = log_p.masked_fill(log_p < threshold, float("-inf"))
        log_p = log_p - log_p.logsumexp(-1, keepdim=True)
    n = q.shape[0]
    q_law = draft_law_logits(
        q.log().float(),
        torch.full((n,), q_top_k or 2**30),
        torch.ones(n),
        torch.zeros(n),
    ).double()
    return log_p, q_law - q_law.logsumexp(-1, keepdim=True)


def _contexts(text: torch.Tensor, ends: torch.Tensor, width: int) -> torch.Tensor:
    """Contexts of positions ``ends`` [rows]: the ``width`` tokens before."""
    index = ends.unsqueeze(1) - width + torch.arange(width)
    tokens = text.gather(1, index.clamp_min(0))
    return torch.where(index >= 0, tokens, -1)


def _prev(text: torch.Tensor, ends: torch.Tensor, V: int) -> torch.Tensor:
    index = (ends - 1).clamp_min(0)
    return torch.where(ends >= 1, text.gather(1, index.unsqueeze(1))[:, 0], V)


def _emitted_repeats(
    text: torch.Tensor,
    n: torch.Tensor,
    contexts: torch.Tensor,
    earlier: torch.Tensor | None,
    width: int,
) -> torch.Tensor:
    """Native ``single_turn``: the row's context is the context of one of the
    sequence's ``n`` generated positions, or of an earlier path row
    (``earlier`` [rows, j, width])."""
    T = text.shape[1]
    positions = torch.arange(T)
    index = positions.unsqueeze(1) - width + torch.arange(width)  # [T, width]
    hist = text[:, index.clamp_min(0)]  # [rows, T, width]
    hist = torch.where(index >= 0, hist, -1)
    seen = (hist == contexts.unsqueeze(1)).all(-1) & (positions < n.unsqueeze(1))
    repeated = seen.any(1)
    if earlier is not None and earlier.shape[1] > 0:
        repeated |= (earlier == contexts.unsqueeze(1)).all(-1).any(1)
    return repeated


def path_law(log_p: torch.Tensor, V: int, length: int) -> torch.Tensor:
    law = torch.zeros(V**length, dtype=torch.float64)
    for index, path in enumerate(itertools.product(range(V), repeat=length)):
        prev, logp = V, 0.0
        for token in path:
            logp += log_p[prev, token].item()
            prev = token
        law[index] = math.exp(logp)
    return law


def stop_law(
    log_p: torch.Tensor, q_law: torch.Tensor, V: int, K: int, drafts: str, law: str
) -> torch.Tensor:
    """The first round's law of the stopping row I (drafts kept) [K + 1]: the
    verification law averaged over the drafts (keyed and sampled drafts are
    exact samples of the draft law; greedy drafts its argmax chain)."""
    total = torch.zeros(K + 1, dtype=torch.float64)
    if drafts == "greedy":
        chains = []
        prev, chain = V, []
        for _ in range(K):
            token = int(q_law[prev].argmax())
            chain.append(token)
            prev = token
        chains = [(tuple(chain), 1.0)]
        law = "point"
    else:
        chains = []
        for d in itertools.product(range(V), repeat=K):
            prev = [V] + list(d)
            weight = math.prod(q_law[prev[i], t].exp().item() for i, t in enumerate(d))
            if weight > 0:
                chains.append((d, weight))
    for d, weight in chains:
        prev = [V] + list(d)
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
            torch.tensor([list(d) + [0]]),
            torch.tensor([K]),
            law,
        )
        total += weight * stop[0]
    return total


def make_verifier(arm: str, aggregator: str, width: int):
    spec = ARMS[arm]
    if spec.get("lookahead"):
        return LookaheadGumbelWatermarker(KEY, width, aggregator=PASS_TO_LOOKAHEAD[aggregator])
    plain = spec.get("residual_race", True) is False
    return BlockVerifiedGumbelVerificationWatermarker(
        KEY,
        width,
        law=spec["law"],
        pass_aggregator="none" if (plain or spec.get("fresh_lanes")) else aggregator,
        keyed_drafts=spec["drafts"] == "keyed",
        fresh_lanes=spec.get("fresh_lanes", False),
        residual_race=spec.get("residual_race", True),
    )


def run(
    arm: str,
    aggregator: str = "identity",
    *,
    num_sequences: int = 100_000,
    length: int = 3,
    K: int = 2,
    V: int = 4,
    seed: int = 0,
    width: int = 1,
    mask: str = "none",
    random_mask: float = 0.0,
    p_top_k: int | None = None,
    q_top_k: int | None = None,
    sharpness: float = 1.5,
    near_deterministic: bool = True,
    variable_k: bool = False,
    verifier=None,
) -> SimpleNamespace:
    """Decode ``length`` tokens per sequence with arm ``arm`` (``ARMS``);
    returns the paths [N, length], the first round's stopping rows [N] and
    drafts-per-round, the laws, and the verifier (module docstring)."""
    spec = ARMS[arm]
    log_p, q_law = toy_laws(V, seed, sharpness, p_top_k, q_top_k, near_deterministic)
    N = num_sequences
    generator = torch.Generator().manual_seed(seed + 1)
    salts = torch.randint(10, 2**40, (N,), generator=generator)
    seeds = torch.randint(0, 2**62, (N,), generator=generator)
    T = length + K + 1
    text = torch.full((N, T), -1, dtype=torch.long)
    n = torch.zeros(N, dtype=torch.long)
    if verifier is None:
        verifier = make_verifier(arm, aggregator, width + 1)
    fresh_noise = mask == "strict"
    if spec["drafts"] == "keyed":
        drafter = BlockVerifiedGumbelDraftWatermarker(
            KEY, width + 1, fresh_draft_noise=fresh_noise
        )
    elif spec["drafts"] == "sampled":
        drafter = SampledDraftWatermarker(width + 1)
    else:
        drafter = None
    evaluated = EvaluatedContexts(N, 4 * T * (K + 1), torch.device("cpu"))
    first_stop = None
    first_k = None
    key_counts = torch.zeros(2, dtype=torch.long)

    def salted(contexts: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        return torch.cat([salts[rows].unsqueeze(1), contexts], dim=1)

    while True:
        active = (n < length).nonzero().squeeze(1)
        A = active.numel()
        if A == 0:
            break
        n_a = n[active]
        Ks = (
            torch.randint(0, K + 1, (A,), generator=generator)
            if variable_k
            else torch.full((A,), K)
        )
        ext = text[active].clone()
        drafts = torch.zeros(A, K, dtype=torch.long)
        draft_ctx = torch.zeros(A, K, width, dtype=torch.long)
        rows_a = torch.arange(A)
        for j in range(K):
            ends = n_a + j
            ctx = _contexts(ext, ends, width)
            draft_ctx[:, j] = ctx
            law = q_law[_prev(ext, ends, V)].float()
            repeated = torch.zeros(A, dtype=torch.bool)
            if mask in ("emitted", "strict"):
                repeated = _emitted_repeats(ext, n_a, ctx, draft_ctx[:, :j], width)
                if mask == "strict":
                    repeated |= evaluated.contains(salted(ctx, active), active)
            elif mask == "random":
                repeated = torch.rand(A, generator=generator) < random_mask
            if drafter is None:
                tokens = law.argmax(-1)
            else:
                sampler = ToySampler(
                    expanded_idx_mapping=rows_a,
                    temperatures=torch.ones(A),
                    seeds=seeds[active],
                    positions=ends,
                    is_drafting=True,
                    logits_cache_col=torch.full((A,), j),
                    generator=generator,
                )
                if getattr(drafter, "samples_draft_rows", False):
                    tokens = drafter.sample_draft_rows(
                        law, salted(ctx, active), sampler, repeated, torch.ones(A, dtype=torch.bool)
                    )
                else:
                    keyed = drafter._sample_watermarked(law, salted(ctx, active)).token_ids
                    tokens = torch.where(repeated, sampler(law), keyed)
            drafts[:, j] = tokens
            ext.scatter_(1, ends.unsqueeze(1), tokens.unsqueeze(1))

        sizes = Ks + 1
        request = torch.repeat_interleave(rows_a, sizes)
        starts = torch.cumsum(sizes, 0) - sizes
        local = torch.arange(int(sizes.sum())) - starts[request]
        ends = n_a[request] + local
        ext_rows = ext[request]
        ctx = _contexts(ext_rows, ends, width)
        prev = _prev(ext_rows, ends, V)
        is_draft = local < Ks[request]
        draft_tokens = torch.where(is_draft, drafts[request, local.clamp_max(K - 1)], 0)
        repeated = torch.zeros(len(local), dtype=torch.bool)
        if mask in ("emitted", "strict"):
            earlier = draft_ctx[request]
            earlier_mask = torch.arange(K).unsqueeze(0) < local.unsqueeze(1)
            earlier = torch.where(earlier_mask.unsqueeze(-1), earlier, -7)
            repeated = _emitted_repeats(ext_rows, n_a[request], ctx, earlier, width)
            if mask == "strict":
                repeated |= evaluated.contains(salted(ctx, active[request]), active[request])
        elif mask == "random":
            repeated = torch.rand(len(local), generator=generator) < random_mask
        cu = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(sizes, 0)])
        logits = log_p[prev].float()
        contexts = salted(ctx, active[request])
        if spec.get("lookahead"):
            output = verifier.verification_logits(
                logits, contexts, draft_tokens, cu.to(torch.int32), repeated
            )
        else:
            before = (
                verifier.key_share_stats() if not spec.get("residual_race", True) else None
            )
            output = verifier.draft_law_verification_logits(
                logits,
                contexts,
                draft_tokens,
                cu.to(torch.int32),
                torch.ones(len(local), dtype=torch.bool),
                repeated,
                q_law[prev].float(),
                RandomSampler(
                    expanded_idx_mapping=request.to(torch.int32),
                    temperatures=torch.ones(A),
                    seeds=seeds[active],
                    positions=ends,
                ),
            )
            if before is not None:
                after = verifier.key_share_stats()
                key_counts += torch.tensor([after[0] - before[0], after[1] - before[1]])
        if mask == "strict":
            evaluated.commit(
                contexts, active[request], local, torch.ones(len(local), dtype=torch.bool)
            )
        # Rejection sampling: point masses pass; rows that keep their logits
        # are point-mass verified with fresh randomness (greedy drafts).
        point = torch.isfinite(output).sum(-1) == 1
        own = ToySampler(
            expanded_idx_mapping=request,
            temperatures=torch.ones(A),
            seeds=seeds[active],
            positions=ends,
            generator=generator,
        )(output.double())
        chosen = torch.where(point, output.argmax(-1), own)
        mismatch = ~is_draft | (chosen != draft_tokens)
        stop = torch.full((A,), K + 1).scatter_reduce(
            0, request, torch.where(mismatch, local, K + 1), reduce="amin"
        )
        emitted = local <= stop[request]
        tokens = torch.where(local < stop[request], draft_tokens, chosen)
        updated = text[active]
        updated[request[emitted], ends[emitted]] = tokens[emitted]
        text[active] = updated
        n[active] = n_a + stop + 1
        if first_stop is None:
            first_stop, first_k = stop.clone(), Ks.clone()
    return SimpleNamespace(
        paths=text[:, :length],
        first_stop=first_stop,
        first_k=first_k,
        log_p=log_p,
        q_law=q_law,
        verifier=verifier,
        key_counts=key_counts,
        V=V,
        K=K,
        length=length,
    )


def chisquare_p(counts: torch.Tensor, law: torch.Tensor) -> float:
    """Pearson's p-value of ``counts`` against ``law`` (cells outside the
    law's support must be empty: p = 0 otherwise)."""
    from scipy import stats

    support = law > 0
    if counts[~support].sum() > 0:
        return 0.0
    observed = counts[support]
    expected = law[support] / law[support].sum() * counts.sum()
    # Cells expected fewer than 5 times are pooled into one (Pearson's
    # statistic is not chi-square distributed on near-empty cells).
    rare = expected < 5
    if rare.any():
        observed = torch.cat([observed[~rare], observed[rare].sum().view(1)])
        expected = torch.cat([expected[~rare], expected[rare].sum().view(1)])
    if observed.numel() < 2:
        return 1.0
    return float(stats.chisquare(observed.numpy(), expected.numpy()).pvalue)


def law_p_values(result: SimpleNamespace) -> dict[str, float]:
    """Chi-square p-values of the paths against the target: the joint law,
    every position's marginal and every adjacent pair."""
    V, length, paths = result.V, result.length, result.paths
    law = path_law(result.log_p, V, length).view((V,) * length)
    codes = torch.zeros(paths.shape[0], dtype=torch.long)
    for t in range(length):
        codes = codes * V + paths[:, t]
    out = {"joint": chisquare_p(torch.bincount(codes, minlength=V**length).double(), law.flatten())}
    for t in range(length):
        others = tuple(u for u in range(length) if u != t)
        marginal = law.sum(others) if others else law
        out[f"pos{t}"] = chisquare_p(
            torch.bincount(paths[:, t], minlength=V).double(), marginal
        )
    for t in range(length - 1):
        others = tuple(u for u in range(length) if u not in (t, t + 1))
        pair = (law.sum(others) if others else law).flatten()
        out[f"pair{t}"] = chisquare_p(
            torch.bincount(paths[:, t] * V + paths[:, t + 1], minlength=V * V).double(),
            pair,
        )
    return out


def stop_p_value(result: SimpleNamespace, drafts: str, law: str) -> tuple[float, float, float]:
    """(chi-square p of the first round's stopping rows against the
    verification law's, mean kept, the law's mean kept)."""
    expected = stop_law(result.log_p, result.q_law, result.V, result.K, drafts, law)
    counts = torch.bincount(result.first_stop, minlength=result.K + 1).double()
    mean = float(result.first_stop.double().mean())
    return chisquare_p(counts, expected), mean, float((expected * torch.arange(result.K + 1)).sum())
