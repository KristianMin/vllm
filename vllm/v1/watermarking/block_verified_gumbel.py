# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Block-verified Gumbel-max (``block_verified_gumbel``): speculative decoding
for the Gumbel-max watermark with keyed drafts, block verification and the
tau-pass lookahead.

One key, ``gumbel``'s: a candidate ``v`` of a row with context ``c`` has the
lane ``E(v) = -log U(c, v) ~ Exp(1)``, with ``U`` the Gumbel-max PRF's uniform,
so Gumbel-max's token is ``argmin_v E(v) / p(v)`` and ``GumbelWatermarkDetector``
(and its weighted variants) score every emitted token on its own lane, as for
``gumbel``. Without speculation this is ``gumbel``.

Draft. ``d_i = argmin_v E_i(v) / q_i(v)``: Gumbel-max sampling of the draft law
``q_i`` with the target's key at row ``i``'s context (the text and ``d_<i``,
the context the target's row ``i`` sees). ``q_i`` is the drafter's
distribution after temperature and then the request's min-p, top-k and top-p
(``draft_law_logits``, which keeps at most ``draft_law_width`` candidates), so
the drafts follow the target's truncation. vLLM caches the raw draft logits
and verification recomputes ``q`` from that cache with the same function.

Verify. With ``tau_i = E_i(d_i) / q_i(d_i)`` the draft race's time, the
leftovers ``L_i(v) = E_i(v) - tau_i q_i(v)`` (``E`` on the bonus row) and
``pi`` the outcome law of block verification (Sun et al.;
``verification_law="block"``) or of token-level speculative sampling
(``"token"``) given the drafts, zero on the drafted tokens, the round stops at
row ``I`` and emits ``y`` by the backward chain over rows
(``residual_race_fast``, ``textseal2_verification.residual_race``
vectorized)::

    M_i = min_v L_i(v) / pi_i(v),
    C_i = psi(tau_i, S_{i+1} P(I > i)) / P(I > i),
    stop at i iff M_i < C_i,   S_i = min(M_i, C_i).

``pass_aggregator`` is ``psi``: ``none`` (``S_{i+1}`` alone: one race over all
leftovers, TextSeal2's residual race), ``identity`` (``tau_i`` alone: the
draft keeps its own clock), ``max`` (``2 min(a, b)``), ``tsum`` (lookahead
Gumbel-max's table) and ``expsum``. Given the drafts, ``tau_i ~ Exp(1)`` and
the leftovers are independent Exp(1) clocks (memorylessness of the draft race;
TextSeal2, arXiv:2610.40613, Lemma 1), so ``M_i ~ Exp(P(I = i))`` and ``C_i ~
Exp(P(I > i))`` are independent and every ``psi`` keeps ``(I, y) ~ pi``: the
output follows the target and the acceptance is the verification law's, the
same for every ``psi``. With point-mass drafts the chain is lookahead
Gumbel-max (``identity`` its ``pi1``, ``none`` its ``pi2``, ``max`` and
``tsum`` its namesakes). ``verification_law="shared"`` keeps the keyed drafts
but verifies them on the shared clocks: every row emits its own Gumbel-max
token ``argmin E / p`` and a draft is kept iff it equals it (the baseline,
whose acceptance ``P(argmin E/q = argmin E/p)`` is below ``sum min(p, q)``).

Candidates. Verification works on the target's support (at most
``target_slots`` candidates, the rest of it merged into one slot whose lane is
the minimum of ``E / p`` over it, exact for exponential races) and the draft
law's support; ``p`` and ``q`` vanish elsewhere.

Masked rows. Repeated contexts (``deduplicate_contexts``) race on fresh lanes
``F(v) = -log U_F((seed, position, step), v)`` of a public PRF of the request
seed, the row's position and its depth in the round, which every round
reaches with a new (position, depth) pair. The drafter sampled such a row's
draft from ``q`` with vLLM's own noise, which verification cannot see, so
verification races it on the clocks a draft race has given its winner:
``tau = F(d)`` and the leftovers ``F(v)``, ``v != d``. Exact; the row is not
watermarked, as a repeated context in ``gumbel`` (the detector skips it).
Requests that do not watermark (opted out or greedy) keep their logits and
vLLM's rejection sampling, with temperature-only drafts. The row of a
request's first invalid draft (a placeholder) ends its round as its bonus row.

Consistency. A keyed draft that is not its race's winner on the recomputed law
(the drafter races in float32 and verification in float64, so near ties; or
the drafter took the row for a repeated context and verification did not) was
not raced on its keyed lanes, so its row is verified on fresh lanes like a
repeated context. Both treatments hand the race independent Exp(1) clocks
given the drafts, so the choice between them, although it reads the lanes,
keeps the rule exact. A draft outside the draft law (only from a bug or a
vocabulary mismatch) makes its request use the shared clocks, exact for any
drafts. ``fallback_stats`` counts both, logged every
``VLLM_BV_GUMBEL_REPORT_EVERY`` verification steps (1000);
``VLLM_BV_GUMBEL_STRICT=1`` raises on a draft outside the draft law.
``VLLM_BV_GUMBEL_COMPILE=1`` runs the race through ``torch.compile`` on CUDA
(opt-in; eager torch by default).

As for the lookahead watermarks, ``context_width`` should be at least the
number of drafts: the rule reads the lanes of rows it then discards
(``deduplicate_contexts="strict"`` masks every context an earlier
verification step evaluated, which makes the rule exact for any width).

The family. The same verifier runs the rest of the speculative family (codes
``<draft><verification>.<aggregator>``):

- ``verification_law="point"`` (kpr): the residual race at point-mass
  verification's law (``textseal2_verification.point_law``): the pass weight
  ``p_i(d_i)``, a stop with ``v != d_i`` at weight ``p_i(v)``. With
  ``identity`` this is, row by row, the target's own race on (``tau_i`` at the
  draft, the leftovers elsewhere); the acceptance is sampled drafts' with
  point-mass verification.
- ``keyed_drafts=False`` (sp, st, sb): the drafts are Gumbel-max samples of the
  draft law on fresh draft noise (``SampledDraftWatermarker``, a public PRF of
  the request seed, position and draft step, fresh in every round), which does
  not depend on the key. Given them the keyed lanes are an i.i.d. Exp(1) field,
  so verification races the raw lanes (no leftovers) and a row's pass clock
  reads its drafted token's raw lane ``E_i(d_i)`` (``residual_race_fast(...,
  raw=True)``): lookahead Gumbel-max's recursion with the verification law as
  the rows' weights, exact by its argument, with the acceptance of token-level
  / block / point-mass verification. With ``point`` it is lookahead
  Gumbel-max on the sampled drafts. Repeated rows race on fresh lanes.
- ``fresh_lanes`` (nowm.sp, nowm.st, nowm.sb; sampled drafts): every row races
  on fresh lanes, an exact sample of the verification law without a
  watermark; without speculation vLLM samples.
- ``residual_race=False`` (kb.plain): plain block (or token-level)
  verification of the keyed drafts (``plain_block_decision``): the stopping row
  ``I ~ P(I = i)`` from a fresh draw (public PRF ``_STOP_PRF``), the bonus row's
  Gumbel-max token with the key when every draft is kept, else the Gumbel-max
  token of the residual law with key B (``derive_watermark_key(key,
  KEY_B_DOMAIN)``; repeated rows race on fresh key-B lanes). The drafter never
  reads the bonus row's lanes or key B's, so the round has block verification's
  law exactly. The verifier counts the keyed emitted tokens per key
  (``key_share_stats``; ``VLLM_BV_GUMBEL_KEY_LOG``) for
  ``FusedTwoKeyGumbelWatermarkDetector``.
- ``fresh_draft_noise`` (``deduplicate_contexts="strict"``): the keyed
  drafter's repeated rows draw fresh draft noise, not vLLM's, which repeats at
  a position drafted again in a later round.
"""

import json
import math
import os
import time
from dataclasses import dataclass

import torch

from vllm.config.watermarking import WatermarkPRFName, derive_watermark_key
from vllm.logger import init_logger
from vllm.v1.watermarking.detector import WatermarkDetection
from vllm.v1.watermarking.gumbel import (
    GumbelWatermarkDetector,
    GumbelWatermarker,
    _gamma_survival_integer_shape,
)
from vllm.v1.watermarking.prfs import PhiloxPRF, WatermarkPRF, create_prf
from vllm.v1.watermarking.textseal2_verification import (
    PASS_AGGREGATORS,
    PASS_SOURCES,
    RESIDUAL_LAWS,
    pass_aggregate,
    race,
)
from vllm.v1.watermarking.watermarker import (
    RandomSampler,
    SupportsSpeculativeDecoding,
    Watermarker,
    WatermarkSample,
)

logger = init_logger(__name__)

# "block", "token" and "point" race on the leftovers (keyed drafts) or the raw
# clocks (sampled drafts); "shared" is the keyed drafts' baseline.
VERIFICATION_LAWS = RESIDUAL_LAWS + ("shared",)
# Candidates the draft law keeps at most (before ties at its threshold).
DRAFT_LAW_WIDTH = 1024
# Target candidates verification races individually; the rest of a larger
# support shares one slot.
TARGET_SLOTS = 1024
# Philox key of the fresh lanes of masked rows. Public: the lanes need no
# secrecy, only independence from the watermark key.
_FRESH_LANE_PRF = PhiloxPRF(0x4256474D_46524553)
# Public Philox keys of the other fresh randomness, on the same (seed,
# position, depth) contexts: the key-B fresh lanes of a masked row in plain
# block verification, its stopping row's draw, and the noise of drafts that do
# not depend on the key (sampled drafts; masked rows of keyed drafts under
# deduplicate_contexts "strict").
_FRESH_LANE_KEY_B_PRF = PhiloxPRF(0x4256474D_4B455942)
_STOP_PRF = PhiloxPRF(0x4256474D_53544F50)
_DRAFT_NOISE_PRF = PhiloxPRF(0x4256474D_44524654)
# residual_race=False: key B of the token after a rejection.
KEY_B_DOMAIN = b"block_verified_gumbel_key_b"
STRICT_ENV_VAR = "VLLM_BV_GUMBEL_STRICT"
REPORT_EVERY_ENV_VAR = "VLLM_BV_GUMBEL_REPORT_EVERY"
# residual_race=False: a file the verifier keeps the running counts of keyed
# emitted tokens under key A and key B in (JSON), for the detector's weight.
KEY_LOG_ENV_VAR = "VLLM_BV_GUMBEL_KEY_LOG"
# Opt-in: torch.compile the race on CUDA (fewer kernel launches; compiles once
# per aggregator, ~20 s). Eager torch is the default.
COMPILE_ENV_VAR = "VLLM_BV_GUMBEL_COMPILE"


def draft_law_logits(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
    width: int = DRAFT_LAW_WIDTH,
) -> torch.Tensor:
    """The draft law's logits: ``logits`` [rows, vocab] (temperature-scaled)
    with -inf outside the law's support, which is every token at least as
    likely as the ``n``-th most likely, ``n`` the number of the ``width`` most
    likely tokens that survive the request's ``min_p``, then ``top_k``, then
    ``top_p`` on the survivors' renormalized probabilities, as vLLM truncates
    the target. Thresholds are values, so ties never depend on the order a
    kernel returns them in, and the drafter and verification get the same law
    from the same logits. No host synchronization (CUDA-graph safe)."""
    logits = logits.float()
    logits = logits.masked_fill(logits.isnan(), float("-inf"))
    num_rows, vocab_size = logits.shape
    width = min(width, vocab_size)
    values = torch.topk(logits, width, dim=-1).values  # descending
    ranks = torch.arange(width, device=logits.device)
    keep = torch.isfinite(values)
    keep &= ranks.unsqueeze(0) < top_k.to(torch.int64).clamp_min(1).unsqueeze(1)
    keep &= values >= values[:, :1] + torch.log(min_p.float()).unsqueeze(1)
    probs = torch.softmax(
        torch.where(keep, values.double(), float("-inf")), dim=-1
    ).nan_to_num(0.0)
    before = torch.cumsum(probs, dim=-1) - probs
    top_p = top_p.double().unsqueeze(1)
    keep &= (before < top_p) | (top_p >= 1.0)
    num_kept = keep.sum(dim=-1).clamp_min(1)
    threshold = values.gather(1, (num_kept - 1).unsqueeze(1))
    return logits.masked_fill(logits < threshold, float("-inf"))


def fresh_contexts(
    seeds: torch.Tensor, positions: torch.Tensor, steps: torch.Tensor
) -> torch.Tensor:
    """[rows, 4] contexts of the fresh-lane PRF: the request seed's two words,
    the row's position and its depth in the round."""
    seeds = seeds.long()
    return torch.stack(
        [
            seeds & 0xFFFFFFFF,
            (seeds >> 32) & 0xFFFFFFFF,
            positions.long().expand_as(seeds),
            steps.long().expand_as(seeds),
        ],
        dim=-1,
    )


def _lanes(
    prf: WatermarkPRF, contexts: torch.Tensor, ids: torch.Tensor
) -> torch.Tensor:
    """``-log U`` of every (row context, candidate) pair, float64 [rows, C]."""
    return -torch.log(prf.uniform(contexts, ids).double())


def _gumbel_max(
    prf: WatermarkPRF, logits: torch.Tensor, contexts: torch.Tensor
) -> torch.Tensor:
    """Gumbel-max sampling of ``logits`` on the lanes of ``prf`` at
    ``contexts`` (``GumbelWatermarker``'s rule, any PRF and contexts)."""
    if type(prf) is PhiloxPRF and logits.device.type == "cuda":
        from vllm.v1.worker.gpu.sample.watermark import philox_gumbel_sample

        return philox_gumbel_sample(logits, contexts, prf.key)
    vocabulary = torch.arange(logits.shape[-1], device=logits.device)
    uniforms = prf.uniform(contexts, vocabulary)
    uniforms = uniforms.clamp_min(torch.finfo(torch.float32).tiny)
    return torch.argmax(logits - torch.log(-torch.log(uniforms)), dim=-1)


def draft_noise_contexts(random_sampler: RandomSampler) -> torch.Tensor:
    """[rows, 4] contexts of the fresh draft noise of the drafter's rows: the
    request seed, the row's position and its draft step, which no two rounds
    of a request share (a later round starts at a later position)."""
    requests = random_sampler.expanded_idx_mapping.long().clamp_min(0)
    steps = random_sampler.logits_cache_col
    if steps is None:
        raise ValueError("fresh draft noise needs each row's draft step")
    steps = torch.as_tensor(steps, device=requests.device).long()
    return fresh_contexts(
        random_sampler.seeds[requests],
        random_sampler.positions,
        steps.expand_as(requests),
    )


def _sample_draft_rows(
    prf: WatermarkPRF | None,
    logits: torch.Tensor,
    contexts: torch.Tensor,
    random_sampler: RandomSampler,
    repeated: torch.Tensor,
    request_enabled: torch.Tensor,
) -> torch.Tensor:
    """Draft tokens: the requests that do not watermark sample ``logits`` with
    vLLM's noise; the others race on ``prf``'s lanes at ``contexts`` (``prf``
    None: every such row on fresh draft noise), their ``repeated`` rows on
    fresh draft noise. Caches the draft logits as vLLM's sampler does."""
    fresh_ctx = draft_noise_contexts(random_sampler)
    on_fresh = request_enabled & (repeated if prf is not None else True)
    if logits.device.type == "cuda" and (prf is None or type(prf) is PhiloxPRF):
        from vllm.v1.worker.gpu.sample.watermark import philox_gumbel_sample

        # One launch with vLLM's state: the opted-out rows' sample and the cache.
        first_prf = _DRAFT_NOISE_PRF if prf is None else prf
        first_ctx = fresh_ctx if prf is None else contexts
        tokens = philox_gumbel_sample(
            logits,
            first_ctx,
            first_prf.key,
            skip_mask=~request_enabled,
            expanded_idx_mapping=random_sampler.expanded_idx_mapping,
            temperatures=random_sampler.temperatures,
            seeds=random_sampler.seeds,
            positions=random_sampler.positions,
            use_fp64=random_sampler.use_fp64,
            is_drafting=random_sampler.is_drafting,
            logits_cache=random_sampler.logits_cache,
            logits_cache_col=random_sampler.logits_cache_col,
            logits_cache_source=random_sampler.logits_cache_source,
        )
        if prf is None:
            return tokens
        fresh = philox_gumbel_sample(logits, fresh_ctx, _DRAFT_NOISE_PRF.key)
        return torch.where(on_fresh, fresh, tokens)
    # Torch: vLLM's sampler for the opted-out rows (it also fills the cache).
    plain = random_sampler(logits)
    fresh = _gumbel_max(_DRAFT_NOISE_PRF, logits, fresh_ctx)
    tokens = fresh if prf is None else _gumbel_max(prf, logits, contexts)
    tokens = torch.where(on_fresh, fresh, tokens)
    return torch.where(request_enabled, tokens, plain)


class BlockVerifiedGumbelDraftWatermarker(GumbelWatermarker):
    """The drafter's side: Gumbel-max sampling with the target's key on the
    draft law, whose logits ``DraftWatermarker`` takes from
    ``draft_law_logits`` for the watermarked requests' rows.

    ``fresh_draft_noise`` (``deduplicate_contexts="strict"``): repeated rows of
    watermarked requests draw fresh draft noise (``draft_noise_contexts``)
    instead of vLLM's, which repeats at a position drafted again in a later
    round."""

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
        draft_law_width: int = DRAFT_LAW_WIDTH,
        fresh_draft_noise: bool = False,
    ) -> None:
        super().__init__(key, context_width, prf)
        if draft_law_width < 1:
            raise ValueError("draft_law_width must be positive")
        self.draft_law_width = draft_law_width
        self.fresh_draft_noise = fresh_draft_noise

    @property
    def samples_draft_rows(self) -> bool:
        return self.fresh_draft_noise

    def draft_law_logits(
        self,
        logits: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        min_p: torch.Tensor,
    ) -> torch.Tensor:
        return draft_law_logits(logits, top_k, top_p, min_p, self.draft_law_width)

    def sample_draft_rows(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler,
        skip_mask: torch.Tensor,
        request_enabled: torch.Tensor,
    ) -> torch.Tensor:
        """Draft tokens of ``DraftWatermarker`` (``samples_draft_rows``):
        ``skip_mask`` rows of ``request_enabled`` requests are repeated."""
        return _sample_draft_rows(
            self.prf, logits, contexts, random_sampler, skip_mask, request_enabled
        )


class FreshNoiseDraftWatermarker(Watermarker):
    """Drafts that do not depend on the key: every watermarked request's
    draft is a Gumbel-max sample of its draft distribution on fresh draft
    noise (``draft_noise_contexts``), so the draft is an exact sample of that
    distribution, fresh in every round (vLLM's own noise repeats at a position
    drafted again). The other requests draft with vLLM's noise. The draft
    distribution is the drafter's temperature-only one (``SampledDraftWatermarker``:
    the draft law)."""

    samples_draft_rows = True

    def __init__(self, context_width: int = 4) -> None:
        self._context_width = context_width

    @property
    def context_width(self) -> int:
        return self._context_width

    def sample_draft_rows(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler,
        skip_mask: torch.Tensor,
        request_enabled: torch.Tensor,
    ) -> torch.Tensor:
        return _sample_draft_rows(
            None, logits, contexts, random_sampler, skip_mask, request_enabled
        )

    def _sample_watermarked(
        self, logits: torch.Tensor, contexts: torch.Tensor
    ) -> WatermarkSample:
        raise NotImplementedError(
            "sampled drafts need the request seeds and positions: "
            "DraftWatermarker calls sample_draft_rows"
        )


class SampledDraftWatermarker(FreshNoiseDraftWatermarker):
    """The drafter's side of ``keyed_drafts=False``: every watermarked
    request's draft is a Gumbel-max sample of its draft law on fresh draft
    noise (``draft_noise_contexts``), which does not depend on the key, so
    the draft is an exact sample of the draft law, fresh in every round. The
    other requests draft with vLLM's noise on the temperature-only law."""

    def __init__(
        self, context_width: int = 4, draft_law_width: int = DRAFT_LAW_WIDTH
    ) -> None:
        if draft_law_width < 1:
            raise ValueError("draft_law_width must be positive")
        super().__init__(context_width)
        self.draft_law_width = draft_law_width

    def draft_law_logits(
        self,
        logits: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        min_p: torch.Tensor,
    ) -> torch.Tensor:
        return draft_law_logits(logits, top_k, top_p, min_p, self.draft_law_width)


class KeyShareCounter:
    """Running counts of the keyed emitted tokens under key A and key B, for
    the fused two-key detectors; with the environment variable named by
    ``_init_key_counts`` set to a path, the counts are kept in that file
    (JSON). Subclasses call ``_init_key_counts`` and ``_count_keys``."""

    def _init_key_counts(self, env_var: str = KEY_LOG_ENV_VAR) -> None:
        self.key_log_path = os.environ.get(env_var) or None
        self._key_counts: torch.Tensor | None = None
        self._key_pinned: torch.Tensor | None = None
        self._key_event = None
        self._key_written: list[int] | None = None

    def _count_keys(self, n_a: torch.Tensor, n_b: torch.Tensor) -> None:
        """Accumulate the keyed emitted tokens under key A and key B on the
        device; with ``VLLM_BV_GUMBEL_KEY_LOG``, refresh the log file."""
        counts = torch.stack([n_a.long(), n_b.long()])
        self._key_counts = (
            counts if self._key_counts is None else self._key_counts + counts
        )
        if self.key_log_path is not None:
            self._flush_key_log()

    def _flush_key_log(self, sync: bool = False) -> None:
        """Write the running counts to ``key_log_path``. On CUDA without a
        host synchronization: the counts go to a pinned buffer by a
        non-blocking copy, which a later call writes out once its event has
        completed (so the file lags by a call or two)."""
        counts = self._key_counts
        if counts is None:
            return
        if sync or counts.device.type != "cuda":
            self._write_key_log([int(v) for v in counts.tolist()])
            return
        if self._key_event is not None and self._key_event.query():
            assert self._key_pinned is not None
            self._write_key_log([int(v) for v in self._key_pinned.tolist()])
            self._key_event = None
        if self._key_event is None:
            if self._key_pinned is None:
                self._key_pinned = torch.zeros(2, dtype=torch.long, pin_memory=True)
            self._key_pinned.copy_(counts, non_blocking=True)
            self._key_event = torch.cuda.Event()
            self._key_event.record(torch.cuda.current_stream(counts.device))

    def _write_key_log(self, values: list[int]) -> None:
        if values == self._key_written or self.key_log_path is None:
            return
        tmp = f"{self.key_log_path}.tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(
                {
                    "key_a": values[0],
                    "key_b": values[1],
                    "pid": os.getpid(),
                    "time": time.time(),
                },
                f,
            )
        os.replace(tmp, self.key_log_path)
        self._key_written = values

    def key_share_stats(self) -> tuple[int, int]:
        """(keyed emitted tokens under key A, under key B) since construction
        (synchronizes)."""
        if self._key_counts is None:
            return 0, 0
        n_a, n_b = (int(v) for v in self._key_counts.tolist())
        return n_a, n_b



class BlockVerifiedGumbelVerificationWatermarker(GumbelWatermarker, KeyShareCounter):
    """The verifier's side; without speculation it samples like ``gumbel``.

    ``draft_law_verification_logits`` decides each round of the keyed drafts
    of ``BlockVerifiedGumbelDraftWatermarker`` (module docstring) and returns
    point masses, which vLLM's rejection sampling emits unchanged up to the
    first rejected draft. ``verification_logits`` does the same for drafts
    that do not depend on the key, as point masses (greedy drafts; lookahead
    Gumbel-max's rule), with its rows in ``skip_mask`` on fresh lanes from a
    generator the watermarker owns (it receives no request seeds).
    """

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
        law: str = "block",
        pass_aggregator: str = "none",
        pass_source: str = "own",
        draft_law_width: int = DRAFT_LAW_WIDTH,
        target_slots: int = TARGET_SLOTS,
        keyed_drafts: bool = True,
        fresh_lanes: bool = False,
        residual_race: bool = True,
    ) -> None:
        super().__init__(key, context_width, prf)
        if law not in VERIFICATION_LAWS:
            raise ValueError(f"verification_law must be one of {VERIFICATION_LAWS}")
        if pass_aggregator not in PASS_AGGREGATORS:
            raise ValueError(f"pass_aggregator must be one of {PASS_AGGREGATORS}")
        if pass_source not in PASS_SOURCES:
            raise ValueError(f"pass_source must be one of {PASS_SOURCES}")
        if draft_law_width < 1 or target_slots < 1:
            raise ValueError("draft_law_width and target_slots must be positive")
        _check_rule(law, pass_aggregator, keyed_drafts, fresh_lanes, residual_race)
        self.law = law
        self.pass_aggregator = pass_aggregator
        self.pass_source = pass_source
        self.draft_law_width = draft_law_width
        self.target_slots = target_slots
        self.keyed_drafts = keyed_drafts
        self.fresh_lanes = fresh_lanes
        self.residual_race = residual_race
        # Plain block verification: the second key, of the token after a
        # rejection.
        self.key_b_prf: WatermarkPRF | None = None
        if not residual_race:
            self.key_b_prf = create_prf("philox", derive_watermark_key(key, KEY_B_DOMAIN))
        self.strict = os.environ.get(STRICT_ENV_VAR, "0") == "1"
        self.report_every = int(os.environ.get(REPORT_EVERY_ENV_VAR, "1000"))
        self.compile_race = os.environ.get(COMPILE_ENV_VAR, "0") == "1"
        self._init_key_counts()
        self._generators: dict[torch.device, torch.Generator] = {}
        self._counts: torch.Tensor | None = None
        self._calls = 0

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
    ) -> WatermarkSample:
        if self.fresh_lanes:
            # The unwatermarked baseline: vLLM samples every row itself.
            if random_sampler is None:
                raise ValueError("fresh_lanes samples with vLLM's random sampler")
            return WatermarkSample(random_sampler(logits), logits)
        return super().sample(logits, contexts, random_sampler, skip_mask)

    def draft_law_logits(
        self,
        logits: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        min_p: torch.Tensor,
    ) -> torch.Tensor:
        return draft_law_logits(logits, top_k, top_p, min_p, self.draft_law_width)

    # ---- entry points ---------------------------------------------------------

    def verification_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        draft_tokens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        skip_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Point-mass drafts that do not depend on the key (as
        ``SupportsVerificationLogits``). A request is decided by the rule when
        one of its rows is watermarked; its rows in ``skip_mask`` race on
        fresh lanes. The other requests keep their logits."""
        num_rows = logits.shape[0]
        if num_rows == 0:
            return logits
        device = logits.device
        request, local, _ = _layout(cu_num_logits, num_rows)
        num_requests = cu_num_logits.shape[0] - 1
        watermarked = torch.zeros(num_requests, dtype=torch.long, device=device)
        watermarked.index_add_(0, request, (~skip_mask).long())
        generator = self._generators.get(device)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(0x4256474D_47454E)
            self._generators[device] = generator
        seeds = torch.randint(
            -(2**62), 2**62, (num_rows,), device=device, generator=generator
        )
        fresh = fresh_contexts(seeds, torch.zeros_like(seeds), local)
        tokens, raced = self._decide(
            logits,
            contexts,
            draft_tokens,
            cu_num_logits,
            watermarked > 0,
            skip_mask,
            None,
            fresh,
        )
        return _point_masses(logits, tokens, raced)

    def draft_law_verification_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        draft_tokens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        request_enabled: torch.Tensor,
        repeated: torch.Tensor,
        draft_law: torch.Tensor,
        random_sampler: RandomSampler,
    ) -> torch.Tensor:
        """As ``SupportsDraftLawVerification``: the keyed drafts' rounds.
        ``request_enabled`` [rows] marks the rows of requests that watermark
        (their drafts raced on ``draft_law``), ``repeated`` [rows] the rows on
        fresh lanes and ``draft_law`` [rows, vocab] each draft row's
        ``draft_law_logits``."""
        num_rows = logits.shape[0]
        if num_rows == 0:
            return logits
        bounds = cu_num_logits.long()
        requests = random_sampler.expanded_idx_mapping.long().clamp_min(0)
        _, local, _ = _layout(cu_num_logits, num_rows)
        fresh = fresh_contexts(
            random_sampler.seeds[requests], random_sampler.positions, local
        )
        tokens, raced = self._decide(
            logits,
            contexts,
            draft_tokens,
            cu_num_logits,
            request_enabled[bounds[:-1].clamp_max(num_rows - 1)],
            repeated,
            draft_law,
            fresh,
        )
        return _point_masses(logits, tokens, raced)

    # ---- the rule ---------------------------------------------------------------

    def _decide(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        draft_tokens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        raced_requests: torch.Tensor,
        fresh_rows: torch.Tensor,
        draft_law: torch.Tensor | None,
        fresh_ctx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Each row's point mass and whether its request is decided here.

        Rows are in path order: each request's draft rows, then its bonus row.
        ``draft_law`` None means point-mass drafts."""
        num_rows, vocab_size = logits.shape
        device = logits.device
        logits = logits.float()
        logits = logits.masked_fill(logits.isnan(), float("-inf"))
        request, local, sizes = _layout(cu_num_logits, num_rows)
        num_requests = sizes.numel()

        # Drafts per request in the rule: up to the first invalid draft.
        drafts = draft_tokens.long()
        last = sizes - 1
        valid = (drafts >= 0) & (drafts < vocab_size)
        big = torch.full((num_requests,), num_rows, dtype=torch.long, device=device)
        first_bad = big.scatter_reduce(
            0,
            request,
            torch.where((local < last[request]) & ~valid, local, num_rows),
            reduce="amin",
        )
        K = torch.minimum(last, first_bad).clamp_min(0)
        is_draft = local < K[request]
        in_round = local <= K[request]
        drafts = torch.where(is_draft, drafts, 0)
        raced_rows = raced_requests[request]
        # fresh_lanes: every row races on fresh lanes (nothing is watermarked).
        fresh_rows = (fresh_rows | self.fresh_lanes) & raced_rows

        # One host synchronization: the batch's shapes.
        live_rows = raced_rows & in_round
        count_p = torch.isfinite(logits).sum(dim=-1)
        max_count_p = torch.where(live_rows, count_p, 0).max()
        if draft_law is not None:
            draft_law = draft_law.float()
            draft_law = draft_law.masked_fill(draft_law.isnan(), float("-inf"))
            count_q = torch.isfinite(draft_law).sum(dim=-1)
            max_count_q = torch.where(live_rows & is_draft, count_q, 0).max()
        else:
            max_count_q = (live_rows & is_draft).any().long()
        num_raced, max_count_p, max_count_q, depth = torch.stack(
            [raced_requests.long().sum(), max_count_p, max_count_q, sizes.max()]
        ).tolist()
        if num_raced == 0:
            return torch.zeros(num_rows, dtype=torch.long, device=device), raced_rows

        # Candidate slots: the target's support (its top slots), the draft
        # law's support without them, and one slot for the rest of the target.
        width_p = max(1, min(max_count_p, self.target_slots, vocab_size))
        use_rest = max_count_p > width_p
        ids_p = torch.topk(logits, width_p, dim=-1).indices
        if max_count_q > 0:
            if draft_law is None:
                ids_q = drafts.unsqueeze(1)
            else:
                ids_q = torch.topk(draft_law, max_count_q, dim=-1).indices
            in_p = torch.zeros(num_rows, vocab_size, dtype=torch.bool, device=device)
            in_p.scatter_(1, ids_p, True)
            duplicate = in_p.gather(1, ids_q)
            ids = torch.cat([ids_p, ids_q], dim=1)
            live = torch.cat([torch.ones_like(ids_p, dtype=torch.bool), ~duplicate], 1)
        else:
            ids, live = ids_p, torch.ones_like(ids_p, dtype=torch.bool)
        target = logits.gather(1, ids).double().masked_fill(~live, float("-inf"))
        # Every slot's keyed lane and fresh lane; the rest slot's token is the
        # winner of its own race, so it depends on the lanes it races on.
        hit = (ids == drafts.unsqueeze(1)) & live
        keyed_lanes = _lanes(self.prf, contexts, ids)
        fresh_lanes = _lanes(_FRESH_LANE_PRF, fresh_ctx, ids)
        keyed_ids = fresh_ids = ids
        # Plain block verification: key B's lanes, and their fresh lanes.
        plain = not self.residual_race and self.law != "shared"
        if plain:
            assert self.key_b_prf is not None
            key_b_lanes = _lanes(self.key_b_prf, contexts, ids)
            fresh_b_lanes = _lanes(_FRESH_LANE_KEY_B_PRF, fresh_ctx, ids)
            key_b_ids = fresh_b_ids = ids
        if use_rest:
            log_mass, keyed_rest, keyed_rest_lane, fresh_rest, fresh_rest_lane = (
                self._rest(logits, ids, contexts, fresh_ctx)
            )
            keyed_ids = torch.cat([ids, keyed_rest.unsqueeze(1)], dim=1)
            fresh_ids = torch.cat([ids, fresh_rest.unsqueeze(1)], dim=1)
            live = torch.cat([live, torch.isfinite(log_mass).unsqueeze(1)], dim=1)
            hit = torch.cat([hit, torch.zeros_like(hit[:, :1])], dim=1)
            target = torch.cat([target, log_mass.unsqueeze(1)], dim=1)
            keyed_lanes = torch.cat([keyed_lanes, keyed_rest_lane.unsqueeze(1)], 1)
            fresh_lanes = torch.cat([fresh_lanes, fresh_rest_lane.unsqueeze(1)], 1)
            if plain:
                assert self.key_b_prf is not None
                _, b_rest, b_rest_lane, fb_rest, fb_rest_lane = self._rest(
                    logits,
                    ids,
                    contexts,
                    fresh_ctx,
                    self.key_b_prf,
                    _FRESH_LANE_KEY_B_PRF,
                )
                key_b_ids = torch.cat([ids, b_rest.unsqueeze(1)], dim=1)
                fresh_b_ids = torch.cat([ids, fb_rest.unsqueeze(1)], dim=1)
                key_b_lanes = torch.cat([key_b_lanes, b_rest_lane.unsqueeze(1)], 1)
                fresh_b_lanes = torch.cat([fresh_b_lanes, fb_rest_lane.unsqueeze(1)], 1)
        P = torch.softmax(target, dim=-1).nan_to_num(0.0)

        def treat(on_fresh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """(lanes, slot token ids) with the rows ``on_fresh`` on fresh lanes."""
            rows = on_fresh.unsqueeze(1)
            return (
                torch.where(rows, fresh_lanes, keyed_lanes),
                torch.where(rows, fresh_ids, keyed_ids),
            )

        # The draft law on the slots (zero at bonus rows and padding).
        found = hit.any(dim=1)
        dpos = hit.to(torch.uint8).argmax(dim=1)
        if draft_law is None:
            Q = hit.double()
        else:
            law = draft_law.gather(1, keyed_ids).double()
            Q = torch.softmax(law.masked_fill(~live, float("-inf")), dim=-1)
            Q = Q.nan_to_num(0.0)
        Q = Q * (is_draft & found).unsqueeze(1)

        # Shared clocks: every row's own Gumbel-max winner (repeated contexts
        # on fresh lanes), exact for any drafts.
        lanes, slot_ids = treat(fresh_rows)
        shared = slot_ids.gather(1, race(lanes, P).unsqueeze(1)).squeeze(1)
        fallback = torch.zeros(num_requests, dtype=torch.bool, device=device)
        unkeyed = torch.zeros(num_rows, dtype=torch.bool, device=device)
        if self.law == "shared":
            tokens = shared
        else:
            # A draft outside the draft law cannot be verified as its sample:
            # its request uses the shared clocks (a decision the lanes do not
            # enter). A keyed draft that is not its race's winner on the law
            # was not raced on these lanes (a repeated context at the drafter,
            # or a float32 near tie): it is verified on fresh lanes instead,
            # as are the repeated contexts, which keeps every row's clocks
            # independent Exp(1) given the drafts.
            q_at_draft = Q.gather(1, dpos.unsqueeze(1)).squeeze(1)
            in_law = found & (q_at_draft > 0)
            outside = live_rows & is_draft & ~in_law
            counts = torch.zeros(num_requests, dtype=torch.long, device=device)
            counts.index_add_(0, request, outside.long())
            fallback = counts > 0
            if self.keyed_drafts and self.residual_race:
                winner = race(lanes, Q) == dpos
                unkeyed = live_rows & is_draft & ~fresh_rows & in_law & ~winner
            on_fresh = fresh_rows | unkeyed
            rule_lanes, rule_ids = treat(on_fresh)
            if self.keyed_drafts and self.residual_race:
                # On fresh lanes F a draft row races on the clocks of a draft
                # race given its winner d: tau = F(d) and the leftovers F(v),
                # v != d.
                tau = rule_lanes.gather(1, dpos.unsqueeze(1))
                conditional = (rule_lanes + tau * Q).scatter(
                    1, dpos.unsqueeze(1), tau * q_at_draft.unsqueeze(1)
                )
                rule_lanes = torch.where(
                    (on_fresh & is_draft & in_law).unsqueeze(1),
                    conditional,
                    rule_lanes,
                )
            # Sampled drafts never touched the lanes: the race reads the raw
            # lanes, keyed or (repeated contexts) fresh.

            slot_rows = torch.full(
                (num_requests, depth), -1, dtype=torch.long, device=device
            )
            slot_rows[request, local] = torch.arange(num_rows, device=device)
            index = slot_rows.clamp_min(0)
            pad = ((slot_rows < 0) | ~in_round[index]).unsqueeze(2)
            pad2 = pad.squeeze(2)
            if plain:
                lanes_b = torch.where(
                    on_fresh.unsqueeze(1), fresh_b_lanes, key_b_lanes
                )
                ids_b = torch.where(on_fresh.unsqueeze(1), fresh_b_ids, key_b_ids)
                stop_draw = _lanes(
                    _STOP_PRF, fresh_ctx, torch.zeros_like(ids[:, :1])
                ).squeeze(1)
                stopping, column = plain_block_decision(
                    rule_lanes[index].masked_fill(pad, 1.0),
                    lanes_b[index].masked_fill(pad, 1.0),
                    stop_draw[index].masked_fill(pad2, 1.0),
                    P[index].masked_fill(pad, 0.0),
                    Q[index].masked_fill(pad, 0.0),
                    dpos[index].masked_fill(pad2, 0),
                    K,
                    self.law,
                )
                stop_rows = index.gather(1, stopping.unsqueeze(1)).squeeze(1)
                bonus = stopping >= K
                emitted = torch.where(
                    bonus,
                    rule_ids[stop_rows].gather(1, column.unsqueeze(1)).squeeze(1),
                    ids_b[stop_rows].gather(1, column.unsqueeze(1)).squeeze(1),
                )
                # Keyed emitted tokens: the kept drafts and the bonus token on
                # rows that race on the key (key A), the token after a
                # rejection on such a row (key B).
                keyed_rows = (~on_fresh)[index] & ~pad2
                kept = keyed_rows & (
                    torch.arange(depth, device=device).unsqueeze(0)
                    < stopping.unsqueeze(1)
                )
                at_stop = keyed_rows.gather(1, stopping.unsqueeze(1)).squeeze(1)
                decided = raced_requests & ~fallback
                n_a = (kept.sum(dim=1) + (at_stop & bonus)) * decided
                n_b = (at_stop & ~bonus) * decided
                self._count_keys(n_a.sum(), n_b.sum())
            else:
                stopping, column = _race(device, self.compile_race)(
                    rule_lanes[index].masked_fill(pad, 1.0),
                    P[index].masked_fill(pad, 0.0),
                    Q[index].masked_fill(pad, 0.0),
                    dpos[index].masked_fill(pad2, 0),
                    K,
                    self.law,
                    self.pass_aggregator,
                    self.pass_source,
                    not self.keyed_drafts,
                )
                stop_rows = index.gather(1, stopping.unsqueeze(1)).squeeze(1)
                emitted = rule_ids[stop_rows].gather(1, column.unsqueeze(1)).squeeze(1)
            stop = stopping[request]
            tokens = torch.where(
                local < stop,
                drafts,
                torch.where(local == stop, emitted[request], shared),
            )
            tokens = torch.where(fallback[request], shared, tokens)
        self._count(fallback & raced_requests, unkeyed, raced_requests)
        return tokens, raced_rows

    def _rest(
        self,
        logits: torch.Tensor,
        ids: torch.Tensor,
        contexts: torch.Tensor,
        fresh_ctx: torch.Tensor,
        keyed_prf: WatermarkPRF | None = None,
        fresh_prf: WatermarkPRF = _FRESH_LANE_PRF,
    ) -> tuple[torch.Tensor, ...]:
        """The slot of the target's support outside ``ids``: its log-mass and,
        on the keyed lanes and on the fresh lanes, its token (the winner of its
        own race, ``argmin E / p`` over the rest) and its lane ``mass *
        E(token) / p(token)``, an Exp(1) clock that races like the merged
        lanes. ``keyed_prf`` (default the key's) and ``fresh_prf`` choose the
        two PRFs."""
        vocab_size = logits.shape[1]
        rest = logits.scatter(1, ids, float("-inf"))
        log_mass = torch.logsumexp(rest.double(), dim=-1)
        on_cuda = logits.device.type == "cuda"

        def winner(prf: WatermarkPRF, ctx: torch.Tensor) -> tuple[torch.Tensor, ...]:
            if type(prf) is PhiloxPRF and on_cuda:
                from vllm.v1.worker.gpu.sample.watermark import philox_gumbel_sample

                token = philox_gumbel_sample(rest, ctx, prf.key)
            else:
                vocabulary = torch.arange(vocab_size, device=logits.device)
                u = prf.uniform(ctx, vocabulary)
                u = u.clamp_min(torch.finfo(torch.float32).tiny)
                token = torch.argmax(rest - torch.log(-torch.log(u)), dim=-1)
            lane = _lanes(prf, ctx, token.unsqueeze(1)).squeeze(1)
            token_logit = rest.gather(1, token.unsqueeze(1)).squeeze(1).double()
            finite = torch.isfinite(log_mass) & torch.isfinite(token_logit)
            scale = torch.exp(log_mass - torch.where(finite, token_logit, 0.0))
            return token, torch.where(finite, lane * scale, 1.0)

        keyed_token, keyed_lane = winner(
            self.prf if keyed_prf is None else keyed_prf, contexts
        )
        fresh_token, fresh_lane = winner(fresh_prf, fresh_ctx)
        return log_mass, keyed_token, keyed_lane, fresh_token, fresh_lane

    # ---- bookkeeping ------------------------------------------------------------

    def _count(
        self, fallback: torch.Tensor, unkeyed: torch.Tensor, raced: torch.Tensor
    ) -> None:
        """Accumulate the counters on the device; log them every
        ``report_every`` calls (one synchronization)."""
        if self.strict and bool(fallback.any()):
            raise RuntimeError(
                "block_verified_gumbel: a draft outside the draft law (the drafter "
                "and verification computed different draft laws)"
            )
        counts = torch.stack([fallback.sum(), unkeyed.sum(), raced.sum()])
        self._counts = counts if self._counts is None else self._counts + counts
        self._calls += 1
        if self.report_every > 0 and self._calls % self.report_every == 0:
            fallbacks, unkeyed_rows, num_raced = self.fallback_stats()
            logger.info(
                "block_verified_gumbel: of %d verified requests, %d had a draft "
                "outside the draft law (shared clocks) and %d draft rows lost "
                "their keyed race (fresh clocks)",
                num_raced,
                fallbacks,
                unkeyed_rows,
            )

    def fallback_stats(self) -> tuple[int, int, int]:
        """(requests verified on the shared clocks because a draft lies outside
        the draft law, keyed draft rows verified on fresh clocks because the
        draft is not its race's winner, requests verified) since construction
        (synchronizes)."""
        if self._counts is None:
            return 0, 0, 0
        fallbacks, unkeyed_rows, num_raced = self._counts.tolist()
        return fallbacks, unkeyed_rows, num_raced

    # ---- plain block verification: the keys' shares --------------------------

def _check_rule(
    law: str,
    pass_aggregator: str,
    keyed_drafts: bool,
    fresh_lanes: bool,
    residual_race: bool,
) -> None:
    """The combinations of the speculative rule's switches (as
    ``WatermarkConfig`` validates them)."""
    if fresh_lanes and (keyed_drafts or pass_aggregator != "none"):
        raise ValueError(
            "fresh_lanes is the unwatermarked baseline of the sampled drafts: "
            "keyed_drafts=False and pass_aggregator 'none'"
        )
    if not keyed_drafts and law == "shared":
        raise ValueError("verification_law 'shared' verifies keyed drafts")
    if not residual_race and (
        not keyed_drafts or law not in ("block", "token") or pass_aggregator != "none"
    ):
        raise ValueError(
            "residual_race=False is plain block or token-level verification of "
            "keyed drafts, without a pass_aggregator"
        )


class BlockVerifiedGumbelWatermarker(GumbelWatermarker, SupportsSpeculativeDecoding):
    """``block_verified_gumbel``: Gumbel-max whose speculative decoding drafts
    on the key (``draft_watermarker``) and verifies with the block-verified
    rule (``target_watermarker``); see the module docstring. Without
    speculation it samples like ``gumbel``; detection is
    ``GumbelWatermarkDetector`` with the same key and context width.

    The switches of the speculative rule (module docstring, "The family"):
    ``keyed_drafts=False`` samples the drafts from the draft law on fresh
    noise (``SampledDraftWatermarker``) and verifies on the raw lanes;
    ``fresh_lanes`` (with sampled drafts) verifies on fresh lanes only, the
    unwatermarked baseline, which also samples without speculation as vLLM
    does; ``residual_race=False`` is plain block (or token-level) verification
    of the keyed drafts with a second key, detected by
    ``FusedTwoKeyGumbelWatermarkDetector``; ``fresh_draft_noise`` draws the
    keyed drafter's repeated rows on fresh draft noise
    (``deduplicate_contexts="strict"``)."""

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRFName = "philox",
        law: str = "block",
        pass_aggregator: str = "none",
        pass_source: str = "own",
        draft_law_width: int = DRAFT_LAW_WIDTH,
        target_slots: int = TARGET_SLOTS,
        keyed_drafts: bool = True,
        fresh_lanes: bool = False,
        residual_race: bool = True,
        fresh_draft_noise: bool = False,
    ) -> None:
        super().__init__(key, context_width, prf)
        self.fresh_lanes = fresh_lanes
        self.draft_watermarker: Watermarker
        if keyed_drafts:
            self.draft_watermarker = BlockVerifiedGumbelDraftWatermarker(
                key, context_width, prf, draft_law_width, fresh_draft_noise
            )
        else:
            self.draft_watermarker = SampledDraftWatermarker(
                context_width, draft_law_width
            )
        self.target_watermarker = BlockVerifiedGumbelVerificationWatermarker(
            key,
            context_width,
            prf,
            law,
            pass_aggregator,
            pass_source,
            draft_law_width,
            target_slots,
            keyed_drafts,
            fresh_lanes,
            residual_race,
        )

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
    ) -> WatermarkSample:
        if self.fresh_lanes:
            if random_sampler is None:
                raise ValueError("fresh_lanes samples with vLLM's random sampler")
            return WatermarkSample(random_sampler(logits), logits)
        return super().sample(logits, contexts, random_sampler, skip_mask)


def outcome_law_fast(
    P: torch.Tensor,
    Q: torch.Tensor,
    dpos: torch.Tensor,
    K: torch.Tensor,
    law: str = "block",
) -> tuple[torch.Tensor, torch.Tensor]:
    """``textseal2_verification.outcome_law`` without its loops over depth:
    block verification's running ratio ``c_i = min(1, c_{i-1} p_i(d_i) /
    q_i(d_i))`` is ``exp(S_i - max(0, max_{j <= i} S_j))`` for the partial
    sums ``S`` of the log-ratios, and the stopping law is a reverse product.
    Same inputs and outputs (pi [B, D, C], stop [B, D])."""
    B, D, C = P.shape
    depth = torch.arange(D, device=P.device).unsqueeze(0)
    is_draft = depth < K.unsqueeze(1)
    is_bonus = depth == K.unsqueeze(1)
    pos = dpos.clamp(0, C - 1).unsqueeze(2)
    pd = P.gather(2, pos).squeeze(2)
    qd = Q.gather(2, pos).squeeze(2)
    ratio = pd / torch.where(qd > 0, qd, 1.0)
    if law == "token":
        accept = torch.where(is_draft, ratio.clamp(max=1.0), 1.0)
        reach = torch.cumprod(
            torch.cat([torch.ones_like(accept[:, :1]), accept[:, :-1]], dim=1), dim=1
        )
        u = (P - Q).clamp_min(0.0)
        z = u.sum(dim=2, keepdim=True)
        r = torch.where(z > 0, u / torch.where(z > 0, z, 1.0), 0.0)
        stop = torch.where(
            is_draft, reach * (1.0 - accept), torch.where(is_bonus, reach, 0.0)
        )
        R = torch.where(is_draft.unsqueeze(2), r, P)
    elif law == "block":
        log_ratio = torch.where(is_draft, torch.log(ratio), 0.0)
        sums = torch.cumsum(log_ratio, dim=1)
        peak = torch.cummax(sums, dim=1).values.clamp_min(0.0)
        c = torch.exp(sums - peak)  # c_i, after draft i
        c_prev = torch.cat([torch.ones_like(c[:, :1]), c[:, :-1]], dim=1)
        u = (c_prev.unsqueeze(2) * P - Q).clamp_min(0.0)
        z = u.sum(dim=2)
        r = torch.where(
            z.unsqueeze(2) > 0, u / torch.where(z > 0, z, 1.0).unsqueeze(2), 0.0
        )
        den = z + 1.0 - c_prev
        h_draft = torch.where(den > 0, z / torch.where(den > 0, den, 1.0), 1.0)
        h_draft = torch.where(depth == 0, 1.0, h_draft)
        h = torch.where(is_draft, h_draft, torch.where(is_bonus, c_prev, 0.0)).clamp(
            0.0, 1.0
        )
        # tail_i = prod_{j > i} (1 - h_j), multiplied from the deepest row up.
        survive = torch.flip(torch.cumprod(torch.flip(1.0 - h, [1]), dim=1), [1])
        tail = torch.cat([survive[:, 1:], torch.ones_like(survive[:, :1])], dim=1)
        stop = torch.where(depth <= K.unsqueeze(1), h * tail, 0.0)
        R = torch.where(
            is_draft.unsqueeze(2), r, torch.where(is_bonus.unsqueeze(2), P, 0.0)
        )
    elif law == "point":
        # Point-mass verification: keep d_i with probability p_i(d_i), else
        # stop with v at weight p_i(v); P(I = i) = reach_i (1 - p_i(d_i)).
        accept = torch.where(is_draft, pd.clamp(0.0, 1.0), 1.0)
        reach = torch.cumprod(
            torch.cat([torch.ones_like(accept[:, :1]), accept[:, :-1]], dim=1), dim=1
        )
        stop = torch.where(
            is_draft, reach * (1.0 - accept), torch.where(is_bonus, reach, 0.0)
        )
        live = (is_draft | is_bonus).unsqueeze(2)
        pi = torch.where(live, reach.unsqueeze(2) * P, 0.0)
        hit = torch.zeros_like(pi, dtype=torch.bool).scatter_(2, pos, True)
        return pi.masked_fill(hit & is_draft.unsqueeze(2), 0.0), stop
    else:
        raise ValueError(f"law must be one of {RESIDUAL_LAWS}")
    pi = stop.unsqueeze(2) * R
    hit = torch.zeros_like(pi, dtype=torch.bool).scatter_(2, pos, True)
    return pi.masked_fill(hit & is_draft.unsqueeze(2), 0.0), stop


def residual_race_fast(
    E: torch.Tensor,
    P: torch.Tensor,
    Q: torch.Tensor,
    dpos: torch.Tensor,
    K: torch.Tensor,
    law: str = "block",
    pass_aggregator: str = "none",
    pass_source: str = "own",
    raw: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``textseal2_verification.residual_race`` on ``outcome_law_fast``, with
    the backward chain vectorized over depth where it does not carry state
    (``none``: one race; ``identity`` with ``own``: every row's pass clock is
    its own draft time). Same inputs and outputs. ``raw`` is
    ``use_leftovers=False, pass_time="raw"``: drafts that do not depend on the
    key race on the raw lanes, and a row's pass clock reads its drafted
    token's raw lane."""
    pi, stop = outcome_law_fast(P, Q, dpos, K, law)
    B, D, C = E.shape
    depth = torch.arange(D, device=E.device).unsqueeze(0)
    is_draft = depth < K.unsqueeze(1)
    pos = dpos.clamp(0, C - 1).unsqueeze(2)
    if raw:
        tau = torch.where(is_draft, E.gather(2, pos).squeeze(2), 0.0)
        lanes = E
    else:
        qd = Q.gather(2, pos).squeeze(2)
        tau = E.gather(2, pos).squeeze(2) / torch.where(qd > 0, qd, 1.0)
        tau = torch.where(is_draft & (qd > 0), tau, 0.0)
        lanes = (E - tau.unsqueeze(2) * Q).clamp_min(0.0)
    ratio = torch.where(pi > 0, lanes / torch.where(pi > 0, pi, 1.0), torch.inf)
    if pass_aggregator == "none":
        flat = ratio.reshape(B, D * C).argmin(dim=1)
        return flat // C, flat % C
    M, col = ratio.min(dim=2)
    after = (1.0 - torch.cumsum(stop, dim=1)).clamp_min(0.0)
    if pass_aggregator == "identity" and pass_source == "own":
        # S_{i+1} is finite whenever P(I > i) > 0, so only tau_i enters.
        later = is_draft & (after > 0)
        clock = torch.where(later, tau / torch.where(later, after, 1.0), torch.inf)
        stop_here = (clock > M) | (torch.isinf(clock) & torch.isfinite(M))
        first = torch.where(stop_here, depth, D - 1).min(dim=1).values
        return first, col.gather(1, first.unsqueeze(1)).squeeze(1)
    inf = torch.full((B,), torch.inf, dtype=E.dtype, device=E.device)
    S = inf.clone()
    stopping = torch.full((B,), D - 1, dtype=torch.long, device=E.device)
    stopping_col = col[:, D - 1].clone()
    for i in range(D - 1, -1, -1):
        a_i = after[:, i]
        has_later = is_draft[:, i] & (a_i > 0) & torch.isfinite(S)
        unit_s = torch.where(has_later, S * a_i, inf)
        if pass_source == "own":
            unit = pass_aggregate(tau[:, i], unit_s, pass_aggregator)
        else:
            child_tau = tau[:, i + 1] if i + 1 < D else torch.zeros_like(tau[:, i])
            unit = torch.where(
                i + 1 < K, pass_aggregate(child_tau, unit_s, pass_aggregator), unit_s
            )
        clock = torch.where(has_later, unit / torch.where(a_i > 0, a_i, 1.0), inf)
        M_i = M[:, i]
        stop_here = (M_i < clock) | (torch.isinf(clock) & torch.isfinite(M_i))
        stopping = torch.where(stop_here, i, stopping)
        stopping_col = torch.where(stop_here, col[:, i], stopping_col)
        S = torch.minimum(M_i, clock)
    return stopping, stopping_col


def plain_block_decision(
    E_A: torch.Tensor,
    E_B: torch.Tensor,
    F: torch.Tensor,
    P: torch.Tensor,
    Q: torch.Tensor,
    dpos: torch.Tensor,
    K: torch.Tensor,
    law: str = "block",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain block (or token-level) verification of keyed drafts
    (``residual_race=False``; the reference's Meta baseline recipe): the
    stopping depth ``I = argmin_i F_i / P(I = i)`` from fresh Exp(1) draws
    ``F`` [B, D], then at the bonus row (``I = K``) the key-A race
    ``argmin E_A / p_K``, else the key-B race on the residual law,
    ``argmin E_B / R_I`` (``pi_I`` is proportional to ``R_I``). Inputs as
    ``residual_race_fast``; returns (I [B], candidate [B])."""
    pi, stop = outcome_law_fast(P, Q, dpos, K, law)
    B, D, C = E_A.shape
    stopping = race(F, stop)
    rows = torch.arange(B, device=E_A.device)
    bonus = stopping >= K
    lanes = torch.where(
        bonus.unsqueeze(1), E_A[rows, stopping], E_B[rows, stopping]
    )
    return stopping, race(lanes, pi[rows, stopping])


_compiled_race = None


def _race(device: torch.device, compile_race: bool):
    """``residual_race_fast``, compiled on CUDA when asked."""
    global _compiled_race
    if not compile_race or device.type != "cuda":
        return residual_race_fast
    if _compiled_race is None:
        _compiled_race = torch.compile(residual_race_fast, dynamic=True)
    return _compiled_race


def _layout(
    cu_num_logits: torch.Tensor, num_rows: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(request [rows], depth in the request [rows], rows per request)."""
    bounds = cu_num_logits.long()
    sizes = bounds.diff()
    request = torch.repeat_interleave(
        torch.arange(sizes.numel(), device=bounds.device), sizes, output_size=num_rows
    )
    local = torch.arange(num_rows, device=bounds.device) - bounds[:-1][request]
    return request, local, sizes


def _point_masses(
    logits: torch.Tensor, tokens: torch.Tensor, raced: torch.Tensor
) -> torch.Tensor:
    point_masses = torch.full_like(logits, float("-inf")).scatter_(
        1, tokens.unsqueeze(1), 0.0
    )
    return torch.where(raced.unsqueeze(1), point_masses, logits)


# ---- plain block verification's detector ---------------------------------------


def _gammaincc(shape: float | torch.Tensor, x: float) -> torch.Tensor:
    shape = torch.as_tensor(shape, dtype=torch.float64)
    return torch.special.gammaincc(shape, torch.full_like(shape, x))


def fused_gamma_sf(n: int, s: float, a: float) -> float:
    """``P(S >= s)`` for ``S = sum_{j <= n} (a X_j + (1 - a) Y_j)``, X, Y
    i.i.d. Exp(1): the exact null of the fused two-key score. With ``t1 =
    max(a, 1 - a) >= t2 = min(a, 1 - a) > 0``, Gamma(n, scale t1) is a negative
    binomial mixture of Gamma(n + k, scale t2), ``k ~ NB(n, r = t2 / t1)``, so
    ``S ~ Gamma(2n + k, t2)`` and ``P(S >= s) = sum_k P(k) Q(2n + k, s /
    t2)``. The sum runs to 40 standard deviations past the mixture's mean
    (the terms are below the NB tail there); ``r`` below 1e-5 (a share within
    1e-5 of 0 or 1) uses Gamma(n, t1)."""
    n = int(n)
    s = float(s)
    a = float(a)
    if not 0.0 <= a <= 1.0:
        raise ValueError(f"the key-A share must lie in [0, 1], got {a}")
    if n <= 0 or not s > 0.0:
        return 1.0
    t1, t2 = max(a, 1.0 - a), min(a, 1.0 - a)
    if t2 <= 0.0 or t2 / t1 < 1e-5:
        return float(_gammaincc(float(n), s / t1))
    if t1 - t2 < 1e-12:
        return float(_gammaincc(2.0 * n, s / t2))
    r = t2 / t1
    mean = n * (1.0 - r) / r
    sd = math.sqrt(n * (1.0 - r)) / r
    k = torch.arange(int(mean + 40.0 * sd + 100.0), dtype=torch.float64)
    log_pmf = (
        torch.lgamma(n + k)
        - torch.lgamma(k + 1.0)
        - math.lgamma(n)
        + n * math.log(r)
        + k * math.log1p(-r)
    )
    total = (torch.exp(log_pmf) * _gammaincc(2.0 * n + k, s / t2)).sum()
    return float(total.clamp(0.0, 1.0))


@dataclass(frozen=True)
class FusedWatermarkDetection(WatermarkDetection):
    """``WatermarkDetection`` of the fused two-key test, with key A's own
    test and Meta's moment-matched p-value."""

    p_value_key_a: float = 1.0
    p_value_moment_matched: float = 1.0
    score_key_a: float = 0.0
    score_key_b: float = 0.0
    key_share_a: float = 1.0


class FusedTwoKeyGumbelWatermarkDetector(GumbelWatermarkDetector):
    """Detection of ``block_verified_gumbel`` with ``residual_race=False``
    (Meta's detector of its speculative Gumbel baselines): each scored
    (context, token) pair scores ``a s_A + (1 - a) s_B``, ``s_K = -log(1 -
    U_K)`` its Gumbel-max score under key K (A the watermark key, B the key
    ``derive_watermark_key(key, KEY_B_DOMAIN)``), ``a`` = ``key_share_a``, the
    run's share of key-A tokens among the keyed emitted tokens (the verifier's
    ``key_share_stats``, rounded to two decimals). The p-value of the sum is
    exact (``fused_gamma_sf``); ``p_value_key_a`` is ``GumbelWatermarkDetector``'s
    test with key A alone and ``p_value_moment_matched`` Meta's Gamma(n / v,
    scale v), ``v = a^2 + (1 - a)^2``. Any ``a`` fixed before the text is seen
    gives an exact test."""

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        key_share_a: float = 0.5,
        p_value_threshold: float = 0.01,
        prf: WatermarkPRFName = "philox",
        deduplicate_contexts: bool = True,
    ) -> None:
        if not 0.0 <= key_share_a <= 1.0:
            raise ValueError("key_share_a must lie in [0, 1]")
        super().__init__(key, context_width, p_value_threshold, prf, deduplicate_contexts)
        self.key_b_prf = create_prf(prf, derive_watermark_key(key, KEY_B_DOMAIN))
        self.key_share_a = float(key_share_a)

    def detect(self, token_ids: list[int]) -> FusedWatermarkDetection:
        contexts, targets = self._prepare_inputs(token_ids)
        n = len(targets)
        a = self.key_share_a
        if n == 0:
            return FusedWatermarkDetection(0.0, 1.0, 0, False, key_share_a=a)
        s_a = float(self._score_tokens(contexts, targets).sum())
        u_b = self.key_b_prf.uniform(contexts, targets.unsqueeze(-1)).squeeze(-1)
        s_b = float((-torch.log1p(-u_b.to(torch.float64))).sum())
        score = a * s_a + (1.0 - a) * s_b
        p_value = fused_gamma_sf(n, score, a)
        variance = a * a + (1.0 - a) ** 2
        return FusedWatermarkDetection(
            score=score,
            p_value=p_value,
            num_scored_tokens=n,
            is_watermarked=p_value <= self.p_value_threshold,
            p_value_key_a=_gamma_survival_integer_shape(s_a, n),
            p_value_moment_matched=float(_gammaincc(n / variance, score / variance)),
            score_key_a=s_a,
            score_key_b=s_b,
            key_share_a=a,
        )


def measured_key_share(n_a: int, n_b: int) -> float:
    """The detector's weight from the verifier's counts: the share of key-A
    tokens among the keyed emitted tokens, rounded to two decimals (Meta)."""
    if n_a < 0 or n_b < 0 or n_a + n_b == 0:
        raise ValueError(f"no keyed emitted tokens counted (key A {n_a}, B {n_b})")
    return round(n_a / (n_a + n_b), 2)


__all__ = [
    "BlockVerifiedGumbelDraftWatermarker",
    "BlockVerifiedGumbelVerificationWatermarker",
    "BlockVerifiedGumbelWatermarker",
    "DRAFT_LAW_WIDTH",
    "FusedTwoKeyGumbelWatermarkDetector",
    "FusedWatermarkDetection",
    "KEY_B_DOMAIN",
    "KEY_LOG_ENV_VAR",
    "KeyShareCounter",
    "FreshNoiseDraftWatermarker",
    "SampledDraftWatermarker",
    "TARGET_SLOTS",
    "VERIFICATION_LAWS",
    "draft_law_logits",
    "draft_noise_contexts",
    "fresh_contexts",
    "fused_gamma_sf",
    "measured_key_share",
    "outcome_law_fast",
    "plain_block_decision",
    "residual_race_fast",
]
