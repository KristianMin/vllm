# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Two-key block verification (``two_key_block_verification``): the
speculative baselines of the TextSeal2 reference implementation
(``textseal.watermarking.speculative.BaselineRound``,
https://github.com/facebookresearch/textseal) for ``dual_key_gumbel`` (the
reference's ``textseal``), ``gaussian_copula`` (``automark``), ``textgrain``
and ``synthid``.

Draft. Each draft is the scheme's watermarked sample of the draft law ``q``
(the drafter's distribution after temperature and then the request's min-p,
top-k and top-p, as the reference truncates it) under key A; for
``dual_key_gumbel`` under a key chosen by a coin, A with probability
``1 - alpha`` (the reference's ``mixing_alpha`` is ``1 - alpha``).

Verify. Block verification (or token-level verification) on the raw laws
given the drafts: the stopping row ``I ~ P(I = i)`` from fresh randomness, the
drafts below it kept, and the token emitted at it is the scheme's watermarked
sample of the row's residual law, under key B at a draft row
(``dual_key_gumbel``: the key the draft did not use) and at the bonus row of
the target's law under the drafts' key choice (key A; ``dual_key_gumbel``: a
coin of its own).

The schemes' watermarked samples (``_Scheme``): Gumbel-max sampling on the key's
lanes (``dual_key_gumbel``), the Gaussian copula race on the key's uniforms and
fresh normal noise (``gaussian_copula``), and the watermarked law of the key
(``textgrain``, ``synthid``: ``watermark_logits``) sampled on fresh noise. All
fresh randomness (the coin, the stopping row, the copula noise, the sampling
noise, the samples of repeated rows) is a public PRF of the request seed, the
row's position and its draft step, which no two rounds share.

Repeated contexts (``deduplicate_contexts``) and rows whose context reaches
before the output are sampled without the watermark, for every key (the
reference tracks each (context, key) pair: a context it used under key A may
still take key B; here a repeated context takes neither). The reference marks
every keyed row a round read, which is ``deduplicate_contexts="strict"``.

Law. As the reference's baselines, the output follows the target on average
over keys: each draft is the scheme's sample of ``q``, which is ``q`` on
average over key A, independently across contexts, and the emitted token's
key is independent of the drafts' (or fresh), so the round has block
verification's law on average. For a fixed key it does not (the drafts are
not samples of ``q``), as for the schemes without speculation (``textgrain``
and ``synthid`` are not distortion-free for a fixed key either).

Detection. Tokens are keyed under A or B; ``FusedTwoKeyWatermarkDetector``
scores each pair ``a s_A + (1 - a) s_B`` with ``a`` the run's share of key-A
tokens among the keyed emitted tokens, which the verifier counts
(``key_share_stats``; ``VLLM_TWO_KEY_BLOCK_KEY_LOG``), as the reference's
two-key tests do.
"""

import math
from dataclasses import dataclass

import torch

from vllm.config.watermarking import (
    TWO_KEY_BLOCK_ALGORITHMS,
    WatermarkPRFName,
    derive_watermark_key,
)
from vllm.v1.watermarking.block_verified_gumbel import (
    DRAFT_LAW_WIDTH,
    KeyShareCounter,
    _gumbel_max,
    draft_law_logits,
    draft_noise_contexts,
    fresh_contexts,
    fused_gamma_sf,
)
from vllm.v1.watermarking.detector import WatermarkDetection, WatermarkDetector
from vllm.v1.watermarking.gaussian_copula import gumbels
from vllm.v1.watermarking.gumbel import GumbelWatermarkDetector
from vllm.v1.watermarking.prfs import PhiloxPRF, create_prf
from vllm.v1.watermarking.synthid import SynthIDWatermarkDetector, SynthIDWatermarker
from vllm.v1.watermarking.textgrain import (
    TextGrainWatermarkDetector,
    TextGrainWatermarker,
)
from vllm.v1.watermarking.textseal2_verification import (
    LAWS,
    draw_stop,
    outcome_law,
)
from vllm.v1.watermarking.watermarker import (
    RandomSampler,
    SupportsSpeculativeDecoding,
    Watermarker,
    WatermarkSample,
)

# Key B of the schemes keyed by the master key itself (key A).
KEY_B_DOMAIN = b"two_key_block_key_b"
KEY_LOG_ENV_VAR = "VLLM_TWO_KEY_BLOCK_KEY_LOG"
# Public Philox keys of the fresh randomness, on (seed, position, step):
# the coin of dual_key_gumbel's key, the stopping row, the drafter's and the
# verifier's sampling noise, and the copula's normal noise of each.
_COIN_PRF = PhiloxPRF(0x324B4231_434F494E)
_STOP_PRF = PhiloxPRF(0x324B4231_53544F50)
_DRAFT_NOISE_PRF = PhiloxPRF(0x324B4231_44524654)
_VERIFY_NOISE_PRF = PhiloxPRF(0x324B4231_56455249)
_DRAFT_COPULA_PRF = PhiloxPRF(0x324B4231_44434F50)
_VERIFY_COPULA_PRF = PhiloxPRF(0x324B4231_56434F50)


def two_key_block_keys(algorithm: str, key: int) -> tuple[int, int]:
    """Keys A and B of ``algorithm`` from the master ``key``: dual-key
    Gumbel-max's own two keys, else the key itself and a key derived from
    it (``KEY_B_DOMAIN``)."""
    if algorithm == "dual_key_gumbel":
        return derive_watermark_key(key, b"key_a"), derive_watermark_key(key, b"key_b")
    return key, derive_watermark_key(key, KEY_B_DOMAIN)


class _Scheme:
    """A scheme's watermarked sample of a law under key A or key B."""

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        key_b: torch.Tensor,
        noise_contexts: torch.Tensor,
        verifying: bool,
    ) -> torch.Tensor:
        """Tokens [rows]: row ``r`` samples ``softmax(logits[r])`` with the
        watermark under key B if ``key_b[r]`` else key A, its fresh
        randomness from ``noise_contexts[r]`` (the drafter's or, if
        ``verifying``, the verifier's PRFs)."""
        raise NotImplementedError


class _GumbelScheme(_Scheme):
    def __init__(self, keys: tuple[int, int], prf: WatermarkPRFName) -> None:
        self.prfs = [create_prf(prf, key) for key in keys]

    def sample(self, logits, contexts, key_b, noise_contexts, verifying):
        a = _gumbel_max(self.prfs[0], logits, contexts)
        b = _gumbel_max(self.prfs[1], logits, contexts)
        return torch.where(key_b, b, a)


class _CopulaScheme(_Scheme):
    def __init__(
        self, keys: tuple[int, int], prf: WatermarkPRFName, correlation: float
    ) -> None:
        self.prfs = [create_prf(prf, key) for key in keys]
        self.correlation = correlation

    def sample(self, logits, contexts, key_b, noise_contexts, verifying):
        vocabulary = torch.arange(logits.shape[-1], device=logits.device)
        u = torch.where(
            key_b.unsqueeze(-1),
            self.prfs[1].uniform(contexts, vocabulary),
            self.prfs[0].uniform(contexts, vocabulary),
        ).double()
        noise_prf = _VERIFY_COPULA_PRF if verifying else _DRAFT_COPULA_PRF
        noise = torch.special.ndtri(noise_prf.uniform(noise_contexts, vocabulary).double())
        noised = logits.double() + gumbels(u, noise, self.correlation)
        return torch.argmax(noised, dim=-1)


class _LawScheme(_Scheme):
    """Schemes whose watermark is a keyed law (``watermark_logits``)."""

    def __init__(self, watermarkers: tuple[Watermarker, Watermarker]) -> None:
        self.watermarkers = watermarkers

    def sample(self, logits, contexts, key_b, noise_contexts, verifying):
        law = torch.where(
            key_b.unsqueeze(-1),
            self.watermarkers[1].watermark_logits(logits, contexts),
            self.watermarkers[0].watermark_logits(logits, contexts),
        )
        noise_prf = _VERIFY_NOISE_PRF if verifying else _DRAFT_NOISE_PRF
        return _gumbel_max(noise_prf, law, noise_contexts)


def make_scheme(
    algorithm: str,
    key: int,
    context_width: int,
    prf: WatermarkPRFName = "philox",
    correlation: float = 0.98551,
    textgrain_args: dict | None = None,
    synthid_args: dict | None = None,
) -> _Scheme:
    keys = two_key_block_keys(algorithm, key)
    if algorithm == "dual_key_gumbel":
        return _GumbelScheme(keys, prf)
    if algorithm == "gaussian_copula":
        return _CopulaScheme(keys, prf, correlation)
    if algorithm == "textgrain":
        args = textgrain_args or {}
        return _LawScheme(
            tuple(TextGrainWatermarker(k, context_width, prf, **args) for k in keys)
        )
    if algorithm == "synthid":
        args = synthid_args or {}
        return _LawScheme(
            tuple(SynthIDWatermarker(k, context_width, prf, **args) for k in keys)
        )
    raise ValueError(f"two-key block verification supports {TWO_KEY_BLOCK_ALGORITHMS}")


@dataclass(frozen=True)
class _Protocol:
    """The reference's choice of keys: the drafts' (and the bonus row's) key
    and the residual row's."""

    coin: bool  # dual_key_gumbel: key A with probability 1 - alpha
    alpha: float = 0.0

    def draft_key_b(self, noise_contexts: torch.Tensor) -> torch.Tensor:
        """Whether each row's draft (or bonus token) takes key B."""
        if not self.coin:
            return torch.zeros(
                noise_contexts.shape[0], dtype=torch.bool, device=noise_contexts.device
            )
        zero = torch.zeros(1, dtype=torch.long, device=noise_contexts.device)
        u = _COIN_PRF.uniform(noise_contexts, zero).squeeze(-1).double()
        return u >= 1.0 - self.alpha

    def residual_key_b(self, draft_key_b: torch.Tensor) -> torch.Tensor:
        """Whether the token after a rejected draft takes key B: the key the
        draft did not use (coin), else key B."""
        return ~draft_key_b if self.coin else torch.ones_like(draft_key_b)


def _full_context(contexts: torch.Tensor) -> torch.Tensor:
    return (contexts >= 0).all(dim=-1)


class TwoKeyBlockDraftWatermarker(Watermarker):
    """The drafter's side: each watermarked request's draft is the scheme's
    watermarked sample of the draft law under the protocol's key; repeated
    rows and rows without a full context sample the draft law on fresh noise.
    The other requests draft with vLLM's noise on the temperature-only law."""

    samples_draft_rows = True

    def __init__(
        self,
        scheme: _Scheme,
        protocol: _Protocol,
        context_width: int = 4,
        draft_law_width: int = DRAFT_LAW_WIDTH,
    ) -> None:
        if draft_law_width < 1:
            raise ValueError("draft_law_width must be positive")
        self.scheme = scheme
        self.protocol = protocol
        self._context_width = context_width
        self.draft_law_width = draft_law_width

    @property
    def context_width(self) -> int:
        return self._context_width

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
        # vLLM's sampler for the opted-out requests (it also fills the cache).
        plain = random_sampler(logits)
        noise_contexts = draft_noise_contexts(random_sampler)
        keyed = request_enabled & ~skip_mask & _full_context(contexts)
        key_b = self.protocol.draft_key_b(noise_contexts)
        watermarked = self.scheme.sample(logits, contexts, key_b, noise_contexts, False)
        fresh = _gumbel_max(_DRAFT_NOISE_PRF, logits, noise_contexts)
        tokens = torch.where(keyed, watermarked, fresh)
        return torch.where(request_enabled, tokens, plain)

    def _sample_watermarked(
        self, logits: torch.Tensor, contexts: torch.Tensor
    ) -> WatermarkSample:
        raise NotImplementedError(
            "two-key block drafts need the request seeds and positions: "
            "DraftWatermarker calls sample_draft_rows"
        )


class TwoKeyBlockVerificationWatermarker(Watermarker, KeyShareCounter):
    """The verifier's side: block (``law="block"``) or token-level
    (``"token"``) verification of the drafts on the raw laws, the stopping row
    from fresh randomness and the token emitted there the scheme's watermarked
    sample under the protocol's key (module docstring). Every watermarked
    request's round is decided so, ending at its first invalid draft; the
    other requests keep their logits and vLLM's rejection sampling. Without
    speculation it samples like ``base`` (the algorithm's own watermarker)."""

    def __init__(
        self,
        base: Watermarker,
        scheme: _Scheme,
        protocol: _Protocol,
        law: str = "block",
        draft_law_width: int = DRAFT_LAW_WIDTH,
    ) -> None:
        if law not in LAWS:
            raise ValueError(f"law must be one of {LAWS}")
        if draft_law_width < 1:
            raise ValueError("draft_law_width must be positive")
        self.base = base
        self.scheme = scheme
        self.protocol = protocol
        self.law = law
        self.draft_law_width = draft_law_width
        self._init_key_counts(KEY_LOG_ENV_VAR)

    @property
    def context_width(self) -> int:
        return self.base.context_width

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
    ) -> WatermarkSample:
        return self.base.sample(logits, contexts, random_sampler, skip_mask)

    def _sample_watermarked(
        self, logits: torch.Tensor, contexts: torch.Tensor
    ) -> WatermarkSample:
        return self.base._sample_watermarked(logits, contexts)

    def draft_law_logits(
        self,
        logits: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        min_p: torch.Tensor,
    ) -> torch.Tensor:
        return draft_law_logits(logits, top_k, top_p, min_p, self.draft_law_width)

    def verification_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        draft_tokens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        skip_mask: torch.Tensor,
    ) -> torch.Tensor:
        raise ValueError(
            "two-key block verification needs probabilistic drafting "
            "(draft_sample_method='probabilistic')"
        )

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
        num_rows, vocab_size = logits.shape
        if num_rows == 0:
            return logits
        device = logits.device
        bounds = cu_num_logits.long()
        sizes = bounds.diff()
        num_requests = sizes.numel()
        depth = int(sizes.max())
        request = torch.repeat_interleave(torch.arange(num_requests, device=device), sizes)
        local = torch.arange(num_rows, device=device) - bounds[:-1][request]
        requests = random_sampler.expanded_idx_mapping.long().clamp_min(0)
        noise_contexts = fresh_contexts(
            random_sampler.seeds[requests], random_sampler.positions, local
        )
        draft_tokens = draft_tokens.long()
        is_bonus = local == (sizes - 1)[request]
        valid = is_bonus | ((draft_tokens >= 0) & (draft_tokens < vocab_size))
        # Drafts per request: up to the first invalid one (its bonus row).
        invalid_at = torch.where(valid, torch.iinfo(torch.long).max, local)
        K = torch.minimum(
            sizes - 1,
            torch.full((num_requests,), torch.iinfo(torch.long).max, device=device)
            .scatter_reduce(0, request, invalid_at, reduce="amin"),
        )
        in_round = local <= K[request]
        is_draft = local < K[request]
        drafts = torch.where(is_draft, draft_tokens, 0).clamp(0, vocab_size - 1)

        P = torch.softmax(logits.double().nan_to_num(nan=float("-inf")), dim=-1)
        Q = torch.softmax(draft_law.double().nan_to_num(nan=float("-inf")), dim=-1)
        Q = Q * is_draft.unsqueeze(1)
        slots = torch.full((num_requests, depth), -1, dtype=torch.long, device=device)
        slots[request[in_round], local[in_round]] = torch.arange(num_rows, device=device)[
            in_round
        ]
        pad = (slots < 0).unsqueeze(2)
        index = slots.clamp_min(0)
        pi, stop = outcome_law(
            P[index].masked_fill(pad, 0.0),
            Q[index].masked_fill(pad, 0.0),
            drafts[index],
            K,
            self.law,
        )
        zero = torch.zeros(1, dtype=torch.long, device=device)
        u = _STOP_PRF.uniform(noise_contexts[index[:, 0]], zero).squeeze(-1)
        stopping = draw_stop(stop, u.double())
        req = torch.arange(num_requests, device=device)
        stop_row = index[req, stopping]  # [requests] row of the stopping depth

        # The emitted token's key and law.
        keyed = request_enabled & ~repeated & _full_context(contexts)
        draft_key_b = self.protocol.draft_key_b(noise_contexts)
        at_bonus = stopping == K
        key_b = torch.where(
            at_bonus,
            draft_key_b[stop_row],
            self.protocol.residual_key_b(draft_key_b[stop_row]),
        )
        law = pi[req, stopping]
        law_logits = torch.where(
            law > 0, torch.log(torch.where(law > 0, law, 1.0)), float("-inf")
        )
        stop_contexts = contexts[stop_row]
        stop_noise = noise_contexts[stop_row]
        watermarked = self.scheme.sample(
            law_logits.float(), stop_contexts, key_b, stop_noise, True
        )
        fresh = _gumbel_max(_VERIFY_NOISE_PRF, law_logits.float(), stop_noise)
        stop_keyed = keyed[stop_row]
        emitted = torch.where(stop_keyed, watermarked, fresh)

        tokens = torch.where(is_draft, drafts, 0)
        tokens = torch.where(local == stopping[request], emitted[request], tokens)
        decided = torch.zeros(num_requests, dtype=torch.long, device=device)
        decided.index_add_(0, request, request_enabled.long())
        decided_rows = (decided > 0)[request]
        point_masses = torch.full_like(logits, float("-inf")).scatter_(
            1, tokens.unsqueeze(1), 0.0
        )
        # The keyed emitted tokens per key: kept drafts and the stopping token.
        kept = decided_rows & keyed & (local < stopping[request])
        last = decided[req] > 0
        n_b = (kept & draft_key_b).sum() + (last & stop_keyed & key_b).sum()
        n_a = (kept & ~draft_key_b).sum() + (last & stop_keyed & ~key_b).sum()
        self._count_keys(n_a, n_b)
        return torch.where(decided_rows.unsqueeze(1), point_masses, logits)


class TwoKeyBlockWatermarker(Watermarker, SupportsSpeculativeDecoding):
    """An algorithm with the reference's speculative baseline: without
    speculation ``base`` (the algorithm's watermarker), under speculative
    decoding ``TwoKeyBlockDraftWatermarker`` and
    ``TwoKeyBlockVerificationWatermarker``."""

    def __init__(
        self,
        base: Watermarker,
        algorithm: str,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRFName = "philox",
        law: str = "block",
        alpha: float = 0.5,
        correlation: float = 0.98551,
        textgrain_args: dict | None = None,
        synthid_args: dict | None = None,
        draft_law_width: int = DRAFT_LAW_WIDTH,
    ) -> None:
        if algorithm not in TWO_KEY_BLOCK_ALGORITHMS:
            raise ValueError(
                f"two-key block verification supports {TWO_KEY_BLOCK_ALGORITHMS}"
            )
        self.base = base
        scheme = make_scheme(
            algorithm, key, context_width, prf, correlation, textgrain_args, synthid_args
        )
        protocol = _Protocol(coin=algorithm == "dual_key_gumbel", alpha=alpha)
        self.draft_watermarker = TwoKeyBlockDraftWatermarker(
            scheme, protocol, context_width, draft_law_width
        )
        self.target_watermarker = TwoKeyBlockVerificationWatermarker(
            base, scheme, protocol, law, draft_law_width
        )

    @property
    def context_width(self) -> int:
        return self.base.context_width

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
    ) -> WatermarkSample:
        return self.base.sample(logits, contexts, random_sampler, skip_mask)

    def _sample_watermarked(
        self, logits: torch.Tensor, contexts: torch.Tensor
    ) -> WatermarkSample:
        return self.base._sample_watermarked(logits, contexts)


# ---- detection --------------------------------------------------------------


def _binomial_mixture_sf(n_trials: int, score: float, a: float) -> float:
    """``P(a S_A + (1 - a) S_B >= score)`` for independent ``S_A, S_B ~
    Binomial(n_trials, 1/2)`` (exact, by summing over ``S_A``)."""
    if n_trials <= 0:
        return 1.0
    k = torch.arange(n_trials + 1, dtype=torch.float64)
    log_pmf = (
        math.lgamma(n_trials + 1.0)
        - torch.lgamma(k + 1)
        - torch.lgamma(n_trials - k + 1)
        - n_trials * math.log(2.0)
    )
    pmf = log_pmf.exp()
    cdf = torch.cumsum(pmf, 0)
    if a >= 1.0:
        return float(pmf[k >= score - 1e-9].sum().clamp(0, 1))
    if a <= 0.0:
        return float(pmf[k >= score - 1e-9].sum().clamp(0, 1))
    # S_B >= (score - a k) / (1 - a)
    need = torch.ceil((score - a * k) / (1.0 - a) - 1e-9).clamp(0, n_trials + 1).long()
    tail = torch.where(need > 0, 1.0 - cdf[(need - 1).clamp(0, n_trials)], 1.0)
    tail = torch.where(need > n_trials, 0.0, tail)
    return float((pmf * tail).sum().clamp(0.0, 1.0))


@dataclass(frozen=True)
class TwoKeyDetection(WatermarkDetection):
    """``WatermarkDetection`` of the fused two-key test, with key A's own
    test and the scores of each key."""

    p_value_key_a: float = 1.0
    score_key_a: float = 0.0
    score_key_b: float = 0.0
    key_share_a: float = 1.0


class FusedTwoKeyWatermarkDetector:
    """The fused two-key test of the two-key block baselines: every scored
    (context, token) pair (deduplicated as the scheme's detector does) scores
    ``a s_A + (1 - a) s_B`` with ``s_K`` the scheme's token score under key K
    and ``a`` = ``key_share_a`` (the verifier's ``key_share_stats``,
    ``block_verified_gumbel.measured_key_share``). Exp(1) scores
    (``dual_key_gumbel``, ``gaussian_copula``: the Gumbel score of the key's
    uniform; ``textgrain``: the cell score) take the exact p-value of
    ``fused_gamma_sf``; SynthID's g-value counts the exact binomial mixture.
    ``p_value_key_a`` is the scheme's test with key A alone."""

    def __init__(
        self,
        algorithm: str,
        key: int,
        context_width: int = 4,
        key_share_a: float = 0.5,
        p_value_threshold: float = 0.01,
        prf: WatermarkPRFName = "philox",
        num_blocks: int = 16,
        num_columns: int = 16,
        depth: int = 30,
    ) -> None:
        if not 0.0 <= key_share_a <= 1.0:
            raise ValueError("key_share_a must lie in [0, 1]")
        keys = two_key_block_keys(algorithm, key)
        self.algorithm = algorithm
        self.key_share_a = float(key_share_a)
        self.p_value_threshold = p_value_threshold
        self.binomial = algorithm == "synthid"
        self.depth = depth
        if algorithm in ("dual_key_gumbel", "gaussian_copula"):
            self.detectors = [
                GumbelWatermarkDetector(k, context_width, p_value_threshold, prf)
                for k in keys
            ]
        elif algorithm == "textgrain":
            self.detectors = [
                TextGrainWatermarkDetector(
                    k, context_width, p_value_threshold, prf,
                    num_blocks=num_blocks, num_columns=num_columns,
                )
                for k in keys
            ]
        elif algorithm == "synthid":
            self.detectors = [
                SynthIDWatermarkDetector(k, context_width, p_value_threshold, prf, depth=depth)
                for k in keys
            ]
        else:
            raise ValueError(f"two-key block verification supports {TWO_KEY_BLOCK_ALGORITHMS}")

    def detect(self, token_ids: list[int]) -> TwoKeyDetection:
        first: WatermarkDetector = self.detectors[0]
        contexts, targets = first._prepare_inputs(token_ids)
        n = len(targets)
        a = self.key_share_a
        if n == 0:
            return TwoKeyDetection(0.0, 1.0, 0, False, key_share_a=a)
        s_a = float(first._score_tokens(contexts, targets).sum())
        s_b = float(self.detectors[1]._score_tokens(contexts, targets).sum())
        score = a * s_a + (1.0 - a) * s_b
        if self.binomial:
            p_value = _binomial_mixture_sf(n * self.depth, score, a)
        else:
            p_value = fused_gamma_sf(n, score, a)
        return TwoKeyDetection(
            score=score,
            p_value=p_value,
            num_scored_tokens=n,
            is_watermarked=p_value <= self.p_value_threshold,
            p_value_key_a=first._get_p_value(s_a, n),
            score_key_a=s_a,
            score_key_b=s_b,
            key_share_a=a,
        )


__all__ = [
    "FusedTwoKeyWatermarkDetector",
    "KEY_B_DOMAIN",
    "KEY_LOG_ENV_VAR",
    "TWO_KEY_BLOCK_ALGORITHMS",
    "TwoKeyBlockDraftWatermarker",
    "TwoKeyBlockVerificationWatermarker",
    "TwoKeyBlockWatermarker",
    "TwoKeyDetection",
    "make_scheme",
    "two_key_block_keys",
]
