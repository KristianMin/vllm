# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""TextSeal2 watermark generation and detection, from "TextSeal2:
Distortion-Free LLM Watermarking with Copula Mixing and Memoryless Races"
(arXiv:2610.40613) and its reference implementation (watermark_type
``textseal2``, https://github.com/facebookresearch/textseal).

TextSeal2 is the Gaussian copula watermark (``gaussian_copula``) run on a
virtual key that mixes two keys. Every candidate token ``v`` gets the keyed
uniforms ``U_A(v)`` and ``U_B(v)`` of keys A and B, merged into the standard
normal ``G*_v = (ndtri(U_A(v)) + ndtri(U_B(v))) / sqrt(2)``, which is coupled
to fresh standard normal noise ``N_v``:

    z_v = correlation * G*_v + sqrt(1 - correlation**2) * N_v
    token = argmax_v log p_v - log(-log Phi(z_v))

Every ``z_v`` is standard normal, so the token follows ``p``, the model's
distribution after temperature scaling and top-k/top-p filtering. Detection
scores each distinct (context, token) pair with ``-log Phi(-G*)``, which needs
both keys (private), or with ``-log(1 - U_A)``, which needs key A alone
(public). Both scores are Exp(1) without the watermark, and the public one
carries ``1 / sqrt(2)`` of the private signal.

Speculative decoding follows the paper's rule (Sec. 3.2): the drafter races
each draft on its own distribution ``q`` with the same kind of clocks,
``d = argmin_v E_v / q_v`` with ``E_v = -log Phi(z_v)``, and verification runs
one more race on the leftover clocks ``E - tau q`` of all rows of the round,
at the speeds of block verification's outcome law
(``textseal2_verification``). It picks the row where the drafted path stops
and the token emitted there; the rows become point masses, which rejection
sampling emits unchanged. The output follows the target, the acceptance is
block verification's, and every emitted token is keyed. The noise ``N`` must
be the same in a draft's race and in the residual race, so both draw it from a
public PRF of the request seed, the row's position and its draft step, which
together identify the round. With unwatermarked drafts the residual race runs
on them as point masses instead (the reference's unkeyed-draft mode).
"""

import dataclasses
import math
from collections.abc import Sequence

import numpy as np
import torch

from vllm.config.watermarking import WatermarkPRFName, derive_watermark_key
from vllm.v1.watermarking.detector import (
    Context,
    WatermarkDetection,
    WatermarkDetector,
)
from vllm.v1.watermarking.gumbel import (
    _entropy_array,
    _gamma_survival_integer_shape,
    _validate_context_width,
)
from vllm.v1.watermarking.localized import (
    LocalizedWatermarkDetection,
    adaptive_ensemble_detection,
    entropy_weights,
    weighted_gamma_p,
)
from vllm.v1.watermarking.prfs import PhiloxPRF, WatermarkPRF, create_prf
from vllm.v1.watermarking.textseal2_verification import (
    LAWS,
    PASS_AGGREGATORS,
    PASS_SOURCES,
    race,
    residual_race,
)
from vllm.v1.watermarking.watermarker import (
    RandomSampler,
    SupportsSpeculativeDecoding,
    Watermarker,
    WatermarkSample,
)

# The paper's main correlation, the reference's default.
DEFAULT_CORRELATION = 0.9
# Keys A (public) and B come from the master key as for dual-key Gumbel-max.
KEY_DOMAINS = (b"key_a", b"key_b")
# Seeds the fresh noise off CUDA. Public: the noise needs no secrecy.
_FRESH_NOISE_SEED = 0x54535332_4E4F4953
_VERIFICATION_NOISE_SEED = 0x54535332_56455249
# Philox key of the noise a draft race shares with its verification. Public:
# the noise needs no secrecy, only independence from keys A and B.
_RESIDUAL_NOISE_PRF = PhiloxPRF(0x54535332_52455344)
# Elements of each float64 [rows, vocab] tensor of the speculative rule.
_CHUNK_ELEMENTS = 1 << 24


def textseal2_keys(key: int) -> list[int]:
    """Keys A and B of a master key; key A is the public key."""
    return [derive_watermark_key(key, domain) for domain in KEY_DOMAINS]


def virtual_normals(uniforms: Sequence[torch.Tensor]) -> torch.Tensor:
    """The virtual key ``M^(-1/2) * sum_j ndtri(U_j)`` of the keyed uniforms of
    ``M`` keys, in float64. It is standard normal and independent across
    (context, token) pairs, and needs every key."""
    total = None
    for u in uniforms:
        normal = torch.special.ndtri(u.double())
        total = normal if total is None else total + normal
    assert total is not None
    return total / math.sqrt(len(uniforms))


def race_gumbels(
    virtual: torch.Tensor, noise: torch.Tensor, correlation: float
) -> torch.Tensor:
    """Gumbel noise ``-log(-log Phi(z))`` for ``z = correlation * virtual +
    sqrt(1 - correlation**2) * noise``, given the virtual key and fresh
    standard normal ``noise``. ``-log Phi(z)`` is the token's Exp(1) clock."""
    z = virtual
    if correlation < 1:
        z = correlation * virtual + math.sqrt(1.0 - correlation * correlation) * noise
    return -torch.log(-torch.special.log_ndtr(z))


def race_clocks(
    virtual: torch.Tensor,
    noise: torch.Tensor,
    correlation: float,
    keyed: torch.Tensor,
) -> torch.Tensor:
    """The Exp(1) clocks ``-log Phi(z)`` of the speculative rule's races, with
    ``z = correlation * virtual + sqrt(1 - correlation**2) * noise`` on
    ``keyed`` rows and ``z = noise`` on the others, in float64."""
    z = virtual
    if correlation < 1:
        z = correlation * virtual + math.sqrt(1.0 - correlation * correlation) * noise
    return -torch.special.log_ndtr(torch.where(keyed.unsqueeze(-1), z, noise))


def residual_noise(
    seeds: torch.Tensor,
    positions: torch.Tensor,
    steps: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """The noise ``N`` of the speculative rule's races: [rows, vocab_size]
    standard normal float64, a public PRF of each row's request seed, position
    and draft step. The drafter samples the draft for position ``L + j`` of a
    round at position ``L - 1 + j`` and step ``j``, and verification row ``j``
    has the same position and step, so the two read the same noise, while
    another round, whose committed length differs, reads other noise."""
    seeds = seeds.long()
    contexts = torch.stack(
        [
            seeds & 0xFFFFFFFF,
            (seeds >> 32) & 0xFFFFFFFF,
            positions.long().expand_as(seeds),
            steps.long().expand_as(seeds),
        ],
        dim=-1,
    )
    vocabulary = torch.arange(vocab_size, device=contexts.device)
    return torch.special.ndtri(
        _RESIDUAL_NOISE_PRF.uniform(contexts, vocabulary).double()
    )


def _request_chunks(sizes: list[int], vocab_size: int) -> list[tuple[int, int]]:
    """Ranges (first, end) of whole requests holding at most
    ``_CHUNK_ELEMENTS`` logits each, or one request."""
    chunks, first, rows = [], 0, 0
    for index, size in enumerate(sizes):
        if rows and (rows + size) * vocab_size > _CHUNK_ELEMENTS:
            chunks.append((first, index))
            first, rows = index, 0
        rows += size
    if rows:
        chunks.append((first, len(sizes)))
    return chunks


def _fresh_normals(
    generators: dict[torch.device, torch.Generator],
    shape: torch.Size,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    generator = generators.get(device)
    if generator is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        generators[device] = generator
    return torch.randn(shape, dtype=torch.float64, device=device, generator=generator)


class TextSeal2Watermarker(Watermarker):
    """TextSeal2 watermark: the Gaussian copula watermark on the virtual key of
    keys A and B, see the module docstring.

    The context is the last ``context_width`` tokens; rows whose context
    reaches before the output are sampled without the watermark, as are
    repeated contexts under context deduplication (the paper's rule for
    repeated windows).

    Unlike the reference, the PRF is vLLM's Philox with keys A and B derived
    from the master key, the fresh noise is the request's sampling noise on
    CUDA (elsewhere it comes from a seeded generator owned by the
    watermarker), and top-k, top-p and temperature are vLLM's sampling
    parameters.
    """

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRFName = "philox",
        correlation: float = DEFAULT_CORRELATION,
    ) -> None:
        if not 0 < correlation <= 1:
            raise ValueError("correlation must lie in (0, 1]")
        _validate_context_width(context_width)
        key_a, key_b = textseal2_keys(key)
        self.prf = create_prf(prf, key_a)
        self.key_b_prf = create_prf(prf, key_b)
        self._context_width = context_width
        self.correlation = correlation
        self._generators: dict[torch.device, torch.Generator] = {}

    @property
    def context_width(self) -> int:
        return self._context_width

    def virtual_key(
        self, contexts: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        """``G*`` of every (context, token) pair, as ``WatermarkPRF.uniform``
        broadcasts them, in float64."""
        return virtual_normals(
            [prf.uniform(contexts, token_ids) for prf in (self.prf, self.key_b_prf)]
        )

    def fresh_noise(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        """Standard normal float64 noise for the torch path."""
        return _fresh_normals(self._generators, shape, device, _FRESH_NOISE_SEED)

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
    ) -> WatermarkSample:
        """As ``Watermarker.sample``. With a ``random_sampler``, rows whose
        context reaches before the output are sampled by it."""
        if random_sampler is not None:
            partial = (contexts < 0).any(dim=-1)
            skip_mask = partial if skip_mask is None else skip_mask | partial
        if (
            type(self.prf) is PhiloxPRF
            and type(self.key_b_prf) is PhiloxPRF
            and logits.device.type == "cuda"
            and isinstance(random_sampler, RandomSampler)
            and not random_sampler.is_drafting
            and random_sampler.logits_cache is None
        ):
            from vllm.v1.worker.gpu.sample.textseal2_watermark import (
                philox_textseal2_sample,
            )

            assert skip_mask is not None
            token_ids = philox_textseal2_sample(
                logits,
                contexts,
                self.prf.key,
                self.key_b_prf.key,
                self.correlation,
                skip_mask,
                random_sampler.expanded_idx_mapping,
                random_sampler.temperatures,
                random_sampler.seeds,
                random_sampler.positions,
                random_sampler.use_fp64,
            )
            return WatermarkSample(token_ids, logits)
        return super().sample(logits, contexts, random_sampler, skip_mask)

    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> WatermarkSample:
        vocabulary = torch.arange(logits.shape[-1], device=logits.device)
        virtual = self.virtual_key(contexts, vocabulary)
        noise = self.fresh_noise(virtual.shape, logits.device)
        noised = logits.double() + race_gumbels(virtual, noise, self.correlation)
        return WatermarkSample(torch.argmax(noised, dim=-1), logits)


class TextSeal2DraftWatermarker(TextSeal2Watermarker):
    """The drafter's side of TextSeal2's speculative rule.

    Every draft is the race ``argmin_v E_v / q_v`` on the draft distribution
    ``q`` (temperature only, as vLLM drafts), with the clocks of
    ``race_clocks`` on ``residual_noise``, which verification recomputes. Rows
    without a full context or in ``skip_mask`` race on the noise alone, which
    is plain sampling from ``q``; greedy rows take the draft's argmax. The
    random sampler writes the draft logits verification reads.
    """

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
    ) -> WatermarkSample:
        if not isinstance(random_sampler, RandomSampler) or not (
            random_sampler.is_drafting and random_sampler.logits_cache_col is not None
        ):
            raise ValueError("TextSeal2 drafts need the drafter's random sampler")
        # Writes the logits cache and draws the greedy rows.
        plain = random_sampler(logits)
        num_rows, vocab_size = logits.shape
        requests = random_sampler.expanded_idx_mapping.long().clamp_min(0)
        steps = torch.as_tensor(
            random_sampler.logits_cache_col, device=logits.device
        ).long()
        steps = steps.expand(num_rows) if steps.ndim == 0 else steps
        keyed = (contexts >= 0).all(dim=-1)
        if skip_mask is not None:
            keyed &= ~skip_mask
        greedy = random_sampler.temperatures[requests] == 0
        if (
            type(self.prf) is PhiloxPRF
            and type(self.key_b_prf) is PhiloxPRF
            and logits.device.type == "cuda"
        ):
            # A kernel, so that the drafter's CUDA graphs can capture the race.
            from vllm.v1.worker.gpu.sample.textseal2_watermark import (
                philox_textseal2_draft,
            )

            seeds = random_sampler.seeds[requests].long()
            raced = philox_textseal2_draft(
                logits,
                contexts,
                self.prf.key,
                self.key_b_prf.key,
                self.correlation,
                keyed,
                torch.stack(
                    [
                        seeds & 0xFFFFFFFF,
                        (seeds >> 32) & 0xFFFFFFFF,
                        random_sampler.positions.long(),
                        steps,
                    ],
                    dim=-1,
                ),
                _RESIDUAL_NOISE_PRF.key,
            )
            return WatermarkSample(torch.where(greedy, plain, raced), logits)
        vocabulary = torch.arange(vocab_size, device=logits.device)
        raced = torch.empty(num_rows, dtype=torch.int64, device=logits.device)
        chunk = max(1, _CHUNK_ELEMENTS // max(vocab_size, 1))
        for start in range(0, num_rows, chunk):
            rows = slice(start, start + chunk)
            noise = residual_noise(
                random_sampler.seeds[requests[rows]],
                random_sampler.positions[rows],
                steps[rows],
                vocab_size,
            )
            clocks = race_clocks(
                self.virtual_key(contexts[rows], vocabulary),
                noise,
                self.correlation,
                keyed[rows],
            )
            # argmin E / q = argmax log q - log E
            raced[rows] = torch.argmax(
                logits[rows].double() - torch.log(clocks), dim=-1
            )
        return WatermarkSample(torch.where(greedy, plain, raced), logits)


class TextSeal2VerificationWatermarker(TextSeal2Watermarker):
    """The verifier's side of TextSeal2's speculative rule; without
    speculation it samples like ``TextSeal2Watermarker``.

    ``residual_verification_logits`` runs the residual race on the keyed drafts
    of ``TextSeal2DraftWatermarker``, with the drafts' distributions and the
    shared noise. ``verification_logits`` runs it on unwatermarked drafts as
    point masses, on raw clocks with fresh noise from a generator the
    watermarker owns (the reference's unkeyed-draft mode, exact only for drafts
    that do not depend on the key). ``law`` is the verification law the race
    follows, ``pass_aggregator`` and ``pass_source`` TextSeal2's lookahead on
    it (``textseal2_verification``).

    A request is decided by the race when one of its rows is watermarked and
    every draft is a valid token; its rows without a full context or in
    ``skip_mask`` race on the noise alone, as the drafter raced them. The other
    requests keep their logits. As for the lookahead watermarks,
    ``context_width`` should be at least the number of drafts: the race reads
    the clocks of rows it then discards.
    """

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRFName = "philox",
        correlation: float = DEFAULT_CORRELATION,
        keyed_drafts: bool = True,
        law: str = "block",
        pass_aggregator: str = "none",
        pass_source: str = "own",
    ) -> None:
        super().__init__(key, context_width, prf, correlation)
        if law not in LAWS:
            raise ValueError(f"law must be one of {LAWS}")
        if pass_aggregator not in PASS_AGGREGATORS:
            raise ValueError(f"pass_aggregator must be one of {PASS_AGGREGATORS}")
        if pass_source not in PASS_SOURCES:
            raise ValueError(f"pass_source must be one of {PASS_SOURCES}")
        self.keyed_drafts = keyed_drafts
        # Unwatermarked drafts need no draft watermarker.
        self.verifies_unwatermarked_drafts = not keyed_drafts
        self.law = law
        self.pass_aggregator = pass_aggregator
        self.pass_source = pass_source
        self._verification_generators: dict[torch.device, torch.Generator] = {}

    def verification_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        draft_tokens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        skip_mask: torch.Tensor,
    ) -> torch.Tensor:
        def noise(rows: slice, vocab_size: int) -> torch.Tensor:
            shape = torch.Size([rows.stop - rows.start, vocab_size])
            return _fresh_normals(
                self._verification_generators,
                shape,
                logits.device,
                _VERIFICATION_NOISE_SEED,
            )

        return self._residual_logits(
            logits, contexts, draft_tokens, cu_num_logits, skip_mask, None, noise
        )

    def residual_verification_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        draft_tokens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        skip_mask: torch.Tensor,
        draft_logits: torch.Tensor,
        random_sampler: RandomSampler,
    ) -> torch.Tensor:
        if not self.keyed_drafts:
            return self.verification_logits(
                logits, contexts, draft_tokens, cu_num_logits, skip_mask
            )
        requests = random_sampler.expanded_idx_mapping.long().clamp_min(0)
        bounds = cu_num_logits.long()
        starts = bounds[:-1].repeat_interleave(bounds.diff())
        steps = torch.arange(logits.shape[0], device=logits.device) - starts

        def noise(rows: slice, vocab_size: int) -> torch.Tensor:
            return residual_noise(
                random_sampler.seeds[requests[rows]],
                random_sampler.positions[rows],
                steps[rows],
                vocab_size,
            )

        return self._residual_logits(
            logits,
            contexts,
            draft_tokens,
            cu_num_logits,
            skip_mask,
            draft_logits,
            noise,
        )

    def _residual_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        draft_tokens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        skip_mask: torch.Tensor,
        draft_logits: torch.Tensor | None,
        noise,
    ) -> torch.Tensor:
        num_rows, vocab_size = logits.shape
        if num_rows == 0:
            return logits
        device = logits.device
        bounds = cu_num_logits.long().tolist()
        sizes = [end - start for start, end in zip(bounds, bounds[1:])]
        draft_tokens = draft_tokens.long()
        is_bonus = torch.zeros(num_rows, dtype=torch.bool, device=device)
        is_bonus[cu_num_logits.long()[1:] - 1] = True
        valid_draft = is_bonus | ((draft_tokens >= 0) & (draft_tokens < vocab_size))
        output = logits.clone()
        for first, end in _request_chunks(sizes, vocab_size):
            rows = slice(bounds[first], bounds[end])
            tokens = self._decide(
                logits[rows],
                contexts[rows],
                draft_tokens[rows],
                skip_mask[rows],
                None if draft_logits is None else draft_logits[rows],
                noise(rows, vocab_size),
                sizes[first:end],
            )
            request = torch.repeat_interleave(
                torch.arange(end - first, device=device),
                torch.tensor(sizes[first:end], device=device),
            )
            # Raced: a row is watermarked and every draft is valid.
            counts = torch.zeros(2, end - first, dtype=torch.long, device=device)
            counts[0].index_add_(0, request, (~skip_mask[rows]).long())
            counts[1].index_add_(0, request, (~valid_draft[rows]).long())
            raced = ((counts[0] > 0) & (counts[1] == 0))[request]
            point_masses = torch.full_like(logits[rows], float("-inf")).scatter_(
                1, tokens.unsqueeze(1), 0.0
            )
            output[rows] = torch.where(raced.unsqueeze(1), point_masses, logits[rows])
        return output

    def _decide(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        draft_tokens: torch.Tensor,
        skip_mask: torch.Tensor,
        draft_logits: torch.Tensor | None,
        noise: torch.Tensor,
        sizes: list[int],
    ) -> torch.Tensor:
        """Each row's point mass for whole requests (rows in path order): its
        draft below the stopping row, the emitted token at it, and above it its
        draft, or at the bonus row its own race's winner."""
        num_rows, vocab_size = logits.shape
        device = logits.device
        vocabulary = torch.arange(vocab_size, device=device)
        keyed = ~skip_mask & (contexts >= 0).all(dim=-1)
        E = race_clocks(
            self.virtual_key(contexts, vocabulary), noise, self.correlation, keyed
        )
        P = torch.softmax(logits.double().nan_to_num(nan=float("-inf")), dim=-1)

        sizes_t = torch.tensor(sizes, device=device)
        num_requests, depth = len(sizes), max(sizes)
        request = torch.repeat_interleave(
            torch.arange(num_requests, device=device), sizes_t
        )
        starts = torch.cumsum(sizes_t, 0) - sizes_t
        local = torch.arange(num_rows, device=device) - starts[request]
        K = sizes_t - 1
        is_draft = local < K[request]
        drafts = torch.where(is_draft, draft_tokens, 0).clamp(0, vocab_size - 1)
        if draft_logits is None:
            # Point-mass drafts: the leftovers are the raw clocks.
            Q = torch.zeros_like(P).scatter_(1, drafts.unsqueeze(1), 1.0)
        else:
            Q = torch.softmax(
                draft_logits.double().nan_to_num(nan=float("-inf")), dim=-1
            )
        Q = Q * is_draft.unsqueeze(1)

        slots = torch.full((num_requests, depth), -1, dtype=torch.long, device=device)
        slots[request, local] = torch.arange(num_rows, device=device)
        pad = (slots < 0).unsqueeze(2)
        index = slots.clamp_min(0)
        stopping, col = residual_race(
            E[index].masked_fill(pad, 1.0),
            P[index].masked_fill(pad, 0.0),
            Q[index].masked_fill(pad, 0.0),
            drafts[index],
            K,
            self.law,
            self.pass_aggregator,
            self.pass_source,
        )
        tokens = torch.where(is_draft, drafts, race(E, P))
        return torch.where(local == stopping[request], col[request], tokens)


class TextSeal2SpeculativeWatermarker(
    TextSeal2Watermarker, SupportsSpeculativeDecoding
):
    """TextSeal2 with the paper's speculative rule: the drafter races the
    drafts on the key (``draft_watermarker``) and verification runs the
    residual race (``target_watermarker``). Without speculation it samples like
    ``TextSeal2Watermarker``."""

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRFName = "philox",
        correlation: float = DEFAULT_CORRELATION,
        law: str = "block",
        pass_aggregator: str = "none",
        pass_source: str = "own",
    ) -> None:
        super().__init__(key, context_width, prf, correlation)
        self.draft_watermarker = TextSeal2DraftWatermarker(
            key, context_width, prf, correlation
        )
        self.target_watermarker = TextSeal2VerificationWatermarker(
            key,
            context_width,
            prf,
            correlation,
            keyed_drafts=True,
            law=law,
            pass_aggregator=pass_aggregator,
            pass_source=pass_source,
        )


class TextSeal2WatermarkDetector(WatermarkDetector):
    """Detect a TextSeal2 watermark.

    Every distinct (context, token) pair with a full context scores
    ``-log Phi(-G*)`` on the virtual key of keys A and B, which is Exp(1)
    without the watermark; the p-value is the Gamma tail of the summed score.
    The public detector (``public=True``, or ``detect_public``) scores
    ``-log(1 - U_A)`` instead, which needs only the public key A and carries
    ``1 / sqrt(2)`` of the signal.
    """

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        p_value_threshold: float = 0.01,
        prf: WatermarkPRFName = "philox",
        public: bool = False,
    ) -> None:
        """With ``public``, ``key`` is the public key A and the detector scores
        ``-log(1 - U_A)``; otherwise ``key`` is the master key."""
        _validate_context_width(context_width)
        super().__init__(context_width, p_value_threshold, deduplicate_contexts=False)
        self.prf_name = prf
        keys = [key] if public else textseal2_keys(key)
        self.prfs: list[WatermarkPRF] = [create_prf(prf, k) for k in keys]

    @property
    def public_key(self) -> int:
        """Key A, all the public detector needs."""
        prf = self.prfs[0]
        assert isinstance(prf, PhiloxPRF)
        return prf.key

    def detect_public(
        self, token_ids: list[int], entropies: Sequence[float] | None = None
    ) -> WatermarkDetection:
        """``detect`` with the public detector."""
        return TextSeal2WatermarkDetector(
            self.public_key,
            self.context_width,
            self.p_value_threshold,
            self.prf_name,
            public=True,
        ).detect(token_ids, entropies)

    def detect(
        self, token_ids: list[int], entropies: Sequence[float] | None = None
    ) -> WatermarkDetection:
        """Detect the watermark in ``token_ids``.

        With ``entropies``, where ``entropies[t]`` is the entropy of the
        distribution token ``t`` was sampled from, this runs TextSeal's
        entropy-weighted test (arXiv:2605.12456, Eq. 4-6) instead, which
        weights each pair's score by its entropy, min-max normalized over the
        scored pairs.
        """
        if entropies is None:
            return super().detect(token_ids)
        entropy_array = _entropy_array(entropies, len(token_ids))
        contexts, targets, positions = self._pairs(token_ids)
        if len(targets) == 0:
            return WatermarkDetection(0.0, 1.0, 0, False)
        scores = self._score_tokens(contexts, targets).numpy()
        weights = entropy_weights(entropy_array[positions])
        p_value = weighted_gamma_p(scores, weights, 1.0)
        return WatermarkDetection(
            score=float(np.sum(weights * scores)),
            p_value=p_value,
            num_scored_tokens=len(scores),
            is_watermarked=p_value <= self.p_value_threshold,
        )

    def detect_localized(
        self,
        token_ids: list[int],
        entropies: Sequence[float] | None = None,
        min_length: int = 50,
        max_regions: int = 5,
    ) -> LocalizedWatermarkDetection:
        """Detect the watermark in all of ``token_ids`` or in parts of it.

        Runs TextSeal's adaptive ensemble (arXiv:2605.12456, Sec. 3.3) on the
        tokens with a full context, each (context, token) pair counted once
        per test: the global test, the best window of at least ``min_length``
        tokens and a union of up to ``max_regions`` windows, corrected for the
        windows and tests tried. ``entropies`` adds the entropy-weighted tests.
        Regions and ``token_labels`` are in positions of ``token_ids``.
        """
        width = self.context_width
        num_scored = max(len(token_ids) - width, 0)
        weights = None
        if entropies is not None:
            weights = entropy_weights(_entropy_array(entropies, len(token_ids))[width:])
        if num_scored == 0:
            return adaptive_ensemble_detection([])
        contexts = torch.tensor(
            [token_ids[stop - width : stop] for stop in range(width, len(token_ids))],
            dtype=torch.int64,
        )
        targets = torch.tensor(token_ids[width:], dtype=torch.int64)
        dedup_keys: list[tuple[Context, int]] = [
            (tuple(context), target)
            for context, target in zip(contexts.tolist(), targets.tolist())
        ]
        result = adaptive_ensemble_detection(
            self._score_tokens(contexts, targets).numpy(),
            p_value_threshold=self.p_value_threshold,
            min_length=min_length,
            max_regions=max_regions,
            dedup_keys=dedup_keys,
            weights=weights,
            base_variance=1.0,
        )
        return dataclasses.replace(
            result,
            regions=[(s + width, e + width, p) for s, e, p in result.regions],
            token_labels=[0] * width + result.token_labels,
        )

    def _pairs(
        self, token_ids: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Each distinct (context, token) pair with a full context, at its
        first occurrence: contexts, tokens and positions in ``token_ids``."""
        width = self.context_width
        pairs: dict[tuple[Context, int], int] = {}
        for stop in range(width, len(token_ids)):
            pairs.setdefault(
                (tuple(token_ids[stop - width : stop]), token_ids[stop]), stop
            )
        if not pairs:
            return (
                torch.empty((0, width), dtype=torch.int64),
                torch.empty(0, dtype=torch.int64),
                np.empty(0, dtype=np.int64),
            )
        contexts, targets = zip(*pairs)
        return (
            torch.tensor(contexts, dtype=torch.int64),
            torch.tensor(targets, dtype=torch.int64),
            np.fromiter(pairs.values(), dtype=np.int64, count=len(pairs)),
        )

    def _prepare_inputs(
        self, token_ids: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        contexts, targets, _ = self._pairs(token_ids)
        return contexts, targets

    def _score_tokens(
        self, contexts: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        virtual = virtual_normals(
            [
                prf.uniform(contexts, targets.unsqueeze(-1)).squeeze(-1)
                for prf in self.prfs
            ]
        )
        return -torch.special.log_ndtr(-virtual)

    def _aggregate_scores(self, token_scores: torch.Tensor) -> float:
        return token_scores.sum().item()

    def _get_p_value(self, score: float, num_scored_tokens: int) -> float:
        return _gamma_survival_integer_shape(score, num_scored_tokens)
