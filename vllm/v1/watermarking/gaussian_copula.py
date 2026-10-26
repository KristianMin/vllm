# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Gaussian copula watermark generation and detection, adapted from
``ComponentWatermark`` in AutoMark (Gloaguen et al., 2026),
https://arxiv.org/abs/2609.37310."""

import math
from collections.abc import Sequence
from typing import Any

import torch

from vllm.config.watermarking import WatermarkPRFName
from vllm.v1.watermarking.concordmark import _WEIGHT_LEVELS, _weighted_tail
from vllm.v1.watermarking.detector import WatermarkDetection
from vllm.v1.watermarking.gumbel import (
    GumbelWatermarkDetector,
    _validate_context_width,
)
from vllm.v1.watermarking.prfs import PhiloxPRF, WatermarkPRF, create_prf
from vllm.v1.watermarking.watermarker import (
    RandomSampler,
    Watermarker,
    WatermarkSample,
)
from vllm.v1.watermarking.weighting import WordStartWeights, surprisal_levels

# AutoMark's coupling for its copula, chosen to match the quality of TextSeal
# at alpha 0.1.
DEFAULT_CORRELATION = 0.98551
# Only the most likely tokens receive keyed noise; the tail is sampled privately.
NUM_CANDIDATES = 4096
# Seeds the fresh noise off CUDA. Public: the noise needs no secrecy.
_FRESH_NOISE_SEED = 0x434F50554C4131


def gumbels(u: torch.Tensor, noise: torch.Tensor, correlation: float) -> torch.Tensor:
    """Gumbel noise ``-log(-log Phi(z))`` for ``z = correlation * ndtri(u) +
    sqrt(1 - correlation**2) * noise``, given keyed uniforms ``u`` and fresh
    standard normal ``noise``."""
    z = correlation * torch.special.ndtri(u) + math.sqrt(1 - correlation**2) * noise
    # log Phi(z) is stable even in the lower tail; its negative is Exp(1).
    log_u = torch.special.log_ndtr(z)
    return -torch.log(-log_u)


class GaussianCopulaWatermarker(Watermarker):
    """Gaussian copula watermark: Gumbel-max sampling whose Gumbel noise is tied
    to the watermark key by a Gaussian copula.

    Every token ``u`` gets ``z_u = correlation * ndtri(U_u) + sqrt(1 -
    correlation**2) * N_u``, where ``U_u`` is the PRF uniform of the token and
    its context and ``N_u`` is fresh standard normal noise, and the sampled
    token is ``argmax_u log p_u - log(-log Phi(z_u))``. Each ``z_u`` is
    standard normal, so the token is distributed as ``p``, the model's
    distribution after temperature scaling and top-k/top-p filtering, while it
    favors tokens with a high ``U_u``. The context is the last
    ``context_width`` tokens; rows whose context reaches before the output are
    sampled without the watermark.

    Keyed noise is limited to the ``NUM_CANDIDATES`` most likely tokens.
    An independent plain sample chooses the head or tail, preserving their
    probability masses and the original tail distribution.

    Unlike the reference, the PRF is vLLM's Philox, the fresh noise is the
    request's sampling noise on CUDA (elsewhere it comes from a seeded
    generator owned by the watermarker), and top-k and temperature are vLLM's
    sampling parameters.
    """

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
        correlation: float = DEFAULT_CORRELATION,
    ) -> None:
        if not 0 < correlation <= 1:
            raise ValueError("correlation must lie in (0, 1]")
        _validate_context_width(context_width)
        self.prf = create_prf(prf, key) if isinstance(prf, str) else prf
        self._context_width = context_width
        self.correlation = correlation
        self._generators: dict[torch.device, torch.Generator] = {}

    @property
    def context_width(self) -> int:
        return self._context_width

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
            and logits.device.type == "cuda"
            and isinstance(random_sampler, RandomSampler)
            and not random_sampler.is_drafting
            and random_sampler.logits_cache is None
        ):
            from vllm.v1.worker.gpu.sample.gaussian_copula_watermark import (
                philox_gaussian_copula_sample,
            )

            assert skip_mask is not None
            token_ids = philox_gaussian_copula_sample(
                logits,
                contexts,
                self.prf.key,
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
        capped = logits.shape[-1] > NUM_CANDIDATES
        if capped:
            candidate_logits, vocabulary = logits.topk(
                NUM_CANDIDATES, dim=-1, sorted=False
            )
        else:
            candidate_logits = logits
            vocabulary = torch.arange(logits.shape[-1], device=logits.device)
        u = self.prf.uniform(contexts, vocabulary).double()
        generator = self._generators.get(logits.device)
        if generator is None:
            generator = torch.Generator(device=logits.device)
            generator.manual_seed(_FRESH_NOISE_SEED)
            self._generators[logits.device] = generator
        noise = torch.randn(
            u.shape, dtype=torch.float64, device=logits.device, generator=generator
        )
        noised = candidate_logits.double() + gumbels(u, noise, self.correlation)
        token_ids = torch.argmax(noised, dim=-1)
        if capped:
            token_ids = vocabulary.gather(-1, token_ids.unsqueeze(-1))[:, 0]
            # An independent plain draw chooses the head or its original tail.
            plain = torch.multinomial(
                logits.float().softmax(-1), 1, generator=generator
            )
            in_head = (vocabulary == plain).any(dim=-1)
            token_ids = torch.where(in_head, token_ids, plain[:, 0])
        return WatermarkSample(token_ids, logits)


class GaussianCopulaWatermarkDetector(GumbelWatermarkDetector):
    """Detect a Gaussian copula watermark.

    Every distinct (context, token) pair with a full context scores
    ``-log(1 - U)``, which is Exp(1) without the watermark; the p-value is the
    Gamma tail of the summed score, assuming ideal independent PRF uniforms.

    Two weighted tests sum ``w * -log(1 - U)`` instead, with weights from 1 to
    3.5 in steps of 0.25, and report the tail of that sum for independent
    Exp(1) scores (saddlepoint approximation): AutoMark's surprisal weights,
    given a detection model's log-probabilities (``detect(token_ids,
    logprobs)``), or text-only weights, ``word_start_weight`` for tokens that
    start a word or follow a sentence end (``tokenizer`` required).
    """

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        p_value_threshold: float = 0.01,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
        tokenizer: Any = None,
        word_start_weight: float = 1.0,
    ) -> None:
        super().__init__(
            key, context_width, p_value_threshold, prf, deduplicate_contexts=False
        )
        self.word_starts = None
        if word_start_weight != 1.0:
            if tokenizer is None:
                raise ValueError("word_start_weight needs the tokenizer")
            self.word_starts = WordStartWeights(tokenizer, word_start_weight)

    def detect(
        self,
        token_ids: list[int],
        logprobs: Sequence[float | None] | None = None,
    ) -> WatermarkDetection:
        """Test generated ``token_ids``, excluding the prompt. With
        ``logprobs``, ``logprobs[i]`` is a detection model's log-probability
        of ``token_ids[i]`` (or None), and each score weighs ``1 + min(-logprob,
        2.5)``, rounded to steps of 0.25."""
        if logprobs is None and self.word_starts is None:
            return super().detect(token_ids)
        if logprobs is not None and self.word_starts is not None:
            raise ValueError("logprobs and word_start_weight are exclusive")
        if logprobs is not None and len(logprobs) != len(token_ids):
            raise ValueError(
                f"logprobs must hold one value per token, got {len(logprobs)} "
                f"for {len(token_ids)} tokens"
            )
        positions = self._scored_positions(token_ids)
        if not positions:
            return WatermarkDetection(0.0, 1.0, 0, False)
        contexts, targets = self._pairs(token_ids, positions)
        if logprobs is not None:
            levels = surprisal_levels([logprobs[p] for p in positions])
        else:
            assert self.word_starts is not None
            levels = self.word_starts.levels(token_ids, positions)
        scores = self._score_tokens(contexts, targets)
        score = (_WEIGHT_LEVELS[levels] * scores).sum().item()
        hist = torch.bincount(levels, minlength=len(_WEIGHT_LEVELS))
        p_value = _weighted_tail(score, hist)
        return WatermarkDetection(
            score, p_value, len(positions), p_value <= self.p_value_threshold
        )

    def _scored_positions(self, token_ids: list[int]) -> list[int]:
        """Positions of the first occurrence of each (context, token) pair
        with a full context."""
        width = self.context_width
        first: dict[tuple[tuple[int, ...], int], int] = {}
        for stop in range(width, len(token_ids)):
            pair = (tuple(token_ids[stop - width : stop]), token_ids[stop])
            first.setdefault(pair, stop)
        return list(first.values())

    def _pairs(
        self, token_ids: list[int], positions: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        width = self.context_width
        if not positions:
            return (
                torch.empty((0, width), dtype=torch.int64),
                torch.empty(0, dtype=torch.int64),
            )
        return (
            torch.tensor(
                [token_ids[p - width : p] for p in positions], dtype=torch.int64
            ),
            torch.tensor([token_ids[p] for p in positions], dtype=torch.int64),
        )

    def _prepare_inputs(
        self, token_ids: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._pairs(token_ids, self._scored_positions(token_ids))
