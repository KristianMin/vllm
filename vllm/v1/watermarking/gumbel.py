# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Gumbel-max watermark generation and detection primitives."""

import math
import warnings
from collections.abc import Sequence

import numpy as np
import torch

from vllm.config.watermarking import WatermarkPRFName, derive_watermark_key
from vllm.v1.watermarking.detector import (
    Context,
    WatermarkDetection,
    WatermarkDetector,
)
from vllm.v1.watermarking.localized import (
    LocalizedWatermarkDetection,
    _dedup_window,
    adaptive_ensemble_detection,
    entropy_weights,
    weighted_gamma_p,
)
from vllm.v1.watermarking.prfs import PhiloxPRF, WatermarkPRF, create_prf
from vllm.v1.watermarking.watermarker import (
    RandomSampler,
    SupportsSpeculativeDecoding,
    Watermarker,
    WatermarkSample,
)


def _gamma_survival_integer_shape(score: float, shape: int) -> float:
    """Return the survival probability of a Gamma(shape, 1) variable."""
    if shape <= 0 or score <= 0:
        return 1.0
    log_score = math.log(score)
    log_term = 0.0
    max_log_term = 0.0
    log_terms = [log_term]
    for index in range(1, shape):
        log_term += log_score - math.log(index)
        log_terms.append(log_term)
        max_log_term = max(max_log_term, log_term)
    log_sum = max_log_term + math.log(
        sum(math.exp(term - max_log_term) for term in log_terms)
    )
    return min(1.0, math.exp(min(0.0, -score + log_sum)))


class GumbelWatermarker(Watermarker):
    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
    ) -> None:
        prf = create_prf(prf, key) if isinstance(prf, str) else prf
        _validate_context_width(context_width)
        self.prf = prf
        self._context_width = context_width

    @property
    def context_width(self) -> int:
        return self._context_width

    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> WatermarkSample:
        if type(self.prf) is PhiloxPRF and logits.device.type == "cuda":
            from vllm.v1.worker.gpu.sample.watermark import philox_gumbel_sample

            return WatermarkSample(
                philox_gumbel_sample(logits, contexts, self.prf.key), logits
            )
        vocabulary = torch.arange(logits.shape[-1], device=logits.device)
        uniforms = self.prf.uniform(contexts, vocabulary)
        uniforms = uniforms.clamp_min(torch.finfo(torch.float32).tiny)
        noise = -torch.log(-torch.log(uniforms))
        return WatermarkSample(torch.argmax(logits + noise, dim=-1), logits)

    def _try_sample_mixed(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        skip_mask: torch.Tensor,
        random_sampler: RandomSampler,
    ) -> WatermarkSample | None:
        if type(self.prf) is not PhiloxPRF or logits.device.type != "cuda":
            return None

        from vllm.v1.worker.gpu.sample.watermark import philox_gumbel_sample

        token_ids = philox_gumbel_sample(
            logits,
            contexts,
            self.prf.key,
            skip_mask=skip_mask,
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
        return WatermarkSample(token_ids, logits)


class DualKeyGumbelWatermarker(Watermarker, SupportsSpeculativeDecoding):
    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRFName = "philox",
        alpha: float = 0.1,
    ) -> None:
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha
        self.draft_watermarker = GumbelWatermarker(
            derive_watermark_key(key, b"key_a"), context_width, prf
        )
        self.target_watermarker = GumbelWatermarker(
            derive_watermark_key(key, b"key_b"), context_width, prf
        )
        self._routing_logits_cache: dict[torch.device, torch.Tensor] = {}

    def _routing_logits(self, device: torch.device) -> torch.Tensor:
        # Building this tensor on every CUDA call stages a pageable H2D copy and
        # synchronizes the stream.
        cached = self._routing_logits_cache.get(device)
        if cached is None:
            cached = torch.tensor(
                [1 - self.alpha, self.alpha],
                dtype=torch.float32,
                device=device,
            ).log()
            self._routing_logits_cache[device] = cached
        return cached

    @property
    def context_width(self) -> int:
        return self.draft_watermarker.context_width

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
    ) -> WatermarkSample:
        if self.alpha == 0:
            return self.draft_watermarker.sample(
                logits, contexts, random_sampler, skip_mask
            )
        if self.alpha == 1:
            return self.target_watermarker.sample(
                logits, contexts, random_sampler, skip_mask
            )
        if random_sampler is None:
            raise ValueError("dual-key Gumbel routing requires a random sampler")

        key_a_sample = self.draft_watermarker.sample(
            logits, contexts, random_sampler, skip_mask
        )
        key_b_sample = self.target_watermarker.sample(
            logits, contexts, random_sampler, skip_mask
        )
        routing_logits = self._routing_logits(logits.device).expand(logits.shape[0], -1)
        use_key_a = random_sampler(routing_logits) == 0
        return WatermarkSample(
            torch.where(
                use_key_a,
                key_a_sample.token_ids,
                key_b_sample.token_ids,
            ),
            logits,
        )

    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> WatermarkSample:
        raise ValueError("dual-key Gumbel routing requires a random sampler")


class GumbelWatermarkDetector(WatermarkDetector):
    def __init__(
        self,
        key: int,
        context_width: int = 4,
        p_value_threshold: float = 0.01,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
        deduplicate_contexts: bool = True,
    ) -> None:
        prf = create_prf(prf, key) if isinstance(prf, str) else prf
        _validate_context_width(context_width)
        super().__init__(context_width, p_value_threshold, deduplicate_contexts)
        self.prf = prf

    def _score_tokens(
        self, contexts: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        uniforms = self.prf.uniform(contexts, targets.unsqueeze(-1)).squeeze(-1)
        return -torch.log1p(-uniforms.to(torch.float64))

    def _get_p_value(self, score: float, num_scored_tokens: int) -> float:
        return _gamma_survival_integer_shape(score, num_scored_tokens)

    def _aggregate_scores(self, token_scores: torch.Tensor) -> float:
        return token_scores.sum().item()


class DualKeyGumbelWatermarkDetector(GumbelWatermarkDetector):
    """Detect a dual-key watermark with weighted early fusion.

    ``alpha`` is the weight assigned to key B; key A receives ``1 - alpha``.
    """

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        p_value_threshold: float = 0.01,
        prf: WatermarkPRFName = "philox",
        deduplicate_contexts: bool = True,
        alpha: float = 0.1,
    ) -> None:
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        super().__init__(
            derive_watermark_key(key, b"key_a"),
            context_width,
            p_value_threshold,
            prf,
            deduplicate_contexts,
        )
        self.key_b_prf = create_prf(prf, derive_watermark_key(key, b"key_b"))
        self.alpha = alpha

    @property
    def _null_variance(self) -> float:
        return (1 - self.alpha) ** 2 + self.alpha**2

    def detect(
        self, token_ids: list[int], entropies: Sequence[float] | None = None
    ) -> WatermarkDetection:
        """Detect the watermark in ``token_ids``.

        With ``entropies``, where ``entropies[t]`` is the entropy of the
        distribution token ``t`` was sampled from, this runs TextSeal's
        entropy-weighted test (arXiv:2605.12456, Eq. 4-6) instead, which
        weights each token's score by its entropy, min-max normalized within
        the text.
        """
        if entropies is None:
            return super().detect(token_ids)
        entropy_array = _entropy_array(entropies, len(token_ids))
        if not token_ids:
            return WatermarkDetection(0.0, 1.0, 0, False)
        scores, dedup_keys = self._position_scores(token_ids)
        scores, scored_entropies = _dedup_window(
            dedup_keys, scores, entropy_array, 0, len(scores)
        )
        assert scored_entropies is not None
        weights = entropy_weights(scored_entropies)
        p_value = weighted_gamma_p(scores, weights, self._null_variance)
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

        Runs TextSeal's adaptive ensemble (arXiv:2605.12456, Sec. 3.3): the
        global test, the best window of at least ``min_length`` tokens and a
        union of up to ``max_regions`` windows, corrected for the windows and
        tests tried. ``entropies`` adds the entropy-weighted tests.
        """
        weights = None
        if entropies is not None:
            weights = entropy_weights(_entropy_array(entropies, len(token_ids)))
        if not token_ids:
            return adaptive_ensemble_detection([])
        scores, dedup_keys = self._position_scores(token_ids)
        return adaptive_ensemble_detection(
            scores,
            p_value_threshold=self.p_value_threshold,
            min_length=min_length,
            max_regions=max_regions,
            dedup_keys=dedup_keys,
            weights=weights,
            base_variance=self._null_variance,
        )

    def _position_scores(
        self, token_ids: list[int]
    ) -> tuple[np.ndarray, list[Context] | None]:
        """Score every token, repeated contexts included. Return the scores
        and, when contexts are deduplicated, each token's context as its
        deduplication key."""
        padded = [-1] * self.context_width + list(token_ids)
        contexts = [
            tuple(padded[position : position + self.context_width])
            for position in range(len(token_ids))
        ]
        scores = self._score_tokens(
            torch.tensor(contexts, dtype=torch.int64),
            torch.tensor(token_ids, dtype=torch.int64),
        )
        return scores.numpy(), contexts if self.deduplicate_contexts else None

    def _score_tokens(
        self, contexts: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        scores = torch.stack(
            [
                -torch.log1p(
                    -prf.uniform(contexts, targets.unsqueeze(-1))
                    .squeeze(-1)
                    .to(torch.float64)
                )
                for prf in (self.prf, self.key_b_prf)
            ],
            dim=-1,
        )
        weights = scores.new_tensor([1 - self.alpha, self.alpha])
        return scores @ weights

    def _get_p_value(self, score: float, num_scored_tokens: int) -> float:
        variance = self._null_variance
        if self.alpha in (0, 0.5, 1):
            return _gamma_survival_integer_shape(
                score / variance, round(num_scored_tokens / variance)
            )
        shape = torch.tensor(num_scored_tokens / variance, dtype=torch.float64)
        scaled_score = torch.tensor(score / variance, dtype=torch.float64)
        return torch.special.gammaincc(shape, scaled_score).item()


def _entropy_array(entropies: Sequence[float], num_tokens: int) -> np.ndarray:
    entropy_array = np.asarray(entropies, dtype=np.float64)
    if entropy_array.shape != (num_tokens,):
        raise ValueError(
            "entropies must hold one value per token, got shape "
            f"{entropy_array.shape} for {num_tokens} tokens"
        )
    return entropy_array


def _validate_context_width(context_width: int) -> None:
    if context_width < 1:
        raise ValueError("context_width must be positive")
    if context_width > 16:
        warnings.warn(
            "context_width values greater than 16 reduce robustness to edits because "
            "each changed token affects more subsequent watermark contexts",
            stacklevel=3,
        )
