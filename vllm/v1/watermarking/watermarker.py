# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample


@dataclass(frozen=True)
class WatermarkSample:
    token_ids: torch.Tensor
    logits: torch.Tensor


@dataclass(frozen=True)
class RandomSampler:
    expanded_idx_mapping: torch.Tensor
    temperatures: torch.Tensor
    seeds: torch.Tensor
    positions: torch.Tensor
    use_fp64: bool = False
    is_drafting: bool = False
    logits_cache: torch.Tensor | None = None
    logits_cache_col: torch.Tensor | None = None
    logits_cache_source: torch.Tensor | None = None

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return gumbel_sample(
            logits,
            self.expanded_idx_mapping,
            self.temperatures,
            self.seeds,
            self.positions,
            apply_temperature=False,
            is_drafting=self.is_drafting,
            logits_cache=self.logits_cache,
            logits_cache_col=self.logits_cache_col,
            logits_cache_source=self.logits_cache_source,
            use_fp64=self.use_fp64,
        )


class Watermarker(ABC):
    stateful: bool = False
    """Whether sampling updates per-request state reset by ``add_request``."""

    @property
    @abstractmethod
    def context_width(self) -> int:
        raise NotImplementedError

    def add_request(self, req_idx: int) -> None:
        """Reset the state of a request slot."""
        return None

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
    ) -> WatermarkSample:
        if skip_mask is not None:
            if random_sampler is None:
                raise ValueError("skip_mask requires a random sampler")
            mixed = self._try_sample_mixed(logits, contexts, skip_mask, random_sampler)
            if mixed is not None:
                return mixed

        watermarked = self._sample_watermarked(logits, contexts)
        if skip_mask is None:
            return watermarked

        assert random_sampler is not None
        token_ids = torch.where(
            skip_mask, random_sampler(logits), watermarked.token_ids
        )
        output_logits = watermarked.logits
        if output_logits is not logits:
            output_logits = torch.where(skip_mask.unsqueeze(-1), logits, output_logits)
        return WatermarkSample(token_ids, output_logits)

    @abstractmethod
    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> WatermarkSample:
        raise NotImplementedError

    def _try_sample_mixed(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        skip_mask: torch.Tensor,
        random_sampler: RandomSampler,
    ) -> WatermarkSample | None:
        return None


@runtime_checkable
class SupportsSpeculativeDecoding(Protocol):
    """A watermarker compatible with standard speculative rejection sampling."""

    draft_watermarker: Watermarker
    target_watermarker: Watermarker


@runtime_checkable
class SupportsVerificationLogits(Protocol):
    """A watermarker that changes the logits speculative decoding verifies
    drafts against, instead of keying the recovery draw."""

    def verification_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        draft_tokens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        skip_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Logits to verify each row against.

        ``logits`` [num_logits, vocab] are the processed target logits of the
        verification rows: each request's drafted rows followed by its bonus
        row, request ``r`` owning rows ``cu_num_logits[r]:cu_num_logits[r + 1]``.
        Row ``i``'s context ends with the drafts before it and
        ``draft_tokens[i]`` is the draft it verifies (arbitrary at bonus rows).
        Rows in ``skip_mask`` must keep their logits.
        """
        ...


@runtime_checkable
class SupportsResidualVerification(Protocol):
    """A watermarker whose verification also reads the drafts' distributions
    and the request seeds and positions the drafts were raced with
    (TextSeal2's speculative rule). Used instead of ``verification_logits``
    when the drafts were sampled probabilistically."""

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
        """As ``SupportsVerificationLogits.verification_logits``.
        ``draft_logits`` [num_logits, vocab] are the temperature-scaled draft
        logits each draft row's draft was sampled from (arbitrary at bonus
        rows), and ``random_sampler`` holds each row's request seed and
        position."""
        ...


@runtime_checkable
class SupportsDraftLaw(Protocol):
    """A draft watermarker whose drafts race on a draft law truncated like the
    target, instead of the drafter's temperature-only distribution."""

    def draft_law_logits(
        self,
        logits: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        min_p: torch.Tensor,
    ) -> torch.Tensor:
        """The draft law's logits, -inf outside its support, from
        temperature-scaled draft ``logits`` [rows, vocab] and each row's
        request's ``top_k``, ``top_p`` and ``min_p`` [rows]. Must be a
        deterministic function of its inputs without host synchronization: the
        drafter calls it (possibly inside a CUDA graph) and verification calls
        it again on the cached draft logits."""
        ...


@runtime_checkable
class SupportsDraftLawVerification(SupportsDraftLaw, Protocol):
    """A watermarker that verifies keyed drafts raced on ``draft_law_logits``
    (block_verified_gumbel). Used instead of ``verification_logits`` when the
    drafts were sampled probabilistically."""

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
        """As ``SupportsVerificationLogits.verification_logits``, with the
        request-level ``request_enabled`` [num_logits] (watermarked, not
        greedy: those requests' drafts raced on the draft law, the others' on
        the temperature-only draft distribution) and the row-level
        ``repeated`` [num_logits] (repeated contexts) instead of
        ``skip_mask``, each row's ``draft_law`` [num_logits, vocab] (arbitrary
        at bonus rows) and ``random_sampler`` with each row's request seed and
        position."""
        ...
