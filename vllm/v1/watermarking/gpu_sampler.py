# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.watermarking.concordmark import ConcordMarkWatermarker
from vllm.v1.watermarking.evaluated_contexts import EvaluatedContexts
from vllm.v1.watermarking.watermarker import (
    RandomSampler,
    SupportsDraftLawVerification,
    SupportsResidualVerification,
    SupportsVerificationLogits,
    Watermarker,
)
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.watermark import repeated_context_mask

logger = init_logger(__name__)


class GPUWatermarkSampler(Sampler):
    concordmark: ConcordMarkWatermarker | None = None

    def __init__(
        self,
        watermarker: Watermarker,
        *args,
        deduplicate_contexts: Literal[
            "none", "single_turn", "all", "strict"
        ] = "single_turn",
        deduplicate_contexts_max_history: int | None = 8192,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.watermarker = watermarker
        if isinstance(watermarker, ConcordMarkWatermarker):
            self.concordmark = watermarker
        # A context-free watermark, such as unigram, has no context to repeat.
        if watermarker.context_width == 0:
            deduplicate_contexts = "none"
        self.deduplicate_contexts = deduplicate_contexts
        self.deduplicate_contexts_max_history = deduplicate_contexts_max_history
        # "strict": the contexts earlier verification steps evaluated.
        self.evaluated_contexts: EvaluatedContexts | None = None
        if deduplicate_contexts == "strict":
            self.evaluated_contexts = EvaluatedContexts(
                self.sampling_states.max_num_reqs,
                deduplicate_contexts_max_history or 8192,
                self.req_states.all_token_ids.gpu.device,
            )
        self.watermarking = UvaBackedTensor(
            self.sampling_states.max_num_reqs, dtype=torch.bool
        )
        self.watermarking.np.fill(True)
        self.watermarking.copy_to_uva()

    @property
    def stateful(self) -> bool:
        return getattr(self.watermarker, "stateful", False)

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        super().add_request(req_idx, sampling_params)
        self.watermarking.np[req_idx] = sampling_params.watermarking
        if self.evaluated_contexts is not None:
            self.evaluated_contexts.reset(req_idx)
        if self.stateful:
            self.watermarker.add_request(req_idx)
            resumed = (
                self.req_states.prefill_len.np[req_idx]
                > self.req_states.prompt_len.np[req_idx]
            )
            if resumed and self.watermarking.np[req_idx]:
                # The per-request watermark state does not survive preemption.
                self.watermarking.np[req_idx] = False
                logger.warning_once(
                    "A preempted request resumed; its remaining tokens are "
                    "not watermarked."
                )
        if sampling_params.watermarking and sampling_params.temperature == 0:
            logger.warning_once(
                "Watermarking is enabled, but greedy decoding "
                "(temperature=0) cannot be watermarked. This request will use "
                "ordinary greedy sampling."
            )

    def apply_staged_writes(self) -> None:
        super().apply_staged_writes()
        self.watermarking.copy_to_uva()

    def _sample_random(
        self,
        processed_logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        top_k: torch.Tensor | None,
        top_p: torch.Tensor | None,
        use_fused_sampler: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        enabled = self.watermarking.np[idx_mapping_np] & (
            self.sampling_states.temperature.np[idx_mapping_np] != 0
        )
        if not np.any(enabled):
            return super()._sample_random(
                processed_logits,
                expanded_idx_mapping,
                idx_mapping_np,
                pos,
                top_k,
                top_p,
                use_fused_sampler,
            )

        processed_logits = apply_top_k_top_p(processed_logits, top_k, top_p)
        repeated_contexts = None
        if self.concordmark is not None:
            # ConcordMark contexts never repeat; it skips rows without one.
            contexts, repeated_contexts = self.concordmark.contexts_from_history(
                self.req_states.all_token_ids.gpu,
                expanded_idx_mapping,
                self.req_states.prompt_len.gpu,
                self.req_states.total_len.gpu,
                self.sampling_states.seeds.gpu,
            )
        else:
            contexts = self._get_contexts(expanded_idx_mapping)
            if self.deduplicate_contexts != "none":
                repeated_contexts = self._get_repeated_contexts(
                    expanded_idx_mapping, contexts
                )

        temperatures = self.sampling_states.temperature.gpu[expanded_idx_mapping]
        needs_mixed_sampling = (
            repeated_contexts is not None or not np.all(enabled) or self.stateful
        )
        skip_mask = None
        if needs_mixed_sampling:
            watermarking = self.watermarking.gpu[expanded_idx_mapping] & (
                temperatures != 0
            )
            if repeated_contexts is not None:
                watermarking &= ~repeated_contexts
            if self.stateful:
                # Partial prefill chunks emit no token and must not update state.
                prefill_lens = self.req_states.prefill_len.gpu[expanded_idx_mapping]
                watermarking &= pos + 1 >= prefill_lens
            skip_mask = ~watermarking

        random_sampler = RandomSampler(
            expanded_idx_mapping=expanded_idx_mapping,
            temperatures=self.sampling_states.temperature.gpu,
            seeds=self.sampling_states.seeds.gpu,
            positions=pos,
            use_fp64=self.use_fp64_gumbel,
        )
        output = self.watermarker.sample(
            processed_logits,
            contexts,
            random_sampler,
            skip_mask=skip_mask,
        )
        sampled = output.token_ids
        output_logits = output.logits
        sampled = torch.where(
            temperatures == 0,
            processed_logits.argmax(dim=-1),
            sampled,
        )
        return sampled, output_logits

    def watermark_verification_logits(
        self,
        processed_logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        draft_sampled: torch.Tensor,
        cu_num_logits: torch.Tensor,
        draft_logits: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The logits rejection sampling verifies, from the watermarker's
        ``verification_logits``, or its ``draft_law_verification_logits`` or
        ``residual_verification_logits`` when it has one and the drafts were
        sampled probabilistically.

        ``draft_sampled`` holds each row's input token, so the draft row ``i``
        verifies is ``draft_sampled[i + 1]``, and each row's context ends with
        the drafted tokens before it. ``draft_logits`` is the drafter's
        pre-temperature logits cache [max_num_reqs, num_speculative_steps,
        vocab] and ``pos`` each row's position. Opted-out, greedy and
        deduplicated rows keep their logits.
        """
        assert isinstance(self.watermarker, SupportsVerificationLogits)
        contexts = self._get_contexts(
            expanded_idx_mapping, expanded_local_pos, draft_sampled
        )
        enabled = self.watermarking.gpu[expanded_idx_mapping] & (
            self.sampling_states.temperature.gpu[expanded_idx_mapping] != 0
        )
        evaluated = getattr(self, "evaluated_contexts", None)
        # The rows of the watermarked requests; every one is evaluated, whatever
        # the rule then emits, so later steps treat its context as repeated.
        evaluated_rows = enabled.clone() if evaluated is not None else None
        if (
            draft_logits is not None
            and pos is not None
            and isinstance(self.watermarker, SupportsDraftLawVerification)
        ):
            verification_logits = self._draft_law_verification_logits(
                processed_logits,
                expanded_idx_mapping,
                expanded_local_pos,
                draft_sampled,
                cu_num_logits,
                draft_logits,
                pos,
                contexts,
                enabled,
            )
            if evaluated is not None:
                evaluated.commit(
                    contexts, expanded_idx_mapping, expanded_local_pos, evaluated_rows
                )
            return verification_logits
        if self.deduplicate_contexts != "none":
            enabled &= ~self._get_repeated_contexts(
                expanded_idx_mapping, contexts, expanded_local_pos
            )
        if (
            draft_logits is not None
            and pos is not None
            and isinstance(self.watermarker, SupportsResidualVerification)
        ):
            requests = expanded_idx_mapping.long().clamp_min(0)
            steps = expanded_local_pos.long().clamp(0, draft_logits.shape[1] - 1)
            temperatures = self.sampling_states.temperature.gpu[requests]
            temperatures = torch.where(temperatures == 0, 1, temperatures)
            row_draft_logits = draft_logits[
                requests, steps, : processed_logits.shape[-1]
            ].float() / temperatures.unsqueeze(-1)
            return self.watermarker.residual_verification_logits(
                processed_logits,
                contexts,
                draft_sampled.roll(-1),
                cu_num_logits,
                ~enabled,
                row_draft_logits,
                RandomSampler(
                    expanded_idx_mapping=expanded_idx_mapping,
                    temperatures=self.sampling_states.temperature.gpu,
                    seeds=self.sampling_states.seeds.gpu,
                    positions=pos,
                    use_fp64=self.use_fp64_gumbel,
                ),
            )
        verification_logits = self.watermarker.verification_logits(
            processed_logits,
            contexts,
            draft_sampled.roll(-1),
            cu_num_logits,
            ~enabled,
        )
        if evaluated is not None:
            evaluated.commit(
                contexts, expanded_idx_mapping, expanded_local_pos, evaluated_rows
            )
        return verification_logits

    def _draft_law_verification_logits(
        self,
        processed_logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        draft_sampled: torch.Tensor,
        cu_num_logits: torch.Tensor,
        draft_logits: torch.Tensor,
        pos: torch.Tensor,
        contexts: torch.Tensor,
        request_enabled: torch.Tensor,
    ) -> torch.Tensor:
        """``draft_law_verification_logits`` with each row's draft law, from
        the cached draft logits as the drafter computed it: temperature, then
        the request's truncation (``draft_law_logits``)."""
        assert isinstance(self.watermarker, SupportsDraftLawVerification)
        requests = expanded_idx_mapping.long().clamp_min(0)
        steps = expanded_local_pos.long().clamp(0, draft_logits.shape[1] - 1)
        states = self.sampling_states
        temperatures = states.temperature.gpu[requests]
        temperatures = torch.where(temperatures == 0, 1, temperatures)
        row_draft_logits = draft_logits[
            requests, steps, : processed_logits.shape[-1]
        ].float() / temperatures.unsqueeze(-1)
        draft_law = self.watermarker.draft_law_logits(
            row_draft_logits,
            states.top_k.gpu[requests],
            states.top_p.gpu[requests],
            states.min_p.gpu[requests],
        )
        if self.deduplicate_contexts != "none":
            repeated = self._get_repeated_contexts(
                expanded_idx_mapping, contexts, expanded_local_pos
            )
        else:
            repeated = torch.zeros_like(request_enabled)
        return self.watermarker.draft_law_verification_logits(
            processed_logits,
            contexts,
            draft_sampled.roll(-1),
            cu_num_logits,
            request_enabled,
            repeated,
            draft_law,
            RandomSampler(
                expanded_idx_mapping=expanded_idx_mapping,
                temperatures=states.temperature.gpu,
                seeds=states.seeds.gpu,
                positions=pos,
                use_fp64=self.use_fp64_gumbel,
            ),
        )

    def _get_repeated_contexts(
        self,
        expanded_idx_mapping: torch.Tensor,
        contexts: torch.Tensor,
        expanded_local_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        repeated = repeated_context_mask(
            self.req_states.all_token_ids.gpu,
            expanded_idx_mapping,
            self.req_states.prompt_len.gpu,
            self.req_states.total_len.gpu,
            contexts,
            self.deduplicate_contexts_max_history,
            include_prompt=self.deduplicate_contexts == "all",
            skip_partial_context=(
                self.deduplicate_contexts == "all" and expanded_local_pos is None
            ),
            history_offsets=expanded_local_pos,
            local_positions=expanded_local_pos,
            num_speculative_steps=(
                self.num_speculative_tokens if expanded_local_pos is not None else 0
            ),
        )
        evaluated = getattr(self, "evaluated_contexts", None)
        if evaluated is not None:
            repeated = repeated | evaluated.contains(contexts, expanded_idx_mapping)
        return repeated

    def _get_contexts(
        self,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor | None = None,
        draft_sampled: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build contexts for flattened verification rows.

        With committed tokens ``[10, 11]``, drafts ``[20, 21]``, and width 2,
        local positions ``[0, 1, 2]`` use contexts ``[10, 11]``, ``[11, 20]``,
        and ``[20, 21]``. The first verification row precedes the first draft.
        """
        context_width = self.watermarker.context_width
        req_indices = expanded_idx_mapping.to(torch.int64)
        valid_reqs = req_indices >= 0
        safe_req_indices = req_indices.clamp_min(0)
        total_lens = self.req_states.total_len.gpu[safe_req_indices].to(torch.int64)
        prompt_lens = self.req_states.prompt_len.gpu[safe_req_indices].to(torch.int64)
        if expanded_local_pos is None:
            expanded_local_pos = torch.zeros_like(req_indices)
        else:
            expanded_local_pos = expanded_local_pos.to(torch.int64)
        offsets = torch.arange(
            -context_width, 0, dtype=torch.int64, device=req_indices.device
        )
        positions = (
            total_lens.unsqueeze(-1) + expanded_local_pos.unsqueeze(-1) + offsets
        )
        committed = positions < total_lens.unsqueeze(-1)
        valid_committed = (
            valid_reqs.unsqueeze(-1)
            & committed
            & (positions >= prompt_lens.unsqueeze(-1))
        )
        committed_contexts = self.req_states.all_token_ids.gpu[
            safe_req_indices.unsqueeze(-1),
            positions.clamp(0, self.req_states.all_token_ids.gpu.shape[1] - 1),
        ]
        contexts = torch.where(valid_committed, committed_contexts, -1)
        if draft_sampled is None:
            return contexts

        row_indices = torch.arange(
            len(req_indices), dtype=torch.int64, device=req_indices.device
        )
        draft_offsets = positions - total_lens.unsqueeze(-1)
        draft_indices = (
            row_indices.unsqueeze(-1)
            - expanded_local_pos.unsqueeze(-1)
            + draft_offsets
            + 1
        )
        valid_drafts = (
            valid_reqs.unsqueeze(-1)
            & ~committed
            & (draft_offsets < expanded_local_pos.unsqueeze(-1))
        )
        draft_contexts = draft_sampled[draft_indices.clamp(0, len(draft_sampled) - 1)]
        return torch.where(valid_drafts, draft_contexts, contexts)
