# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Triton kernel for the TextSeal2 watermark: the Gaussian copula kernel on
the virtual key of two Philox keys."""

import math

import torch

from vllm.triton_utils import tl, tldevice, triton
from vllm.v1.watermarking.prfs.philox import _UINT32_MASK
from vllm.v1.worker.gpu.sample.gaussian_copula_watermark import _log_ndtr
from vllm.v1.worker.gpu.sample.gumbel import (
    gumbel_noised_argmax,
    murmur3_uniform32,
    murmur3_uniform64,
)
from vllm.v1.worker.gpu.sample.watermark import (
    _philox_candidate_words,
    _philox_context_state,
    _uint32_to_uniform,
)


@triton.jit
def _philox_block_uniforms(
    context_ptr,
    key_0,
    key_1,
    block_index,
    BLOCK_SIZE: tl.constexpr,
    CONTEXT_WIDTH: tl.constexpr,
):
    """``PhiloxPRF.uniform`` of one context for the tokens of one vocabulary
    block, in token order."""
    state_0, state_1, state_2, state_3 = _philox_context_state(
        context_ptr, key_0, key_1, CONTEXT_WIDTH
    )
    groups = block_index * (BLOCK_SIZE // 4) + tl.arange(0, BLOCK_SIZE // 4)
    output_0, output_1, output_2, output_3 = _philox_candidate_words(
        groups, state_0, state_1, state_2, state_3, key_0, key_1
    )
    # Lay the per-group words out in token order: 4g+0, 4g+1, 4g+2, 4g+3.
    return _uint32_to_uniform(
        tl.interleave(
            tl.interleave(output_0, output_2),
            tl.interleave(output_1, output_3),
        )
    )


@triton.jit(
    do_not_specialize=[
        "key_a_0_value",
        "key_a_1_value",
        "key_b_0_value",
        "key_b_1_value",
    ]
)
def _philox_textseal2_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    logits_ptr,
    logits_stride,
    contexts_ptr,
    context_stride,
    skip_mask_ptr,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    key_a_0_value,
    key_a_1_value,
    key_b_0_value,
    key_b_1_value,
    keyed_scale,
    fresh_scale,
    vocab_size,
    CONTEXT_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE % 4 == 0)
    row = tl.program_id(0).to(tl.int64)
    block_index = tl.program_id(1)
    candidate = block_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = candidate < vocab_size

    req_state_idx = tl.load(expanded_idx_mapping_ptr + row).to(tl.int64)
    valid_req = req_state_idx >= 0
    temp = tl.load(temp_ptr + req_state_idx, mask=valid_req, other=0.0).to(tl.float32)
    seed = tl.load(seeds_ptr + req_state_idx, mask=valid_req, other=0)
    pos = tl.load(pos_ptr + row)
    logits = tl.load(
        logits_ptr + row * logits_stride + candidate, mask=mask, other=float("-inf")
    ).to(tl.float32)

    if tl.load(skip_mask_ptr + row):
        # Sampled without the watermark, as gumbel_sample.
        value, index = gumbel_noised_argmax(
            logits,
            candidate,
            mask,
            seed,
            pos,
            temp,
            IS_DRAFTING=False,
            USE_FP64=USE_FP64,
            APPLY_TEMPERATURE=False,
        )
    else:
        context_ptr = contexts_ptr + row * context_stride
        u_a = _philox_block_uniforms(
            context_ptr,
            key_a_0_value.to(tl.uint32),
            key_a_1_value.to(tl.uint32),
            block_index,
            BLOCK_SIZE,
            CONTEXT_WIDTH,
        )
        u_b = _philox_block_uniforms(
            context_ptr,
            key_b_0_value.to(tl.uint32),
            key_b_1_value.to(tl.uint32),
            block_index,
            BLOCK_SIZE,
            CONTEXT_WIDTH,
        )
        # The fresh noise is the request's own sampling noise, oriented as in
        # the Gaussian copula kernel, so that with a vanishing correlation the
        # token is the one gumbel_sample draws.
        if USE_FP64:
            noise = tldevice.normcdfinv(murmur3_uniform64(seed, pos, candidate))
            logits = logits.to(tl.float64)
            u_a = u_a.to(tl.float64)
            u_b = u_b.to(tl.float64)
        else:
            fresh = tl.minimum(
                murmur3_uniform32(seed, pos, candidate), 0.9999999403953552
            )
            noise = -tldevice.normcdfinv(fresh)
        # keyed_scale = correlation / sqrt(2) turns the sum of the two keyed
        # normals into correlation times the virtual key.
        z = (
            keyed_scale * (tldevice.normcdfinv(u_a) + tldevice.normcdfinv(u_b))
            + fresh_scale * noise
        )
        logits = tl.where(logits != logits, float("-inf"), logits)
        values = tl.where(mask, logits - tl.log(-_log_ndtr(z)), float("-inf"))
        value, index = tl.max(values, axis=0, return_indices=True)
    tl.store(
        local_argmax_ptr + row * local_argmax_stride + block_index,
        block_index * BLOCK_SIZE + index,
    )
    tl.store(local_max_ptr + row * local_max_stride + block_index, value)


def philox_textseal2_sample(
    logits: torch.Tensor,
    contexts: torch.Tensor,
    key_a: int,
    key_b: int,
    correlation: float,
    skip_mask: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperatures: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    use_fp64: bool = False,
) -> torch.Tensor:
    """The token ``TextSeal2Watermarker`` samples in each row of
    temperature-scaled ``logits`` with Philox keys A and B, with the fresh
    noise of the given sampler inputs; rows in ``skip_mask`` get the token
    ``gumbel_sample`` draws."""
    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    if contexts.stride(-1) != 1:
        contexts = contexts.contiguous()
    skip_mask = skip_mask.contiguous()
    expanded_idx_mapping = expanded_idx_mapping.contiguous()
    positions = positions.contiguous()
    num_tokens, vocab_size = logits.shape
    if num_tokens == 0:
        return logits.new_empty(0, dtype=torch.int64)
    block_size = 1024
    num_blocks = triton.cdiv(vocab_size, block_size)
    local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max = logits.new_empty(
        num_tokens, num_blocks, dtype=torch.float64 if use_fp64 else torch.float32
    )
    _philox_textseal2_kernel[(num_tokens, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        logits,
        logits.stride(0),
        contexts,
        contexts.stride(0),
        skip_mask,
        expanded_idx_mapping,
        seeds,
        positions,
        temperatures,
        key_a & _UINT32_MASK,
        key_a >> 32,
        key_b & _UINT32_MASK,
        key_b >> 32,
        correlation / math.sqrt(2.0),
        math.sqrt(1.0 - correlation * correlation),
        vocab_size,
        CONTEXT_WIDTH=contexts.shape[-1],
        BLOCK_SIZE=block_size,
        USE_FP64=use_fp64,
    )
    max_block_index = local_max.argmax(dim=-1, keepdim=True)
    return local_argmax.gather(dim=-1, index=max_block_index).view(-1)


@triton.jit(
    do_not_specialize=[
        "key_a_0_value",
        "key_a_1_value",
        "key_b_0_value",
        "key_b_1_value",
        "noise_key_0_value",
        "noise_key_1_value",
    ]
)
def _philox_textseal2_draft_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    logits_ptr,
    logits_stride,
    contexts_ptr,
    context_stride,
    keyed_ptr,
    noise_contexts_ptr,
    key_a_0_value,
    key_a_1_value,
    key_b_0_value,
    key_b_1_value,
    noise_key_0_value,
    noise_key_1_value,
    keyed_scale,
    fresh_scale,
    vocab_size,
    CONTEXT_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block_index = tl.program_id(1)
    candidate = block_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = candidate < vocab_size
    logits = tl.load(
        logits_ptr + row * logits_stride + candidate, mask=mask, other=float("-inf")
    ).to(tl.float64)
    # The noise is PhiloxPRF.uniform of the row's (seed, position, step) words.
    noise = tldevice.normcdfinv(
        _philox_block_uniforms(
            noise_contexts_ptr + row * 4,
            noise_key_0_value.to(tl.uint32),
            noise_key_1_value.to(tl.uint32),
            block_index,
            BLOCK_SIZE,
            4,
        ).to(tl.float64)
    )
    z = noise
    if tl.load(keyed_ptr + row):
        context_ptr = contexts_ptr + row * context_stride
        u_a = _philox_block_uniforms(
            context_ptr,
            key_a_0_value.to(tl.uint32),
            key_a_1_value.to(tl.uint32),
            block_index,
            BLOCK_SIZE,
            CONTEXT_WIDTH,
        ).to(tl.float64)
        u_b = _philox_block_uniforms(
            context_ptr,
            key_b_0_value.to(tl.uint32),
            key_b_1_value.to(tl.uint32),
            block_index,
            BLOCK_SIZE,
            CONTEXT_WIDTH,
        ).to(tl.float64)
        z = (
            keyed_scale * (tldevice.normcdfinv(u_a) + tldevice.normcdfinv(u_b))
            + fresh_scale * noise
        )
    logits = tl.where(logits != logits, float("-inf"), logits)
    values = tl.where(mask, logits - tl.log(-_log_ndtr(z)), float("-inf"))
    value, index = tl.max(values, axis=0, return_indices=True)
    tl.store(
        local_argmax_ptr + row * local_argmax_stride + block_index,
        block_index * BLOCK_SIZE + index,
    )
    tl.store(local_max_ptr + row * local_max_stride + block_index, value)


def philox_textseal2_draft(
    logits: torch.Tensor,
    contexts: torch.Tensor,
    key_a: int,
    key_b: int,
    correlation: float,
    keyed: torch.Tensor,
    noise_contexts: torch.Tensor,
    noise_key: int,
) -> torch.Tensor:
    """The draft ``TextSeal2DraftWatermarker`` races in each row of
    temperature-scaled draft ``logits``, in float64: ``argmax_v logit_v -
    log(-log Phi(z_v))`` with the noise of ``PhiloxPRF(noise_key)`` on the
    row's ``noise_contexts`` words, keyed by Philox keys A and B on
    ``keyed`` rows and on the noise alone elsewhere."""
    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    if contexts.stride(-1) != 1:
        contexts = contexts.contiguous()
    keyed = keyed.contiguous()
    noise_contexts = noise_contexts.to(torch.int64).contiguous()
    num_tokens, vocab_size = logits.shape
    if num_tokens == 0:
        return logits.new_empty(0, dtype=torch.int64)
    block_size = 1024
    num_blocks = triton.cdiv(vocab_size, block_size)
    local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max = logits.new_empty(num_tokens, num_blocks, dtype=torch.float64)
    _philox_textseal2_draft_kernel[(num_tokens, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        logits,
        logits.stride(0),
        contexts,
        contexts.stride(0),
        keyed,
        noise_contexts,
        key_a & _UINT32_MASK,
        key_a >> 32,
        key_b & _UINT32_MASK,
        key_b >> 32,
        noise_key & _UINT32_MASK,
        noise_key >> 32,
        correlation / math.sqrt(2.0),
        math.sqrt(1.0 - correlation * correlation),
        vocab_size,
        CONTEXT_WIDTH=contexts.shape[-1],
        BLOCK_SIZE=block_size,
    )
    max_block_index = local_max.argmax(dim=-1, keepdim=True)
    return local_argmax.gather(dim=-1, index=max_block_index).view(-1)
