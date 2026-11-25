# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

import torch

from vllm.config.watermarking import WatermarkConfig
from vllm.v1.watermarking.block_verified_gumbel import BlockVerifiedGumbelWatermarker
from vllm.v1.watermarking.chi_squared import ChiSquaredWatermarker
from vllm.v1.watermarking.concordmark import ConcordMarkWatermarker
from vllm.v1.watermarking.dipmark_morphmark import (
    DiPMarkWatermarker,
    MorphMarkWatermarker,
)
from vllm.v1.watermarking.gaussian_copula import (
    DEFAULT_CORRELATION as COPULA_DEFAULT_CORRELATION,
)
from vllm.v1.watermarking.gaussian_copula import GaussianCopulaWatermarker
from vllm.v1.watermarking.gumbel import (
    DualKeyGumbelWatermarker,
    GumbelWatermarker,
)
from vllm.v1.watermarking.lookahead import (
    LookaheadDualKeyGumbelWatermarker,
    LookaheadGaussianCopulaWatermarker,
    LookaheadGumbelWatermarker,
    LookaheadTextSeal2Watermarker,
)
from vllm.v1.watermarking.lookahead_ppl import LookaheadPPLSoftWatermarker
from vllm.v1.watermarking.lookahead_textgrain import LookaheadTextGrainWatermarker
from vllm.v1.watermarking.multibit_red_green import MultibitRedGreenWatermarker
from vllm.v1.watermarking.perplexity import PPLHardWatermarker, PPLSoftWatermarker
from vllm.v1.watermarking.red_green import RedGreenWatermarker, UnigramWatermarker
from vllm.v1.watermarking.rotation_first import RotationFirstWatermarker
from vllm.v1.watermarking.synthid import SynthIDWatermarker
from vllm.v1.watermarking.textgrain import TextGrainWatermarker
from vllm.v1.watermarking.textseal2 import (
    DEFAULT_CORRELATION as TEXTSEAL2_DEFAULT_CORRELATION,
)
from vllm.v1.watermarking.textseal2 import (
    TextSeal2SpeculativeWatermarker,
    TextSeal2VerificationWatermarker,
)
from vllm.v1.watermarking.watermarker import Watermarker

if TYPE_CHECKING:
    from vllm.config import ModelConfig


def create_watermarker(
    config: WatermarkConfig,
    model_config: "ModelConfig | None" = None,
    max_num_reqs: int | None = None,
    device: torch.device | None = None,
) -> Watermarker:
    watermarker = _create_watermarker(config, model_config, max_num_reqs, device)
    if not config.two_key_block_verification:
        return watermarker
    from vllm.v1.watermarking.two_key_block import TwoKeyBlockWatermarker

    return TwoKeyBlockWatermarker(
        watermarker,
        config.algorithm,
        config.key,
        config.context_width,
        config.prf,
        law=config.verification_law,
        alpha=config.alpha,
        correlation=getattr(watermarker, "correlation", 0.98551),
        textgrain_args={
            "entropy_budget": config.entropy_budget,
            "num_blocks": config.num_blocks,
            "num_columns": config.num_columns,
        },
        synthid_args={"depth": config.depth, "num_leaves": config.num_leaves},
    )


def _create_watermarker(
    config: WatermarkConfig,
    model_config: "ModelConfig | None" = None,
    max_num_reqs: int | None = None,
    device: torch.device | None = None,
) -> Watermarker:
    if config.algorithm == "gumbel":
        return GumbelWatermarker(config.key, config.context_width, config.prf)
    if config.algorithm == "block_verified_gumbel":
        return BlockVerifiedGumbelWatermarker(
            config.key,
            config.context_width,
            config.prf,
            law=config.verification_law,
            pass_aggregator=config.pass_aggregator,
            pass_source=config.pass_source,
            keyed_drafts=config.keyed_drafts,
            fresh_lanes=config.fresh_lanes,
            residual_race=config.residual_race,
            fresh_draft_noise=config.deduplicate_contexts == "strict",
        )
    if config.algorithm == "dual_key_gumbel":
        return DualKeyGumbelWatermarker(
            config.key,
            config.context_width,
            config.prf,
            config.alpha,
        )
    # Each algorithm has its own default delta and correlation.
    delta: dict[str, Any] = {} if config.delta is None else {"delta": config.delta}
    if config.correlation is not None:
        correlation = config.correlation
    elif config.algorithm in ("textseal2", "lookahead_textseal2"):
        correlation = TEXTSEAL2_DEFAULT_CORRELATION
    else:
        correlation = COPULA_DEFAULT_CORRELATION
    if config.algorithm == "red_green":
        return RedGreenWatermarker(
            config.key,
            config.context_width,
            config.prf,
            gamma=0.25 if config.gamma is None else config.gamma,
            **delta,
        )
    if config.algorithm == "unigram":
        return UnigramWatermarker(
            config.key,
            config.prf,
            gamma=0.5 if config.gamma is None else config.gamma,
            **delta,
        )
    if config.algorithm == "dipmark":
        return DiPMarkWatermarker(
            config.key,
            config.context_width,
            config.prf,
            alpha=config.dipmark_alpha,
            gamma=0.5 if config.gamma is None else config.gamma,
        )
    if config.algorithm == "morphmark":
        return MorphMarkWatermarker(
            config.key,
            config.context_width,
            config.prf,
            k=config.morphmark_k,
            p_0=config.morphmark_p0,
            gamma=0.5 if config.gamma is None else config.gamma,
        )
    if config.algorithm == "multibit_red_green":
        return MultibitRedGreenWatermarker(
            config.key,
            config.context_width,
            config.prf,
            message=config.message,
            message_bits=config.message_bits,
            **delta,
        )
    if config.algorithm == "lookahead_gumbel":
        return LookaheadGumbelWatermarker(
            config.key,
            config.context_width,
            config.prf,
            config.aggregator,
            config.aggregator_lambda,
        )
    if config.algorithm == "lookahead_dual_key_gumbel":
        return LookaheadDualKeyGumbelWatermarker(
            config.key,
            config.context_width,
            config.prf,
            config.alpha,
            config.aggregator,
            config.aggregator_lambda,
        )
    if config.algorithm == "lookahead_gaussian_copula":
        return LookaheadGaussianCopulaWatermarker(
            config.key,
            config.context_width,
            config.prf,
            correlation,
            config.aggregator,
            config.aggregator_lambda,
        )
    if config.algorithm == "lookahead_textseal2":
        return LookaheadTextSeal2Watermarker(
            config.key,
            config.context_width,
            config.prf,
            correlation,
            config.aggregator,
            config.aggregator_lambda,
        )
    if config.algorithm == "lookahead_ppl_soft":
        return LookaheadPPLSoftWatermarker(
            config.key,
            config.context_width,
            config.prf,
            config.epsilon,
            config.aggregator,
            config.aggregator_lambda,
        )
    if config.algorithm == "chi_squared":
        return ChiSquaredWatermarker(
            config.key, config.context_width, config.prf, **delta
        )
    if config.algorithm == "ppl_hard":
        return PPLHardWatermarker(
            config.key, config.context_width, config.prf, config.epsilon
        )
    if config.algorithm == "ppl_soft":
        return PPLSoftWatermarker(
            config.key, config.context_width, config.prf, config.epsilon
        )
    if config.algorithm == "synthid":
        return SynthIDWatermarker(
            config.key,
            config.context_width,
            config.prf,
            config.depth,
            config.num_leaves,
        )
    if config.algorithm == "textgrain":
        return TextGrainWatermarker(
            config.key,
            config.context_width,
            config.prf,
            config.entropy_budget,
            config.num_blocks,
            config.num_columns,
        )
    if config.algorithm == "lookahead_textgrain":
        return LookaheadTextGrainWatermarker(
            config.key,
            config.context_width,
            config.prf,
            config.entropy_budget,
            config.num_blocks,
            config.num_columns,
            config.lookahead_reward_samples,
        )
    if config.algorithm == "gaussian_copula":
        return GaussianCopulaWatermarker(
            config.key,
            config.context_width,
            config.prf,
            correlation,
        )
    if config.algorithm == "textseal2":
        speculative_rule = {
            "law": config.verification_law,
            "pass_aggregator": config.pass_aggregator,
            "pass_source": config.pass_source,
        }
        if config.keyed_drafts or config.spec_decision == "j_only":
            return TextSeal2SpeculativeWatermarker(
                config.key,
                config.context_width,
                config.prf,
                correlation,
                **speculative_rule,
                spec_decision=config.spec_decision,
                truncate_drafts=config.truncate_drafts,
            )
        return TextSeal2VerificationWatermarker(
            config.key,
            config.context_width,
            config.prf,
            correlation,
            keyed_drafts=False,
            **speculative_rule,
        )
    if config.algorithm == "rotation_first":
        from vllm.tokenizers import cached_tokenizer_from_config

        if model_config is None or max_num_reqs is None or device is None:
            raise ValueError(
                "rotation_first watermarking requires the model config, "
                "max_num_reqs and device"
            )
        tokenizer = cached_tokenizer_from_config(model_config)
        if tokenizer is None:
            raise ValueError(
                "rotation_first watermarking requires a tokenizer and does not "
                "support skip_tokenizer_init"
            )
        return RotationFirstWatermarker(
            config.key,
            tokenizer,
            model_config.get_vocab_size(),
            max_num_reqs,
            device,
            config.prf,
            config.lexical_letters,
        )
    if config.algorithm == "concordmark":
        return ConcordMarkWatermarker(
            config.key,
            config.context_width,
            config.prf,
            config.deduplicate_contexts_max_history,
        )
    raise ValueError(f"Unknown watermarking algorithm: {config.algorithm}")
