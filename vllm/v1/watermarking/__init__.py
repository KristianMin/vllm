# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.watermarking import derive_watermark_key
from vllm.v1.watermarking.binomial import BinomialScoreWatermarkDetector
from vllm.v1.watermarking.block_verified_gumbel import (
    BlockVerifiedGumbelDraftWatermarker,
    BlockVerifiedGumbelVerificationWatermarker,
    BlockVerifiedGumbelWatermarker,
    FusedTwoKeyGumbelWatermarkDetector,
    FusedWatermarkDetection,
    FreshNoiseDraftWatermarker,
    SampledDraftWatermarker,
)
from vllm.v1.watermarking.chi_squared import ChiSquaredWatermarker
from vllm.v1.watermarking.concordmark import (
    ConcordMarkWatermarkDetector,
    ConcordMarkWatermarker,
)
from vllm.v1.watermarking.detector import WatermarkDetection, WatermarkDetector
from vllm.v1.watermarking.dipmark_morphmark import (
    DiPMarkWatermarkDetector,
    DiPMarkWatermarker,
    MorphMarkWatermarkDetector,
    MorphMarkWatermarker,
)
from vllm.v1.watermarking.factory import create_watermarker
from vllm.v1.watermarking.gaussian_copula import (
    GaussianCopulaWatermarkDetector,
    GaussianCopulaWatermarker,
)
from vllm.v1.watermarking.gumbel import (
    DualKeyGumbelWatermarkDetector,
    DualKeyGumbelWatermarker,
    GumbelWatermarkDetector,
    GumbelWatermarker,
    WeightedGumbelWatermarkDetector,
)
from vllm.v1.watermarking.localized import LocalizedWatermarkDetection
from vllm.v1.watermarking.lookahead import (
    LookaheadDualKeyGumbelWatermarker,
    LookaheadGaussianCopulaWatermarker,
    LookaheadGumbelWatermarker,
    LookaheadTextSeal2Watermarker,
)
from vllm.v1.watermarking.lookahead_ppl import LookaheadPPLSoftWatermarker
from vllm.v1.watermarking.lookahead_textgrain import LookaheadTextGrainWatermarker
from vllm.v1.watermarking.multibit_red_green import (
    MultibitRedGreenWatermarkDetector,
    MultibitRedGreenWatermarker,
    MultibitWatermarkDetection,
)
from vllm.v1.watermarking.perplexity import PPLHardWatermarker, PPLSoftWatermarker
from vllm.v1.watermarking.prfs import (
    PhiloxPRF,
    WatermarkPRF,
    create_prf,
)
from vllm.v1.watermarking.red_green import (
    RedGreenWatermarkDetector,
    RedGreenWatermarker,
    UnigramWatermarkDetector,
    UnigramWatermarker,
)
from vllm.v1.watermarking.rotation_first import (
    RotationFirstWatermarkDetector,
    RotationFirstWatermarker,
)
from vllm.v1.watermarking.synthid import (
    SynthIDWatermarkDetector,
    SynthIDWatermarker,
)
from vllm.v1.watermarking.textgrain import (
    TextGrainWatermarkDetector,
    TextGrainWatermarker,
)
from vllm.v1.watermarking.textseal2 import (
    TextSeal2DraftLawDraftWatermarker,
    TextSeal2DraftLawVerificationWatermarker,
    TextSeal2SpeculativeWatermarker,
    TextSeal2VerificationWatermarker,
    TextSeal2WatermarkDetector,
    TextSeal2Watermarker,
)
from vllm.v1.watermarking.two_key_block import (
    FusedTwoKeyWatermarkDetector,
    TwoKeyBlockDraftWatermarker,
    TwoKeyBlockVerificationWatermarker,
    TwoKeyBlockWatermarker,
    TwoKeyDetection,
)
from vllm.v1.watermarking.watermarker import (
    SupportsDraftLaw,
    SupportsDraftLawVerification,
    SupportsResidualVerification,
    SupportsSpeculativeDecoding,
    SupportsVerificationLogits,
    Watermarker,
    WatermarkSample,
)

__all__ = [
    "ConcordMarkWatermarkDetector",
    "ConcordMarkWatermarker",
    "BinomialScoreWatermarkDetector",
    "BlockVerifiedGumbelDraftWatermarker",
    "BlockVerifiedGumbelVerificationWatermarker",
    "BlockVerifiedGumbelWatermarker",
    "ChiSquaredWatermarker",
    "DiPMarkWatermarkDetector",
    "DiPMarkWatermarker",
    "DualKeyGumbelWatermarkDetector",
    "FusedTwoKeyGumbelWatermarkDetector",
    "FusedWatermarkDetection",
    "DualKeyGumbelWatermarker",
    "GaussianCopulaWatermarkDetector",
    "GaussianCopulaWatermarker",
    "GumbelWatermarkDetector",
    "WeightedGumbelWatermarkDetector",
    "GumbelWatermarker",
    "MultibitRedGreenWatermarkDetector",
    "MultibitRedGreenWatermarker",
    "MultibitWatermarkDetection",
    "LookaheadDualKeyGumbelWatermarker",
    "LookaheadGaussianCopulaWatermarker",
    "LookaheadGumbelWatermarker",
    "LookaheadTextSeal2Watermarker",
    "LookaheadPPLSoftWatermarker",
    "LookaheadTextGrainWatermarker",
    "PPLHardWatermarker",
    "PPLSoftWatermarker",
    "LocalizedWatermarkDetection",
    "MorphMarkWatermarkDetector",
    "MorphMarkWatermarker",
    "PhiloxPRF",
    "RotationFirstWatermarkDetector",
    "RotationFirstWatermarker",
    "RedGreenWatermarkDetector",
    "RedGreenWatermarker",
    "FreshNoiseDraftWatermarker",
    "SampledDraftWatermarker",
    "SupportsDraftLaw",
    "SupportsDraftLawVerification",
    "SupportsResidualVerification",
    "SupportsSpeculativeDecoding",
    "SupportsVerificationLogits",
    "SynthIDWatermarkDetector",
    "SynthIDWatermarker",
    "TextGrainWatermarkDetector",
    "TextGrainWatermarker",
    "TextSeal2DraftLawDraftWatermarker",
    "TextSeal2DraftLawVerificationWatermarker",
    "TextSeal2SpeculativeWatermarker",
    "FusedTwoKeyWatermarkDetector",
    "TwoKeyBlockDraftWatermarker",
    "TwoKeyBlockVerificationWatermarker",
    "TwoKeyBlockWatermarker",
    "TwoKeyDetection",
    "TextSeal2VerificationWatermarker",
    "TextSeal2WatermarkDetector",
    "TextSeal2Watermarker",
    "UnigramWatermarkDetector",
    "UnigramWatermarker",
    "WatermarkDetection",
    "WatermarkDetector",
    "WatermarkPRF",
    "Watermarker",
    "WatermarkSample",
    "create_watermarker",
    "create_prf",
    "derive_watermark_key",
]
