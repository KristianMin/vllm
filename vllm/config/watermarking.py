# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from vllm.config.utils import config
from vllm.logger import init_logger

logger = init_logger(__name__)

WatermarkingAlgorithm = Literal[
    "gumbel",
    "dual_key_gumbel",
    "red_green",
    "unigram",
    "dipmark",
    "morphmark",
    "multibit_red_green",
    "lookahead_gumbel",
    "lookahead_dual_key_gumbel",
    "lookahead_gaussian_copula",
    "lookahead_textseal2",
    "lookahead_ppl_soft",
    "lookahead_textgrain",
    "chi_squared",
    "ppl_hard",
    "ppl_soft",
    "synthid",
    "textgrain",
    "gaussian_copula",
    "textseal2",
    "block_verified_gumbel",
    "rotation_first",
    "concordmark",
]
LookaheadAggregator = Literal[
    "pi1", "pi2", "max", "tsum", "min", "lin", "expsum", "gauss", "optpath"
]
# The aggregators each lookahead algorithm implements.
LOOKAHEAD_GUMBEL_AGGREGATORS = ("pi1", "pi2", "max", "tsum", "min", "lin", "expsum")
# Aggregators that carry the realized score of the path below up the rows,
# lookahead_gumbel only (its score is the Gumbel-max detection score).
LOOKAHEAD_GUMBEL_PATH_AGGREGATORS = ("optpath",)
LOOKAHEAD_PPL_AGGREGATORS = ("pi1", "pi2", "tsum", "max", "gauss", "optpath")
WatermarkPRFName = Literal["philox"]
RotationFirstLexicalLetters = Literal["latin", "latin_cyrillic"]
WatermarkContextScope = Literal["none", "single_turn", "all", "strict"]
# Algorithms whose speculative verification implements deduplicate_contexts
# "strict" (contexts evaluated in earlier verification steps are repeated too).
STRICT_DEDUPLICATION_ALGORITHMS = ("lookahead_gumbel", "block_verified_gumbel")

_MIN_RECOMMENDED_DEDUP_HISTORY = 1024


def derive_watermark_key(key: int, domain: bytes) -> int:
    digest = hashlib.sha256(domain + key.to_bytes(8, "big")).digest()
    return int.from_bytes(digest[:8], "big")


@config
class WatermarkConfig:
    """Configuration for text watermark generation."""

    key: int = Field(ge=0, repr=False, exclude=True)
    """Secret key used to watermark generated text."""
    algorithm: WatermarkingAlgorithm = "gumbel"
    """Algorithm used to watermark generated text."""
    alpha: float = Field(default=0.1, ge=0, le=1)
    """Probability of selecting key B for dual-key watermarking (`dipmark`'s
    interval parameter is `dipmark_alpha`)."""
    correlation: float | None = Field(default=None, gt=0, le=1)
    """Keyed share of the Gaussian behind the Gumbel noise of the Gaussian
    copula watermarks; 1 makes the noise fully keyed. Each algorithm has its
    own default: 0.98551 for `gaussian_copula` and `lookahead_gaussian_copula`,
    AutoMark's value chosen to match the quality of TextSeal at alpha 0.1, and
    0.9 for `textseal2` and `lookahead_textseal2`, the TextSeal2 paper's."""
    delta: float | None = Field(default=None, gt=0)
    """Strength of the logit-bias and chi-squared watermarks; each algorithm
    has its own default: the bias added to green-list tokens for `red_green`
    and `unigram` (2.0) and for every message bit a token agrees with for
    `multibit_red_green` (0.5), and how strongly `chi_squared` favors tokens
    with high watermark scores (1.0)."""
    gamma: float | None = Field(default=None, gt=0, lt=1)
    """Fraction of the vocabulary in each green list (the tokens whose PRF
    value is below `gamma`), whose default depends on the algorithm: 0.25 for
    `red_green` and 0.5 for `unigram`, `dipmark` and `morphmark` (TextSeal's
    value). `dipmark` generation does not depend on it: it is the green-list
    fraction its detector counts, the end of DiPMark's keyed permutation."""
    dipmark_alpha: float = Field(default=0.2, gt=0, le=0.5)
    """Interval parameter of `dipmark` (TextSeal's `alpha`, default 0.2): the
    reweighting removes the first `dipmark_alpha` of the cumulative
    probability in the keyed permutation and doubles the last `dipmark_alpha`.
    Larger values give a stronger watermark; the law is unchanged on average
    over keys for every value. Values above 0.5 are equivalent to
    `1 - dipmark_alpha` and are rejected."""
    morphmark_k: float = Field(default=1.3, gt=0)
    """Slope of `morphmark`'s adaptive strength (TextSeal's `k_morphmark`,
    default 1.3): when the green mass `P_G` of the next-token law exceeds
    `morphmark_p0`, a fraction `r = min(morphmark_k * P_G, 1 - 1e-6)` of the
    red mass moves to the green tokens."""
    morphmark_p0: float = Field(default=0.15, ge=0, lt=1)
    """Green-mass threshold of `morphmark` (TextSeal's `p_0`, default 0.15):
    at or below it the strength is 1e-6, which leaves the law unchanged up to
    a total variation of 1e-6."""
    message: int = Field(default=0, ge=0)
    """Message embedded by multi-bit red-green watermarking, as an unsigned
    integer of `message_bits` bits."""
    message_bits: int = Field(default=16, ge=1, le=24)
    """Number of message bits embedded by multi-bit red-green watermarking."""
    aggregator: LookaheadAggregator = "tsum"
    """How a lookahead watermark combines a drafted token's own randomness with
    what the row below sends up. `lookahead_gumbel`, `lookahead_dual_key_gumbel`,
    `lookahead_gaussian_copula` and `lookahead_textseal2`: `pi1` is the plain
    watermark, `pi2` takes the child's race value, `max`, `tsum`, `min`, `lin`
    and `expsum` mix both while keeping the result a standard Gumbel.
    `lookahead_ppl_soft`: `pi1` is plain `ppl_soft`, `pi2`, `tsum`, `max` and
    `gauss` combine the rank of the drafted token's score with the rank the row
    below sends up, and `optpath` races it on its own score plus the realized
    score of the path below."""
    aggregator_lambda: float = Field(default=1.0, gt=0)
    """Weight of the child's value in the `lin` (`lookahead_gumbel` and its
    dual-key, Gaussian copula and TextSeal2 variants) and `gauss`
    (`lookahead_ppl_soft`) lookahead aggregators."""
    keyed_drafts: bool = True
    """Whether `textseal2` under speculative decoding races each draft on the
    key, the paper's rule: verification then runs TextSeal2's residual race on
    the leftovers of the drafts' clocks, which keeps block verification's
    acceptance and watermarks every emitted token. False leaves the drafts
    unwatermarked and runs the residual race on them as point masses, which
    accepts each draft with its target probability. For
    `block_verified_gumbel`, false samples the drafts from the draft law with
    fresh noise that does not depend on the key, and verification runs
    lookahead Gumbel-max's recursion on the raw keyed clocks with the
    verification law as the rows' weights (no residual race): the acceptance of
    unwatermarked verification with the same draft law."""
    verification_law: Literal["block", "token", "shared", "point"] = "block"
    """Verification law the residual race of `textseal2` and
    `block_verified_gumbel` follows under speculative decoding: `block`
    verification (the TextSeal2 paper's rule) or `token`-level verification,
    which accepts fewer drafts. `shared` (`block_verified_gumbel` only) is the
    baseline that verifies the keyed drafts on the shared clocks: every row
    emits its own Gumbel-max token and a draft is kept iff it equals it.
    `point` (`block_verified_gumbel` only) is point-mass verification's law:
    a draft is kept with its target probability, otherwise the row emits
    another token with its target probability."""
    residual_race: bool = True
    """Whether `block_verified_gumbel` decides each round of its keyed drafts
    by the residual race. False is plain block (or token-level) verification
    of the keyed drafts: the stopping row is drawn with fresh randomness that
    does not depend on the key, the token after the kept drafts is the bonus
    row's Gumbel-max sample with the key when every draft is kept, else a
    Gumbel-max sample of the verification law's residual distribution with a
    second key; detect with `FusedTwoKeyGumbelWatermarkDetector`. Requires
    `pass_aggregator="none"` and keyed drafts."""
    fresh_lanes: bool = False
    """`block_verified_gumbel` with `keyed_drafts=false` only: verify on fresh
    clocks that do not depend on the key, so nothing is watermarked. The
    unwatermarked baseline of the sampled-draft rule: speculative decoding with
    drafts sampled from the draft law and the verification law of
    `verification_law`. Requires `pass_aggregator="none"`."""
    pass_aggregator: Literal["none", "identity", "tsum", "expsum", "max"] = "none"
    """The lookahead on the residual race of `textseal2` and
    `block_verified_gumbel` under speculative decoding (not in the TextSeal2
    paper): `none` is the one-shot race on the leftovers (the paper's rule);
    `identity`, `tsum`, `expsum` and `max` put each draft race's time into the
    race's backward chain (`identity` alone, the others mixed with the later
    rows' time), which keeps the output distribution and the acceptance and
    raises the scores of kept drafts. With point-mass drafts, `identity`,
    `none`, `max` and `tsum` are lookahead Gumbel-max's `pi1`, `pi2`, `max` and
    `tsum`."""
    pass_source: Literal["own", "child"] = "own"
    """Whose draft race time `pass_aggregator` mixes into a row's pass clock:
    the row's own draft's (`own`) or the next draft's (`child`)."""
    lookahead_reward_samples: int = Field(default=64, ge=2)
    """Size of each of the two reference samples `lookahead_textgrain` draws of
    a child's expected-reward message to calibrate its parent (each costs that
    many extra coupling solves per drafted row). Exactness holds for every
    size; larger samples quantize the message less."""
    epsilon: float = Field(default=0.0, ge=0)
    """Perplexity budget of the perplexity watermarks: how many nats the
    expected surprisal of a sampled token may exceed the entropy of the model's
    next-token distribution, for every key (`ppl_hard`) or on average over keys
    (`ppl_soft`)."""
    depth: int = Field(default=30, ge=1, le=48)
    """Number of tournament layers of SynthID watermarking."""
    num_leaves: int = Field(default=2, ge=2)
    """Number of draws per tournament match of SynthID watermarking. More than
    two strengthens the watermark but distorts the output distribution."""
    entropy_budget: float = Field(default=0.2, ge=0, le=1)
    """Average fraction of the next-token entropy textGrain watermarking
    removes. A larger budget gives a stronger watermark and more distortion."""
    num_blocks: int = Field(default=16, ge=2)
    """Number of keyed vocabulary blocks of textGrain watermarking."""
    num_columns: int = Field(default=16, ge=2)
    """Number of keyed columns textGrain couples the blocks to."""
    context_width: int = Field(default=4, ge=1)
    """Number of prior tokens used by the watermark PRF. Ignored by `unigram`,
    whose green list takes no context, and by `rotation_first`, which keys
    clocks by the previous token class. For `concordmark`, the widest window of
    its context ladder. `lookahead_textseal2`, and `textseal2` under
    speculative decoding, should use at least `num_speculative_tokens`: then a
    row discarded in one verification step and a row of the next step at the
    same position differ in the token where the output left the drafted path,
    so keyed values that a discarded race already used do not come back at that
    position. The same holds for `block_verified_gumbel`."""
    deduplicate_contexts: WatermarkContextScope = "single_turn"
    """Which history is searched for a repeated context before a token is
    sampled; a repeated context is sampled without the watermark. `none`
    disables the search. `single_turn` (default) searches this request's
    generated tokens. `all` also searches the prompt and samples the first
    `context_width` generated tokens without watermarking. `strict`
    (`lookahead_gumbel` and `block_verified_gumbel` only) searches like
    `single_turn` and also treats as repeated every context of a row an
    earlier speculative verification step of the request evaluated (rejected
    rows and rows after the first rejection included), whose keyed values a
    lookahead or residual race has already read. Always `none` for
    `rotation_first`. Ignored by `concordmark`, which builds its own contexts."""
    deduplicate_contexts_max_history: int | None = Field(default=8192, ge=1)
    """Number of most recent history positions searched (default 8192), or
    `None` for the whole scope. Each position is compared over the
    `context_width` tokens before it. Ignored when `deduplicate_contexts` is
    `none`, except by `concordmark`, which counts window occurrences over this
    many positions."""
    prf: WatermarkPRFName = "philox"
    """Pseudorandom function used by the watermarking algorithm."""
    lexical_letters: RotationFirstLexicalLetters = "latin"
    """Letters of the words `rotation_first` groups into lexical classes:
    `latin` (a-z) or `latin_cyrillic` (also the lowercase Cyrillic letters
    а-я and ѐ-џ). Detection must use the same setting."""
    allow_target_only_watermarking: bool = False
    """Allow speculative decoding without watermarking draft tokens."""

    @property
    def supports_speculative_decoding(self) -> bool:
        # textGrain verifies drafts against its watermarked distribution.
        return self.algorithm in (
            "dual_key_gumbel",
            "lookahead_gumbel",
            "lookahead_dual_key_gumbel",
            "lookahead_gaussian_copula",
            "lookahead_textseal2",
            "lookahead_ppl_soft",
            "lookahead_textgrain",
            "textgrain",
            "textseal2",
            "block_verified_gumbel",
        )

    @property
    def verifies_unwatermarked_drafts(self) -> bool:
        # Drafts are sampled without the key and verified against the
        # watermark, so the output stays exact however they are sampled.
        return self.algorithm in (
            "lookahead_gumbel",
            "lookahead_dual_key_gumbel",
            "lookahead_gaussian_copula",
            "lookahead_textseal2",
            "lookahead_ppl_soft",
            "lookahead_textgrain",
            "textgrain",
        )

    def _validate_block_verified_gumbel(self) -> None:
        """The combinations of `block_verified_gumbel`'s speculative rule."""
        bvg = self.algorithm == "block_verified_gumbel"
        if self.verification_law == "point" and not bvg:
            raise ValueError(
                "verification_law='point' is only defined for block_verified_gumbel"
            )
        if not self.residual_race:
            if not bvg:
                raise ValueError(
                    "residual_race=false is only defined for block_verified_gumbel"
                )
            if not self.keyed_drafts or self.verification_law not in (
                "block",
                "token",
            ):
                raise ValueError(
                    "residual_race=false verifies keyed drafts by block or "
                    "token-level verification"
                )
            if self.pass_aggregator != "none":
                raise ValueError(
                    "residual_race=false has no pass_aggregator (no residual race)"
                )
        if self.fresh_lanes:
            if not bvg or self.keyed_drafts:
                raise ValueError(
                    "fresh_lanes is the unwatermarked baseline of "
                    "block_verified_gumbel with keyed_drafts=false"
                )
            if self.pass_aggregator != "none":
                raise ValueError("fresh_lanes requires pass_aggregator='none'")
        if bvg and not self.keyed_drafts and self.verification_law == "shared":
            raise ValueError(
                "verification_law='shared' verifies keyed drafts (keyed_drafts=true)"
            )

    @model_validator(mode="after")
    def validate_watermark_settings(self) -> Self:
        if self.key > 2**64 - 1:
            raise ValueError("philox keys must fit in 64 bits")
        if self.message >= 2**self.message_bits:
            raise ValueError("message must fit in message_bits bits")
        if (
            self.algorithm
            in (
                "lookahead_gumbel",
                "lookahead_dual_key_gumbel",
                "lookahead_gaussian_copula",
                "lookahead_textseal2",
            )
            and self.aggregator not in LOOKAHEAD_GUMBEL_AGGREGATORS
            and not (
                self.algorithm == "lookahead_gumbel"
                and self.aggregator in LOOKAHEAD_GUMBEL_PATH_AGGREGATORS
            )
        ):
            raise ValueError(
                f"{self.algorithm} supports the aggregators "
                f"{LOOKAHEAD_GUMBEL_AGGREGATORS}"
                + (
                    f" and {LOOKAHEAD_GUMBEL_PATH_AGGREGATORS}"
                    if self.algorithm == "lookahead_gumbel"
                    else ""
                )
            )
        if (
            self.algorithm == "lookahead_ppl_soft"
            and self.aggregator not in LOOKAHEAD_PPL_AGGREGATORS
        ):
            raise ValueError(
                "lookahead_ppl_soft supports the aggregators "
                f"{LOOKAHEAD_PPL_AGGREGATORS}"
            )
        if self.verification_law == "shared" and self.algorithm != (
            "block_verified_gumbel"
        ):
            raise ValueError(
                "verification_law='shared' is only defined for block_verified_gumbel"
            )
        self._validate_block_verified_gumbel()
        if (
            self.deduplicate_contexts == "strict"
            and self.algorithm not in STRICT_DEDUPLICATION_ALGORITHMS
        ):
            raise ValueError(
                "deduplicate_contexts='strict' is only implemented for "
                f"{STRICT_DEDUPLICATION_ALGORITHMS}"
            )
        if self.algorithm == "rotation_first":
            self.deduplicate_contexts = "none"
            return self
        history_is_too_short = (
            self.deduplicate_contexts_max_history is not None
            and self.deduplicate_contexts_max_history < _MIN_RECOMMENDED_DEDUP_HISTORY
        )
        if self.algorithm in (
            "gumbel",
            "dual_key_gumbel",
            "lookahead_gumbel",
            "lookahead_dual_key_gumbel",
            "ppl_hard",
            "ppl_soft",
            "lookahead_ppl_soft",
            "gaussian_copula",
            "lookahead_gaussian_copula",
            "textseal2",
            "lookahead_textseal2",
            "block_verified_gumbel",
        ) and (self.deduplicate_contexts == "none" or history_is_too_short):
            # These favor the same token whenever the key and context recur.
            name = {
                "ppl_hard": "Hard perplexity",
                "ppl_soft": "Soft perplexity",
                "lookahead_ppl_soft": "Lookahead soft perplexity",
                "gaussian_copula": "Gaussian copula",
                "lookahead_gaussian_copula": "Lookahead Gaussian copula",
                "textseal2": "TextSeal2",
                "lookahead_textseal2": "Lookahead TextSeal2",
            }.get(self.algorithm, "Gumbel-max")
            logger.warning_once(
                f"{name} watermarking with context deduplication "
                "disabled or limited to fewer than "
                f"{_MIN_RECOMMENDED_DEDUP_HISTORY} positions may increase the "
                "frequency of degenerate generations, including repetition loops. "
                "Use deduplicate_contexts='single_turn' or 'all' with "
                "deduplicate_contexts_max_history at least "
                f"{_MIN_RECOMMENDED_DEDUP_HISTORY} or null to mitigate this.",
                scope="global",
            )
        return self
