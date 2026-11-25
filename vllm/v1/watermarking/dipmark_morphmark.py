# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DiPMark and MorphMark: green-list reweighting watermarks, ported from Meta's
TextSeal (``textseal/watermarking/generator.py``: ``DipmarkGenerator`` and
``MorphMarkGenerator``).

Both reweight the model's next-token law with a green list drawn from the key
and the context, and both are detected, as in TextSeal, by the green-list
binomial test of the red-green watermark (``RedGreenWatermarkDetector``). Like
``red_green``, the reweighting is keyed with vLLM's PRF and contexts, runs after
temperature, min-p, top-k and top-p, and leaves the sampling itself to vLLM's
random sampler. Each token's PRF value ``u`` decides both schemes' green list:
a token is green when ``u < gamma``.

- DiPMark (Wu et al., 2023, https://arxiv.org/abs/2310.07710) orders the
  vocabulary by descending ``u`` (so the green tokens come last) and moves the
  mass of the first ``alpha`` of the cumulative distribution in that order to
  its last ``alpha``. Averaged over the key's permutations, the reweighted law
  is the model's law.
- MorphMark (Wang et al., 2025, https://arxiv.org/abs/2505.11541) moves a
  fraction ``r = min(k * P_G, 1 - 1e-6)`` of the red mass onto the green
  tokens, proportionally to their probabilities, where ``P_G`` is the green
  mass; ``r = 1e-6`` when ``P_G <= p_0``. TextSeal's code uses this linear
  strength (the paper's ``exp(k * P_G) - 1`` is commented out there).
"""

import torch

from vllm.config.watermarking import WatermarkPRFName
from vllm.v1.watermarking.gumbel import _validate_context_width
from vllm.v1.watermarking.prfs import WatermarkPRF, create_prf
from vllm.v1.watermarking.red_green import (
    RedGreenWatermarkDetector,
    _validate_green_list,
)
from vllm.v1.watermarking.watermarker import (
    RandomSampler,
    Watermarker,
    WatermarkSample,
)

# Rows are reweighted in chunks of at most this many logits, which bounds the
# transient memory of the [rows, vocab] uniforms, sort and float64 cumulative
# sums.
_MAX_CHUNK_ELEMENTS = 1 << 23

# TextSeal's MorphMark strength floor and ceiling.
MORPHMARK_EPS = 1e-6


def dipmark_permutation(uniforms: torch.Tensor) -> torch.Tensor:
    """DiPMark's order of the vocabulary: the token ids [rows, vocab] by
    descending PRF value, as TextSeal's ``torch.argsort(rs, descending=True)``,
    so the green tokens (``u < gamma``) come last. Ties, which the 24-bit
    uniforms make rare, go to the smaller token id (stable sort), the same on
    every device."""
    return torch.sort(uniforms, dim=-1, descending=True, stable=True).indices


def dipmark_reweight(
    probs: torch.Tensor, permutation: torch.Tensor, alpha: float
) -> torch.Tensor:
    """DiPMark's alpha-reweighting of ``probs`` [rows, vocab] for the token
    order ``permutation`` [rows, vocab], in float64 and vocabulary order.

    With ``F`` the cumulative distribution in permutation order, the
    reweighted cumulative distribution is ``max(F - alpha, 0) + max(F - (1 -
    alpha), 0)``: the mass in ``[0, alpha]`` is removed and the mass in ``[1 -
    alpha, 1]`` doubled. This is TextSeal's ``dip_reweight_probs`` (smooth
    boundary masks, renormalized) in closed form."""
    shuffled = probs.gather(-1, permutation).double()
    upper = shuffled.cumsum(dim=-1)
    upper = upper / upper[..., -1:]
    lower = torch.nn.functional.pad(upper[..., :-1], (1, 0))

    def ramp(cdf: torch.Tensor) -> torch.Tensor:
        return (cdf - alpha).clamp_min(0) + (cdf - (1 - alpha)).clamp_min(0)

    # ramp is nondecreasing, so every mass is >= 0; they telescope to 1.
    reweighted = ramp(upper) - ramp(lower)
    return torch.empty_like(reweighted).scatter_(-1, permutation, reweighted)


def morphmark_strength(green_mass: torch.Tensor, k: float, p_0: float) -> torch.Tensor:
    """MorphMark's adaptive strength ``r`` for the green mass ``P_G``, as
    TextSeal computes it: ``k * P_G`` when ``P_G > p_0``, else ``1e-6``, at
    most ``1 - 1e-6``."""
    strength = torch.where(
        green_mass > p_0,
        k * green_mass,
        torch.full_like(green_mass, MORPHMARK_EPS),
    )
    return strength.clamp(max=1 - MORPHMARK_EPS)


def morphmark_logits(
    logits: torch.Tensor, green: torch.Tensor, k: float, p_0: float
) -> torch.Tensor:
    """MorphMark's reweighting of ``logits`` [rows, vocab] for the green list
    ``green`` [rows, vocab], as float32 log-probabilities.

    TextSeal multiplies the green probabilities by ``1 + r (1 - P_G) / P_G``
    and the red ones by ``1 - r``. Equivalently the green group's mass becomes
    ``P_G + r (1 - P_G)`` and the red group's ``(1 - r)(1 - P_G)``, each split
    in proportion to the model's probabilities; this computes it in that form,
    from log-sum-exps, so a vanishing ``P_G`` cannot overflow. A row whose
    support is all green or all red keeps its law (TextSeal's formula is the
    identity when ``P_G = 1`` and undefined, 0/0, when ``P_G = 0``).
    """
    logits = logits.float()
    neg_inf = torch.tensor(-torch.inf, device=logits.device)
    lse = torch.logsumexp(logits, dim=-1, keepdim=True)
    lse_green = torch.logsumexp(
        torch.where(green, logits, neg_inf), dim=-1, keepdim=True
    )
    lse_red = torch.logsumexp(torch.where(green, neg_inf, logits), dim=-1, keepdim=True)
    green_mass = (lse_green - lse).double().exp()
    red_mass = (lse_red - lse).double().exp()
    strength = morphmark_strength(green_mass, k, p_0)
    # The groups' new log-masses; each token keeps its log-share of its group.
    log_green_mass = torch.log(green_mass + strength * red_mass).float()
    log_red_factor = torch.log1p(-strength).float()
    reweighted = torch.where(
        green,
        (logits - lse_green) + log_green_mass,
        (logits - lse) + log_red_factor,
    )
    mixed = torch.isfinite(lse_green) & torch.isfinite(lse_red)
    return torch.where(mixed, reweighted, logits)


class _GreenListReweightWatermarker(Watermarker):
    """A watermark that reweights the processed law with the green list of
    red-green (PRF value below ``gamma``) and samples the result with vLLM's
    random sampler, as ``RedGreenWatermarker`` does with its bias."""

    algorithm_name: str

    def __init__(
        self,
        key: int,
        context_width: int,
        prf: WatermarkPRF | WatermarkPRFName,
        gamma: float,
    ) -> None:
        prf = create_prf(prf, key) if isinstance(prf, str) else prf
        _validate_context_width(context_width)
        _validate_green_list(gamma)
        self.prf = prf
        self._context_width = context_width
        self.gamma = gamma

    @property
    def context_width(self) -> int:
        return self._context_width

    def uniforms(self, contexts: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """Each context's PRF value of every token: [num_rows, vocab_size]."""
        vocabulary = torch.arange(vocab_size, device=contexts.device)
        return self.prf.uniform(contexts, vocabulary)

    def green_list(self, contexts: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """Boolean mask of each context's green tokens, the tokens the detector
        counts: [num_rows, vocab_size]."""
        return self.uniforms(contexts, vocab_size) < self.gamma

    def _reweight(self, logits: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor:
        """Watermarked float32 logits of float32 ``logits`` [rows, vocab] given
        each row's PRF values."""
        raise NotImplementedError

    def watermark_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        skip_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return float32 logits of the watermarked law for every row not in
        ``skip_mask``; skipped rows keep their logits."""
        logits = logits.float()
        num_rows, vocab_size = logits.shape
        chunk = max(1, _MAX_CHUNK_ELEMENTS // max(vocab_size, 1))
        output = torch.empty_like(logits)
        for start in range(0, num_rows, chunk):
            end = min(start + chunk, num_rows)
            output[start:end] = self._reweight(
                logits[start:end], self.uniforms(contexts[start:end], vocab_size)
            )
        if skip_mask is not None:
            output = torch.where(skip_mask.unsqueeze(-1), logits, output)
        return output

    def verification_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        draft_tokens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        skip_mask: torch.Tensor,
    ) -> torch.Tensor:
        # As for red-green, the watermark is a change of distribution:
        # verifying against it makes every emitted token follow it.
        return self.watermark_logits(logits, contexts, skip_mask)

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
    ) -> WatermarkSample:
        if random_sampler is None:
            raise ValueError(
                f"{self.algorithm_name} watermarking requires a random sampler"
            )
        token_ids = random_sampler(self.watermark_logits(logits, contexts, skip_mask))
        # The logits before the watermark, as for red-green, so processed
        # logprobs do not reveal the green list.
        return WatermarkSample(token_ids, logits)

    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> WatermarkSample:
        raise ValueError(
            f"{self.algorithm_name} watermarking requires a random sampler"
        )


class DiPMarkWatermarker(_GreenListReweightWatermarker):
    """DiPMark of Wu et al. (2023), https://arxiv.org/abs/2310.07710, as in
    Meta's TextSeal (``DipmarkGenerator``).

    The key and context order the vocabulary by descending PRF value, and the
    processed law (after temperature and top-k/top-p, as in TextSeal, which
    reweights after top-p) is alpha-reweighted in that order
    (``dipmark_reweight``). The reweighting is unbiased over the key: averaged
    over the permutations it returns the model's law. ``gamma`` does not change
    generation; it is the detector's green-list fraction: the green tokens
    (PRF value below ``gamma``) are the last ones of the permutation, where the
    reweighting moves the mass.
    """

    algorithm_name = "DiPMark"

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
        alpha: float = 0.2,
        gamma: float = 0.5,
    ) -> None:
        if not 0 < alpha <= 0.5:
            raise ValueError("DiPMark's alpha must be in (0, 0.5]")
        super().__init__(key, context_width, prf, gamma)
        self.alpha = alpha

    def _reweight(self, logits: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        reweighted = dipmark_reweight(probs, dipmark_permutation(uniforms), self.alpha)
        return reweighted.log().float()


class MorphMarkWatermarker(_GreenListReweightWatermarker):
    """MorphMark of Wang et al. (2025), https://arxiv.org/abs/2505.11541, as in
    Meta's TextSeal (``MorphMarkGenerator``).

    The green list holds the tokens whose PRF value is below ``gamma``. The
    processed law's green mass ``P_G`` sets the strength ``r = k * P_G`` (``1e-6``
    when ``P_G <= p_0``, at most ``1 - 1e-6``), and a fraction ``r`` of the red
    mass moves to the green tokens (``morphmark_logits``). TextSeal computes
    ``P_G`` on the temperature law and applies top-p after the reweighting;
    here, as for every vLLM watermark, ``P_G`` and the reweighting are on the
    law after temperature, min-p, top-k and top-p. Without truncation the two
    coincide.
    """

    algorithm_name = "MorphMark"

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
        k: float = 1.30,
        p_0: float = 0.15,
        gamma: float = 0.5,
    ) -> None:
        if k <= 0:
            raise ValueError("MorphMark's k must be positive")
        if not 0 <= p_0 < 1:
            raise ValueError("MorphMark's p_0 must be in [0, 1)")
        super().__init__(key, context_width, prf, gamma)
        self.k = k
        self.p_0 = p_0

    def _reweight(self, logits: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor:
        return morphmark_logits(logits, uniforms < self.gamma, self.k, self.p_0)


class DiPMarkWatermarkDetector(RedGreenWatermarkDetector):
    """Detect DiPMark, as TextSeal does, with the green-list binomial test of
    ``RedGreenWatermarkDetector``: a token is green when its PRF value is below
    ``gamma``, the last ``gamma`` share of the vocabulary in DiPMark's order.
    ``gamma`` defaults to TextSeal's 0.5."""

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        p_value_threshold: float = 0.01,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
        deduplicate_contexts: bool = True,
        gamma: float = 0.5,
    ) -> None:
        super().__init__(
            key, context_width, p_value_threshold, prf, deduplicate_contexts, gamma
        )


class MorphMarkWatermarkDetector(RedGreenWatermarkDetector):
    """Detect MorphMark, as TextSeal does, with the green-list binomial test of
    ``RedGreenWatermarkDetector`` on MorphMark's green list. ``gamma`` defaults
    to TextSeal's 0.5."""

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        p_value_threshold: float = 0.01,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
        deduplicate_contexts: bool = True,
        gamma: float = 0.5,
    ) -> None:
        super().__init__(
            key, context_width, p_value_threshold, prf, deduplicate_contexts, gamma
        )
