# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Entropy-weighted and localized watermark detection of TextSeal.

From "TextSeal: A Localized LLM Watermark for Provenance & Distillation
Protection" (arXiv:2605.12456). Adapted from the reference implementation,
textseal/watermarking/localized_detection.py in
https://github.com/facebookresearch/textseal (Copyright (c) Meta Platforms,
Inc. and affiliates; Apache-2.0).

The scores are per-token detection scores with mean 1 and variance
``base_variance`` without the watermark, such as dual-key Gumbel-max's fused
scores. Entropy weighting (Eq. 4) gives low-entropy tokens, which carry little
watermark, less weight; the weighted sum is tested against a moment-matched
Gamma distribution (Eq. 5-6). Localized detection (Sec. 3.3) also tests the
best dyadic window and a greedy union of windows, each Bonferroni-corrected,
and corrects the smallest p-value for the k tests taken:

    log10 p = min(log10 p_global, log10 p_single, log10 p_multi) + log10(k)

with k = 3, or 4 with entropy weights, since the global test then runs both
weighted and unweighted.
"""

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from vllm.v1.watermarking.detector import WatermarkDetection

# Entropy weight w = W_FLOOR + W_SCALE * normalized entropy (Eq. 4).
W_FLOOR, W_SCALE = 0.1, 0.9
# Normalization range when all entropies of a text are (nearly) equal.
FLAT_ENTROPY_RANGE = (0.0, 5.0)
# Moving average that labels the watermarked tokens of a detected text.
SMOOTHER_WINDOW = 20
SMOOTHER_THRESHOLD = 1.2


@dataclass(frozen=True)
class LocalizedWatermarkDetection(WatermarkDetection):
    """Result of localized detection; ``score`` is ``-log10(p_value)``.

    ``winner`` names the test with the smallest p-value: ``global_weighted``,
    ``global_unweighted``, ``localized_single`` or ``localized_multi``. When a
    localized test wins, ``regions`` lists the spans it found as
    ``(start, end, p_value)`` token positions, ``end`` exclusive. When the text
    is watermarked, ``token_labels`` marks with 1 the tokens whose moving
    average score is high.
    """

    global_p_value: float
    localized_p_value: float
    winner: str
    regions: list[tuple[int, int, float]]
    token_labels: list[int]


def entropy_weights(entropies: np.ndarray) -> np.ndarray:
    """Eq. 4: min-max normalize the entropies within the text, then map them
    to ``W_FLOOR + W_SCALE * x``."""
    e = np.asarray(entropies, dtype=np.float64)
    if e.size == 0:
        return e
    e_min, e_max = float(e.min()), float(e.max())
    if e_max - e_min < 1e-6:
        e_min, e_max = FLAT_ENTROPY_RANGE
    ratio = np.clip((e - e_min) / (e_max - e_min), 0.0, 1.0)
    return W_FLOOR + W_SCALE * ratio


def _gammaincc(shape: float, x: float) -> float:
    return torch.special.gammaincc(
        torch.tensor(shape, dtype=torch.float64), torch.tensor(x, dtype=torch.float64)
    ).item()


def _gammaln(x: int) -> float:
    return math.lgamma(x) if x > 0 else math.inf


def unweighted_gamma_p(
    score_sum: float, n: int, base_var: float, eps: float = 1e-300
) -> float:
    """P(sum >= score_sum) without the watermark, where the sum of ``n``
    scores is Gamma(n / base_var, base_var)."""
    if n <= 0:
        return 1.0
    return max(_gammaincc(n / base_var, score_sum / base_var), eps)


def weighted_gamma_p(
    scores: np.ndarray, weights: np.ndarray, base_var: float, eps: float = 1e-300
) -> float:
    """Moment-matched Gamma p-value of ``sum(weights * scores)``, whose mean is
    ``sum(weights)`` and variance ``base_var * sum(weights**2)`` without the
    watermark (Eq. 5-6)."""
    if len(scores) == 0:
        return 1.0
    w_sum = float(np.sum(weights * scores))
    mu = float(np.sum(weights))
    var = float(np.sum(weights**2)) * base_var
    if var <= 1e-10 or mu <= 0:
        return 1.0
    return max(_gammaincc(mu**2 / var, w_sum / (var / mu)), eps)


def _dyadic_windows(n: int, min_length: int) -> list[tuple[int, int]]:
    """Candidate windows: lengths from ``min_length``, doubling, with stride
    half the length. The search and the Bonferroni count share them."""
    out: list[tuple[int, int]] = []
    if n < min_length:
        return out
    max_power = int(math.floor(math.log2(n)))
    min_power = max(0, int(math.ceil(math.log2(min_length))))
    for power in range(min_power, max_power + 1):
        length = 1 << power
        if length < min_length or length > n:
            continue
        stride = max(1, length // 2)
        for start in range(0, n - length + 1, stride):
            out.append((start, start + length))
    return out


def _count_windows(n: int, min_length: int) -> int:
    return max(1, len(_dyadic_windows(n, min_length)))


def _best_window(
    scores: np.ndarray, min_length: int, base_var: float
) -> tuple[int, int, float]:
    """The dyadic window with the highest z-score of its unweighted sum."""
    n = len(scores)
    windows = _dyadic_windows(n, min_length)
    if not windows:
        return 0, n, float("-inf")
    P = np.zeros(n + 1, dtype=np.float64)
    np.cumsum(scores, out=P[1:])
    best = (0, n, float("-inf"))
    for s, e in windows:
        z = (P[e] - P[s] - (e - s)) / math.sqrt((e - s) * base_var)
        if z > best[2]:
            best = (s, e, z)
    return best


def _dedup_window(
    dedup_keys: Sequence[Hashable] | None,
    scores: np.ndarray,
    weights: np.ndarray | None,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Keep the first position of each key in ``[start, end)``."""
    if dedup_keys is None:
        return scores[start:end], (weights[start:end] if weights is not None else None)
    keep = []
    seen = set()
    for pos in range(start, end):
        key = dedup_keys[pos]
        if key in seen:
            continue
        seen.add(key)
        keep.append(pos)
    keep_array = np.asarray(keep, dtype=np.int64)
    return scores[keep_array], (weights[keep_array] if weights is not None else None)


def _verified_pvalue(
    dedup_keys: Sequence[Hashable] | None,
    scores: np.ndarray,
    weights: np.ndarray | None,
    start: int,
    end: int,
    base_var: float,
    num_tests: int,
) -> float:
    """Bonferroni-corrected p-value of the deduplicated window."""
    s, w = _dedup_window(dedup_keys, scores, weights, start, end)
    n_eff = len(s)
    if n_eff == 0:
        return 1.0
    if w is not None and len(w) == n_eff:
        raw = weighted_gamma_p(s, w, base_var)
    else:
        raw = unweighted_gamma_p(float(np.sum(s)), n_eff, base_var)
    return min(1.0, raw * max(num_tests, 1))


def _greedy_multi_regions(
    scores: np.ndarray,
    weights: np.ndarray | None,
    min_length: int,
    base_var: float,
    max_regions: int,
    dedup_keys: Sequence[Hashable] | None,
    p_value_threshold: float,
    num_tests: int,
) -> list[tuple[int, int, float]]:
    """Repeatedly take the best window of the residual scores, accept it while
    its p-value is below ``p_value_threshold``, and mask it to the null mean.
    Windows are searched on unweighted scores and verified with the weights."""
    scores_work = scores.copy()
    found: list[tuple[int, int, float]] = []
    masks: list[tuple[int, int]] = []
    for _ in range(max_regions):
        s, e, _ = _best_window(scores_work, min_length, base_var)
        # Windows at different scales can partially overlap a masked one.
        for m_s, m_e in masks:
            if s < m_e and e > m_s:
                left, right = m_s - s, e - m_e
                if left >= right and left >= min_length:
                    e = m_s
                elif right >= min_length:
                    s = m_e
                else:
                    s = e = 0
                    break
        if e <= s or (e - s) < min_length:
            break
        p = _verified_pvalue(dedup_keys, scores, weights, s, e, base_var, num_tests)
        if p >= p_value_threshold:
            break
        found.append((s, e, p))
        scores_work[s:e] = 1.0
        masks.append((s, e))
    return found


def _multi_region_corrected_log_p(
    regions: list[tuple[int, int, float]],
    scores: np.ndarray,
    weights: np.ndarray | None,
    dedup_keys: Sequence[Hashable] | None,
    base_var: float,
    n: int,
    min_length: int,
    y_max: int,
) -> tuple[float, int, list[tuple[int, int, float]]]:
    """Test the union of the y most significant regions for every y, each
    corrected by C(M, y) * y_max for the M windows; return the best log10
    p-value, its y and its regions."""
    if not regions:
        return 0.0, 0, []
    regions = sorted(regions, key=lambda r: r[2])
    M = _count_windows(n, min_length)
    log_ymax = math.log10(max(y_max, 1))
    best_log_p, best_y, best_regs = 0.0, 0, []
    doc_seen: set[Hashable] = set()
    union_scores: list[float] = []
    union_weights: list[float] = []
    for y, (s, e, _) in enumerate(regions[:y_max], start=1):
        for pos in range(s, e):
            if dedup_keys is not None:
                key = dedup_keys[pos]
                if key in doc_seen:
                    continue
                doc_seen.add(key)
            union_scores.append(scores[pos])
            if weights is not None:
                union_weights.append(weights[pos])
        n_eff = len(union_scores)
        if n_eff == 0:
            continue
        if weights is not None and len(union_weights) == n_eff:
            raw = weighted_gamma_p(
                np.asarray(union_scores), np.asarray(union_weights), base_var
            )
        else:
            raw = unweighted_gamma_p(float(sum(union_scores)), n_eff, base_var)
        log_cmy = _gammaln(M + 1) - _gammaln(y + 1) - _gammaln(M - y + 1)
        log_cmy /= math.log(10)
        log_corrected = math.log10(max(raw, 1e-300)) + log_cmy + log_ymax
        if log_corrected < best_log_p or best_y == 0:
            best_log_p, best_y, best_regs = log_corrected, y, list(regions[:y])
    return best_log_p, best_y, best_regs


def _smooth_labels(
    scores: np.ndarray,
    weights: np.ndarray | None,
    window: int,
    threshold: float,
    min_weight_sum: float = 0.1,
) -> list[int]:
    """Label each position 1 when the (weighted) moving average of the scores
    exceeds ``threshold`` times the null mean."""
    n = len(scores)
    if n == 0:
        return []
    half = window // 2
    if weights is None or len(weights) != n:
        padded = np.pad(scores, (half, half), mode="edge")
        kernel = np.ones(window) / window
        smoothed = np.convolve(padded, kernel, mode="valid")[:n]
    else:
        smoothed = np.empty(n)
        for t in range(n):
            a, b = max(0, t - half), min(n, t + half + 1)
            w_sum = float(np.sum(weights[a:b]))
            if w_sum < min_weight_sum:
                smoothed[t] = 1.0
            else:
                smoothed[t] = float(np.sum(scores[a:b] * weights[a:b])) / w_sum
    return (smoothed > threshold).astype(int).tolist()


def adaptive_ensemble_detection(
    scores: Sequence[float] | np.ndarray,
    *,
    p_value_threshold: float = 0.01,
    min_length: int = 50,
    max_regions: int = 5,
    dedup_keys: Sequence[Hashable] | None = None,
    weights: Sequence[float] | np.ndarray | None = None,
    base_variance: float = 1.0,
) -> LocalizedWatermarkDetection:
    """TextSeal's adaptive ensemble (Sec. 3.3) over per-position scores.

    Args:
        scores: per-position scores, with mean 1 without the watermark.
        p_value_threshold: significance level of the result and of accepting
            a region.
        min_length: the shortest window tried.
        max_regions: the most regions the greedy multi-region test unites.
        dedup_keys: one key per position; positions that share a key are
            scored once per test. None scores every position.
        weights: per-position entropy weights (Eq. 4), or None.
        base_variance: variance of a score without the watermark.

    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    n = len(scores_arr)
    if n == 0:
        return LocalizedWatermarkDetection(
            0.0, 1.0, 0, False, 1.0, 1.0, "global", [], []
        )
    if dedup_keys is not None and len(dedup_keys) != n:
        raise ValueError("dedup_keys must hold one key per score")

    weights_arr = None
    if weights is not None and len(weights) > 0:
        weights_arr = np.asarray(weights, dtype=np.float64)
        if len(weights_arr) != n:
            raise ValueError("weights must hold one weight per score")
    use_weighted = weights_arr is not None

    # Global test on the deduplicated scores.
    g_scores, g_weights = _dedup_window(dedup_keys, scores_arr, weights_arr, 0, n)
    global_unweighted = unweighted_gamma_p(
        float(np.sum(g_scores)), len(g_scores), base_variance
    )
    if g_weights is not None:
        global_weighted = weighted_gamma_p(g_scores, g_weights, base_variance)
        global_pvalue = min(global_weighted, global_unweighted)
    else:
        global_weighted = global_unweighted
        global_pvalue = global_unweighted

    # Best single window.
    M = _count_windows(n, min_length)
    s, e, _ = _best_window(scores_arr, min_length, base_variance)
    if e > s and (e - s) >= min_length:
        single_pvalue = _verified_pvalue(
            dedup_keys, scores_arr, weights_arr, s, e, base_variance, M
        )
        single_regions = [(s, e, single_pvalue)]
    else:
        single_pvalue, single_regions = 1.0, []

    # Greedy multi-region.
    raw_regions = _greedy_multi_regions(
        scores_arr,
        weights_arr,
        min_length,
        base_variance,
        max_regions,
        dedup_keys,
        p_value_threshold,
        M,
    )
    if len(raw_regions) > 1:
        multi_log_p, _, multi_regs = _multi_region_corrected_log_p(
            raw_regions,
            scores_arr,
            weights_arr,
            dedup_keys,
            base_variance,
            n,
            min_length,
            max_regions,
        )
    else:
        multi_log_p, multi_regs = 0.0, []
    multi_pvalue = min(1.0, 10**multi_log_p) if multi_log_p < 0 else 1.0

    localized_pvalue = min(single_pvalue, multi_pvalue)
    localized_regs = single_regions if single_pvalue <= multi_pvalue else multi_regs

    k = 4 if use_weighted else 3
    candidates_log = {
        "global_weighted": math.log10(max(global_weighted, 1e-300)),
        "global_unweighted": math.log10(max(global_unweighted, 1e-300)),
        "localized_single": math.log10(max(single_pvalue, 1e-300)),
        "localized_multi": math.log10(max(multi_pvalue, 1e-300)),
    }
    if not use_weighted:
        del candidates_log["global_weighted"]
    final_log_pvalue = min(0.0, min(candidates_log.values()) + math.log10(k))
    final_pvalue = 10**final_log_pvalue
    winner = min(candidates_log, key=candidates_log.__getitem__)

    detection = final_pvalue < p_value_threshold
    if detection:
        labels = _smooth_labels(
            scores_arr, weights_arr, SMOOTHER_WINDOW, SMOOTHER_THRESHOLD
        )
    else:
        labels = [0] * n
    return LocalizedWatermarkDetection(
        score=-final_log_pvalue,
        p_value=final_pvalue,
        num_scored_tokens=n,
        is_watermarked=detection,
        global_p_value=global_pvalue,
        localized_p_value=localized_pvalue,
        winner=winner,
        regions=[] if winner.startswith("global") else localized_regs,
        token_labels=labels,
    )
