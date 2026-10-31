# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lookahead Gumbel-max watermark for speculative decoding.

Without speculation it samples like Gumbel-max. In speculative verification
the rows of each request's drafted path, its drafted rows and then its bonus
row, are decided together from the bonus row up: every row races on
``logit + gumbel`` of its candidates, except that a row's drafted child races
on ``aggregate(child gumbel, race value of the row below)``. The rows become
point masses at their winners, so rejection sampling emits the winners along
the path up to the first rejected draft. Each aggregator maps two independent
standard Gumbels to a standard Gumbel, so every row's race stays an exact
Gumbel-max sample of its distribution as long as the drafts do not depend on
the key. Detection is Gumbel-max detection.

The rule needs only that every row races on standard Gumbels independent across
its candidates and of the rows below, so it applies unchanged to the
Gumbel-max watermarks that draw their Gumbels differently: TextSeal's dual-key
Gumbel-max, whose rows race on key A's or, with probability ``alpha``, key B's
Gumbels, the Gaussian copula watermark, whose Gumbels mix the keyed uniform
with fresh Gaussian noise, and TextSeal2, the copula on the virtual key of two
keys.

The ``optpath`` aggregator (lookahead Gumbel-max only) races the drafted child
on the realized detection score of what the rows emit at and below it,
``X = s(G_child) + R``, with ``s(g) = -log(1 - exp(-exp(-g)))`` the Gumbel-max
score of a token, standardized to a Gumbel with the law of ``X`` given the path
emitted below: still a standard Gumbel given that path, so every row stays exact.
It keeps per-request state up the rows, so it runs in torch, not in the kernels.
"""

import functools
import math

import numpy as np
import torch

from vllm.config.watermarking import WatermarkPRFName, derive_watermark_key
from vllm.v1.watermarking.gaussian_copula import (
    DEFAULT_CORRELATION,
    GaussianCopulaWatermarker,
)
from vllm.v1.watermarking.gaussian_copula import gumbels as copula_gumbels
from vllm.v1.watermarking.gumbel import DualKeyGumbelWatermarker, GumbelWatermarker
from vllm.v1.watermarking.prfs import PhiloxPRF, WatermarkPRF, create_prf
from vllm.v1.watermarking.textseal2 import (
    DEFAULT_CORRELATION as TEXTSEAL2_DEFAULT_CORRELATION,
)
from vllm.v1.watermarking.textseal2 import TextSeal2Watermarker
from vllm.v1.watermarking.textseal2 import race_gumbels as textseal2_gumbels
from vllm.v1.watermarking.watermarker import RandomSampler, WatermarkSample

AGGREGATORS = ("pi1", "pi2", "max", "tsum", "min", "lin", "expsum")
# Aggregators that carry state up a request's rows instead of combining two
# Gumbels; torch only.
PATH_AGGREGATORS = ("optpath",)
TABULATED_AGGREGATORS = ("tsum", "min", "lin")
_REARRANGED_GRID = 65537
_REARRANGED_SAMPLES = 2_000_000
# Seed the fresh randomness of verification, which receives no per-request
# seeds. Public: the randomness needs no secrecy.
_KEY_COIN_SEED = 0x4C4B41434F494E
_COPULA_NOISE_SEED = 0x4C4B41434F50
_TEXTSEAL2_NOISE_SEED = 0x4C4B54533200


@functools.cache
def tsum_table() -> tuple[float, float, np.ndarray]:
    """``T = F1^{-1} o F2`` on a uniform grid as (lo, hi, values), F2 the CDF of
    the sum of two independent standard Gumbels, so ``T(A + B)`` is a standard
    Gumbel."""
    lo, hi, n = -20.0, 60.0, 16001
    xs = np.linspace(lo, hi, n)
    step = xs[1] - xs[0]
    density = np.exp(-xs - np.exp(-xs))
    cdf = np.empty(n)
    for i in range(0, n, 256):
        chunk = xs[i : i + 256, None]
        cdf[i : i + 256] = (np.exp(-np.exp(-(chunk - xs[None, :]))) * density).sum(
            1
        ) * step
    cdf = np.maximum.accumulate(np.clip(cdf, 1e-12, 1.0 - 1e-12))
    return lo, hi, -np.log(-np.log(cdf))


@functools.cache
def rearranged_table(name: str, lam: float) -> tuple[float, float, np.ndarray]:
    """``F1^{-1} o F_psi`` on a uniform grid as (lo, hi, values), for
    ``psi = min(a, b)`` or ``a + lam * b`` on independent standard Gumbels,
    with ``F_psi`` estimated by Monte Carlo."""
    a, b = np.random.default_rng(12345).gumbel(size=(2, _REARRANGED_SAMPLES))
    psi = np.sort(np.minimum(a, b) if name == "min" else a + lam * b)
    quantiles = (np.arange(psi.size) + 0.5) / psi.size
    grid = np.linspace(psi[0], psi[-1], _REARRANGED_GRID)
    uniforms = np.clip(np.interp(grid, psi, quantiles), 1e-12, 1.0 - 1e-12)
    return float(psi[0]), float(psi[-1]), -np.log(-np.log(uniforms))


class LookaheadGumbelWatermarker(GumbelWatermarker):
    """Gumbel-max that decides each speculative path backwards, see the module
    docstring."""

    # Verification picks every emitted token from the target's races, and the
    # rule is exact only for drafts that do not depend on the key.
    verifies_unwatermarked_drafts = True
    # Whether the races' Gumbels are Philox Gumbels of ``prf``, which the CUDA
    # kernels draw themselves; subclasses with other noise race in torch.
    races_on_prf_gumbels = True

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
        aggregator: str = "tsum",
        aggregator_lambda: float = 1.0,
    ) -> None:
        super().__init__(key, context_width, prf)
        if aggregator not in AGGREGATORS + PATH_AGGREGATORS:
            raise ValueError(
                f"aggregator must be one of {AGGREGATORS + PATH_AGGREGATORS}"
            )
        if aggregator_lambda <= 0:
            raise ValueError("aggregator_lambda must be positive")
        self.aggregator = aggregator
        self.aggregator_lambda = aggregator_lambda
        self._tables: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    def table(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """The aggregator's float64 table and (lo, hi, 1 / grid step); a dummy
        for aggregators without one."""
        if device not in self._tables:
            if self.aggregator == "tsum":
                lo, hi, values = tsum_table()
            elif self.aggregator in TABULATED_AGGREGATORS:
                lam = self.aggregator_lambda if self.aggregator == "lin" else 0.0
                lo, hi, values = rearranged_table(self.aggregator, lam)
            else:
                lo, hi, values = 0.0, 1.0, np.zeros(2)
            inv_step = (values.size - 1) / (hi - lo)
            self._tables[device] = (
                torch.from_numpy(np.ascontiguousarray(values)).to(device),
                torch.tensor([lo, hi, inv_step], dtype=torch.float64, device=device),
            )
        return self._tables[device]

    def aggregate(self, own: torch.Tensor, child: torch.Tensor) -> torch.Tensor:
        """The drafted child's race statistic from its own Gumbel and the race
        value of the row below, in float32 like the kernel."""
        if self.aggregator == "pi1":
            return own
        if self.aggregator == "pi2":
            return child
        if self.aggregator == "max":
            return torch.maximum(own, child) - math.log(2.0)
        if self.aggregator == "expsum":
            s = torch.exp(-own.double()) + torch.exp(-child.double())
            series = s * s / 2 - s * s * s / 3 + s * s * s * s / 4
            argument = torch.where(s < 1e-3, series, s - torch.log(1.0 + s))
            return (-torch.log(argument)).float()
        if self.aggregator == "tsum":
            psi = own + child
        elif self.aggregator == "min":
            psi = torch.minimum(own, child)
        else:
            psi = own + self.aggregator_lambda * child
        table, params = self.table(own.device)
        lo, hi, inv_step = params.tolist()
        t = (psi.double().clamp(lo, hi) - lo) * inv_step
        position = t.floor().clamp(0, table.numel() - 2)
        index = position.long()
        y0 = table[index]
        return (y0 + (t - position) * (table[index + 1] - y0)).float()

    @staticmethod
    def children(
        draft_tokens: torch.Tensor, cu_num_logits: torch.Tensor
    ) -> torch.Tensor:
        """Each row's drafted child, -1 at the bonus rows."""
        bonus = torch.zeros(
            draft_tokens.shape[0], dtype=torch.bool, device=draft_tokens.device
        )
        bonus.index_fill_(0, cu_num_logits[1:].long() - 1, True)
        return torch.where(bonus, -1, draft_tokens.long())

    def verification_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        draft_tokens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        skip_mask: torch.Tensor,
    ) -> torch.Tensor:
        children = self.children(draft_tokens, cu_num_logits)
        if (
            self.races_on_prf_gumbels
            and self.aggregator in AGGREGATORS
            and type(self.prf) is PhiloxPRF
            and logits.device.type == "cuda"
        ):
            from vllm.v1.worker.gpu.sample.lookahead_watermark import (
                philox_lookahead_tokens,
                point_mass_logits,
            )

            table, params = self.table(logits.device)
            chosen = philox_lookahead_tokens(
                logits,
                contexts,
                children,
                cu_num_logits,
                skip_mask,
                self.prf.key,
                self.aggregator,
                table,
                params,
                self.aggregator_lambda,
            )
            return point_mass_logits(logits, chosen, skip_mask)
        chosen = self.lookahead_tokens(
            logits, contexts, children, cu_num_logits, skip_mask
        )
        point_masses = torch.full_like(logits, float("-inf")).scatter_(
            1, chosen.unsqueeze(1), 0.0
        )
        return torch.where(skip_mask.unsqueeze(1), logits, point_masses)

    def lookahead_tokens(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        children: torch.Tensor,
        cu_num_logits: torch.Tensor,
        skip_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Torch reference of ``philox_lookahead_tokens``, racing on
        ``race_gumbels``. Like the chain kernel it walks every request's rows
        from the bonus row up, all requests at once."""
        logits = logits.float()
        logits = logits.masked_fill(logits.isnan(), float("-inf"))
        num_rows, vocab_size = logits.shape
        device = logits.device
        if num_rows == 0:
            return torch.zeros(0, dtype=torch.int64, device=device)
        gumbels = self.race_gumbels(contexts, vocab_size)
        vocabulary = torch.arange(vocab_size, device=device)
        is_child = vocabulary == children.unsqueeze(-1)
        others, other_tokens = (
            (logits + gumbels).masked_fill(is_child, float("-inf")).max(dim=-1)
        )
        lse = torch.logsumexp(logits, dim=-1)
        optpath = (
            _OptPath(cu_num_logits.numel() - 1, device)
            if self.aggregator == "optpath"
            else None
        )
        if optpath is not None:
            # The best other token's own Gumbel and log-probability.
            other_logits = logits.gather(-1, other_tokens.unsqueeze(-1)).squeeze(-1)
            other_gumbels = gumbels.gather(-1, other_tokens.unsqueeze(-1)).squeeze(-1)
        safe = children.clamp(0, vocab_size - 1).unsqueeze(-1)
        child_gumbels = gumbels.gather(-1, safe).squeeze(-1)
        child_logits = logits.gather(-1, safe).squeeze(-1)

        # Row ``ends[r] - 1 - step`` of request r, or none past its first row.
        bounds = cu_num_logits.long()
        ends = bounds[1:]
        sizes = ends - bounds[:-1]
        steps = torch.arange(int(sizes.max()), device=device)
        exists = steps.unsqueeze(0) < sizes.unsqueeze(1)
        rows = torch.where(exists, ends.unsqueeze(1) - 1 - steps.unsqueeze(0), 0)
        in_support = (
            (children >= 0) & (children < vocab_size) & (child_logits > float("-inf"))
        )

        chosen = other_tokens.clone()
        value = torch.zeros(sizes.numel(), device=device)
        has_value = torch.zeros(sizes.numel(), dtype=torch.bool, device=device)
        for step in range(steps.numel()):
            row = rows[:, step]
            # The bonus row (step 0) explores no child.
            explores = exists[:, step] & in_support[row] & (step > 0)
            active = exists[:, step] & ~skip_mask[row]
            own = child_gumbels[row]
            link = active & has_value & explores
            if optpath is not None:
                if step > 0:
                    own = torch.where(link, optpath.combine(own).float(), own)
            else:
                own = torch.where(link, self.aggregate(own, value), own)
            child_race = own + child_logits[row]
            take_child = explores & (child_race > others[row])
            chosen[row[take_child]] = children[row[take_child]].long()
            race_value = torch.where(take_child, child_race, others[row]) - lse[row]
            if optpath is not None:
                child_logp = child_logits[row] - lse[row]
                optpath.observe(
                    step == 0,
                    link & take_child,
                    torch.where(take_child, child_gumbels[row], other_gumbels[row]),
                    torch.where(take_child, child_logp, other_logits[row] - lse[row]),
                    child_logp,
                )
            value = torch.where(active, race_value, value)
            has_value = active
        return chosen

    def race_gumbels(self, contexts: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """The standard Gumbels every row races its candidates on, independent
        across rows and candidates: [num_rows, vocab_size] float32."""
        vocabulary = torch.arange(vocab_size, device=contexts.device)
        uniforms = self.prf.uniform(contexts, vocabulary)
        return -torch.log(-torch.log(uniforms))


def _exp_score(g: torch.Tensor) -> torch.Tensor:
    """The Gumbel-max detection score of a token with race Gumbel ``g``: Exp(1)."""
    return -torch.log((-torch.expm1(-torch.exp(-g.double()))).clamp_min(1e-300))


def _interp_rows(q: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    """Row-wise linear interpolation of ``ys`` over increasing ``xs`` at ``q``
    ([B, n] each, or [n] for ``xs``/``ys``), clamped at the ends."""
    num_rows = q.shape[0]
    xs = (xs.expand(num_rows, -1) if xs.dim() == 1 else xs).contiguous()
    ys = (ys.expand(num_rows, -1) if ys.dim() == 1 else ys).contiguous()
    clamped = torch.maximum(torch.minimum(q, xs[:, -1:]), xs[:, :1]).contiguous()
    index = torch.searchsorted(xs, clamped, right=True).clamp(1, xs.shape[1] - 1)
    x0, x1 = xs.gather(1, index - 1), xs.gather(1, index)
    y0, y1 = ys.gather(1, index - 1), ys.gather(1, index)
    dx = x1 - x0
    t = torch.where(
        dx > 0, (clamped - x0) / torch.where(dx > 0, dx, torch.ones_like(dx)), 1.0
    )
    return torch.where(q < xs[:, :1], ys[:, :1], y0 + t * (y1 - y0))


def _cumtrapz(g: torch.Tensor, step: float) -> torch.Tensor:
    increments = 0.5 * (g[:, 1:] + g[:, :-1]) * step
    return torch.cat([torch.zeros_like(g[:, :1]), increments.cumsum(dim=1)], dim=1)


class _OptPath:
    """State of the ``optpath`` aggregator for a batch of requests, walked from
    each request's bonus row up (see the module docstring).

    ``R`` is the realized score at and below the last row raced, and ``r`` that
    score as an increasing function of the row's race value on the grid ``x``.
    Given the path emitted below, ``X = s(G_child) + R`` is an Exp(1) score plus
    an independent increasing function of a standard Gumbel, whose CDF is
    ``F_X(x) = exp(-x) * int_{-inf}^{x} exp(t) F_W(t) dt``; the drafted child
    races on ``F1^{-1}(F_X(X))``, a standard Gumbel given that path."""

    LO, HI, N = -20.0, 60.0, 4097

    def __init__(self, num_requests: int, device: torch.device) -> None:
        self.x = torch.linspace(
            self.LO, self.HI, self.N, dtype=torch.float64, device=device
        )
        self.h = (self.HI - self.LO) / (self.N - 1)
        self.ex = torch.exp(self.x)
        self.emx = torch.exp(-self.x)
        self.idx = torch.arange(self.N, dtype=torch.float64, device=device)
        self.r = torch.zeros(num_requests, self.N, dtype=torch.float64, device=device)
        self.R = torch.zeros(num_requests, dtype=torch.float64, device=device)
        self.Tv: torch.Tensor | None = None
        self.X: torch.Tensor | None = None

    def _transform(self) -> torch.Tensor:
        """``F1^{-1} o F_X`` on the grid, per request."""
        num = self.r.shape[0]
        r_inv = _interp_rows(self.x.expand(num, -1), self.r + 1e-9 * self.idx, self.x)
        cdf_w = torch.exp(-torch.exp(-r_inv))
        sf_w = -torch.expm1(-torch.exp(-r_inv))
        cdf_x = (self.emx * _cumtrapz(self.ex * cdf_w, self.h)).clamp(1e-300, 1 - 1e-16)
        sf_x = (self.emx * _cumtrapz(self.ex * sf_w, self.h)).clamp(1e-300, 1 - 1e-16)
        tv = torch.where(
            cdf_x < 0.5, -torch.log(-torch.log(cdf_x)), -torch.log(-torch.log1p(-sf_x))
        )
        return torch.cummax(tv, dim=1).values

    def combine(self, child_gumbels: torch.Tensor) -> torch.Tensor:
        """Every request's drafted-child statistic at this row (used where linked)."""
        self.Tv = self._transform()
        self.X = _exp_score(child_gumbels) + self.R
        return _interp_rows(self.X.unsqueeze(1), self.x, self.Tv).squeeze(1)

    def observe(
        self,
        leaf: bool,
        accepted: torch.Tensor,
        winner_gumbels: torch.Tensor,
        winner_logps: torch.Tensor,
        child_logps: torch.Tensor,
    ) -> None:
        """Update the state after the row's race: a rejected or bonus row emits
        only its winner, an accepted drafted child its score and the path below."""
        winner_scores = _exp_score(winner_gumbels)
        rho_plain = _exp_score(self.x.unsqueeze(0) - winner_logps.double().unsqueeze(1))
        if leaf:
            self.R = winner_scores
            self.r = torch.cummax(rho_plain, dim=1).values
            return
        assert self.Tv is not None and self.X is not None
        a = child_logps.double().exp().clamp(1e-12, 1.0 - 1e-12)
        shifted = self.x.unsqueeze(0) - torch.log(a).unsqueeze(1)
        t_inv = _interp_rows(shifted, self.Tv + 1e-12 * self.idx, self.x).clamp_min(0.0)
        self.R = torch.where(accepted, self.X, winner_scores)
        self.r = torch.cummax(
            torch.where(accepted.unsqueeze(1), t_inv, rho_plain), dim=1
        ).values


def _fresh_generator(
    generators: dict[torch.device, torch.Generator], device: torch.device, seed: int
) -> torch.Generator:
    generator = generators.get(device)
    if generator is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        generators[device] = generator
    return generator


class LookaheadDualKeyGumbelWatermarker(LookaheadGumbelWatermarker):
    """TextSeal's dual-key Gumbel-max (``dual_key_gumbel``) with lookahead
    verification.

    Without speculation it samples like ``dual_key_gumbel``: each token races on
    key A's Gumbels, or with probability ``alpha`` on key B's. In speculative
    verification every row flips that coin first and races on the chosen key's
    Gumbels, and the path is decided as in lookahead Gumbel-max. The coin is
    independent of both keys, so every row stays an exact sample and every
    emitted token, accepted draft or not, carries key B with probability
    ``alpha``. Detection is ``DualKeyGumbelWatermarkDetector``.

    Verification receives no per-request seeds, so its coins come from a
    generator the watermarker owns.
    """

    races_on_prf_gumbels = False

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRFName = "philox",
        alpha: float = 0.1,
        aggregator: str = "tsum",
        aggregator_lambda: float = 1.0,
    ) -> None:
        # Not exposed as draft and target watermarkers: drafts stay unwatermarked.
        self.sampling_watermarker = DualKeyGumbelWatermarker(
            key, context_width, prf, alpha
        )
        super().__init__(
            derive_watermark_key(key, b"key_a"),
            context_width,
            prf,
            aggregator,
            aggregator_lambda,
        )
        self.key_b_prf = create_prf(prf, derive_watermark_key(key, b"key_b"))
        self.alpha = alpha
        self._generators: dict[torch.device, torch.Generator] = {}

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
    ) -> WatermarkSample:
        return self.sampling_watermarker.sample(
            logits, contexts, random_sampler, skip_mask
        )

    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> WatermarkSample:
        return self.sampling_watermarker._sample_watermarked(logits, contexts)

    def race_gumbels(self, contexts: torch.Tensor, vocab_size: int) -> torch.Tensor:
        if self.alpha == 0:
            return super().race_gumbels(contexts, vocab_size)
        vocabulary = torch.arange(vocab_size, device=contexts.device)
        key_b = -torch.log(-torch.log(self.key_b_prf.uniform(contexts, vocabulary)))
        if self.alpha == 1:
            return key_b
        generator = _fresh_generator(self._generators, contexts.device, _KEY_COIN_SEED)
        use_key_b = (
            torch.rand(contexts.shape[0], device=contexts.device, generator=generator)
            < self.alpha
        )
        return torch.where(
            use_key_b.unsqueeze(-1), key_b, super().race_gumbels(contexts, vocab_size)
        )


class LookaheadGaussianCopulaWatermarker(LookaheadGumbelWatermarker):
    """The Gaussian copula watermark (``gaussian_copula``) with lookahead
    verification.

    Without speculation it samples like ``gaussian_copula``. In speculative
    verification every row races on the copula's Gumbels
    ``-log(-log Phi(z_u))`` with ``z_u = correlation * ndtri(U_u) + sqrt(1 -
    correlation**2) * N_u``, standard Gumbels independent across candidates
    and rows, and the path is decided as in lookahead Gumbel-max, so every row
    stays an exact sample. Rows whose context reaches before the output are
    verified without the watermark, as ``gaussian_copula`` samples them.
    Detection is ``GaussianCopulaWatermarkDetector``.

    Verification receives no per-request seeds, so its fresh noise ``N`` comes
    from a generator the watermarker owns.
    """

    races_on_prf_gumbels = False

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
        correlation: float = DEFAULT_CORRELATION,
        aggregator: str = "tsum",
        aggregator_lambda: float = 1.0,
    ) -> None:
        self.sampling_watermarker = GaussianCopulaWatermarker(
            key, context_width, prf, correlation
        )
        super().__init__(
            key,
            context_width,
            self.sampling_watermarker.prf,
            aggregator,
            aggregator_lambda,
        )
        self.correlation = correlation
        self._generators: dict[torch.device, torch.Generator] = {}

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
    ) -> WatermarkSample:
        return self.sampling_watermarker.sample(
            logits, contexts, random_sampler, skip_mask
        )

    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> WatermarkSample:
        return self.sampling_watermarker._sample_watermarked(logits, contexts)

    def verification_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        draft_tokens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        skip_mask: torch.Tensor,
    ) -> torch.Tensor:
        partial = (contexts < 0).any(dim=-1)
        return super().verification_logits(
            logits, contexts, draft_tokens, cu_num_logits, skip_mask | partial
        )

    def race_gumbels(self, contexts: torch.Tensor, vocab_size: int) -> torch.Tensor:
        vocabulary = torch.arange(vocab_size, device=contexts.device)
        uniforms = self.prf.uniform(contexts, vocabulary)
        generator = _fresh_generator(
            self._generators, contexts.device, _COPULA_NOISE_SEED
        )
        noise = torch.randn(uniforms.shape, device=uniforms.device, generator=generator)
        return copula_gumbels(uniforms, noise, self.correlation)


class LookaheadTextSeal2Watermarker(LookaheadGumbelWatermarker):
    """TextSeal2 (``textseal2``) with lookahead verification.

    Without speculation it samples like ``textseal2``. In speculative
    verification every row races on TextSeal2's Gumbels ``-log(-log Phi(z_u))``
    with ``z_u = correlation * G*_u + sqrt(1 - correlation**2) * N_u``, where
    ``G*_u`` is the virtual key of keys A and B. These are standard Gumbels
    independent across candidates and rows, so the path is decided as in
    lookahead Gumbel-max and every row stays an exact sample, for drafts that
    do not depend on the key. Rows whose context reaches before the output are
    verified without the watermark, as ``textseal2`` samples them. Detection is
    ``TextSeal2WatermarkDetector``. This is the rule of the reference's
    ``LookaheadTextSeal2Watermark``, an alternative to the paper's speculative
    rule (keyed drafts and a residual race), not a layer on top of it.

    A discarded row's race decides its parent's, so its keyed values must not
    be raced on again as fresh: ``context_width`` should be at least the number
    of drafts, so that a row of the next step at the same position differs from
    every discarded row in the token where the output left the drafted path.

    Verification receives no per-request seeds, so its fresh noise ``N`` comes
    from a generator the watermarker owns.
    """

    races_on_prf_gumbels = False

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRFName = "philox",
        correlation: float = TEXTSEAL2_DEFAULT_CORRELATION,
        aggregator: str = "tsum",
        aggregator_lambda: float = 1.0,
    ) -> None:
        self.sampling_watermarker = TextSeal2Watermarker(
            key, context_width, prf, correlation
        )
        super().__init__(
            key,
            context_width,
            self.sampling_watermarker.prf,
            aggregator,
            aggregator_lambda,
        )
        self.key_b_prf = self.sampling_watermarker.key_b_prf
        self.correlation = correlation
        self._generators: dict[torch.device, torch.Generator] = {}

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
    ) -> WatermarkSample:
        return self.sampling_watermarker.sample(
            logits, contexts, random_sampler, skip_mask
        )

    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> WatermarkSample:
        return self.sampling_watermarker._sample_watermarked(logits, contexts)

    def verification_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        draft_tokens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        skip_mask: torch.Tensor,
    ) -> torch.Tensor:
        partial = (contexts < 0).any(dim=-1)
        return super().verification_logits(
            logits, contexts, draft_tokens, cu_num_logits, skip_mask | partial
        )

    def fresh_noise(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        """Standard normal float64 noise of the verification races."""
        generator = _fresh_generator(self._generators, device, _TEXTSEAL2_NOISE_SEED)
        return torch.randn(
            shape, dtype=torch.float64, device=device, generator=generator
        )

    def race_gumbels(self, contexts: torch.Tensor, vocab_size: int) -> torch.Tensor:
        vocabulary = torch.arange(vocab_size, device=contexts.device)
        virtual = self.sampling_watermarker.virtual_key(contexts, vocabulary)
        noise = self.fresh_noise(virtual.shape, contexts.device)
        return textseal2_gumbels(virtual, noise, self.correlation).float()
