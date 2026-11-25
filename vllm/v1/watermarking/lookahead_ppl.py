# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lookahead soft-perplexity watermark for speculative decoding.

Without speculation it samples like ``ppl_soft``: the token maximizing
``g_u + beta * log max(p_u, 1e-12)`` over the most probable tokens, with ``g``
the keyed Binomial(num_trials, 1/2) scores. In speculative verification the
rows of each request's drafted path, its drafted rows and then its bonus row,
are decided together from the bonus row up, as lookahead Gumbel-max does: a
row's drafted child races with its score replaced by a fresh
Binomial(num_trials, 1/2) score built from its own score and what the row below
sent up. The rows become point masses at their winners, so rejection sampling
emits the winners along the path up to the first rejected draft.

What a row sends up. The maximum of a binomial race is not independent of its
winner, so the row sends the rank of the winner's score within its law given
the winner (a randomized probability integral transform; for a Gumbel race it
is the race maximum). The rank is uniform and independent of the winning token,
and the drafted child's new score ``F^-1(psi(U_own, V_child))`` is again
Binomial for every uniform-preserving ``psi``: ``pi1`` (plain ``ppl_soft``),
``pi2``, ``tsum``, ``max`` and ``gauss``. ``optpath`` instead sends up the
realized score of everything the row emits at and below itself with its law
given the path emitted below, and races the drafted child on the rank of its
own score plus that realized score: the best exact rule at each row.

Every row's winner keeps the ``ppl_soft`` law as long as the drafts do not
depend on the key, so the text has the law of ``ppl_soft`` decoded token by
token. Detection is the binomial score test of ``ppl_soft``. The randomization
of the ranks comes from a PRF under a key derived from the watermark key, which
the detector never reads.
"""

import math

import torch

from vllm.config.watermarking import (
    LOOKAHEAD_PPL_AGGREGATORS,
    WatermarkPRFName,
    derive_watermark_key,
)
from vllm.v1.watermarking.binomial import DEFAULT_NUM_TRIALS
from vllm.v1.watermarking.perplexity import PPLSoftWatermarker
from vllm.v1.watermarking.prfs import PhiloxPRF, WatermarkPRF

_JITTER_DOMAIN = b"lookahead_ppl_jitter"


def binomial_tables(
    num_trials: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Binomial(num_trials, 1/2) pmf [n + 1] and padded cdf [n + 2], with
    ``cdf_pad[g + 1] = P(G <= g)``, float64."""
    pmf = torch.tensor(
        [math.comb(num_trials, g) / 2**num_trials for g in range(num_trials + 1)],
        dtype=torch.float64,
        device=device,
    )
    cdf_pad = torch.cat(
        [torch.zeros(1, dtype=torch.float64, device=device), pmf.cumsum(0)]
    )
    cdf_pad[-1] = 1.0
    return pmf, cdf_pad


def _gumbelize(u: torch.Tensor) -> torch.Tensor:
    return -torch.log(-torch.log(u.clamp(1e-300, 1.0 - 1e-16)))


def aggregate_uniforms(
    name: str, lam: float, u: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """``psi(u, v)`` for independent uniforms ``u`` (the drafted child's own
    score) and ``v`` (the value of the row below): again a uniform."""
    if name == "pi1":
        return u
    if name == "pi2":
        return v
    if name == "max":
        return torch.maximum(u, v) ** 2
    if name == "tsum":
        # The CDF of a sum of two standard Gumbels, z K1(z) with z = 2 e^{-s/2}.
        s = _gumbelize(u) + _gumbelize(v)
        z = 2.0 * torch.exp(-s / 2.0)
        return (z * torch.special.modified_bessel_k1(z)).clamp(0.0, 1.0)
    if name == "gauss":
        normal = torch.distributions.Normal(
            torch.zeros((), dtype=u.dtype, device=u.device),
            torch.ones((), dtype=u.dtype, device=u.device),
        )
        a = normal.icdf(u.clamp(1e-15, 1.0 - 1e-15))
        b = normal.icdf(v.clamp(1e-15, 1.0 - 1e-15))
        return normal.cdf((a + lam * b) / math.sqrt(1.0 + lam * lam))
    raise ValueError(f"unknown aggregator {name!r}")


class LookaheadPPLSoftWatermarker(PPLSoftWatermarker):
    """``ppl_soft`` that decides each speculative path backwards, see the module
    docstring."""

    # Verification picks every emitted token from the target's races, and the
    # rule is exact only for drafts that do not depend on the key.
    verifies_unwatermarked_drafts = True

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRF | WatermarkPRFName = "philox",
        epsilon: float = 0.0,
        aggregator: str = "tsum",
        aggregator_lambda: float = 1.0,
        num_trials: int = DEFAULT_NUM_TRIALS,
        n_mc: int = 128,
    ) -> None:
        super().__init__(key, context_width, prf, epsilon, num_trials, n_mc)
        if aggregator not in LOOKAHEAD_PPL_AGGREGATORS:
            raise ValueError(f"aggregator must be one of {LOOKAHEAD_PPL_AGGREGATORS}")
        if aggregator_lambda <= 0:
            raise ValueError("aggregator_lambda must be positive")
        self.aggregator = aggregator
        self.aggregator_lambda = aggregator_lambda
        self.jitter_prf = PhiloxPRF(derive_watermark_key(key, _JITTER_DOMAIN))
        self._binomial: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    # ---- the race law ------------------------------------------------------------
    def binomial(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if device not in self._binomial:
            self._binomial[device] = binomial_tables(self.num_trials, device)
        return self._binomial[device]

    def winner_law(
        self, offsets: torch.Tensor, valid: torch.Tensor, winner: torch.Tensor
    ) -> torch.Tensor:
        """Law of the winner's score given that ``winner`` won: [rows, n + 1]."""
        joint = self.winner_joint(offsets, valid, winner)
        return joint / joint.sum(-1, keepdim=True).clamp_min(1e-300)

    def winner_joint(
        self, offsets: torch.Tensor, valid: torch.Tensor, winner: torch.Tensor
    ) -> torch.Tensor:
        """``P(W = winner, S_W = g)`` for g = 0..n: [rows, n + 1].

        ``offsets`` [rows, width] are the candidates' ``beta * log max(p, 1e-12)``
        in race order (decreasing probability, then token id), ``valid`` marks
        the candidates. A candidate earlier in the order beats the winner on a
        tie, a later one loses it, as in ``ppl_soft``'s ``argmax``.
        """
        pmf, cdf_pad = self.binomial(offsets.device)
        n = self.num_trials
        rows, width = offsets.shape
        levels = torch.arange(n + 1, dtype=torch.float64, device=offsets.device)
        c_w = offsets.gather(1, winner.view(-1, 1))  # [rows, 1]
        x = levels.view(1, 1, -1) + (c_w - offsets).unsqueeze(-1)  # [rows, width, n+1]
        order = torch.arange(width, device=offsets.device).view(1, -1, 1)
        earlier = order < winner.view(-1, 1, 1)
        bound = torch.where(earlier, torch.ceil(x) - 1.0, torch.floor(x))
        index = bound.clamp(-1, n).long() + 1
        lose = cdf_pad[index]
        others = valid.unsqueeze(-1) & (order != winner.view(-1, 1, 1))
        lose = torch.where(others, lose, torch.ones_like(lose))
        log_joint = torch.log(lose.clamp_min(1e-300)).sum(1) + torch.log(pmf)
        log_joint = torch.where((lose > 0).all(1), log_joint, float("-inf"))
        return log_joint.exp()

    def rank(
        self, law: torch.Tensor, score: torch.Tensor, jitter: torch.Tensor
    ) -> torch.Tensor:
        """Randomized PIT of the integer ``score`` [rows] under ``law`` [rows, L]."""
        cdf = torch.cat([torch.zeros_like(law[:, :1]), law.cumsum(1)], dim=1)
        index = score.long().clamp(0, law.shape[1] - 1).view(-1, 1)
        low = cdf.gather(1, index).squeeze(1)
        high = cdf.gather(1, index + 1).squeeze(1)
        return (low + jitter * (high - low)).clamp(0.0, 1.0)

    def binomial_quantile(self, u: torch.Tensor) -> torch.Tensor:
        _, cdf_pad = self.binomial(u.device)
        return torch.searchsorted(cdf_pad[1:].contiguous(), u.contiguous()).clamp(
            max=self.num_trials
        )

    def race_terms(
        self, logits: torch.Tensor, contexts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Each row's ``ppl_soft`` race: (candidate token ids in race order, i.e.
        decreasing probability then token id [rows, width]; which of them have
        probability; their offsets ``beta * log max(p, 1e-12)``, float64)."""
        log_probs = logits.float().masked_fill(logits.isnan(), float("-inf"))
        log_probs = log_probs.log_softmax(dim=-1)
        # ppl_soft calibrates and selects over the full sampling support.
        ranked, order = log_probs.sort(dim=-1, descending=True, stable=True)
        ranked = ranked.contiguous()
        beta = self.find_beta_for_logp_constraint(
            log_probs, contexts, ranked=ranked
        ).double()
        # Only candidates with probability can win or block: trim to the widest.
        width = max(1, int((ranked > float("-inf")).sum(1).max().item()))
        ranked, order = ranked[:, :width], order[:, :width]
        valid = ranked > float("-inf")
        offsets = beta.unsqueeze(1) * ranked.double().exp().clamp_min(1e-12).log()
        return order, valid, torch.where(valid, offsets, float("-inf"))

    # ---- verification ------------------------------------------------------------
    def verification_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        draft_tokens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        skip_mask: torch.Tensor,
    ) -> torch.Tensor:
        skip_mask = self._skip_partial_contexts(contexts, skip_mask)
        chosen = self.lookahead_tokens(
            logits, contexts, draft_tokens, cu_num_logits, skip_mask
        )
        if logits.device.type == "cuda":
            from vllm.v1.worker.gpu.sample.lookahead_watermark import point_mass_logits

            return point_mass_logits(logits, chosen, skip_mask)
        point_masses = torch.full_like(logits, float("-inf")).scatter_(
            1, chosen.unsqueeze(1), 0.0
        )
        return torch.where(skip_mask.unsqueeze(1), logits, point_masses)

    @torch.no_grad()
    def lookahead_tokens(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        draft_tokens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        skip_mask: torch.Tensor,
    ) -> torch.Tensor:
        """The chosen token of every verification row, [num_rows] int64; rows in
        ``skip_mask`` race on their own scores and send nothing up."""
        device = logits.device
        num_rows, vocab_size = logits.shape
        chosen = torch.zeros(num_rows, dtype=torch.int64, device=device)
        if num_rows == 0:
            return chosen
        n = self.num_trials
        order, valid, offsets = self.race_terms(logits, contexts)
        scores = self.scores(contexts, vocab_size).gather(1, order).double()
        jitter = self.jitter_prf.uniform(
            contexts, torch.arange(2, device=device)
        ).double()

        # Each request's rows in path order: [requests, depth], -1 past the end.
        bounds = cu_num_logits.long()
        sizes = bounds[1:] - bounds[:-1]
        num_requests = sizes.numel()
        depth = int(sizes.max().item()) if num_requests else 0
        position = torch.arange(depth, device=device)
        rows = bounds[:-1].view(-1, 1) + position.view(1, -1)
        rows = torch.where(position.view(1, -1) < sizes.view(-1, 1), rows, -1)
        safe = rows.clamp(min=0)
        active = (rows >= 0) & ~skip_mask[safe]
        children = torch.where(
            position.view(1, -1) < (sizes - 1).view(-1, 1),
            draft_tokens.long()[safe],
            -1,
        )
        # The child's position in its parent's candidates (-1 when not one).
        hit = order[safe] == children.unsqueeze(-1)
        child_pos = torch.where(
            hit.any(-1) & (children >= 0),
            hit.to(torch.int8).argmax(-1),
            -1,
        )
        child_valid = valid[safe].gather(2, child_pos.clamp(min=0).unsqueeze(-1))
        child_pos = torch.where(child_valid.squeeze(-1), child_pos, -1)
        link = torch.zeros_like(active)
        link[:, :-1] = active[:, :-1] & active[:, 1:] & (child_pos[:, :-1] >= 0)

        pmf, cdf_pad = self.binomial(device)
        optpath = self.aggregator == "optpath"
        num_levels = n * depth + 1
        value = torch.zeros(num_requests, dtype=torch.float64, device=device)
        realized = torch.zeros(num_requests, dtype=torch.float64, device=device)
        law = torch.zeros(num_requests, num_levels, dtype=torch.float64, device=device)
        requests = torch.arange(num_requests, device=device)
        for i in range(depth - 1, -1, -1):
            r = safe[:, i]
            race = scores[r].clone()  # [requests, width]
            linked = link[:, i]
            pos = child_pos[:, i].clamp(min=0)
            total = None
            table_cdf = None
            if self.aggregator != "pi1" and bool(linked.any()):
                own = race[requests, pos]
                if optpath:
                    total = own + realized
                    # The table: law of own + realized given the path below.
                    shifted = torch.nn.functional.pad(law, (n, 0)).unsqueeze(1)
                    table = torch.nn.functional.conv1d(
                        shifted, pmf.flip(0).view(1, 1, -1)
                    ).squeeze(1)
                    table_cdf = torch.cat(
                        [torch.zeros_like(table[:, :1]), table.cumsum(1)], dim=1
                    )
                    table_cdf = table_cdf / table_cdf[:, -1:].clamp_min(1e-300)
                    u = self.rank_from_cdf(table_cdf, total, jitter[r, 1])
                else:
                    own_u = cdf_pad[own.long()] + jitter[r, 1] * pmf[own.long()]
                    u = aggregate_uniforms(
                        self.aggregator, self.aggregator_lambda, own_u, value
                    )
                new = self.binomial_quantile(u.clamp(0.0, 1.0)).double()
                race[requests, pos] = torch.where(linked, new, own)
            winner = (race + offsets[r]).argmax(-1)
            exists = rows[:, i] >= 0
            chosen[r[exists]] = order[r, winner][exists]
            if i == 0:
                break
            # What the row sends up, for a parent that links to it.
            needs = link[:, i - 1]
            if not bool(needs.any()):
                continue
            winner_scores = race[requests, winner]
            winner_law = self.winner_law(offsets[r], valid[r], winner)
            if optpath:
                accepted = linked & (winner == pos)
                own_law = torch.nn.functional.pad(winner_law, (0, num_levels - n - 1))
                new_law = own_law
                if table_cdf is not None and bool(accepted.any()):
                    new_law = torch.where(
                        accepted.unsqueeze(1),
                        self.accepted_law(table_cdf, winner_law, num_levels),
                        own_law,
                    )
                realized = torch.where(
                    accepted, total if total is not None else realized, winner_scores
                )
                law = new_law
            else:
                value = self.rank(winner_law, winner_scores, jitter[r, 0])
        return chosen

    def rank_from_cdf(
        self, cdf: torch.Tensor, x: torch.Tensor, jitter: torch.Tensor
    ) -> torch.Tensor:
        """Randomized PIT of the integer ``x`` [rows] under the padded cdf
        [rows, L + 1]."""
        index = x.long().clamp(0, cdf.shape[1] - 2).view(-1, 1)
        low = cdf.gather(1, index).squeeze(1)
        high = cdf.gather(1, index + 1).squeeze(1)
        return (low + jitter * (high - low)).clamp(0.0, 1.0)

    def accepted_law(
        self, table_cdf: torch.Tensor, winner_law: torch.Tensor, num_levels: int
    ) -> torch.Tensor:
        """Law of the realized total given that the drafted child, racing on
        ``F^-1(PIT(total))``, won with its score distributed as ``winner_law``:
        given the score ``g`` the PIT is uniform on the bin ``[F(g-1), F(g)]``."""
        pmf, cdf_pad = self.binomial(winner_law.device)
        density = winner_law / pmf.view(1, -1)
        cumulative = torch.cat(
            [torch.zeros_like(winner_law[:, :1]), winner_law.cumsum(1)], dim=1
        )
        bins = torch.searchsorted(cdf_pad[1:].contiguous(), table_cdf.contiguous())
        bins = bins.clamp(max=self.num_trials)
        h = cumulative.gather(1, bins) + density.gather(1, bins) * (
            table_cdf - cdf_pad[bins]
        ).clamp_min(0.0)
        out = (h[:, 1:] - h[:, :-1]).clamp_min(0.0)
        out = out / out.sum(1, keepdim=True).clamp_min(1e-300)
        return out[:, :num_levels]
