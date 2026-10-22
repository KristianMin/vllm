# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lookahead textGrain for speculative decoding: expected-reward messages.

Without speculation it samples like textGrain. In speculative verification the
rows of each request's drafted path, its drafted rows and then its bonus row,
are built from the bonus row up, and every row sends its parent the expected
detector score of what it will emit at and below itself:

    M = sum_{b,k} pi(T)_bk A_bk(T),

a function of the row's (modified) cell table T only, not of its column J nor
of the token drawn. A cell's detector score is s(u) = -log(1 - u) (Exp(1)), so
outside the drafted child's block A = s(u). In the drafted child d's block b*
the continuation is C = w M_child with w = P(d) / rho_{b*} (the block's other
tokens are not explored), and A = H^-1(u) with H the CDF of E + C, E ~ Exp(1),
over the reference law of C. The parent's cell (b*, J) becomes
u' = H(s(u) + C): uniform again, so the modified table has the law of a keyed
table and stays independent of J, and every column law Q'(. | J) averages back
to P. The coupling is solved on the modified table with textGrain's own
objective, and rejection sampling verifies the drafts against Q'(. | J): the
token is drawn by vLLM's sampler, as without the lookahead.

The reference law of a child's message has no closed form. It is sampled by
solving the child's coupling on ``reward_samples`` reference tables and
applying the child's message function. The parent then uses the randomized
rank of the child's message among one such sample, which is exactly uniform
because the message and the sample are exchangeable, mapped through the
empirical quantile of a second, independent sample: C is exactly uniform on
that sample, H is a finite mixture of exponentials with a closed-form CDF and
inverse, and the construction is exact for every sample size. The message is
the expected score up to that sample's quantization of C.

This is Section 9 of the conditional-lookahead framework
(newspecwm/lookahead_general_framework.md) with the rank calibration above in
place of the exact reference CDF it assumes. The reference tables and the rank
randomization come from PRFs under keys derived from the watermark key, which
the detector never reads. Drafts must not depend on the key.
"""

import torch

from vllm.config.watermarking import WatermarkPRFName, derive_watermark_key
from vllm.v1.watermarking.prfs import create_prf
from vllm.v1.watermarking.textgrain import (
    EULER_GAMMA,
    GUMBEL_STD,
    U_EPS,
    TextGrainWatermarker,
    entropy,
)

_CALIBRATION_DOMAIN = b"textgrain_lookahead_calibration"
_RANK_DOMAIN = b"textgrain_lookahead_rank"


def exp_score(u: torch.Tensor) -> torch.Tensor:
    """The detector's score of a cell with uniform ``u``: ``-log(1 - u)``."""
    return -torch.log1p(-u.clamp(0.0, 1.0 - U_EPS))


def mixture_prep(
    c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tables of ``H(a) = mean_n P(E + c_n <= a)``, ``E ~ Exp(1)``, for each row of
    ``c`` [R, N]: (sorted c, log cumulative sums of e^c, H at the sorted c)."""
    cs = c.double().sort(dim=1).values
    log_e = torch.logcumsumexp(cs, dim=1)
    n = cs.shape[1]
    h_knots = torch.zeros_like(cs)
    k = torch.arange(1, n, dtype=cs.dtype, device=cs.device)
    h_knots[:, 1:] = (k - torch.exp(log_e[:, :-1] - cs[:, 1:])) / n
    return cs, log_e, h_knots


def mixture_cdf(a: torch.Tensor, prep) -> torch.Tensor:
    """``H(a)`` for ``a`` [R, Q]: on ``[c_k, c_{k+1})`` it is
    ``(k - e^{-a} sum_{j <= k} e^{c_j}) / N``."""
    cs, log_e, _ = prep
    n = cs.shape[1]
    k = torch.searchsorted(cs, a.double().contiguous())
    value = (k.double() - torch.exp(log_e.gather(1, (k - 1).clamp(min=0)) - a)) / n
    return torch.where(k > 0, value, torch.zeros_like(value)).clamp(0.0, 1.0)


def mixture_quantile(u: torch.Tensor, prep) -> torch.Tensor:
    """``H^-1(u)`` for ``u`` [R, Q], the closed-form inverse on the interval
    holding ``u``."""
    cs, log_e, h_knots = prep
    n = cs.shape[1]
    u = u.double().clamp(0.0, 1.0 - U_EPS).contiguous()
    k = torch.searchsorted(h_knots, u, right=True).clamp(1, n)
    return log_e.gather(1, k - 1) - torch.log((k.double() - u * n).clamp_min(1e-300))


def recalibrated_continuation(
    message: torch.Tensor,
    rank_sample: torch.Tensor,
    design_sample: torch.Tensor,
    jitter: torch.Tensor,
) -> torch.Tensor:
    """The empirical quantile of ``design_sample`` [R, N] at the randomized rank
    of ``message`` [R] among ``rank_sample`` [R, N]: exactly uniform on the
    entries of ``design_sample`` when ``message`` and ``rank_sample`` are
    exchangeable."""
    n = rank_sample.shape[1]
    below = (rank_sample < message.unsqueeze(1)).sum(1).double()
    ties = (rank_sample == message.unsqueeze(1)).sum(1).double()
    v = (below + jitter * (ties + 1.0)) / (n + 1)
    index = (v * n).long().clamp(max=n - 1)
    return design_sample.sort(dim=1).values.gather(1, index.view(-1, 1)).squeeze(1)


class LookaheadTextGrainWatermarker(TextGrainWatermarker):
    """textGrain whose verification laws look ahead along each drafted path,
    see the module docstring."""

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        prf: WatermarkPRFName = "philox",
        entropy_budget: float = 0.2,
        num_blocks: int = 16,
        num_columns: int = 16,
        reward_samples: int = 64,
        solver_tol: float = 0.01,
        solver_max_iter: int = 200,
        solver_stall_tol: float = 1e-7,
    ) -> None:
        super().__init__(
            key,
            context_width,
            prf,
            entropy_budget,
            num_blocks,
            num_columns,
            solver_tol,
            solver_max_iter,
            solver_stall_tol,
        )
        if reward_samples < 2:
            raise ValueError("reward_samples must be at least 2")
        self.reward_samples = reward_samples
        self.calibration_prf = create_prf(
            prf, derive_watermark_key(key, _CALIBRATION_DOMAIN)
        )
        self.rank_prf = create_prf(prf, derive_watermark_key(key, _RANK_DOMAIN))
        self.prfs = self.prfs + [self.calibration_prf, self.rank_prf]

    def cell_uniforms(self, contexts: torch.Tensor) -> torch.Tensor:
        """The keyed cell uniforms [N, B, m], float64."""
        cells = torch.arange(self.num_blocks * self.num_columns, device=contexts.device)
        return (
            self.cost_prf.uniform(contexts, cells)
            .double()
            .view(-1, self.num_blocks, self.num_columns)
        )

    @staticmethod
    def standardized(cell_u: torch.Tensor) -> torch.Tensor:
        z = -torch.log(-torch.log(cell_u.clamp(U_EPS, 1.0 - U_EPS)))
        return (z - EULER_GAMMA) / GUMBEL_STD

    def column_log_law(
        self,
        log_p: torch.Tensor,
        blocks: torch.Tensor,
        rho: torch.Tensor,
        pi: torch.Tensor,
        column: torch.Tensor,
    ) -> torch.Tensor:
        """float32 log Q(. | J) from a coupling, as ``watermark_logits``."""
        index = column.view(-1, 1, 1).expand(-1, self.num_blocks, 1)
        conditional = pi.gather(2, index).squeeze(-1) * self.num_columns
        ratio = torch.where(rho > 0, conditional / torch.where(rho > 0, rho, 1.0), 0.0)
        log_ratio = torch.where(
            ratio > 0, torch.log(torch.where(ratio > 0, ratio, 1.0)), float("-inf")
        )
        return log_p + log_ratio.gather(1, blocks).float()

    @staticmethod
    def message(
        cell_u: torch.Tensor, pi: torch.Tensor, b_star: torch.Tensor, prep
    ) -> torch.Tensor:
        """``sum pi A`` for tables [R, T, B, m] (T tables per row) with
        ``A = s(u)`` outside block ``b_star[r]`` and ``H^-1(u)`` inside it;
        ``prep`` holds the mixtures of the rows with ``b_star >= 0``, in order.
        Returns [R, T]."""
        a = exp_score(cell_u)
        has = b_star >= 0
        if bool(has.any()):
            idx = has.nonzero(as_tuple=True)[0]
            bs = b_star[idx]
            n, t, m = idx.numel(), cell_u.shape[1], cell_u.shape[3]
            row_u = cell_u[idx, :, bs, :].reshape(n, t * m)
            a[idx, :, bs, :] = mixture_quantile(row_u, prep).view(n, t, m)
        return (pi * a).sum((-2, -1))

    def verification_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        draft_tokens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        skip_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.lookahead_logits(
            logits, contexts, draft_tokens, cu_num_logits, skip_mask
        )

    @torch.no_grad()
    def lookahead_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        draft_tokens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        skip_mask: torch.Tensor,
    ) -> torch.Tensor:
        """float32 log Q'(. | J) of every verification row; rows in
        ``skip_mask`` keep their logits and send nothing up."""
        device = logits.device
        num_rows, vocab_size = logits.shape
        logits_f32 = logits.float().masked_fill(logits.isnan(), float("-inf"))
        if num_rows == 0:
            return logits_f32
        num_b, num_m = self.num_blocks, self.num_columns
        log_p = torch.log_softmax(logits_f32, dim=-1)
        probs = log_p.double().exp()
        vocabulary = torch.arange(vocab_size, device=device)
        uniforms = self.partition_prf.uniform(contexts, vocabulary).double()
        blocks = (uniforms * num_b).long().clamp_(max=num_b - 1)
        zero = torch.zeros(1, dtype=torch.long, device=device)
        column_u = self.column_prf.uniform(contexts, zero).double().squeeze(-1)
        column = (column_u * num_m).long().clamp_(max=num_m - 1)
        cell_u = self.cell_uniforms(contexts)
        rho = torch.zeros(num_rows, num_b, dtype=torch.float64, device=device)
        rho = rho.scatter_add_(1, blocks, probs)
        rho = rho / rho.sum(-1, keepdim=True)
        token_entropy = entropy(probs)

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
        child_ok = (children >= 0) & (children < vocab_size)
        child_p = probs[safe, children.clamp(0, vocab_size - 1)]
        link = torch.zeros_like(active)
        link[:, :-1] = (
            active[:, :-1] & active[:, 1:] & child_ok[:, :-1] & (child_p[:, :-1] > 0)
        )
        need = torch.zeros_like(active)
        need[:, 1:] = link[:, :-1]

        n_ref = self.reward_samples
        n_cal = 2 * n_ref
        output = logits_f32.clone()
        message = torch.zeros(num_requests, dtype=torch.float64, device=device)
        rank_sample = torch.zeros(
            num_requests, n_ref, dtype=torch.float64, device=device
        )
        design_sample = torch.zeros_like(rank_sample)
        for i in range(depth - 1, -1, -1):
            sel = active[:, i].nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            r = safe[sel, i]
            table = cell_u[r].clone()
            linked = link[sel, i]
            b_star = torch.full((sel.numel(),), -1, dtype=torch.long, device=device)
            prep = None
            if bool(linked.any()):
                li = linked.nonzero(as_tuple=True)[0]
                rl = r[li]
                d = children[sel[li], i]
                bs = blocks[rl, d]
                w = probs[rl, d] / rho[rl, bs]
                jitter = self.rank_prf.uniform(contexts[rl], zero).double().squeeze(-1)
                cont = w * recalibrated_continuation(
                    message[sel[li]],
                    rank_sample[sel[li]],
                    design_sample[sel[li]],
                    jitter,
                )
                prep = mixture_prep(w.unsqueeze(1) * design_sample[sel[li]])
                own = table[li, bs, column[rl]]
                table[li, bs, column[rl]] = mixture_cdf(
                    (exp_score(own) + cont).unsqueeze(1), prep
                ).squeeze(1)
                b_star[li] = bs
            pi, _ = self.solve(rho[r], self.standardized(table), token_entropy[r])
            output[r] = self.column_log_law(log_p[r], blocks[r], rho[r], pi, column[r])
            if i == 0:
                break
            current = self.message(
                table.unsqueeze(1), pi.unsqueeze(1), b_star, prep
            ).squeeze(1)
            # The reference law of this message, for a parent that links here.
            ci = need[sel, i].nonzero(as_tuple=True)[0]
            if ci.numel():
                rc = r[ci]
                indices = torch.arange(n_cal * num_b * num_m, device=device)
                reference = (
                    self.calibration_prf.uniform(contexts[rc], indices)
                    .double()
                    .view(-1, num_b, num_m)
                )
                ref_pi, _ = self.solve(
                    rho[rc].repeat_interleave(n_cal, 0),
                    self.standardized(reference),
                    token_entropy[rc].repeat_interleave(n_cal, 0),
                )
                ref_b = b_star[ci]
                ref_prep = None
                if prep is not None and bool((ref_b >= 0).any()):
                    in_prep = torch.cumsum((b_star >= 0).long(), 0) - 1
                    keep = in_prep[ci[ref_b >= 0]]
                    ref_prep = tuple(x[keep] for x in prep)
                shape = (ci.numel(), n_cal, num_b, num_m)
                messages = self.message(
                    reference.view(shape), ref_pi.view(shape), ref_b, ref_prep
                )
                rank_sample[sel[ci]] = messages[:, :n_ref]
                design_sample[sel[ci]] = messages[:, n_ref:]
            message = torch.zeros_like(message)
            message[sel] = current
        return torch.where(skip_mask.unsqueeze(1), logits_f32, output)
