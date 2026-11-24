# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Verification outcome laws and TextSeal2's residual race (arXiv:2610.40613,
Sec. 3.2, App. A.3-A.4), from the reference's ``watermarking/verification.py``.

Pure torch on padded block tensors, in float64, batched over the B requests of
a verification step:

    P    [B, D, C]  target distribution of each row over C candidates (row i of
                    a request sits at depth i; rows past its bonus row are
                    padding and hold zeros)
    Q    [B, D, C]  draft distribution (zeros at bonus and padding rows)
    E    [B, D, C]  Exp(1) clocks the drafts were raced on
    dpos [B, D]     candidate index of the drafted token at draft rows
    K    [B]        drafts per request; the bonus row sits at depth K

With ``c_0 = 1`` and, for draft row i (depths are 0-based),

    c_i    = min(1, c_{i-1} p_i(d_i) / q_i(d_i))
    u_i(v) = (c_{i-1} p_i(v) - q_i(v))_+,  z_i = sum_v u_i(v),  r_i = u_i / z_i
    h_0 = 1,  h_i = z_i / (z_i + 1 - c_{i-1}),  r_bonus = p_bonus,  h_bonus = c_K
    P(I = i) = h_i prod_{j > i} (1 - h_j),  pi_i(v) = P(I = i) r_i(v),  pi_i(d_i) = 0

is block verification's law of the stopping row ``I`` and the token emitted
there (Sun et al.). The residual race

    (I, y) = argmin_{(i, v): pi_i(v) > 0} (E_i(v) - tau_i q_i(v)) / pi_i(v),
    tau_i  = E_i(d_i) / q_i(d_i)

decides the round: the drafts below ``I`` are kept and ``y`` is emitted. When
each draft ``d_i`` won the race ``argmin_v E_i(v) / q_i(v)`` on the same clocks,
the leftovers ``E - tau q`` are independent Exp clocks of the residual race, so
``(I, y)`` follows ``pi``: the output follows the target, the acceptance is
block verification's, and every emitted token is keyed. ``token_law`` is the
token-level verification law (the paper's Remark 2), raced the same way.

The tau-pass lookahead (the reference's ``pass_aggregator``; not in the paper)
runs the race as a backward chain over depths, where position i stops iff its
own race time ``M_i ~ Exp(P(I = i))`` beats the later positions' time
``S_{i+1} ~ Exp(P(I > i))``, and replaces that pass clock by
``psi(tau_i, S_{i+1} P(I > i)) / P(I > i)`` for an Exp(1)-preserving ``psi``
increasing in both arguments, or ``identity`` (``tau_i`` alone). ``tau_i`` is
independent of position i's leftovers and of the later positions, so the
outcome keeps the law ``pi`` while kept drafts are those whose keyed draft race
was fast. ``pass_source="child"`` uses ``tau_{i+1}`` in position i's pass clock
instead.
"""

import numpy as np
import torch

LAWS = ("block", "token")
PASS_AGGREGATORS = ("none", "identity", "tsum", "expsum", "max")
PASS_SOURCES = ("own", "child")


def pass_aggregate(a: torch.Tensor, b: torch.Tensor, name: str) -> torch.Tensor:
    """``psi(a, b)``, Exp(1) for independent Exp(1) ``a`` and ``b`` and
    increasing in both; infinite when either is. ``identity``: ``a`` (the
    draft keeps its own clock); ``max``: ``2 min(a, b)``;
    ``expsum``: ``s - log(1 + s)`` for ``s = a + b``, whose survival function is
    uniform; ``tsum``: ``exp(-T(-log a - log b))`` with lookahead Gumbel-max's
    table ``T``. In Gumbel units these are lookahead Gumbel-max's ``max``,
    ``expsum`` and ``tsum`` aggregators."""
    a, b = a.double(), b.double()
    if name == "identity":
        return a
    infinite = torch.isinf(a) | torch.isinf(b)
    a = torch.where(infinite, 1.0, a)
    b = torch.where(infinite, 1.0, b)
    if name == "max":
        out = 2.0 * torch.minimum(a, b)
    elif name == "expsum":
        s = a + b
        s2 = s * s
        series = s2 / 2 - s2 * s / 3 + s2 * s2 / 4
        out = torch.where(s < 1e-3, series, s - torch.log1p(s))
    elif name == "tsum":
        g = -torch.log(a.clamp_min(1e-300)) - torch.log(b.clamp_min(1e-300))
        out = torch.exp(-_tsum(g))
    else:
        raise ValueError(f"pass_aggregator must be one of {PASS_AGGREGATORS}")
    return torch.where(infinite, torch.inf, out)


def _tsum(x: torch.Tensor) -> torch.Tensor:
    """Lookahead Gumbel-max's ``T``, piecewise linear on its grid and clamped
    to it."""
    from vllm.v1.watermarking.lookahead import tsum_table

    lo, hi, values = tsum_table()
    xs = torch.from_numpy(np.linspace(lo, hi, values.size)).to(x.device)
    ys = torch.from_numpy(values).to(x.device)
    x = x.clamp(lo, hi).contiguous()
    index = torch.searchsorted(xs, x, right=True).clamp(1, xs.numel() - 1)
    x0, x1 = xs[index - 1], xs[index]
    return ys[index - 1] + (x - x0) / (x1 - x0) * (ys[index] - ys[index - 1])


def _gather(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``x[b, index[b]]`` for x [B, C]."""
    return x.gather(1, index.view(-1, 1)).squeeze(1)


def _zero_draft_slots(
    pi: torch.Tensor, dpos: torch.Tensor, K: torch.Tensor
) -> torch.Tensor:
    """Zero the drafted slot of every draft row: zero in exact arithmetic, and
    rounding must never let a draft's lane, whose leftover is 0, win."""
    B, D, C = pi.shape
    is_draft = torch.arange(D, device=pi.device).unsqueeze(0) < K.unsqueeze(1)
    hit = torch.zeros_like(pi, dtype=torch.bool).scatter_(
        2, dpos.clamp(0, C - 1).unsqueeze(2), True
    )
    return pi.masked_fill(hit & is_draft.unsqueeze(2), 0.0)


def block_law(
    P: torch.Tensor, Q: torch.Tensor, dpos: torch.Tensor, K: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block verification's outcome law given the drafts: (pi [B, D, C], stop
    [B, D]) with ``stop[b, i] = P(I = i)``."""
    B, D, C = P.shape
    c_prev = torch.ones(B, dtype=P.dtype, device=P.device)
    h = torch.zeros(B, D, dtype=P.dtype, device=P.device)
    R = torch.zeros_like(P)
    for i in range(D):
        is_draft = i < K
        is_bonus = i == K
        p, q, pos = P[:, i], Q[:, i], dpos[:, i].clamp(0, C - 1)
        u = (c_prev.unsqueeze(1) * p - q).clamp_min(0.0)
        z = u.sum(1)
        r = torch.where(
            z.unsqueeze(1) > 0, u / torch.where(z > 0, z, 1.0).unsqueeze(1), 0.0
        )
        if i == 0:
            h_draft = torch.ones_like(z)
        else:
            den = z + 1.0 - c_prev
            h_draft = torch.where(den > 0, z / torch.where(den > 0, den, 1.0), 1.0)
        h[:, i] = torch.where(
            is_draft, h_draft, torch.where(is_bonus, c_prev, 0.0)
        ).clamp(0.0, 1.0)
        R[:, i] = torch.where(
            is_draft.unsqueeze(1), r, torch.where(is_bonus.unsqueeze(1), p, 0.0)
        )
        qd, pd = _gather(q, pos), _gather(p, pos)
        ratio = pd / torch.where(qd > 0, qd, 1.0)
        c_prev = torch.where(is_draft, torch.clamp(c_prev * ratio, max=1.0), c_prev)
    stop = torch.zeros(B, D, dtype=P.dtype, device=P.device)
    tail = torch.ones(B, dtype=P.dtype, device=P.device)
    for i in range(D - 1, -1, -1):
        valid = i <= K
        stop[:, i] = torch.where(valid, h[:, i] * tail, 0.0)
        tail = torch.where(valid, tail * (1.0 - h[:, i]), tail)
    return _zero_draft_slots(stop.unsqueeze(2) * R, dpos, K), stop


def token_law(
    P: torch.Tensor, Q: torch.Tensor, dpos: torch.Tensor, K: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-level verification's outcome law given the drafts, as
    ``block_law``: ``pi_i = (prod_{j < i} a_j) (1 - a_i) r_i``, ``pi_bonus =
    (prod_j a_j) p_bonus``, ``a_i = min(1, p_i(d_i) / q_i(d_i))``, ``r_i ~
    (p_i - q_i)_+``."""
    B, D, C = P.shape
    reach = torch.ones(B, dtype=P.dtype, device=P.device)
    pi = torch.zeros_like(P)
    stop = torch.zeros(B, D, dtype=P.dtype, device=P.device)
    for i in range(D):
        is_draft = i < K
        is_bonus = i == K
        p, q, pos = P[:, i], Q[:, i], dpos[:, i].clamp(0, C - 1)
        qd, pd = _gather(q, pos), _gather(p, pos)
        a = torch.clamp(pd / torch.where(qd > 0, qd, 1.0), max=1.0)
        u = (p - q).clamp_min(0.0)
        z = u.sum(1)
        r = torch.where(
            z.unsqueeze(1) > 0, u / torch.where(z > 0, z, 1.0).unsqueeze(1), 0.0
        )
        s = torch.where(is_draft, reach * (1.0 - a), torch.where(is_bonus, reach, 0.0))
        stop[:, i] = s
        pi[:, i] = s.unsqueeze(1) * torch.where(is_draft.unsqueeze(1), r, p)
        reach = torch.where(is_draft, reach * a, reach)
    return _zero_draft_slots(pi, dpos, K), stop


def outcome_law(
    P: torch.Tensor,
    Q: torch.Tensor,
    dpos: torch.Tensor,
    K: torch.Tensor,
    law: str = "block",
) -> tuple[torch.Tensor, torch.Tensor]:
    if law == "block":
        return block_law(P, Q, dpos, K)
    if law == "token":
        return token_law(P, Q, dpos, K)
    raise ValueError(f"law must be one of {LAWS}")


def race(E: torch.Tensor, law: torch.Tensor) -> torch.Tensor:
    """``argmin_v E_v / law_v`` over ``law > 0``, along the last dimension."""
    ratio = torch.where(law > 0, E / torch.where(law > 0, law, 1.0), torch.inf)
    return ratio.argmin(dim=-1)


def draft_times(
    E: torch.Tensor, Q: torch.Tensor, dpos: torch.Tensor, K: torch.Tensor
) -> torch.Tensor:
    """``tau_i = E_i(d_i) / q_i(d_i)``, each draft race's time, at draft rows
    ([B, D]; 0 elsewhere)."""
    B, D, C = E.shape
    pos = dpos.clamp(0, C - 1).unsqueeze(2)
    qd = Q.gather(2, pos).squeeze(2)
    tau = E.gather(2, pos).squeeze(2) / torch.where(qd > 0, qd, 1.0)
    is_draft = torch.arange(D, device=E.device).unsqueeze(0) < K.unsqueeze(1)
    return torch.where(is_draft & (qd > 0), tau, 0.0)


def residual_race(
    E: torch.Tensor,
    P: torch.Tensor,
    Q: torch.Tensor,
    dpos: torch.Tensor,
    K: torch.Tensor,
    law: str = "block",
    pass_aggregator: str = "none",
    pass_source: str = "own",
) -> tuple[torch.Tensor, torch.Tensor]:
    """TextSeal2's decision for a batch of requests: the stopping depth ``I``
    [B] and the candidate emitted there [B]."""
    if pass_aggregator not in PASS_AGGREGATORS:
        raise ValueError(f"pass_aggregator must be one of {PASS_AGGREGATORS}")
    if pass_source not in PASS_SOURCES:
        raise ValueError(f"pass_source must be one of {PASS_SOURCES}")
    pi, stop = outcome_law(P, Q, dpos, K, law)
    tau = draft_times(E, Q, dpos, K)
    # Leftovers E - tau q, at least 0 when the draft won its race.
    lanes = (E - tau.unsqueeze(2) * Q).clamp_min(0.0)
    B, D, C = E.shape
    ratio = torch.where(pi > 0, lanes / torch.where(pi > 0, pi, 1.0), torch.inf)
    if pass_aggregator == "none":
        flat = ratio.reshape(B, D * C).argmin(dim=1)
        return flat // C, flat % C
    # The backward chain: M_i is position i's race time (inf without mass) and
    # after_i = P(I > i).
    M, col = ratio.min(dim=2)
    after = (1.0 - torch.cumsum(stop, dim=1)).clamp_min(0.0)
    inf = torch.full((B,), torch.inf, dtype=E.dtype, device=E.device)
    S = inf.clone()
    stopping = torch.full((B,), D - 1, dtype=torch.long, device=E.device)
    stopping_col = col[:, D - 1].clone()
    for i in range(D - 1, -1, -1):
        is_draft = i < K
        a_i = after[:, i]
        has_later = is_draft & (a_i > 0) & torch.isfinite(S)
        # The later positions' time in Exp(1) units.
        unit_s = torch.where(has_later, S * a_i, inf)
        if pass_source == "own":
            unit = pass_aggregate(tau[:, i], unit_s, pass_aggregator)
        else:
            # The child's draft time; the bonus row has no draft race.
            child_tau = tau[:, i + 1] if i + 1 < D else torch.zeros_like(tau[:, i])
            unit = torch.where(
                i + 1 < K, pass_aggregate(child_tau, unit_s, pass_aggregator), unit_s
            )
        clock = torch.where(has_later, unit / torch.where(a_i > 0, a_i, 1.0), inf)
        M_i = M[:, i]
        stop_here = (M_i < clock) | (torch.isinf(clock) & torch.isfinite(M_i))
        stopping = torch.where(stop_here, i, stopping)
        stopping_col = torch.where(stop_here, col[:, i], stopping_col)
        S = torch.minimum(M_i, clock)
    return stopping, stopping_col
