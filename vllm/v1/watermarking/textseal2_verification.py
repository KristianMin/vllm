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
# Point-mass verification's law (``point_law``): block_verified_gumbel's
# residual race only, not a TextSeal2 rule.
POINT_LAW = "point"
RESIDUAL_LAWS = LAWS + (POINT_LAW,)
PASS_AGGREGATORS = ("none", "identity", "tsum", "expsum", "max")
PASS_SOURCES = ("own", "child")
# Which time of row i enters its pass clock: ``draft``, the draft race's time
# ``tau_i = E_i(d_i) / q_i(d_i)`` (keyed drafts raced on the clocks); ``raw``,
# the drafted token's raw clock ``E_i(d_i)`` (drafts that do not depend on the
# key: given them, the clocks are an i.i.d. Exp(1) field, so ``E_i(d_i)`` is
# independent of row i's other clocks and of the later rows). With ``raw`` and
# ``use_leftovers=False`` the chain is lookahead Gumbel-max's recursion with the
# rows' weights replaced by the verification law.
PASS_TIMES = ("draft", "raw")
# The reference's decision rules for a round (its ``spec_decision``:
# ``verification.decide`` at block verification, ``_verify_tokens`` at token
# level), see ``decide``. ``residual_j`` is the residual race above.
SPEC_DECISIONS = ("residual_j", "fresh_stop", "tau_stop", "agreement", "j_only", "raw_j")
# The decisions defined at token-level verification: the reference's token
# rules "residual" (fresh_stop) and "agreement", and the residual race.
TOKEN_DECISIONS = ("residual_j", "fresh_stop", "agreement")
# The decisions that draw fresh randomness (a uniform per row, ``agreement``
# also fresh Exp(1) clocks).
FRESH_DECISIONS = ("fresh_stop", "agreement")
# The decision whose drafts do not depend on the key (sampled from q).
SAMPLED_DRAFT_DECISIONS = ("j_only",)


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
    pi, stop, _ = block_law_and_hazards(P, Q, dpos, K)
    return pi, stop


def block_law_and_hazards(
    P: torch.Tensor, Q: torch.Tensor, dpos: torch.Tensor, K: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``block_law`` and the hazards ``h`` [B, D] it is built from: ``h_0 =
    1``, ``h_i`` at draft rows, ``h_K = c_K`` at the bonus row, 0 past it;
    ``P(I = i) = h_i prod_{i < j <= K} (1 - h_j)``."""
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
    return _zero_draft_slots(stop.unsqueeze(2) * R, dpos, K), stop, h


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


def point_law(
    P: torch.Tensor, Q: torch.Tensor, dpos: torch.Tensor, K: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Point-mass verification's outcome law given the drafts, as
    ``block_law``: every row compares its draft with an independent pick of the
    target, so row i keeps ``d_i`` with probability ``p_i(d_i)`` and stops with
    ``v != d_i`` with probability ``p_i(v)``: ``pi_i(v) = reach_i p_i(v)``,
    ``pi_bonus = reach_K p_bonus``, ``reach_i = prod_{j < i} p_j(d_j)``. ``Q``
    is not read (``token_law`` with ``q_i`` the point mass at ``d_i``)."""
    B, D, C = P.shape
    reach = torch.ones(B, dtype=P.dtype, device=P.device)
    pi = torch.zeros_like(P)
    stop = torch.zeros(B, D, dtype=P.dtype, device=P.device)
    for i in range(D):
        is_draft = i < K
        is_bonus = i == K
        p, pos = P[:, i], dpos[:, i].clamp(0, C - 1)
        pd = _gather(p, pos).clamp(0.0, 1.0)
        s = torch.where(is_draft, reach * (1.0 - pd), torch.where(is_bonus, reach, 0.0))
        stop[:, i] = s
        live = (is_draft | is_bonus).unsqueeze(1)
        pi[:, i] = torch.where(live, reach.unsqueeze(1) * p, 0.0)
        reach = torch.where(is_draft, reach * pd, reach)
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
    if law == POINT_LAW:
        return point_law(P, Q, dpos, K)
    raise ValueError(f"law must be one of {RESIDUAL_LAWS}")


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


def leftovers(
    E: torch.Tensor, Q: torch.Tensor, dpos: torch.Tensor, K: torch.Tensor
) -> torch.Tensor:
    """The leftover clocks ``E - tau q`` of every row, at least 0 when each
    draft won its race (``E`` at the bonus row, where ``q`` is 0)."""
    return (E - draft_times(E, Q, dpos, K).unsqueeze(2) * Q).clamp_min(0.0)


def residual_race(
    E: torch.Tensor,
    P: torch.Tensor,
    Q: torch.Tensor,
    dpos: torch.Tensor,
    K: torch.Tensor,
    law: str = "block",
    pass_aggregator: str = "none",
    pass_source: str = "own",
    use_leftovers: bool = True,
    pass_time: str = "draft",
) -> tuple[torch.Tensor, torch.Tensor]:
    """TextSeal2's decision for a batch of requests: the stopping depth ``I``
    [B] and the candidate emitted there [B]. ``use_leftovers=False`` races the
    raw clocks ``E`` instead of the leftovers, and ``pass_time="raw"`` feeds
    the drafted token's raw clock into the pass clock (``PASS_TIMES``): both
    for drafts that do not depend on the key."""
    if pass_aggregator not in PASS_AGGREGATORS:
        raise ValueError(f"pass_aggregator must be one of {PASS_AGGREGATORS}")
    if pass_source not in PASS_SOURCES:
        raise ValueError(f"pass_source must be one of {PASS_SOURCES}")
    if pass_time not in PASS_TIMES:
        raise ValueError(f"pass_time must be one of {PASS_TIMES}")
    pi, stop = outcome_law(P, Q, dpos, K, law)
    B, D, C = E.shape
    if pass_time == "raw":
        is_draft_row = torch.arange(D, device=E.device).unsqueeze(0) < K.unsqueeze(1)
        raw = E.gather(2, dpos.clamp(0, C - 1).unsqueeze(2)).squeeze(2)
        tau = torch.where(is_draft_row, raw, 0.0)
    else:
        tau = draft_times(E, Q, dpos, K)
    if use_leftovers:
        lanes = leftovers(E, Q, dpos, K)
    else:
        lanes = E
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


# ---- the reference's other decision rules -----------------------------------
#
# Block verification (the reference's ``verification.decide``), with
# ``pi = W R`` the outcome law and ``E' = E - tau q`` the leftovers:
#
#   fresh_stop  I ~ P(I = i) from a fresh uniform, y = argmin E'_I / pi_I
#               (``ts2b``; at token level the reference's token rule
#               ``residual``: a fresh accept coin per row, the leftover race
#               at the first rejected row)
#   tau_stop    I = max{i >= 1 : 1 - exp(-tau_{i-1}) <= h_i} (0 if none),
#               y = argmin E'_I / pi_I (``taub``): the draft races' times are
#               the stopping rule's uniforms
#   agreement   plain decoding on the same clocks (each row's target race
#               argmin E / p, the round stopping at the first row that
#               disagrees with its draft) outputs O* = (l*, x*) with law P*
#               given the drafts; it is kept with probability
#               min(1, pi(O*) / P*(O*)), else a race on (pi - P*)_+ with
#               fresh clocks at rows <= l* and the leftovers above (``agreeb``)
#   j_only      drafts sampled from q (not keyed), y = argmin E / pi on the
#               raw clocks (``jonly``)
#   raw_j       keyed drafts, the race on the raw clocks (``residjraw``, the
#               reference's biased negative control)
#
# Token level (``_verify_tokens``): ``agreement`` decides each keyed row by
# the race coupling of the draft race and the target race (accept when they
# agree, else with probability beta_d; on rejection the target's winner with
# probability min(1, (p - q)_+ / L) at it, else a fresh draw from the rest),
# and an unkeyed row by a fresh coin min(1, p / q) and a fresh draw from
# (p - q)_+; the bonus row emits its race's winner. All are exact.


def draw_stop(stop: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """The stopping depth with law ``stop`` [B, D] (unnormalized) from the
    uniforms ``u`` [B], by inverse CDF; always a depth with mass."""
    B, D = stop.shape
    cdf = torch.cumsum(stop, dim=1)
    threshold = (u.double() * cdf[:, -1]).unsqueeze(1)
    depth = (cdf <= threshold).sum(dim=1)
    has_mass = stop > 0
    last = D - 1 - has_mass.flip(1).long().argmax(dim=1)
    return torch.minimum(depth, last)


def _race_at(lanes: torch.Tensor, pi: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(lanes.shape[0], device=lanes.device)
    return race(lanes[rows, depth], pi[rows, depth])


def fresh_stop(
    E: torch.Tensor,
    P: torch.Tensor,
    Q: torch.Tensor,
    dpos: torch.Tensor,
    K: torch.Tensor,
    u: torch.Tensor,
    law: str = "block",
) -> tuple[torch.Tensor, torch.Tensor]:
    """``fresh_stop``: the stopping depth from the fresh uniforms ``u`` [B],
    the token from the leftover race at that row."""
    pi, stop = outcome_law(P, Q, dpos, K, law)
    depth = draw_stop(stop, u)
    return depth, _race_at(leftovers(E, Q, dpos, K), pi, depth)


def tau_stop(
    E: torch.Tensor,
    P: torch.Tensor,
    Q: torch.Tensor,
    dpos: torch.Tensor,
    K: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``tau_stop`` (block verification): depth ``i >= 1`` passes when its
    parent draft's race was fast, ``1 - exp(-tau_{i-1}) <= h_i``; the round
    stops at the deepest passing depth (0 if none). ``tau_{i-1} ~ Exp(1)`` is
    independent of the drafts and the leftovers, so this is block
    verification's backward stopping rule on independent uniforms."""
    pi, _, h = block_law_and_hazards(P, Q, dpos, K)
    B, D, _ = E.shape
    tau = draft_times(E, Q, dpos, K)
    depths = torch.arange(1, D, device=E.device).unsqueeze(0)
    passes = (-torch.expm1(-tau[:, :-1]) <= h[:, 1:]) & (depths <= K.unsqueeze(1))
    deepest = torch.cat(
        [torch.zeros_like(K.long()).unsqueeze(1), passes.long() * depths], dim=1
    ).amax(dim=1)
    return deepest, _race_at(leftovers(E, Q, dpos, K), pi, deepest)


def _sorted_race_terms(
    p: torch.Tensor, q: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per row of ``p``, ``q`` [N, C] (float64): the candidates sorted by ``t
    = q / p`` (infinite where ``p`` is 0), and, in that order, ``t``, ``p``,
    ``q``, ``J[v, v]`` and ``Phi(t_v)`` of the race coupling ``x = argmin E /
    q``, ``x_p = argmin E / p`` on shared Exp(1) clocks (the reference's
    ``diag_joint`` and ``phi``)::

        J[v, v] = p_v / (sum_{t_u <= t_v} p_u + sum_{t_u > t_v} q_u / t_v)
        Phi(t) = int_0^t dl / F(l)^2,  F(l) = sum_u max(q_u, l p_u)

    Candidates outside both supports sort last and carry nothing."""
    t = torch.where(p > 0, q / torch.where(p > 0, p, 1.0), torch.inf)
    ts, order = torch.sort(t, dim=1, stable=True)
    ps, qs = p.gather(1, order), q.gather(1, order)
    cum_p = torch.cumsum(ps, dim=1)
    tail_q = qs.sum(dim=1, keepdim=True) - torch.cumsum(qs, dim=1)
    denom = cum_p + tail_q / ts.clamp_min(1e-300)
    diag = torch.where((ps > 0) & (qs > 0), ps / denom, 0.0)
    # Phi: on the k-th segment between consecutive t, F(l) = a + b l.
    zero = torch.zeros_like(ps[:, :1])
    coef_b = torch.cat([zero, cum_p], dim=1)
    coef_a = qs.sum(dim=1, keepdim=True) - torch.cat(
        [zero, torch.cumsum(qs, dim=1)], dim=1
    )
    lo = torch.cat([zero, ts], dim=1)
    hi = torch.cat([ts, torch.full_like(zero, torch.inf)], dim=1)

    def inv(x: torch.Tensor) -> torch.Tensor:
        return torch.where(torch.isinf(x), 0.0, 1.0 / x)

    # Unselected branches may hold inf or nan (an infinite segment).
    linear = (inv(coef_a + coef_b * lo) - inv(coef_a + coef_b * hi)) / coef_b.clamp_min(
        1e-300
    )
    flat = (hi - lo) / coef_a.clamp_min(1e-300) ** 2
    pieces = torch.where(hi > lo, torch.where(coef_b > 0, linear, flat), 0.0)
    phi = torch.cumsum(pieces, dim=1)[:, : ts.shape[1]]
    return order, ts, ps, qs, diag, phi


def _unsort(values: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(values).scatter_(1, order, values)


def joint_rows(
    p: torch.Tensor, q: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``J[w, v]`` for every candidate ``v`` [N, C] and ``J[w, w]`` [N]: the
    probability that the draft race on ``q`` picks ``w[n]`` and the target
    race on ``p`` picks ``v`` (the reference's ``joint_row``)::

        J[w, v] = q_w p_v (Phi(t_w) - Phi(t_v))  for t_v < t_w, else 0"""
    order, _, _, _, diag_s, phi_s = _sorted_race_terms(p, q)
    phi, diag = _unsort(phi_s, order), _unsort(diag_s, order)
    t = torch.where(p > 0, q / torch.where(p > 0, p, 1.0), torch.inf)
    w = w.long().view(-1, 1)
    t_w, phi_w, q_w = t.gather(1, w), phi.gather(1, w), q.gather(1, w)
    row = torch.where(t < t_w, q_w * p * (phi_w - phi), 0.0).clamp_min(0.0)
    j_ww = diag.gather(1, w)
    return row.scatter(1, w, j_ww), j_ww.squeeze(1)


def plain_agreement_law(
    P: torch.Tensor, Q: torch.Tensor, dpos: torch.Tensor, K: torch.Tensor
) -> torch.Tensor:
    """``P*`` [B, D, C]: the law, given the drafts, of the outcome of plain
    decoding on the drafts' clocks: depth ``l`` and the target race's token
    there, ``l`` the first row whose target race disagrees with its draft (the
    bonus row when none does)."""
    B, D, C = P.shape
    rows = P.reshape(B * D, C)
    drafts = dpos.clamp(0, C - 1).reshape(B * D)
    joint, j_dd = joint_rows(rows, Q.reshape(B * D, C), drafts)
    joint, j_dd = joint.view(B, D, C), j_dd.view(B, D)
    q_d = Q.gather(2, dpos.clamp(0, C - 1).unsqueeze(2)).squeeze(2)
    q_d = torch.where(q_d > 0, q_d, 1.0)
    is_draft = torch.arange(D, device=P.device).unsqueeze(0) < K.unsqueeze(1)
    # A_i: every earlier row agreed.
    agree = torch.where(is_draft, j_dd / q_d, 1.0)
    reach = torch.cumprod(torch.cat([torch.ones_like(agree[:, :1]), agree[:, :-1]], 1), 1)
    conditional = joint / q_d.unsqueeze(2)
    conditional = conditional.scatter(2, dpos.clamp(0, C - 1).unsqueeze(2), 0.0)
    is_bonus = torch.arange(D, device=P.device).unsqueeze(0) == K.unsqueeze(1)
    law = torch.where(
        is_draft.unsqueeze(2),
        conditional,
        torch.where(is_bonus.unsqueeze(2), P, 0.0),
    )
    return reach.unsqueeze(2) * law


def agreement_block(
    E: torch.Tensor,
    P: torch.Tensor,
    Q: torch.Tensor,
    dpos: torch.Tensor,
    K: torch.Tensor,
    u: torch.Tensor,
    F: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``agreement`` at block verification: plain decoding's outcome on the
    drafts' clocks, kept with probability ``min(1, pi / P*)`` (uniforms ``u``
    [B]), else a race on ``(pi - P*)_+`` on the fresh clocks ``F`` [B, D, C]
    at rows up to the plain outcome's and the leftovers above."""
    pi, _ = block_law(P, Q, dpos, K)
    B, D, C = E.shape
    rows = torch.arange(B, device=E.device)
    depths = torch.arange(D, device=E.device).unsqueeze(0)
    target = race(E, P)  # [B, D]
    disagree = (depths < K.unsqueeze(1)) & (target != dpos)
    first = torch.where(disagree.any(dim=1), disagree.long().argmax(dim=1), K)
    token = target[rows, first]
    plain = plain_agreement_law(P, Q, dpos, K)
    plain_mass = plain[rows, first, token]
    keep = u.double() * plain_mass < pi[rows, first, token]
    lanes = torch.where(
        (depths <= first.unsqueeze(1)).unsqueeze(2), F, leftovers(E, Q, dpos, K)
    )
    rest = (pi - plain).clamp_min(0.0)
    ratio = torch.where(rest > 0, lanes / torch.where(rest > 0, rest, 1.0), torch.inf)
    flat = ratio.reshape(B, D * C).argmin(dim=1)
    return (
        torch.where(keep, first, flat // C),
        torch.where(keep, token, flat % C),
    )


def _agreement_rows(
    p: torch.Tensor, q: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per row of ``p``, ``q`` [N, C]: ``beta`` [N, C], the probability to
    accept a draft ``w`` its target race disagreed with, and ``L`` [N, C], the
    rejection mass that lands on the target's winner ``v`` (the reference's
    ``verify``)::

        beta_w = (min(p_w, q_w) - J[w, w]) / (q_w - J[w, w])
        L_v = sum_{t_w > t_v} J[w, v] (1 - beta_w)"""
    order, _, ps, qs, diag, phi = _sorted_race_terms(p, q)
    gap = qs - diag
    beta = torch.where(
        gap > 1e-12, (torch.minimum(ps, qs) - diag) / gap.clamp_min(1e-12), 0.0
    )
    weight = qs * (1.0 - beta)

    def later(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(torch.cumsum(torch.flip(x, [1]), 1), [1]) - x

    landing = ps * (later(weight * phi) - phi * later(weight))
    return _unsort(beta, order), _unsort(landing, order).clamp_min(0.0)


def agreement_token(
    E: torch.Tensor,
    P: torch.Tensor,
    Q: torch.Tensor,
    dpos: torch.Tensor,
    K: torch.Tensor,
    keyed: torch.Tensor,
    u: torch.Tensor,
    F: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``agreement`` at token-level verification. ``keyed`` [B, D] marks the
    rows decided by the race coupling, ``u`` [B, D, 2] holds each row's two
    fresh uniforms and ``F`` [B, D, C] its fresh clocks."""
    B, D, C = E.shape
    depths = torch.arange(D, device=E.device).unsqueeze(0)
    is_draft = depths < K.unsqueeze(1)
    drafts = dpos.clamp(0, C - 1)
    p_rows, q_rows = P.reshape(B * D, C), Q.reshape(B * D, C)
    beta, landing = _agreement_rows(p_rows, q_rows)
    beta, landing = beta.view(B, D, C), landing.view(B, D, C)
    target = race(E, P)
    p_d = P.gather(2, drafts.unsqueeze(2)).squeeze(2)
    q_d = Q.gather(2, drafts.unsqueeze(2)).squeeze(2)
    excess = (P - Q).clamp_min(0.0)
    # Keyed rows: the race coupling.
    beta_d = beta.gather(2, drafts.unsqueeze(2)).squeeze(2)
    accept_keyed = (target == dpos) | (u[..., 0].double() < beta_d)
    keep = torch.where(
        landing > 1e-15,
        torch.minimum(torch.ones_like(landing), excess / landing.clamp_min(1e-300)),
        0.0,
    )
    keep_target = keep.gather(2, target.unsqueeze(2)).squeeze(2)
    rest = (excess - landing).clamp_min(0.0)
    rest = torch.where(rest.sum(2, keepdim=True) > 1e-15, rest, excess)
    replaced = torch.where(u[..., 1].double() < keep_target, target, race(F, rest))
    # Unkeyed rows: speculative sampling with fresh randomness.
    ratio = p_d / torch.where(q_d > 0, q_d, 1.0)
    accept_plain = u[..., 0].double() < ratio.clamp(max=1.0)
    residual = torch.where(excess.sum(2, keepdim=True) > 0, excess, P)
    resampled = race(F, residual)
    accept = torch.where(keyed, accept_keyed, accept_plain)
    token = torch.where(keyed, replaced, resampled)
    # The bonus row emits its race's winner.
    token = torch.where(is_draft, token, target)
    rejected = is_draft & ~accept
    depth = torch.where(rejected.any(dim=1), rejected.long().argmax(dim=1), K)
    rows = torch.arange(B, device=E.device)
    return depth, token[rows, depth]


def decide(
    decision: str,
    E: torch.Tensor,
    P: torch.Tensor,
    Q: torch.Tensor,
    dpos: torch.Tensor,
    K: torch.Tensor,
    law: str = "block",
    pass_aggregator: str = "none",
    pass_source: str = "own",
    keyed: torch.Tensor | None = None,
    u: torch.Tensor | None = None,
    F: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The round's stopping depth [B] and emitted candidate [B] under the
    reference's ``decision`` (``SPEC_DECISIONS``). ``u`` [B, D, 2] are fresh
    uniforms per row (``fresh_stop`` and block ``agreement`` read the first of
    depth 0), ``F`` [B, D, C] fresh Exp(1) clocks and ``keyed`` [B, D] the
    rows with keyed clocks (token ``agreement``)."""
    if decision not in SPEC_DECISIONS:
        raise ValueError(f"decision must be one of {SPEC_DECISIONS}")
    if law == "token" and decision not in TOKEN_DECISIONS:
        raise ValueError(f"token-level verification supports {TOKEN_DECISIONS}")
    if law not in LAWS:
        raise ValueError(f"law must be one of {LAWS}")
    if decision != "residual_j" and pass_aggregator != "none":
        raise ValueError("pass_aggregator applies to residual_j only")
    if decision == "residual_j":
        return residual_race(E, P, Q, dpos, K, law, pass_aggregator, pass_source)
    if decision in ("j_only", "raw_j"):
        return residual_race(E, P, Q, dpos, K, law, use_leftovers=False)
    if decision == "tau_stop":
        return tau_stop(E, P, Q, dpos, K)
    if u is None:
        raise ValueError(f"{decision} needs fresh uniforms")
    if decision == "fresh_stop":
        return fresh_stop(E, P, Q, dpos, K, u[:, 0, 0], law)
    if F is None:
        raise ValueError("agreement needs fresh clocks")
    if law == "block":
        return agreement_block(E, P, Q, dpos, K, u[:, 0, 0], F)
    if keyed is None:
        raise ValueError("token-level agreement needs the keyed rows")
    return agreement_token(E, P, Q, dpos, K, keyed, u, F)
