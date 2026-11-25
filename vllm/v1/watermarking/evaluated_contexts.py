# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The contexts earlier speculative verification steps evaluated, for
``deduplicate_contexts="strict"``.

A lookahead or residual race reads the keyed values of every row of a
verification step, rows after the first rejection included, and those rows'
contexts can come back at a later position of the reply. ``single_turn``
deduplication only knows the contexts of emitted positions (and of the earlier
rows of the step's own path), so a context read by a discarded row would be
raced on again with the same keyed values. ``strict`` also treats as repeated
every context a verification row of an earlier step of the request evaluated:
a repeated row races on fresh randomness instead, so no keyed value is read
twice and every row stays an exact sample.

Per request slot, a ring buffer keeps 64-bit hashes of those contexts: a fixed
shape, so lookups need no host synchronization (the drafter may run inside a
CUDA graph). A hash collision or a lookup of a slot's stale entries can only
mark an extra row as repeated, which is exact (whether a row races on fresh
randomness is a function of the contexts alone). Once a request has committed
more contexts than the buffer holds, the oldest are forgotten.
"""

import torch

# 64-bit FNV-1a-style mixing of the context tokens (two's complement wrap).
_HASH_OFFSET = 0x243F6A8885A308D3
_HASH_PRIME = 0x100000001B3


def context_hashes(contexts: torch.Tensor) -> torch.Tensor:
    """A 64-bit hash of every row of ``contexts`` [rows, width] (int64, -1
    padding allowed): int64 [rows], the same on every device."""
    contexts = contexts.long()
    h = torch.full(
        (contexts.shape[0],), _HASH_OFFSET, dtype=torch.int64, device=contexts.device
    )
    for j in range(contexts.shape[1]):
        h = (h ^ contexts[:, j]) * _HASH_PRIME
        h = h ^ (h >> 29)
    return h


class EvaluatedContexts:
    """Hashes of the contexts each request slot's verification steps
    evaluated (module docstring)."""

    def __init__(self, max_num_reqs: int, capacity: int, device: torch.device) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.max_num_reqs = max_num_reqs
        self.capacity = capacity
        # One extra slot takes the writes of padding rows.
        self.hashes = torch.zeros(
            max_num_reqs + 1, capacity, dtype=torch.int64, device=device
        )
        self.counts = torch.zeros(max_num_reqs + 1, dtype=torch.int64, device=device)
        self._columns = torch.arange(capacity, device=device)

    def reset(self, req_idx: int) -> None:
        """Forget a slot's contexts (a new request)."""
        self.counts[req_idx] = 0

    def _slots(self, req_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        req = req_indices.long()
        valid = (req >= 0) & (req < self.max_num_reqs)
        return torch.where(valid, req, self.max_num_reqs), valid

    def contains(self, contexts: torch.Tensor, req_indices: torch.Tensor) -> torch.Tensor:
        """Whether each row's context is among its slot's committed contexts
        (bool [rows]; padding rows, ``req_indices`` -1, are not)."""
        if contexts.shape[0] == 0:
            return torch.zeros(0, dtype=torch.bool, device=contexts.device)
        slots, valid = self._slots(req_indices)
        h = context_hashes(contexts)
        filled = self.counts[slots].clamp_max(self.capacity)
        live = self._columns.unsqueeze(0) < filled.unsqueeze(1)
        hit = (self.hashes[slots] == h.unsqueeze(1)) & live
        return valid & hit.any(dim=1)

    def commit(
        self,
        contexts: torch.Tensor,
        req_indices: torch.Tensor,
        local_positions: torch.Tensor,
        rows: torch.Tensor,
    ) -> None:
        """Remember the contexts of the verification rows in ``rows`` [rows]
        bool. Each request's rows are contiguous with ``local_positions`` 0, 1,
        ...; ``rows`` must select whole requests (all of a request's rows or
        none)."""
        if contexts.shape[0] == 0:
            return
        slots, valid = self._slots(req_indices)
        keep = valid & rows
        slots = torch.where(keep, slots, self.max_num_reqs)
        position = (self.counts[slots] + local_positions.long()) % self.capacity
        self.hashes[slots, position] = context_hashes(contexts)
        self.counts.index_add_(0, slots, keep.long())
        self.counts[self.max_num_reqs] = 0


__all__ = ["EvaluatedContexts", "context_hashes"]
