# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-token weights for weighted detection tests, on ConcordMark's grid of
weights 1, 1.25, ..., 3.5 (levels 0 to 10), whose weighted Exp(1) sums have a
saddlepoint tail (`concordmark._weighted_tail`).

Two weightings: AutoMark's surprisal weights (Gloaguen et al., 2026), from a
detection model's log-probability of each token, and a text-only one that
weighs tokens starting a word or a sentence."""

import math
from collections.abc import Sequence
from typing import Any

import regex as re
import torch

from vllm.v1.watermarking.concordmark import _WEIGHT_LEVELS

_LEVEL_STEP = 0.25
_MAX_SURPRISAL = 2.5
_SENTENCE_END = ".!?:;"
_BYTE_TOKEN = re.compile("<0x([0-9A-Fa-f]{2})>")


def weight_level(weight: float) -> int:
    """The level of a weight on the grid."""
    level = (weight - 1) / _LEVEL_STEP
    if not 0 <= round(level) < len(_WEIGHT_LEVELS) or abs(level - round(level)) > 1e-9:
        raise ValueError(f"weights lie on 1, 1.25, ..., 3.5, got {weight}")
    return round(level)


def surprisal_levels(logprobs: Sequence[float | None]) -> torch.Tensor:
    """AutoMark's weight levels: weight ``1 + min(surprisal, 2.5)`` rounded to
    the grid; tokens without a log-probability weigh 1."""
    levels = [
        0
        if logprob is None or math.isnan(logprob)
        else round(min(max(-logprob, 0.0), _MAX_SURPRISAL) / _LEVEL_STEP)
        for logprob in logprobs
    ]
    return torch.tensor(levels, dtype=torch.int64)


def _byte_decoder() -> dict[str, int]:
    """Inverse of byte-level BPE's byte-to-character map (GPT-2): printable
    bytes stand for themselves, the others for U+0100 onwards in byte order."""
    printable = {
        *range(ord("!"), ord("~") + 1),
        *range(ord("¡"), ord("¬") + 1),
        *range(ord("®"), ord("ÿ") + 1),
    }
    mapping, offset = {}, 0
    for byte in range(256):
        if byte in printable:
            mapping[chr(byte)] = byte
        else:
            mapping[chr(256 + offset)] = byte
            offset += 1
    return mapping


def _token_texts(tokenizer: Any) -> list[str]:
    """Each vocabulary token as text, with a leading space where the token
    starts a word: SentencePiece pieces read with "▁" as a space, byte-level
    BPE pieces through the byte decoder, other tokenizers by decoding."""
    pieces = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
    if any(isinstance(p, str) and p.startswith("▁") for p in pieces):
        texts = []
        for piece in pieces:
            piece = piece or ""
            byte = _BYTE_TOKEN.fullmatch(piece)
            if byte:
                value = int(byte.group(1), 16)
                piece = chr(value) if value < 0x80 else ""
            texts.append(piece.replace("▁", " "))
        return texts
    if any(isinstance(p, str) and p.startswith("Ġ") for p in pieces):
        decoder = _byte_decoder()
        return [
            bytes(decoder.get(c, ord("?")) for c in (piece or "")).decode(
                "utf-8", errors="replace"
            )
            for piece in pieces
        ]
    return [tokenizer.decode([token_id]) for token_id in range(len(tokenizer))]


class WordStartWeights:
    """Text-only weights: ``weight`` for tokens that start a word (they start
    with a space, or follow a token without letters or digits) or follow a
    sentence end (a token ending in one of ``.!?:;``), 1 for tokens after a
    newline, the first token and tokens inside a word."""

    def __init__(self, tokenizer: Any, weight: float = 3.0) -> None:
        self.level = weight_level(weight)
        texts = _token_texts(tokenizer)
        stripped = [text.strip() for text in texts]
        self.has_newline = ["\n" in text for text in texts]
        self.sentence_end = [bool(s) and s[-1] in _SENTENCE_END for s in stripped]
        self.starts_space = [text.startswith(" ") for text in texts]
        self.no_alnum = [bool(s) and not any(c.isalnum() for c in s) for s in stripped]

    def _flag(self, flags: list[bool], token_id: int) -> bool:
        return 0 <= token_id < len(flags) and flags[token_id]

    def levels(
        self, token_ids: Sequence[int], positions: Sequence[int]
    ) -> torch.Tensor:
        """Weight levels of the tokens at ``positions`` of ``token_ids``."""
        levels = []
        for position in positions:
            token = token_ids[position]
            previous = token_ids[position - 1] if position > 0 else None
            if previous is None or self._flag(self.has_newline, previous):
                heavy = False  # after a newline
            elif self._flag(self.sentence_end, previous):
                heavy = True  # after a sentence end
            else:  # a word start, or inside a word
                heavy = self._flag(self.starts_space, token) or self._flag(
                    self.no_alnum, previous
                )
            levels.append(self.level if heavy else 0)
        return torch.tensor(levels, dtype=torch.int64)
