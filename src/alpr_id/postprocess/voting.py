"""Combine several OCR reads of the same vehicle into one plate.

The OCR model gives a probability for every character of its alphabet at every slot. Voting
adds those probabilities up across reads, weighted by how confident each read was, instead of
counting finished strings. A sharp frame that is sure about a character can then outweigh
several blurry frames that lean the other way.

Reads are first grouped by length, because the model writes characters left-aligned: a read
that dropped one character would shift every later slot and vote for the wrong positions.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

MAX_ALTERNATIVES = 3
MIN_RUNNER_UP_SHARE = 0.05


@dataclass(frozen=True)
class OcrRead:
    """One OCR pass over one crop."""

    probs: np.ndarray  # (slots, alphabet), probabilities
    text: str  # decoded characters without padding
    confidence: float  # mean probability of the decoded characters, 0 when text is empty

    @classmethod
    def from_probs(cls, probs: np.ndarray, alphabet: str, pad_char: str) -> OcrRead:
        indices = probs.argmax(axis=-1)
        chars = "".join(alphabet[i] for i in indices).rstrip(pad_char)
        confidence = float(probs.max(axis=-1)[: len(chars)].mean()) if chars else 0.0
        return cls(probs=probs, text=chars, confidence=confidence)


@dataclass(frozen=True)
class VoteResult:
    text: str
    confidence: float
    char_confidences: tuple[float, ...]
    reads_used: int
    reads_total: int
    length_support: float
    alternatives: tuple[str, ...] = field(default_factory=tuple)
    # What the reads together say about every character: one row per character of `text`,
    # one column per letter of the alphabet. Kept for decisions made after the vote.
    shares: np.ndarray | None = field(default=None, repr=False, compare=False)


class PlateVoter:
    """Accumulates reads for one track and reports the current consensus."""

    def __init__(
        self,
        alphabet: str,
        pad_char: str,
        confusable_groups: Sequence[Sequence[str]] = (),
        ambiguity_margin: float = 0.15,
    ) -> None:
        self._alphabet = alphabet
        self._pad_index = alphabet.index(pad_char)
        self._index = {char: i for i, char in enumerate(alphabet)}
        self._groups = [tuple(group) for group in confusable_groups]
        self._margin = ambiguity_margin
        self._reads: list[OcrRead] = []

    def add(self, read: OcrRead) -> None:
        self._reads.append(read)

    @property
    def read_count(self) -> int:
        return len(self._reads)

    def result(self) -> VoteResult | None:
        """Current consensus, or None when no read produced any character."""
        readable = [read for read in self._reads if read.text and read.confidence > 0]
        if not readable:
            return None

        weight_by_length: dict[int, float] = {}
        for read in readable:
            weight_by_length[len(read.text)] = weight_by_length.get(len(read.text), 0.0) + read.confidence
        length = max(weight_by_length, key=lambda n: (weight_by_length[n], n))
        total_weight = sum(weight_by_length.values())
        support = weight_by_length[length] / total_weight

        group = [read for read in readable if len(read.text) == length]
        group_weight = sum(read.confidence for read in group)
        shares = sum(read.confidence * read.probs[:length] for read in group) / group_weight
        shares[:, self._pad_index] = 0.0

        winners = shares.argmax(axis=-1)
        text = "".join(self._alphabet[i] for i in winners)
        char_conf = tuple(float(shares[pos, i]) for pos, i in enumerate(winners))
        confidence = float(np.mean(char_conf)) * support

        return VoteResult(
            text=text,
            confidence=confidence,
            char_confidences=char_conf,
            reads_used=len(group),
            reads_total=len(self._reads),
            length_support=support,
            alternatives=self._alternatives(text, shares),
            shares=shares,
        )

    def _alternatives(self, text: str, shares: np.ndarray) -> tuple[str, ...]:
        """Other readings where a look-alike character was almost as likely as the winner."""
        found: list[tuple[float, str]] = []
        for pos, char in enumerate(text):
            for members in self._groups:
                if char not in members:
                    continue
                winner_share = float(shares[pos, self._index[char]])
                for other in members:
                    if other == char or other not in self._index:
                        continue
                    other_share = float(shares[pos, self._index[other]])
                    gap = winner_share - other_share
                    if other_share >= MIN_RUNNER_UP_SHARE and gap < self._margin:
                        found.append((gap, text[:pos] + other + text[pos + 1 :]))
        found.sort()
        return tuple(dict.fromkeys(alt for _, alt in found))[:MAX_ALTERNATIVES]
