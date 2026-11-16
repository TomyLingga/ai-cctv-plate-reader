"""Position-based correction of look-alike characters, and the full normalisation of a read.

Letters are only turned into digits inside the number, and digits into letters inside the
region code or the series. Which characters form which zone is found by trying every split
of the string and keeping the one that needs the fewest swaps.

Two brakes keep a badly damaged read from being turned into something that merely looks
valid: at most a quarter of the characters may be swapped, and at most half of any zone.
"""
from __future__ import annotations

import math
from collections.abc import Container
from dataclasses import dataclass, field, replace

import numpy as np

from alpr_id.postprocess.plate_format import (
    MAX_LENGTH,
    MIN_LENGTH,
    SPECIAL_CODES,
    PlateParts,
    clean,
    is_known_prefix,
    parse_standard,
    strip_expiry,
)
from alpr_id.postprocess.regions import RegionBook
from alpr_id.postprocess.voting import VoteResult

# Pairs requested for this project: O/0, I/1, B/8, S/5, Z/2. Extend only after the confusion
# matrix from real CCTV crops shows another pair is worth it.
LETTER_TO_DIGIT: dict[str, str] = {"O": "0", "I": "1", "B": "8", "S": "5", "Z": "2"}
DIGIT_TO_LETTER: dict[str, str] = {digit: letter for letter, digit in LETTER_TO_DIGIT.items()}

MAX_SWAP_RATIO = 0.25
MIN_LENGTH_FOR_SWAPS = 5


def max_swaps(length: int) -> int:
    """How many characters may be swapped in a read of this length.

    Short strings get none: with four characters a single swap is already enough to turn
    noise such as 'DOSS' into something plate-shaped.
    """
    return int(length * MAX_SWAP_RATIO) if length >= MIN_LENGTH_FOR_SWAPS else 0


@dataclass(frozen=True)
class Swap:
    index: int
    read: str
    corrected: str


@dataclass(frozen=True)
class PlateResult:
    """A plate after normalisation. `plate` has no spaces, `formatted` has them."""

    plate: str
    formatted: str
    format_valid: bool
    region_known: bool
    raw: str
    parts: PlateParts | None = None
    swaps: tuple[Swap, ...] = field(default_factory=tuple)
    expiry_stripped: bool = False
    # The region code as it was read, when it was not on the list and got replaced.
    region_read: str | None = None

    @property
    def region_corrected(self) -> bool:
        return self.region_read is not None


@dataclass(frozen=True)
class _Candidate:
    parts: PlateParts
    swaps: tuple[Swap, ...]
    known: bool

    @property
    def rank(self) -> tuple[bool, int, int, str]:
        # Known region first, then fewest swaps, then the longer number, then a stable order.
        return (not self.known, len(self.swaps), -len(self.parts.number), self.parts.compact)


def _to_zone(segment: str, offset: int, table: dict[str, str], wanted_digit: bool) -> tuple[str, list[Swap]] | None:
    out: list[str] = []
    swaps: list[Swap] = []
    for i, char in enumerate(segment):
        if char.isdigit() == wanted_digit:
            out.append(char)
        elif char in table:
            out.append(table[char])
            swaps.append(Swap(offset + i, char, table[char]))
        else:
            return None
    if len(swaps) > math.ceil(len(segment) / 2):
        return None
    return "".join(out), swaps


def _candidates(text: str, regions: Container[str] | None) -> list[_Candidate]:
    n = len(text)
    found: list[_Candidate] = []
    for prefix_len in (1, 2):
        for number_len in (1, 2, 3, 4):
            suffix_len = n - prefix_len - number_len
            if not 0 <= suffix_len <= 3:
                continue
            a, b = prefix_len, prefix_len + number_len
            prefix = _to_zone(text[:a], 0, DIGIT_TO_LETTER, wanted_digit=False)
            number = _to_zone(text[a:b], a, LETTER_TO_DIGIT, wanted_digit=True)
            suffix = _to_zone(text[b:], b, DIGIT_TO_LETTER, wanted_digit=False)
            if prefix is None or number is None or suffix is None:
                continue
            swaps = (*prefix[1], *number[1], *suffix[1])
            if len(swaps) > max_swaps(n):
                continue
            parts = PlateParts(prefix[0], number[0], suffix[0])
            found.append(_Candidate(parts, swaps, is_known_prefix(parts.prefix, regions)))
    return found


def correct(text: str, regions: Container[str] | None = None) -> _Candidate | None:
    """Best standard reading of `text`, or None when no acceptable one exists.

    A read that is already standard is still replaced when its region code is unknown and a
    reading with a known code exists, e.g. 'BI234AB' (unknown 'BI') becomes 'B1234AB'.
    """
    if not MIN_LENGTH <= len(text) <= MAX_LENGTH:
        return None
    # A swap is a guess. It is only accepted when it lands on a known region code, and never
    # on diplomatic or state plates, whose numbering does not follow the standard pattern.
    special = text[:2] in SPECIAL_CODES
    candidates = [c for c in _candidates(text, regions) if not c.swaps or (c.known and not special)]
    return min(candidates, key=lambda c: c.rank) if candidates else None


def normalize_plate(raw_text: str, regions: Container[str] | None = None) -> PlateResult:
    """Turn a raw OCR string into a plate: clean, drop expiry digits, correct, validate.

    `regions` is the list of region codes in use; without it the built-in list applies.
    """
    raw = clean(raw_text)
    variants = [(raw, False)]
    stripped, was_stripped = strip_expiry(raw)
    if was_stripped:
        variants.append((stripped, True))

    best: tuple[_Candidate, bool] | None = None
    for text, expiry in variants:
        candidate = correct(text, regions)
        if candidate is None:
            continue
        if best is None:
            best = (candidate, expiry)
        elif best[0].known and len(best[0].swaps) <= 1:
            # Dropping characters is a bigger guess than swapping one look-alike: 'BK1234A8'
            # is 'BK 1234 AB', not 'BK 1234 A' followed by a digit of the expiry line.
            break
        elif (candidate.rank[:2], expiry) < (best[0].rank[:2], best[1]):
            best = (candidate, expiry)

    if best is None:
        return PlateResult(plate=raw, formatted=raw, format_valid=False, region_known=False, raw=raw)

    candidate, expiry = best
    parts = candidate.parts
    return PlateResult(
        plate=parts.compact,
        formatted=parts.formatted,
        format_valid=parse_standard(parts.compact) is not None,
        region_known=candidate.known,
        raw=raw,
        parts=parts,
        swaps=candidate.swaps,
        expiry_stripped=expiry,
    )


def settle_region(plate: PlateResult, vote: VoteResult, alphabet: str, regions: RegionBook) -> tuple[PlateResult, VoteResult]:
    """Replace a region code that is not on the list by the listed code the reads support best.

    'BF 8418 FD' is not a plate that exists: no region uses BF. The reads behind it still say
    how likely every other letter was at that position, and among the listed codes the one
    with the most support is taken, here BK. The number and the series are left alone.

    The confidence of the vote is recomputed with the probability the reads gave to the new
    letters, so a replacement with little support lowers it. `vote.text` keeps the string as
    it was read. Plates outside the standard pattern, special codes and votes without
    character probabilities are returned unchanged.
    """
    parts = plate.parts
    if parts is None or not plate.format_valid or plate.region_known or vote.shares is None:
        return plate, vote
    length = len(parts.prefix)
    if vote.shares.shape[0] < length:
        return plate, vote
    index = {char: i for i, char in enumerate(alphabet)}
    found = regions.nearest(vote.shares[:length], index)
    if found is None:
        return plate, vote
    code = found[0]
    settled = PlateParts(code, parts.number, parts.suffix)
    char_conf = list(vote.char_confidences)
    for position, char in enumerate(code):
        char_conf[position] = float(vote.shares[position, index[char]])
    confidence = float(np.mean(char_conf)) * vote.length_support
    return (
        replace(plate, plate=settled.compact, formatted=settled.formatted, region_known=True, parts=settled,
                region_read=parts.prefix),
        replace(vote, char_confidences=tuple(char_conf), confidence=confidence),
    )
