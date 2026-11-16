import numpy as np
import pytest

from alpr_id.postprocess.voting import OcrRead, PlateVoter

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
PAD = "_"
SLOTS = 10


def make_read(text: str, confidence: float = 0.95, overrides: dict[int, dict[str, float]] | None = None) -> OcrRead:
    """Build a read whose slot probabilities favour `text`.

    `overrides` sets explicit probabilities for some slots, e.g. {6: {"H": 0.5, "M": 0.4}}.
    """
    probs = np.zeros((SLOTS, len(ALPHABET)), dtype=np.float32)
    for slot in range(SLOTS):
        char = text[slot] if slot < len(text) else PAD
        spread = (1.0 - confidence) / (len(ALPHABET) - 1)
        probs[slot, :] = spread
        probs[slot, ALPHABET.index(char)] = confidence
    for slot, values in (overrides or {}).items():
        rest = (1.0 - sum(values.values())) / (len(ALPHABET) - len(values))
        probs[slot, :] = rest
        for char, value in values.items():
            probs[slot, ALPHABET.index(char)] = value
    return OcrRead.from_probs(probs, ALPHABET, PAD)


def voter(**kwargs) -> PlateVoter:
    return PlateVoter(ALPHABET, PAD, confusable_groups=[["H", "M", "W"]], ambiguity_margin=0.15, **kwargs)


def test_read_decodes_text_and_confidence():
    read = make_read("BK1234AB", confidence=0.9)
    assert read.text == "BK1234AB"
    assert read.confidence == pytest.approx(0.9, abs=1e-5)


def test_no_reads_or_only_empty_reads_give_no_result():
    v = voter()
    assert v.result() is None
    v.add(make_read(""))
    assert v.result() is None
    assert v.read_count == 1


def test_single_read_is_returned_as_is():
    v = voter()
    v.add(make_read("BK1234AB", confidence=0.9))
    result = v.result()
    assert result.text == "BK1234AB"
    assert result.confidence == pytest.approx(0.9, abs=1e-3)
    assert (result.reads_used, result.reads_total, result.length_support) == (1, 1, 1.0)
    assert result.alternatives == ()


def test_majority_wins_per_character():
    v = voter()
    for text in ("BK1234AB", "BK1234AB", "BK1284AB"):
        v.add(make_read(text))
    assert v.result().text == "BK1234AB"


def test_one_confident_read_outweighs_two_hesitant_ones():
    # Two blurry frames lean towards H at slot 6, one sharp frame is sure about M.
    v = voter()
    v.add(make_read("BK1234HA", overrides={6: {"H": 0.50, "M": 0.42}}))
    v.add(make_read("BK1234HA", overrides={6: {"H": 0.52, "M": 0.40}}))
    v.add(make_read("BK1234MA", overrides={6: {"M": 0.98, "H": 0.01}}))
    result = v.result()
    assert result.text == "BK1234MA"
    # Counting finished strings would have picked the H reading two to one.


def test_close_call_inside_a_confusable_group_yields_an_alternative():
    v = voter()
    v.add(make_read("BK1234HA", overrides={6: {"H": 0.50, "M": 0.42}}))
    result = v.result()
    assert result.text == "BK1234HA"
    assert result.alternatives == ("BK1234MA",)
    assert result.char_confidences[6] == pytest.approx(0.50, abs=1e-5)


def test_clear_winner_has_no_alternative():
    v = voter()
    v.add(make_read("BK1234HA", overrides={6: {"H": 0.95, "M": 0.02}}))
    assert v.result().alternatives == ()


def test_close_call_between_unrelated_characters_is_not_an_alternative():
    v = voter()
    v.add(make_read("BK1234AB", overrides={7: {"B": 0.50, "8": 0.45}}))
    assert v.result().alternatives == ()


def test_reads_of_another_length_do_not_shift_the_vote():
    v = voter()
    v.add(make_read("BK1234AB"))
    v.add(make_read("BK1234AB"))
    v.add(make_read("BK134AB"))  # dropped a digit: every later slot is shifted
    result = v.result()
    assert result.text == "BK1234AB"
    assert (result.reads_used, result.reads_total) == (2, 3)
    assert result.length_support == pytest.approx(2 / 3, abs=0.01)
    assert result.confidence < 0.95 * 0.7


def test_confidence_is_weighted_by_read_confidence():
    v = voter()
    v.add(make_read("BK1234AB", confidence=0.99))
    v.add(make_read("BK1234AX", confidence=0.40))
    result = v.result()
    assert result.text == "BK1234AB"


def test_padding_never_wins_inside_the_voted_length():
    v = voter()
    v.add(make_read("BK1234AB", overrides={7: {"B": 0.45, "_": 0.44}}))
    assert v.result().text == "BK1234AB"
