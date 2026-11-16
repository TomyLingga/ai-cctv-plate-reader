import pytest

from alpr_id.postprocess.correction import Swap, max_swaps, normalize_plate


@pytest.mark.parametrize(
    ("read", "plate", "formatted"),
    [
        ("BK 1234 ABC", "BK1234ABC", "BK 1234 ABC"),
        ("bk8082xh", "BK8082XH", "BK 8082 XH"),
        ("D 1234 ABC", "D1234ABC", "D 1234 ABC"),  # rejected by the old service: D was missing from its list
        ("F 1234 AB", "F1234AB", "F 1234 AB"),
        ("PA 1234 AB", "PA1234AB", "PA 1234 AB"),
        ("L 12 A", "L12A", "L 12 A"),
        ("B 1 A", "B1A", "B 1 A"),
    ],
)
def test_standard_plates_pass_through_unchanged(read, plate, formatted):
    result = normalize_plate(read)
    assert (result.plate, result.formatted) == (plate, formatted)
    assert result.format_valid and result.region_known
    assert result.swaps == ()


@pytest.mark.parametrize(
    ("read", "plate"),
    [
        ("8K 8227 X0", "BK8227XO"),  # case from timbangan-ocr-service domain_repair
        ("BK I234 ABC", "BK1234ABC"),  # dropped by the old service
        ("BK 1234 A8C", "BK1234ABC"),  # became 'BK 1234 A C' in the old service
        ("81234AB", "B1234AB"),
        ("BKO123AB", "BK0123AB"),
        ("BK12Z4AB", "BK1224AB"),
        ("BK1S34AB", "BK1534AB"),
        ("8K1234ABC", "BK1234ABC"),  # the old service silently returned 'K 1234 ABC'
        ("BK12345", "BK1234S"),  # a trailing S read as 5 must not be taken for an expiry digit
        ("BK12348", "BK1234B"),
        ("BK11225", "BK1122S"),  # even when the last four digits could be a month and year
        ("BK1234A8", "BK1234AB"),
        ("BK1234AB0", "BK1234ABO"),
    ],
)
def test_lookalikes_are_corrected_by_position(read, plate):
    result = normalize_plate(read)
    assert result.plate == plate
    assert result.format_valid and result.region_known
    assert result.swaps


def test_swaps_report_position_and_characters():
    result = normalize_plate("8K8227X0")
    assert result.swaps == (Swap(0, "8", "B"), Swap(7, "0", "O"))
    assert result.raw == "8K8227X0"


def test_unknown_region_is_replaced_when_a_known_reading_exists():
    # 'BI' is not a region code; reading the I as 1 gives region B.
    result = normalize_plate("BI234AB")
    assert result.plate == "B1234AB"
    assert result.region_known


@pytest.mark.parametrize("read", ["BK1234IO", "BL1234IQ", "BK1234OQI", "B1234SOS"])
def test_letters_in_the_series_are_never_turned_into_digits(read):
    result = normalize_plate(read)
    assert result.plate == read
    assert result.swaps == ()


@pytest.mark.parametrize(
    ("read", "plate"),
    [
        ("BK 1234 ABC 09.27", "BK1234ABC"),
        ("BK1234AB0927", "BK1234AB"),
        ("BK 1234 09.27", "BK1234"),
        ("BK1234ABC0", "BK1234ABC"),
        ("BK120925", "BK12"),  # reading it as a plate would need two swaps
    ],
)
def test_expiry_line_digits_are_dropped(read, plate):
    result = normalize_plate(read)
    assert result.plate == plate
    assert result.expiry_stripped and result.format_valid


@pytest.mark.parametrize(
    "read",
    [
        "8039200",  # TNI: 80392-00
        "12345VII",  # Polri: 12345-VII
        "CD12345",  # diplomatic with country code
        "DOSS",  # noise that one swap could turn into 'D 0 SS'
        "BK1234ABCD",
        "ZZZZZZ",
        "",
    ],
)
def test_plates_outside_the_pattern_are_returned_as_read_and_flagged(read):
    result = normalize_plate(read)
    assert result.plate == read
    assert result.formatted == read
    assert not result.format_valid
    assert result.swaps == ()


def test_diplomatic_and_state_plates_that_fit_the_pattern_are_kept():
    for read, formatted in (("CD 12 5", "CD 125"), ("RI 1", "RI 1")):
        result = normalize_plate(read)
        assert result.formatted == formatted
        assert result.format_valid and result.region_known


def test_unknown_region_without_better_reading_stays_valid_but_unknown():
    result = normalize_plate("XX1234AB")
    assert result.plate == "XX1234AB"
    assert result.format_valid
    assert not result.region_known


def test_swap_budget_by_length():
    assert [max_swaps(n) for n in (3, 4, 5, 7, 8, 9)] == [0, 0, 1, 1, 2, 2]


def test_too_many_swaps_leave_the_read_alone():
    # Three of four digits would have to be swapped.
    result = normalize_plate("BKIZS4")
    assert result.plate == "BKIZS4"
    assert not result.format_valid
