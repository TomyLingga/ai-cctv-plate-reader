import pytest

from alpr_id.postprocess.plate_format import (
    REGION_CODES,
    PlateParts,
    clean,
    format_plate,
    is_known_prefix,
    parse_standard,
    strip_expiry,
)


def test_clean_keeps_only_uppercase_letters_and_digits():
    assert clean("bk 1234-abc") == "BK1234ABC"
    assert clean(" B.1234|xy ") == "B1234XY"
    assert clean("") == ""


@pytest.mark.parametrize(
    ("plate", "parts"),
    [
        ("BK1234ABC", PlateParts("BK", "1234", "ABC")),
        ("B1A", PlateParts("B", "1", "A")),
        ("D1234ABC", PlateParts("D", "1234", "ABC")),
        ("BK8082XH", PlateParts("BK", "8082", "XH")),
        ("AB12", PlateParts("AB", "12", "")),
    ],
)
def test_parse_standard_accepts_standard_plates(plate, parts):
    assert parse_standard(plate) == parts


@pytest.mark.parametrize("plate", ["8039200", "BK12345", "ABC123", "BK1234ABCD", "BK", "1234", "12345VII", ""])
def test_parse_standard_rejects_other_shapes(plate):
    assert parse_standard(plate) is None


def test_format_plate_adds_spaces_only_to_standard_plates():
    assert format_plate("BK1234ABC") == "BK 1234 ABC"
    assert format_plate("B1234") == "B 1234"
    assert format_plate("8039200") == "8039200"


def test_region_codes_cover_what_the_old_list_missed():
    assert len(REGION_CODES) == 61
    for code in ("D", "E", "F", "DP", "DW", "PA", "PB", "PG", "PS", "PT", "PY", "BK"):
        assert code in REGION_CODES
    for code in ("DU", "EF", "DF", "DS", "CD", "CC"):
        assert code not in REGION_CODES


def test_known_prefix_includes_diplomatic_and_state_codes():
    assert is_known_prefix("BK")
    assert is_known_prefix("CD")
    assert is_known_prefix("RI")
    assert not is_known_prefix("BI")


@pytest.mark.parametrize(
    ("read", "expected"),
    [
        ("BK1234ABC09", ("BK1234ABC", True)),
        ("BK1234ABC0", ("BK1234ABC", True)),
        ("BK1234AB0927", ("BK1234AB", True)),
        ("BK12340927", ("BK1234", True)),
        ("B1230927", ("B123", True)),
        ("BK1234ABC", ("BK1234ABC", False)),
        ("BK1234", ("BK1234", False)),
        ("CD12345", ("CD12345", False)),
        ("BK12345", ("BK12345", False)),  # 23 is not a month: this is not an expiry date
        ("BK123458", ("BK123458", False)),
        ("8039200", ("8039200", False)),
    ],
)
def test_strip_expiry(read, expected):
    assert strip_expiry(read) == expected
