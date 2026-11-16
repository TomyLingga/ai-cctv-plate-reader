"""Indonesian plate number format: parsing, validation, display format, expiry line removal.

A standard plate is a region code (1-2 letters), a registration number (1-4 digits) and an
optional series (0-3 letters), per Peraturan Kepolisian 7/2021 article 6. Plates outside that
pattern (TNI/Polri, diplomatic with long numbers) are kept as read and flagged invalid.
"""
from __future__ import annotations

import re
from collections.abc import Container
from dataclasses import dataclass

# Region codes from Perpol 7/2021 appendix A (57 codes) plus the four Papua codes introduced
# in 2024 (PG, PS, PT, PY), with the area each one covers. This is the list a new
# installation starts with; the running service keeps its own copy that can be changed
# (storage/regions.py). A plate is never rejected for its code.
REGION_NAMES: dict[str, str] = {
    # Sumatera
    "BL": "Aceh",
    "BK": "Sumatera Utara (Medan dan pesisir timur)",
    "BB": "Sumatera Utara (pesisir barat, Tapanuli)",
    "BA": "Sumatera Barat",
    "BM": "Riau",
    "BP": "Kepulauan Riau (Batam, Tanjung Pinang)",
    "BG": "Sumatera Selatan (Palembang)",
    "BN": "Bangka Belitung",
    "BE": "Lampung",
    "BD": "Bengkulu",
    "BH": "Jambi",
    # Jawa: DKI, Banten, Jawa Barat
    "A": "Banten (Serang, Cilegon, Pandeglang, Lebak)",
    "B": "DKI Jakarta, Depok, Bekasi, Tangerang",
    "D": "Bandung, Cimahi",
    "E": "Cirebon, Indramayu, Majalengka, Kuningan",
    "F": "Bogor, Cianjur, Sukabumi",
    "T": "Karawang, Purwakarta, Subang",
    "Z": "Garut, Tasikmalaya, Sumedang, Ciamis",
    # Jawa: Jawa Tengah, DIY, Jawa Timur
    "G": "Pekalongan, Brebes, Tegal",
    "H": "Semarang, Salatiga, Kendal",
    "K": "Pati, Kudus, Jepara, Grobogan",
    "R": "Banyumas, Cilacap, Purwokerto",
    "AA": "Magelang, Purworejo, Kebumen",
    "AD": "Surakarta, Boyolali, Sragen",
    "AB": "DI Yogyakarta",
    "L": "Surabaya",
    "W": "Sidoarjo, Gresik",
    "N": "Malang, Pasuruan, Probolinggo",
    "S": "Bojonegoro, Mojokerto, Tuban",
    "P": "Besuki, Banyuwangi, Jember",
    "AG": "Kediri, Blitar, Tulungagung",
    "AE": "Madiun, Ngawi, Ponorogo",
    "M": "Madura",
    # Bali dan Nusa Tenggara
    "DK": "Bali",
    "DR": "Lombok",
    "EA": "Sumbawa",
    "DH": "Timor (Kupang)",
    "EB": "Flores",
    "ED": "Sumba",
    # Kalimantan
    "KB": "Kalimantan Barat (Pontianak)",
    "DA": "Kalimantan Selatan (Banjarmasin)",
    "KH": "Kalimantan Tengah (Palangkaraya)",
    "KT": "Kalimantan Timur (Samarinda, Balikpapan)",
    "KU": "Kalimantan Utara",
    # Sulawesi
    "DB": "Sulawesi Utara (Manado)",
    "DL": "Sulawesi Utara (Kepulauan Sangihe)",
    "DM": "Gorontalo",
    "DN": "Sulawesi Tengah (Palu)",
    "DT": "Sulawesi Tenggara (Kendari)",
    "DD": "Sulawesi Selatan (Makassar)",
    "DP": "Sulawesi Selatan (Parepare, Luwu, Toraja)",
    "DW": "Sulawesi Selatan (Bone, Wajo, Soppeng)",
    "DC": "Sulawesi Barat",
    # Maluku dan Papua
    "DE": "Maluku (Ambon)",
    "DG": "Maluku Utara (Ternate)",
    "PA": "Papua (Jayapura)",
    "PB": "Papua Barat",
    "PG": "Papua Pegunungan",
    "PS": "Papua Selatan",
    "PT": "Papua Tengah",
    "PY": "Papua Barat Daya",
}
REGION_CODES: frozenset[str] = frozenset(REGION_NAMES)
REGION_CODE_PATTERN = re.compile(r"^[A-Z]{1,2}$")

# Registration codes that are not regions (diplomatic corps, consulates, honorary consuls,
# state officials). They follow their own numbering and must not be "corrected".
SPECIAL_CODES: frozenset[str] = frozenset({"CD", "CC", "CH", "RI"})

STANDARD_PATTERN = re.compile(r"^([A-Z]{1,2})(\d{1,4})([A-Z]{0,3})$")
_SERIES_THEN_DIGITS = re.compile(r"^([A-Z]{1,2}\d{1,4}[A-Z]{1,3})\d{1,4}$")
_NO_SERIES_LONG_NUMBER = re.compile(r"^([A-Z]{1,2})(\d{5,8})$")

MIN_LENGTH = 3
MAX_LENGTH = 9


@dataclass(frozen=True)
class PlateParts:
    prefix: str
    number: str
    suffix: str

    @property
    def compact(self) -> str:
        return f"{self.prefix}{self.number}{self.suffix}"

    @property
    def formatted(self) -> str:
        return " ".join(part for part in (self.prefix, self.number, self.suffix) if part)


def clean(text: str) -> str:
    """Uppercase and drop everything that is not A-Z or 0-9."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def parse_standard(plate: str) -> PlateParts | None:
    match = STANDARD_PATTERN.match(plate)
    return PlateParts(*match.groups()) if match else None


def is_known_prefix(prefix: str, regions: Container[str] | None = None) -> bool:
    """Whether `prefix` is a listed region code or a special code.

    `regions` is the list in use; without it the built-in list applies.
    """
    return prefix in (REGION_CODES if regions is None else regions) or prefix in SPECIAL_CODES


def format_plate(plate: str) -> str:
    """Return 'BK 1234 ABC' for a standard plate, the input unchanged otherwise."""
    parts = parse_standard(plate)
    return parts.formatted if parts else plate


def strip_expiry(plate: str) -> tuple[str, bool]:
    """Remove digits of the expiry line (month and year) that got appended to the number.

    The expiry line sits under the registration number. An OCR model that reads it too
    produces extra trailing digits, which never occur on a real plate:
    'BK1234ABC09' -> 'BK1234ABC' and 'BK12340927' -> 'BK1234'.

    A plate without series is only shortened when its last four digits can be a month and
    year. 'BK12345' is left alone: it is far more likely 'BK 1234 S' with the S read as 5.
    """
    match = _SERIES_THEN_DIGITS.match(plate)
    if match:
        return match.group(1), True
    match = _NO_SERIES_LONG_NUMBER.match(plate)
    if match and match.group(1) not in SPECIAL_CODES:
        prefix, digits = match.groups()
        if 1 <= int(digits[-4:-2]) <= 12:
            return prefix + digits[:-4], True
    return plate, False
