"""Two rules added after the first nights on the weighbridge.

1. A region code that is not on the list is replaced by the nearest listed one.
2. A plate is read once the vehicle stands still, not while it rolls in or drives off.
"""
import sqlite3

import numpy as np
import pytest
from test_pipeline import ALPHABET, CENTRE_BOX, Harness, ScriptedOcr, make_read

from alpr_id.config import OcrConfig, PostprocessConfig
from alpr_id.postprocess.correction import normalize_plate, settle_region
from alpr_id.postprocess.plate_format import REGION_CODES
from alpr_id.postprocess.regions import RegionBook
from alpr_id.postprocess.voting import OcrRead, PlateVoter
from alpr_id.storage.db import EventStore
from alpr_id.storage.regions import RegionStore

INDEX = {char: i for i, char in enumerate(ALPHABET)}


def read_with(text: str, doubts: dict[int, dict[str, float]], confidence: float = 0.97) -> OcrRead:
    """A read of `text` where some slots are unsure: {slot: {letter: probability, ...}}."""
    probs = make_read(text, confidence).probs.copy()
    for slot, letters in doubts.items():
        rest = (1.0 - sum(letters.values())) / (len(ALPHABET) - len(letters))
        probs[slot, :] = rest
        for letter, probability in letters.items():
            probs[slot, INDEX[letter]] = probability
    return OcrRead.from_probs(probs, ALPHABET, "_")


def vote_of(*reads: OcrRead):
    voter = PlateVoter(ALPHABET, "_")
    for read in reads:
        voter.add(read)
    return voter.result()


# The second letter as the service saw it on the truck it reported as BF 8418 FD.
BF_READ = read_with("BF8418FD", {1: {"F": 0.46, "R": 0.26, "C": 0.08, "K": 0.06, "E": 0.04}})


# -- nearest listed code -------------------------------------------------------------------


def test_nearest_code_is_the_listed_one_the_read_supports_best():
    code, probability = RegionBook().nearest(BF_READ.probs[:2], INDEX)
    assert code == "BK"
    assert probability == pytest.approx(BF_READ.probs[0, INDEX["B"]] * 0.06, rel=1e-3)


def test_only_codes_of_the_same_length_are_compared():
    # 'F' alone is a listed code (Bogor), but taking it would shift every later character.
    one_letter = read_with("C1234AB", {0: {"C": 0.5, "G": 0.3, "D": 0.1}})
    assert RegionBook().nearest(one_letter.probs[:1], INDEX)[0] == "G"
    assert RegionBook(codes={"B", "D"}).nearest(BF_READ.probs[:2], INDEX) is None


def test_code_seen_often_at_the_site_wins_a_close_call():
    # OCR leans slightly to DK (Bali); every truck at this weighbridge so far carried BK.
    unsure = read_with("JK9027AG", {0: {"J": 0.65, "D": 0.072, "B": 0.069}})
    plain = RegionBook(local_boost=2.0)
    assert plain.nearest(unsure.probs[:2], INDEX)[0] == "DK"
    local = RegionBook(local_boost=2.0)
    local.observe("BK", 12)
    assert local.nearest(unsure.probs[:2], INDEX)[0] == "BK"
    assert local.seen() == {"BK": 12}


def test_site_habit_does_not_overrule_a_clear_read():
    # 'BW' is not listed; M was clearly the runner-up, K hardly a candidate.
    clear = read_with("BW1234AB", {1: {"W": 0.55, "M": 0.40, "K": 0.004}})
    local = RegionBook(local_boost=2.0)
    local.observe("BK", 5000)
    assert local.nearest(clear.probs[:2], INDEX)[0] == "BM"


def test_no_listed_code_with_any_support_means_no_answer():
    probs = np.zeros((2, len(ALPHABET)), dtype=np.float32)
    probs[0, INDEX["Q"]] = probs[1, INDEX["Q"]] = 1.0
    assert RegionBook().nearest(probs, INDEX) is None


# -- settling a vote -----------------------------------------------------------------------


def test_unlisted_code_is_replaced_and_the_rest_of_the_plate_is_kept():
    vote = vote_of(BF_READ)
    plate = normalize_plate(vote.text)
    assert (plate.plate, plate.region_known) == ("BF8418FD", False)

    settled, new_vote = settle_region(plate, vote, ALPHABET, RegionBook())
    assert (settled.plate, settled.formatted) == ("BK8418FD", "BK 8418 FD")
    assert settled.region_known and settled.format_valid
    assert settled.region_corrected and settled.region_read == "BF"
    assert settled.raw == "BF8418FD"
    # The string as read stays; the confidence now reflects how little K was supported.
    assert new_vote.text == "BF8418FD"
    assert new_vote.char_confidences[1] == pytest.approx(0.06, rel=1e-3)
    assert new_vote.confidence < vote.confidence
    assert new_vote.char_confidences[2:] == vote.char_confidences[2:]


def test_the_list_decides_which_code_is_nearest():
    vote = vote_of(BF_READ)
    without_bk = RegionBook(codes=REGION_CODES - {"BK"})
    settled, _ = settle_region(normalize_plate(vote.text, without_bk), vote, ALPHABET, without_bk)
    assert settled.plate == "BE8418FD"
    with_bf = RegionBook(codes=REGION_CODES | {"BF"})
    kept, same_vote = settle_region(normalize_plate(vote.text, with_bf), vote, ALPHABET, with_bf)
    assert kept.plate == "BF8418FD" and kept.region_known and not kept.region_corrected
    assert same_vote is vote


@pytest.mark.parametrize(
    "text",
    [
        "BK1234AB",  # listed code
        "CD12345",  # diplomatic: always accepted, never replaced
        "8039200",  # outside the standard pattern
        "12345",
    ],
)
def test_plates_that_need_no_code_are_left_alone(text):
    vote = vote_of(make_read(text, 0.9))
    plate = normalize_plate(vote.text)
    settled, new_vote = settle_region(plate, vote, ALPHABET, RegionBook())
    assert settled == plate
    assert new_vote is vote


def test_normalize_uses_the_list_it_is_given():
    assert not normalize_plate("XX1234AB").region_known
    assert normalize_plate("XX1234AB", RegionBook(codes={"XX"})).region_known
    # Turning the I into a 1 is only accepted when the result has a listed code.
    assert normalize_plate("BI234AB").plate == "B1234AB"
    assert normalize_plate("BI234AB", RegionBook(codes={"D"})).plate == "BI234AB"


# -- in the pipeline -----------------------------------------------------------------------


class ReadsFrom(ScriptedOcr):
    """Answers every crop with the same prepared read."""

    def __init__(self, read: OcrRead):
        super().__init__()
        self._read = read

    def read(self, crops):
        self.crops_read += len(crops)
        return [self._read for _ in crops]


def test_event_carries_the_corrected_code_and_the_string_as_read():
    h = Harness(fake_ocr=ReadsFrom(BF_READ))
    h.frames(100)
    assert len(h.events) == 1
    event = h.events[0]
    assert (event.plate, event.plate_formatted, event.raw) == ("BK8418FD", "BK 8418 FD", "BF8418FD")
    assert event.region_known and event.region_corrected
    assert event.to_dict()["region_corrected"] is True
    current = h.pipeline.visible()[0]
    assert current.plate == "BK8418FD" and current.region_corrected
    # A replaced code does not count as seen: it would vote for itself the next time.
    assert h.pipeline._regions.seen() == {}


def test_a_code_read_as_it_is_counts_as_seen_here():
    h = Harness()
    h.frames(40)
    assert h.events[0].region_corrected is False
    assert h.pipeline._regions.seen() == {"BK": 1}


def test_replacing_can_be_switched_off():
    h = Harness(fake_ocr=ReadsFrom(BF_READ), postprocess=PostprocessConfig(fix_unknown_region=False))
    h.frames(100)
    event = h.events[0]
    assert event.plate == "BF8418FD"
    assert not event.region_known and not event.region_corrected


# -- only a standing vehicle is read -------------------------------------------------------

STOPPED_ONLY = OcrConfig(read_when="stopped")


def test_stopped_only_does_not_report_a_vehicle_that_never_stops():
    h = Harness(ocr=STOPPED_ONLY)
    for n in range(120):  # fifteen seconds rolling back and forth in view
        h.frame(back_and_forth(n))
    assert h.pipeline.in_view(h.now) == 1  # followed all the time
    h.frames(48, box=None)  # then gone: the track closes
    assert h.ocr.crops_read == 0
    assert h.events == []


def test_stopped_only_reads_a_vehicle_when_it_stops_and_not_when_it_drives_off():
    h = Harness(fake_ocr=ScriptedOcr(confidence=0.70), ocr=STOPPED_ONLY)
    for n in range(16):
        h.frame(rolling(n))
    assert h.ocr.crops_read == 0
    h.frames(16, box=rolling(15))  # stands for two seconds
    while_standing = h.ocr.crops_read
    assert while_standing >= 2
    for n in range(16, 40):  # drives on, close past the camera
        h.frame(rolling(n))
    h.frames(48, box=None)
    assert h.ocr.crops_read == while_standing
    assert len(h.events) == 1
    assert h.events[0].stopped is True
    assert h.events[0].bbox[0] == pytest.approx(rolling(15)[0], abs=10)


def test_text_without_a_digit_is_not_a_plate():
    # The lettering on a guard's vest, found by the detector and read by the OCR.
    h = Harness(fake_ocr=ScriptedOcr(text="SATPAN", confidence=0.85), ocr=OcrConfig(retry_after_s=0))
    h.frames(160)  # twenty seconds standing in view
    assert h.ocr.crops_read == 5  # the reads are spent, nothing comes of them
    assert h.events == []
    assert h.pipeline.current(h.now) is None
    assert h.pipeline.last_plate() is None
    assert h.pipeline.visible()[0].plate is None
    h.frames(48, box=None)
    assert h.events == []


# -- standing reads first, one passing read as a fallback (the default) --------------------

# A shorter quiet time than the default keeps the scenarios short. A track closes five
# seconds after it was last seen; a passing read is then held for the quiet time.
WAIT = OcrConfig(read_when="stopped_or_passing", passing_quiet_s=2.0)
GONE = 64  # frames without the plate: eight seconds, past the close and the quiet time
STOPPED_AT_X = 680.0


def rolling(frame_no: int, step_px: float = 30.0) -> tuple[float, float, float, float]:
    """A 160 px wide plate moving right by `step_px` per frame, about 1.5 plate widths a second."""
    x = 200.0 + step_px * frame_no
    return (x, 300.0, x + 160.0, 350.0)


def back_and_forth(frame_no: int) -> tuple[float, float, float, float]:
    """Keeps moving without leaving the picture."""
    phase = frame_no % 40
    return rolling(phase if phase < 20 else 40 - phase)


def test_plate_is_read_after_the_vehicle_stops_and_the_picture_is_from_then():
    h = Harness(ocr=WAIT)
    for n in range(16):  # two seconds rolling in
        h.frame(rolling(n))
    assert h.ocr.crops_read == 0
    assert h.pipeline.in_view(h.now) == 1  # followed, not read yet
    assert h.pipeline.current(h.now) is None
    stopped_at = h.t
    h.frames(24, box=(STOPPED_AT_X, 300.0, STOPPED_AT_X + 160.0, 350.0))

    assert len(h.events) == 1
    # First read once it has stood for most of the still window, the second soon after.
    assert stopped_at + 0.5 <= h.read_times[0] <= stopped_at + 1.0
    assert h.read_times[1] <= stopped_at + 1.6
    # The crop stored with the event shows the plate where it stood, not on the way in.
    assert abs(h.events[0].bbox[0] - STOPPED_AT_X) <= 10


def test_vehicle_that_never_stops_is_not_read_while_it_is_in_view():
    h = Harness(ocr=WAIT)
    for n in range(120):  # fifteen seconds rolling back and forth
        h.frame(back_and_forth(n))
    assert h.ocr.crops_read == 0
    assert h.events == []


def test_vehicle_that_only_passes_gets_one_read_from_its_best_frame():
    h = Harness(ocr=WAIT)
    for n in range(16):
        h.frame(rolling(n))
    h.frames(44, box=None)  # 5.5 s: the track is closed, its read is still held
    assert h.ocr.crops_read == 0
    assert h.events == []
    h.frames(GONE - 44, box=None)
    assert h.ocr.crops_read == 1
    assert len(h.events) == 1
    event = h.events[0]
    assert (event.plate, event.reads, event.stopped) == ("BK1234MA", 1, False)
    assert event.to_dict()["stopped"] is False


def test_passing_read_needs_a_clear_standard_plate():
    unsure = Harness(fake_ocr=ScriptedOcr(confidence=0.70), ocr=WAIT)  # fine for a standing vehicle, not for this
    odd = Harness(fake_ocr=ScriptedOcr(text="8039200", confidence=0.95), ocr=WAIT)  # outside the pattern
    for h in (unsure, odd):
        for n in range(16):
            h.frame(rolling(n))
        h.frames(GONE, box=None)
        assert h.ocr.crops_read == 1
        assert h.events == []


def test_truck_driving_off_under_a_new_track_id_is_not_reported_again():
    h = Harness(ocr=WAIT)
    h.frames(16)  # stands in the middle: read, one event
    assert len(h.events) == 1
    assert h.events[0].stopped is True
    reads = h.ocr.crops_read
    for n in range(16):  # the tracker loses it and finds a "new" plate rolling away
        h.frame(rolling(n))
    h.frames(GONE, box=None)
    assert h.ocr.crops_read == reads
    assert len(h.events) == 1


def test_truck_rolling_in_under_one_id_and_standing_under_another_is_one_event():
    # The plate is lost on the way in, e.g. behind a person, and found again once the truck
    # stands. A different reading of the blurred approach must not become a second event.
    def script(read_number, crop):
        return ("BE1234MA", 0.95) if read_number > 2 else ("BK1234MA", 0.97)

    h = Harness(fake_ocr=ScriptedOcr(script=script), ocr=WAIT)
    for n in range(16):
        h.frame(rolling(n))
    h.frames(46, box=None)  # 5.75 s hidden: the approach track closes and its frame is held
    h.frames(24)  # the truck stands, under a new id, before the held read is released
    h.frames(GONE, box=None)
    assert [e.plate for e in h.events] == ["BK1234MA"]
    assert h.events[0].stopped is True
    assert h.ocr.crops_read == 2  # the two standing reads; the approach was never read


def test_passing_vehicle_long_after_a_standing_one_is_its_own_event():
    def script(read_number, crop):
        return ("BK1234MA", 0.97) if read_number <= 2 else ("BM9719XX", 0.97)

    h = Harness(fake_ocr=ScriptedOcr(script=script), ocr=WAIT)
    h.frames(16)
    h.frames(GONE, box=None)  # the first truck has left, well over the quiet time ago
    for n in range(16):
        h.frame(rolling(n))  # another one passes without stopping
    h.frames(GONE, box=None)
    assert [(e.plate, e.stopped) for e in h.events] == [("BK1234MA", True), ("BM9719XX", False)]


def test_vehicle_driving_off_is_not_read_again():
    # Unsure reads never reach the level for an early event, so reads would go on.
    h = Harness(fake_ocr=ScriptedOcr(confidence=0.70), ocr=WAIT)
    h.frames(16, box=rolling(0))  # standing for two seconds
    while_standing = h.ocr.crops_read
    assert while_standing >= 2
    left_at = h.t
    for n in range(1, 41):  # five seconds driving off from where it stood, still in view
        h.frame(back_and_forth(n))
    assert h.ocr.crops_read == while_standing
    assert all(t < left_at for t in h.read_times)
    h.frames(48, box=None)
    # No late read either: it was read while it stood, and that is what the event is made of.
    assert h.ocr.crops_read == while_standing
    assert len(h.events) == 1
    assert h.events[0].stopped is True
    assert h.events[0].bbox[0] == pytest.approx(200, abs=10)


def test_stop_and_go_is_read_at_each_stop():
    h = Harness(fake_ocr=ScriptedOcr(confidence=0.70), ocr=WAIT)
    h.frames(12)
    first_stop = h.ocr.crops_read
    for n in range(8):
        h.frame(rolling(n))
    h.frames(40, box=rolling(7))  # stands again where it arrived
    assert h.ocr.crops_read > first_stop


def test_box_jitter_on_a_parked_vehicle_still_counts_as_standing():
    h = Harness(ocr=WAIT)
    rng = np.random.default_rng(1)
    x1, y1, x2, y2 = CENTRE_BOX
    for _ in range(24):
        dx, dy = rng.uniform(-4, 4, size=2)  # a few pixels on a 160 px plate
        h.frame((x1 + dx, y1 + dy, x2 + dx, y2 + dy))
    assert len(h.events) == 1


# -- storage -------------------------------------------------------------------------------

OLD_EVENTS_TABLE = """
CREATE TABLE events (
    event_id TEXT PRIMARY KEY, camera_id TEXT NOT NULL, plate TEXT NOT NULL, plate_formatted TEXT NOT NULL,
    confidence REAL NOT NULL, format_valid INTEGER NOT NULL, region_known INTEGER NOT NULL, alternatives TEXT NOT NULL,
    timestamp TEXT NOT NULL, ts_epoch REAL NOT NULL, track_id INTEGER NOT NULL, bbox TEXT NOT NULL, crop_url TEXT,
    snapshot_url TEXT, reads INTEGER NOT NULL, raw TEXT NOT NULL
)
"""


def old_row(event_id: str, plate: str, known: int, epoch: float) -> tuple:
    return (event_id, "cam-2", plate, plate, 0.9, 1, known, "[]", "2026-11-13T21:14:59+07:00", epoch, 1, "[1, 2, 3, 4]",
            None, None, 3, plate)


def test_database_from_before_this_change_gets_the_new_column(tmp_path):
    path = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute(OLD_EVENTS_TABLE)
    marks = ", ".join("?" * 16)
    conn.executemany(f"INSERT INTO events VALUES ({marks})", [
        old_row("a", "BK8857JL", 1, 1.0), old_row("b", "BK8962LL", 1, 2.0), old_row("c", "BM9719XX", 1, 3.0),
        old_row("d", "BF8418FD", 0, 4.0),
    ])
    conn.commit()
    conn.close()

    store = EventStore(path)
    assert store.count() == 4
    assert all(event["region_corrected"] is False for event in store.query())
    assert all(event["stopped"] is None for event in store.query())  # not recorded back then
    # Only codes that were read as listed count; BF was not.
    assert store.region_counts() == {"BK": 2, "BM": 1}
    assert store.region_counts(limit=1) == {"BM": 1}  # the newest listed one
    store.close()
    EventStore(path).close()  # opening twice must not try to add the column again


def test_region_list_starts_with_the_built_in_codes_and_remembers_changes(tmp_path):
    path = tmp_path / "db.sqlite3"
    store = RegionStore(path)
    assert store.codes() == REGION_CODES
    first = store.all()[0]
    assert first == {"code": "A", "name": "Banten (Serang, Cilegon, Pandeglang, Lebak)", "builtin": True}

    assert store.add("XK", "Kode uji") is True
    assert store.add("XK", "lagi") is False
    assert store.remove("DK") is True
    assert store.remove("DK") is False
    store.close()

    reopened = RegionStore(path)
    codes = reopened.codes()
    assert "XK" in codes
    assert "DK" not in codes  # a removed built-in code is not put back by a restart
    assert {row["code"]: row["builtin"] for row in reopened.all()}["XK"] is False
    assert reopened.restore_defaults() == 1
    assert reopened.restore_defaults() == 0
    assert reopened.codes() == REGION_CODES | {"XK"}
    reopened.close()
