"""Read the CSV files of scripts/scan_recording.py: how large a plate must be to be found and to be read.

Four tables, per camera and for day and night:

  1. Detectors on the same key frames: of the plates the OCR could read as a standard plate,
     how many got a box strong enough to start a track from each detector, by plate width.
  2. Reading by plate width: boxes are strung into visits of one truck, a visit's plate is what
     most of its readings agree on, and each single reading is checked against it.
  3. Visits against the events the service stored in the same hours: read right, read
     differently, or not reported at all.
  4. Where standing plates are in the frame: their width, and how many lie inside the focus area.

A visit's plate is only trusted when at least --agree of its readings say the same thing. Plates
no detector proposes in the ROI or the focus area are in none of the tables.

    uv run --frozen --extra cpu python scripts/scan_report.py data/benchmarks/scan-20261118 --events data/rog-20261119/alpr.sqlite3
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from alpr_id.capture.roi import roi_to_pixels
from alpr_id.config import load_config

STRONG = 0.5  # tracking.track_activation_threshold: below it a box starts no track
WIDTHS = [(0, 70), (70, 85), (85, 100), (100, 120), (120, 150), (150, 10_000)]
NIGHT_FROM, NIGHT_UNTIL = 18, 6
FRAME = (2560, 1440)


def bucket(width: float) -> str:
    low, high = next(edge for edge in WIDTHS if edge[0] <= width < edge[1])
    return f"{low}-{high}" if high < 10_000 else f"{low}+"


def iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def is_night(epoch: float) -> bool:
    hour = datetime.fromtimestamp(epoch).hour
    return hour >= NIGHT_FROM or hour < NIGHT_UNTIL


@dataclass
class Sighting:
    """One plate in one key frame, as each detector saw it."""

    camera: str
    epoch: float
    box: tuple[float, float, float, float]
    scores: dict[str, float] = field(default_factory=dict)  # detector -> best of its two views
    text: str = ""  # what the OCR read from the box of the first detector that has one
    ocr_confidence: float = 0.0
    standard: bool = False  # a standard plate with a known region code and six characters or more

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def centre(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2


def load(folder: Path, first: str) -> dict[str, list[Sighting]]:
    """Camera -> sightings in time order. Boxes of the detectors on the same plate are joined into one sighting."""
    per_frame: dict[tuple[str, float], list[Sighting]] = defaultdict(list)
    for path in sorted(folder.glob("cam-*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows.sort(key=lambda row: row["detector"] != first)  # the installed detector's box and reading come first
        for row in rows:
            box = tuple(float(row[key]) for key in ("x1", "y1", "x2", "y2"))
            score = max(float(row["score_roi"]), float(row["score_focus"]))
            frame = per_frame[(row["camera"], float(row["epoch"]))]
            same = next((s for s in frame if iou(s.box, box) >= 0.5), None)
            if same is None:
                same = Sighting(row["camera"], float(row["epoch"]), box)
                frame.append(same)
            same.scores[row["detector"]] = max(score, same.scores.get(row["detector"], 0.0))
            if row["text"] and not same.text:
                same.text = row["text"]
                same.ocr_confidence = float(row["ocr_confidence"])
                same.standard = row["format_valid"] == "1" and row["region_known"] == "1" and len(row["text"]) >= 6
    cameras: dict[str, list[Sighting]] = defaultdict(list)
    for (camera, _), frame in sorted(per_frame.items()):
        cameras[camera].extend(frame)
    return cameras


@dataclass
class Visit:
    sightings: list[Sighting] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.sightings[0].epoch

    @property
    def end(self) -> float:
        return self.sightings[-1].epoch

    def plate(self, agree: int) -> str | None:
        """The text most standard readings of this visit agree on, if at least `agree` do."""
        votes = Counter()
        for sighting in self.sightings:
            if sighting.standard:
                votes[sighting.text] += 1
        if not votes:
            return None
        text, count = votes.most_common(1)[0]
        return text if count >= agree else None


def visits_of(sightings: list[Sighting], gap_s: float) -> list[Visit]:
    """String the sightings of one camera into visits: the same plate, followed from key frame to key frame.

    A sighting continues the visit whose last box it overlaps, or else the nearest one that ended
    less than `gap_s` ago within four plate widths and of a similar size: a truck that is still
    rolling moves several widths between two key frames.
    """
    done: list[Visit] = []
    open_visits: list[Visit] = []
    for sighting in sightings:
        open_visits, closed = ([v for v in open_visits if sighting.epoch - v.end <= gap_s],
                               [v for v in open_visits if sighting.epoch - v.end > gap_s])
        done.extend(closed)
        best, best_cost = None, None
        for visit in open_visits:
            last = visit.sightings[-1]
            if last.epoch == sighting.epoch:
                continue  # two plates in one frame are two trucks
            overlap = iou(last.box, sighting.box)
            reach = ((last.centre[0] - sighting.centre[0]) ** 2 + (last.centre[1] - sighting.centre[1]) ** 2) ** 0.5
            ratio = sighting.width / max(last.width, 1.0)
            if overlap < 0.15 and not (reach <= 4 * max(last.width, sighting.width) and 0.5 <= ratio <= 2.0):
                continue
            cost = reach - 1000 * overlap
            if best_cost is None or cost < best_cost:
                best, best_cost = visit, cost
        if best is None:
            best = Visit()
            open_visits.append(best)
        best.sightings.append(sighting)
    done.extend(open_visits)
    return sorted(done, key=lambda v: v.start)


def merge_by_plate(visits: list[Visit], agree: int, within_s: float) -> list[Visit]:
    """Visits of the same plate that follow each other closely are one truck: the track broke, the truck did not leave."""
    merged: list[Visit] = []
    for visit in visits:
        plate = visit.plate(agree)
        earlier = next((m for m in reversed(merged) if visit.start - m.end <= within_s and plate and m.plate(agree) == plate), None)
        if earlier is None:
            merged.append(visit)
        else:
            earlier.sightings.extend(visit.sightings)
            earlier.sightings.sort(key=lambda s: s.epoch)
    return merged


def table(title: str, header: list[str], rows: list[list]) -> None:
    print(f"\n{title}")
    widths = [max(len(str(row[i])) for row in [header, *rows]) for i in range(len(header))]
    for row in [header, *rows]:
        print("  " + "  ".join(str(cell).ljust(widths[i]) if i == 0 else str(cell).rjust(widths[i]) for i, cell in enumerate(row)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scan", type=Path, help="folder with cam-<n>.csv of scan_recording.py")
    parser.add_argument("--events", type=Path, default=None, help="event database of the service for the same hours")
    parser.add_argument("--installed", default="baru", help="name of the detector the service ran during the recording")
    parser.add_argument("--agree", type=int, default=3, help="readings that must agree before a visit's plate is trusted")
    parser.add_argument("--gap", type=float, default=12.0, help="seconds without a box after which a visit is over")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cameras = load(args.scan, args.installed)
    config = load_config()
    focus = {c.id: roi_to_pixels(c.focus, *FRAME) for c in config.cameras if c.focus is not None}
    detectors = sorted({name for sightings in cameras.values() for s in sightings for name in s.scores},
                       key=lambda name: name != args.installed)
    result: dict = {"detectors": detectors}
    spans = {camera: (sightings[0].epoch, sightings[-1].epoch) for camera, sightings in cameras.items() if sightings}
    print("scan: " + ", ".join(f"{camera} {len(s)} boxes" for camera, s in sorted(cameras.items())))

    # 1. the detectors on the same frames
    rows = []
    for period in ("day", "night"):
        for low_high in WIDTHS:
            name = bucket(low_high[0])
            group = [s for sightings in cameras.values() for s in sightings
                     if s.standard and s.ocr_confidence >= 0.8 and bucket(s.width) == name and is_night(s.epoch) == (period == "night")]
            if group:
                rows.append([f"{period} {name} px", len(group), *(sum(1 for s in group if s.scores.get(d, 0.0) >= STRONG) for d in detectors)])
    table("1. Plates the OCR reads as a standard plate (confidence 0.8 or more): boxes of 0.5 or more per detector",
          ["plate width", "plates", *detectors], rows)
    result["detectors_by_width"] = rows

    # visits
    visits: dict[str, list[Visit]] = {camera: merge_by_plate(visits_of(sightings, args.gap), args.agree, 90.0)
                                      for camera, sightings in cameras.items()}

    # 2. reading by width, inside visits whose plate is trusted
    rows = []
    reading: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0, 0])  # sightings, strong box, read, read right
    for camera_visits in visits.values():
        for visit in camera_visits:
            plate = visit.plate(args.agree)
            if plate is None:
                continue
            for s in visit.sightings:
                tally = reading[("night" if is_night(s.epoch) else "day", bucket(s.width))]
                tally[0] += 1
                tally[1] += s.scores.get(args.installed, 0.0) >= STRONG
                tally[2] += bool(s.text)
                tally[3] += s.text == plate
    for period in ("day", "night"):
        for low_high in WIDTHS:
            name = bucket(low_high[0])
            seen, strong, read, right = reading[(period, name)]
            if seen:
                rows.append([f"{period} {name} px", seen, strong, read, right, f"{right / max(read, 1):.0%}"])
    table(f"2. Single key frames of visits whose plate is known ({args.agree} or more readings agree), installed detector",
          ["plate width", "sightings", "box 0.5+", "read", "read right", "right of read"], rows)
    result["reading_by_width"] = rows

    # 3. visits against events
    events: dict[str, list[tuple[float, str, float]]] = defaultdict(list)
    if args.events:
        connection = sqlite3.connect(f"file:{args.events.as_posix()}?mode=ro", uri=True)
        for camera, plate, epoch, confidence in connection.execute("SELECT camera_id, plate, ts_epoch, confidence FROM events ORDER BY ts_epoch"):
            events[camera].append((epoch, plate, confidence))
        connection.close()
    rows = []
    unreported: list[dict] = []
    wrong: list[dict] = []
    by_width: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # visits, event with the same plate
    by_confidence: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for camera in sorted(visits):
        for period in ("day", "night"):
            kinds = Counter()
            for visit in visits[camera]:
                plate = visit.plate(args.agree)
                if plate is None or is_night(visit.start) != (period == "night"):
                    continue
                near = [e for e in events[camera] if visit.start - 30 <= e[0] <= visit.end + 90]
                best = min((distance(plate, e[1]) for e in near), default=None)
                largest = max(visit.sightings, key=lambda s: s.width)
                widest = largest.width
                where = {"at": largest.epoch, "box": [round(v) for v in largest.box]}
                usual = statistics.median(s.width for s in visit.sightings)
                group = "under 95 px" if usual < 95 else "95-105 px" if usual < 105 else "105-115 px" if usual < 115 else "115 px and more"
                by_width[group][0] += 1
                by_width[group][1] += best == 0
                # the event that goes with this visit: the closest in text, then in time
                own = min(near, key=lambda e: (distance(plate, e[1]), abs(e[0] - visit.start)), default=None)
                if own is None:  # reported under a text too far off to match: take the first event of the visit
                    own = next((e for e in events[camera] if visit.start - 30 <= e[0] <= visit.end + 90), None)
                if own is not None:
                    level = "0.95 and more" if own[2] >= 0.95 else "0.85-0.95" if own[2] >= 0.85 else "0.60-0.85" if own[2] >= 0.6 else "under 0.60"
                    by_confidence[level][0] += 1
                    by_confidence[level][1] += own[1] == plate
                if best == 0:
                    kinds["event, same plate"] += 1
                elif best is not None and best <= 2:
                    kinds["event, other text"] += 1
                    wrong.append({"camera": camera, "time": datetime.fromtimestamp(visit.start).isoformat(timespec="seconds"),
                                  "plate": plate, "event": min(near, key=lambda e: distance(plate, e[1]))[1], "width": round(widest),
                                  "agree": sum(1 for s in visit.sightings if s.text == plate), "read": sum(1 for s in visit.sightings if s.text), **where})
                else:
                    kinds["no event"] += 1
                    unreported.append({"camera": camera, "time": datetime.fromtimestamp(visit.start).isoformat(timespec="seconds"),
                                       "plate": plate, "seconds": round(visit.end - visit.start), "width": round(widest),
                                       "sightings": len(visit.sightings), **where})
            total = sum(kinds.values())
            if total:
                rows.append([f"{camera} {period}", total, kinds["event, same plate"], kinds["event, other text"], kinds["no event"]])
    if args.events:
        table("3. Visits with a known plate against the events of the service",
              ["camera", "visits", "event, same plate", "event, other text", "no event"], rows)
        result["visits_against_events"] = rows
        result["unreported"], result["other_text"] = unreported, wrong
        rows = [[group, *by_width[group], f"{by_width[group][1] / max(by_width[group][0], 1):.0%}"]
                for group in ("under 95 px", "95-105 px", "105-115 px", "115 px and more") if by_width[group][0]]
        table("   The same visits by the plate's usual width while standing", ["width", "visits", "event, same plate", "share"], rows)
        result["visits_by_width"] = rows
        rows = [[level, *by_confidence[level], f"{by_confidence[level][1] / max(by_confidence[level][0], 1):.0%}"]
                for level in ("0.95 and more", "0.85-0.95", "0.60-0.85", "under 0.60") if by_confidence[level][0]]
        table("   ...and by the confidence of their event", ["event confidence", "visits", "event, same plate", "share"], rows)
        result["visits_by_confidence"] = rows
        # events the scan has no known plate for
        rows = []
        for camera in sorted(visits):
            start, end = spans.get(camera, (0, 0))
            own = [e for e in events[camera] if start <= e[0] <= end]
            known = [(v.start, v.end, v.plate(args.agree)) for v in visits[camera] if v.plate(args.agree)]
            matched = sum(1 for e in own if any(a - 30 <= e[0] <= b + 90 and distance(p, e[1]) <= 2 for a, b, p in known))
            rows.append([camera, len(own), matched, len(own) - matched])
        table("   Events in the recorded hours, and whether a visit with a known plate stands behind them",
              ["camera", "events", "with such a visit", "without"], rows)
        result["events_against_visits"] = rows

    # 4. where standing plates are
    rows = []
    for camera in sorted(visits):
        for period in ("day", "night"):
            standing = []
            for visit in visits[camera]:
                if visit.plate(args.agree) is None or is_night(visit.start) != (period == "night") or len(visit.sightings) < 4:
                    continue
                widths = sorted(s.width for s in visit.sightings)
                middle = visit.sightings[len(visit.sightings) // 2]
                area = focus.get(camera)
                inside = area is not None and (area.x1 <= middle.box[0] and area.y1 <= middle.box[1]
                                                and middle.box[2] <= area.x2 and middle.box[3] <= area.y2)
                standing.append((statistics.median(widths), inside))
            if standing:
                values = sorted(width for width, _ in standing)
                rows.append([f"{camera} {period}", len(standing), round(values[len(values) // 10]), round(statistics.median(values)),
                             round(values[-1 - len(values) // 10]), sum(1 for _, inside in standing if inside)])
    table("4. Standing plates (four key frames or more): width in pixels, and how many stand inside the focus area",
          ["camera", "visits", "p10", "median", "p90", "inside focus"], rows)
    result["standing"] = rows

    # 5. plates that stand there with a strong box and never give a plate
    rows = []
    unread: list[dict] = []
    for camera in sorted(visits):
        for period in ("day", "night"):
            group = [v for v in visits[camera] if is_night(v.start) == (period == "night") and len(v.sightings) >= 4
                     and sum(1 for s in v.sightings if s.scores.get(args.installed, 0.0) >= STRONG) >= 4]
            silent = [v for v in group if v.plate(args.agree) is None]
            if group:
                widths = sorted(statistics.median(s.width for s in v.sightings) for v in silent)
                rows.append([f"{camera} {period}", len(group), len(silent), round(statistics.median(widths)) if widths else "-"])
            for visit in silent:
                texts = Counter(s.text for s in visit.sightings if s.text).most_common(3)
                unread.append({"camera": camera, "time": datetime.fromtimestamp(visit.start).isoformat(timespec="seconds"),
                               "seconds": round(visit.end - visit.start), "width": round(statistics.median(s.width for s in visit.sightings)),
                               "at": visit.sightings[len(visit.sightings) // 2].epoch,
                               "box": [round(v) for v in visit.sightings[len(visit.sightings) // 2].box], "texts": texts})
    table(f"5. Things boxed strongly in four key frames or more, and how many never give {args.agree} equal standard readings",
          ["camera", "boxed", "never a plate", "their median width"], rows)
    result["boxed_never_read"], result["never_read"] = rows, unread

    # 6. the two cameras of a weighbridge
    pair = {"cam-1": "cam-2", "cam-2": "cam-1", "cam-3": "cam-4", "cam-4": "cam-3"}
    rows = []
    for camera in sorted(visits):
        other = visits.get(pair.get(camera, ""), [])
        kinds = Counter()
        for visit in visits[camera]:
            plate = visit.plate(args.agree)
            if plate is None:
                continue
            near = [v for v in other if v.start <= visit.end + 120 and v.end >= visit.start - 120]
            plates = [v.plate(args.agree) for v in near if v.plate(args.agree)]
            if plate in plates:
                kinds["same plate"] += 1
            elif any(distance(plate, p) <= 2 for p in plates):
                kinds["nearly the same"] += 1
            elif near:
                kinds["boxes, no such plate"] += 1
            else:
                kinds["nothing"] += 1
        if kinds:
            rows.append([f"{camera} -> {pair[camera]}", sum(kinds.values()), kinds["same plate"], kinds["nearly the same"],
                         kinds["boxes, no such plate"], kinds["nothing"]])
    table("6. While a camera has a visit with a known plate, what the other camera of the same weighbridge has (2 minutes either side)",
          ["cameras", "visits", "same plate", "nearly the same", "boxes, no such plate", "nothing"], rows)
    result["pairs"] = rows

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
