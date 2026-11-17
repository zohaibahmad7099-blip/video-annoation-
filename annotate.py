"""Label persons' behaviour for the Stage A classifier (see docs/LABELLING_GUIDE.md).

  python annotate.py --video "159 Back Cooler_05" --reviewer Zohaib

Reads the reference tracks (outputs/extract/yolo11m_10hz/<name>_tracks.csv) and
writes labels/<video_key>.json after every change.

Navigation   SPACE play/pause   D/A +-1 step   L/J +-2 s   K/H +-10 s
Persons      click a box = select track   N new person from track   G link track to person
             U unlink track   I actor id   T toggle staff
Range        S set start mark at the playhead; a label key then covers mark..playhead
Phase keys   1 normal   2 inspect   3 pick (purchase, no event)   4 pick (current event)
             5 suspicious (current event)   6 basket   7 return   0 unknown
Visibility   V visible   O hands occluded   B body occluded
Events       E new event   W cycle current event   M missed event (no track) over the range
Other        F fill the person's unlabelled time with normal + visible
Y fill the person's unlabelled time with normal + hands occluded
             C save labels   X undo one action   Z reset selected person's labels
             R reset all labels   Q/ESC quit
"""
import argparse
import copy
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from shoplift.annotation import (PHASES, VISIBILITY, MissedEvent, Person, VideoLabels, lint, load,  # noqa: E402
                                 paint, save, value_at)
from shoplift.ingest import CadenceSampler, FileSource  # noqa: E402
from shoplift.label_readiness import completion_check  # noqa: E402
from shoplift.registry import find, load_registry, register_video, save_registry  # noqa: E402

PHASE_KEYS = {ord("1"): ("normal", False), ord("2"): ("inspect", False), ord("3"): ("pick", False),
              ord("4"): ("pick", True), ord("5"): ("suspicious", True), ord("6"): ("basket", False),
              ord("7"): ("return", False), ord("0"): ("unknown", False)}
VIS_KEYS = {ord("v"): "visible", ord("o"): "hands_occluded", ord("b"): "body_occluded"}
PHASE_COLORS = {"normal": (150, 150, 150), "inspect": (255, 200, 0), "pick": (0, 200, 255),
                "suspicious": (0, 0, 255), "basket": (0, 200, 0), "return": (200, 0, 200),
                "unknown": (60, 60, 60)}
VIS_COLORS = {"visible": (200, 200, 200), "hands_occluded": (0, 140, 255), "body_occluded": (0, 0, 160)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def person_color(pid: str):
    rng = np.random.default_rng(abs(hash(pid)) % (2 ** 32))
    return tuple(int(c) for c in rng.integers(80, 255, 3))
def migrate_legacy_phases(labels: VideoLabels) -> None:
    """Convert labels created with the former multi-phase schema."""
    for person in labels.persons:
        for segment in person.phase:
            if segment.value == "none":
                segment.value = "normal"
            elif segment.value in {"conceal", "shoplifting"}:
                segment.value = "suspicious"
def boxes_for(df: pd.DataFrame, track_ids):
    rows = df[df.track_id.isin(track_ids)].sort_values(["timestamp_s", "conf"])
    rows = rows.drop_duplicates("timestamp_s", keep="last")
    return [[round(float(r.timestamp_s), 3), round(float(r.x1), 1), round(float(r.y1), 1),
             round(float(r.x2), 1), round(float(r.y2), 1)] for r in rows.itertuples()]


def decode_frames(path: str, hz: float, width: int):
    """JPEG-encoded display frames on the pipeline's cadence grid: {time: bytes}."""
    src, sampler, frames = FileSource(path), CadenceSampler(hz), {}
    for f in src:
        if not sampler.admit(f.timestamp_s):
            continue
        h, w = f.image.shape[:2]
        small = cv2.resize(f.image, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
        frames[round(f.timestamp_s, 3)] = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]
        if len(frames) % 200 == 0:
            print(f"  decoded {len(frames)} frames ({f.timestamp_s:.0f} s)")
    return frames, w


def video_duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        return count / fps if fps > 0 else 0.0
    finally:
        cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="Registered video name, key or path")
    ap.add_argument("--tracks", type=Path, help="Reference tracks CSV (default: YOLO11 10 Hz extraction)")
    ap.add_argument("--reviewer", default="")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--pose-weights", type=Path, default=ROOT.parent.parent.parent / "models" / "yolo11m-pose.pt")
    ap.add_argument("--device", default="cpu", help="Inference device for automatic track generation")
    ap.add_argument("--video-file", type=Path, help="Local copy of the video (checked against the registry hash)")
    args = ap.parse_args()

    registry_path = ROOT / "labels" / "videos.json"
    legacy_registry_path = ROOT / "labels" / "annoation Json+Videos" / "videos.json"
    if not registry_path.is_file() and legacy_registry_path.is_file():
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(legacy_registry_path.read_text(encoding="utf-8"), encoding="utf-8")
    registry = load_registry(registry_path)
    if args.video_file:
        entry = register_video(args.video_file, registry)
        save_registry(registry_path, registry)
        print(f"Registered new video '{entry.name}' in {registry_path}", flush=True)
    else:
        try:
            entry = find(registry, args.video)
        except KeyError:
            sys.exit(f"Video '{args.video}' is not registered. "
                     "Provide --video-file to register it automatically.")
    candidates = [args.video_file] if args.video_file else [Path(p) if Path(p).is_absolute() else ROOT / p
                                                             for p in entry.paths]
    video_path = next((p for p in candidates if p.is_file()), None)
    if video_path is None:
        sys.exit(f"Video file for '{entry.name}' not found (looked in: {', '.join(map(str, candidates))}). "
                 "Use --video-file.")
    if args.video_file and sha256(video_path) != entry.sha256:
        sys.exit(f"{video_path} is not the registered '{entry.name}' (content hash differs).")
    tracks = args.tracks or ROOT / "outputs" / "extract" / "yolo11m_10hz" / f"{entry.name}_tracks.csv"
    tracks_ready = tracks.is_file()
    duration_s = video_duration(video_path)
    summary = tracks.with_name(tracks.name[:-len("_tracks.csv")] + "_summary.json")
    # The pipeline writes the summary last, so it marks a finished run. Without it, fall back to
    # time coverage (a tracks file legitimately ends early when the last person leaves early).
    if tracks_ready and not summary.is_file():
        try:
            track_times = pd.read_csv(tracks, usecols=["timestamp_s"])
            tracks_ready = (not track_times.empty and
                            float(track_times.timestamp_s.max()) >= duration_s - 1.0 / args.hz)
        except (OSError, ValueError, pd.errors.EmptyDataError):
            tracks_ready = False
    if not tracks_ready:
        print(f"Preparing video '{entry.name}': generating tracking data before labeling...", flush=True)
        try:
            from shoplift.pipeline import Pipeline, PipelineConfig  # needs ultralytics; only for new videos
            tracks.unlink(missing_ok=True)
            Pipeline(PipelineConfig(pose_weights=str(args.pose_weights), device=args.device,
                                    analysis_hz=args.hz), scorer=None).run(
                                        str(video_path), tracks.parent, run_name=entry.name)
        except Exception as exc:
            sys.exit(f"Could not generate reference tracks at {tracks}: {exc}")
        if not tracks.is_file():
            sys.exit(f"Track generation completed without creating {tracks}.")
    label_path = ROOT / "labels" / f"{entry.name}.json"
    legacy_label_path = ROOT / "labels" / f"{entry.name}.json"
    df = pd.read_csv(tracks)
    df["timestamp_s"] = df.timestamp_s.round(3)
    tracks_rel = str(tracks.resolve().relative_to(ROOT)) if tracks.resolve().is_relative_to(ROOT) else str(tracks)
    tracks_sha = sha256(tracks)
    existing_label_path = label_path if label_path.is_file() else legacy_label_path
    if existing_label_path.is_file():
        L = load(existing_label_path)
        L.video_key = entry.name
        L.video_name = entry.name
        migrate_legacy_phases(L)
        if L.reference_tracks_sha256 and L.reference_tracks_sha256 != tracks_sha:
            print("WARNING: reference tracks changed since these labels were made; track ids may differ. "
                  "Person boxes are kept; re-link tracks with G where needed.")
    else:
        L = VideoLabels(entry.name, entry.name, entry.sha256)
    L.reference_tracks, L.reference_tracks_sha256 = tracks_rel, tracks_sha
    if args.reviewer:
        L.reviewer = args.reviewer

    print(f"Decoding {entry.name} at {args.hz:g} Hz for review...")
    frames, src_w = decode_frames(str(video_path), args.hz, args.width)
    times = np.array(sorted(frames))
    scale = args.width / src_w
    by_t = {t: g for t, g in df.groupby("timestamp_s")}
    step = 1.0 / args.hz

    st = {"i": 0, "play": False, "track": None, "person": None, "mark": None, "event": "", "msg": ""}
    undo = []

    def commit(msg):
        save(L, label_path)
        errs, warns = lint(L)
        st["msg"] = f"{msg} | {len(errs)} errors, {len(warns)} warnings"

    def export_labeled_video():
        output_path = ROOT / "labels" / f"{entry.name}_labeled.mp4"
        first = cv2.imdecode(frames[times[0]], cv2.IMREAD_COLOR)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), args.hz,
                                 (first.shape[1], first.shape[0]))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open labeled video output: {output_path}")
        try:
            for frame_time in times:
                image = cv2.imdecode(frames[frame_time], cv2.IMREAD_COLOR)
                group = by_t.get(frame_time)
                if group is not None:
                    for row in group.itertuples():
                        person = L.person_of_track(int(row.track_id))
                        if person is None:
                            continue
                        box = (np.array([row.x1, row.y1, row.x2, row.y2]) * scale).astype(int)
                        phase = value_at(person.phase, frame_time)
                        visibility = value_at(person.visibility, frame_time)
                        label = f"{person.person_id}: {phase.value if phase else '?'}"
                        if visibility:
                            label += f" / {visibility.value}"
                        cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), person_color(person.person_id), 2)
                        cv2.putText(image, label, (box[0], max(18, box[1] - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, person_color(person.person_id), 2,
                                    cv2.LINE_AA)
                writer.write(image)
        finally:
            writer.release()
        return output_path

    def checkpoint():
        undo.append(copy.deepcopy(L))
        del undo[:-50]

    def undo_last():
        if not undo:
            st["msg"] = "nothing to undo"
            return
        restored = undo.pop()
        L.persons, L.missed_events, L.complete = restored.persons, restored.missed_events, restored.complete
        commit("undone")

    def reset_labels():
        checkpoint()
        L.persons = []
        L.missed_events = []
        L.complete = False
        st.update({"i": 0, "track": None, "person": None, "mark": None, "event": ""})
        commit("all labels reset")

    def reset_selected_person_labels():
        if not st["person"]:
            st["msg"] = "select a person first"
            return
        checkpoint()
        person = L.person(st["person"])
        person.phase = []
        person.visibility = []
        commit(f"{person.person_id} labels reset")

    def rng():
        t = times[st["i"]]
        a = st["mark"] if st["mark"] is not None else t
        return min(a, t), max(a, t) + step

    def on_mouse(ev, x, y, *_):
        if ev != cv2.EVENT_LBUTTONDOWN:
            return
        g = by_t.get(times[st["i"]])
        if g is None:
            return
        for r in g.itertuples():
            if r.x1 * scale <= x <= r.x2 * scale and r.y1 * scale + 24 <= y <= r.y2 * scale + 24:
                st["track"] = int(r.track_id)
                p = L.person_of_track(st["track"])
                if p:
                    st["person"] = p.person_id
                st["msg"] = f"track #{st['track']}" + (f" ({p.person_id})" if p else " (unlinked)")
                return

    win = f"Annotate {entry.name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        t = times[st["i"]]
        img = cv2.imdecode(frames[t], cv2.IMREAD_COLOR)
        canvas = np.zeros((img.shape[0] + 24 + 92, img.shape[1], 3), np.uint8)
        canvas[24:24 + img.shape[0]] = img
        g = by_t.get(t)
        if g is not None:
            for r in g.itertuples():
                p = L.person_of_track(int(r.track_id))
                col = person_color(p.person_id) if p else (120, 120, 120)
                thick = 3 if (p and p.person_id == st["person"]) or r.track_id == st["track"] else 1
                b = (np.array([r.x1, r.y1, r.x2, r.y2]) * scale).astype(int) + [0, 24, 0, 24]
                cv2.rectangle(canvas, tuple(b[:2]), tuple(b[2:]), col, thick)
                lab = f"{p.person_id}/#{r.track_id}" if p else f"#{r.track_id}"
                if p:
                    ph, vi = value_at(p.phase, t), value_at(p.visibility, t)
                    lab += f" {ph.value if ph else '?'}{'/' + vi.value[:5] if vi else ''}{' ' + ph.event_id if ph and ph.event_id else ''}"
                cv2.putText(canvas, lab, (b[0], max(36, b[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        # timeline for the selected person
        y0 = 24 + img.shape[0] + 6
        W = canvas.shape[1]
        tx = lambda s: int((s - times[0]) / max(times[-1] - times[0], 1e-6) * (W - 1))
        if st["person"]:
            p = L.person(st["person"])
            if p.span:
                cv2.rectangle(canvas, (tx(p.span[0]), y0), (tx(p.span[1]), y0 + 6), person_color(p.person_id), -1)
            for row, segs, cols in ((1, p.phase, PHASE_COLORS), (2, p.visibility, VIS_COLORS)):
                for s in segs:
                    value = ({"none": "normal", "conceal": "suspicious", "shoplifting": "suspicious"}.get(s.value, s.value)
                             if row == 1 else s.value)
                    cv2.rectangle(canvas, (tx(s.start_s), y0 + 10 + (row - 1) * 18),
                                  (max(tx(s.end_s), tx(s.start_s) + 1), y0 + 24 + (row - 1) * 18),
                                  cols[value], -1)
                    if s.event_id:
                        cv2.putText(canvas, s.event_id, (tx(s.start_s), y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (255, 255, 255), 1)
        for m in L.missed_events:
            cv2.rectangle(canvas, (tx(m.start_s), y0 + 48), (tx(m.end_s), y0 + 54), (0, 0, 255), -1)
        if st["mark"] is not None:
            a, b = rng()
            cv2.rectangle(canvas, (tx(a), y0 + 58), (tx(b), y0 + 64), (0, 255, 255), -1)
        cv2.line(canvas, (tx(t), y0), (tx(t), y0 + 68), (255, 255, 255), 1)
        mark = "-" if st["mark"] is None else f"{st['mark']:.2f}"
        bar = (f"t={t:7.2f}s  person={st['person']}  track=#{st['track']}  event={st['event'] or '-'}  "
               f"mark={mark}  {'COMPLETE ' if L.complete else ''}{st['msg']}")
        cv2.putText(canvas, bar, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, canvas)

        k = cv2.waitKey(int(step * 1000) if st["play"] else 30) & 0xFF
        if st["play"]:
            st["i"] = min(st["i"] + 1, len(times) - 1)
        if k == 255:
            continue
        if k in (ord("q"), 27):
            break
        nav = {ord("d"): 1, ord("a"): -1, ord("l"): int(2 * args.hz), ord("j"): -int(2 * args.hz),
               ord("k"): int(10 * args.hz), ord("h"): -int(10 * args.hz)}
        if k == ord(" "):
            st["play"] = not st["play"]
        elif k in nav:
            st["i"] = int(np.clip(st["i"] + nav[k], 0, len(times) - 1))
        elif k == ord("s"):
            st["mark"] = t
            st["msg"] = f"mark at {t:.2f}s"
        elif k in (8, ord("x")):
            undo_last()
        elif k == ord("n"):
            if st["track"] is None or L.person_of_track(st["track"]):
                st["msg"] = "click an unlinked track first"
                continue
            checkpoint()
            p = Person(L.next_person_id(), [st["track"]], boxes_for(df, [st["track"]]))
            L.persons.append(p)
            st["person"] = p.person_id
            commit(f"new {p.person_id} from #{st['track']}")
        elif k == ord("g"):
            if st["track"] is None or not st["person"] or L.person_of_track(st["track"]):
                st["msg"] = "select a person, then click an unlinked track"
                continue
            checkpoint()
            p = L.person(st["person"])
            p.track_ids.append(st["track"])
            p.boxes = boxes_for(df, p.track_ids)
            commit(f"linked #{st['track']} to {p.person_id}")
        elif k == ord("u"):
            p = L.person_of_track(st["track"]) if st["track"] is not None else None
            if not p:
                continue
            checkpoint()
            p.track_ids.remove(st["track"])
            p.boxes = boxes_for(df, p.track_ids)
            commit(f"unlinked #{st['track']} from {p.person_id}")
        elif k == ord("r"):
            reset_labels()
        elif k == ord("z"):
            reset_selected_person_labels()
        elif k == ord("i") and st["person"]:
            L.person(st["person"]).actor_id = input(f"actor_id for {st['person']} (e.g. A3): ").strip()
            commit("actor id set")
        elif k == ord("t") and st["person"]:
            p = L.person(st["person"])
            p.is_staff = not p.is_staff
            commit(f"{p.person_id} staff={p.is_staff}")
        elif k == ord("e"):
            st["event"] = L.next_event_id()
            st["msg"] = f"current event {st['event']} (used by 4 and 5)"
        elif k == ord("w"):
            ids = L.event_ids()
            if ids:
                st["event"] = ids[(ids.index(st["event"]) + 1) % len(ids)] if st["event"] in ids else ids[0]
                st["msg"] = f"current event {st['event']}"
        elif k in PHASE_KEYS and st["person"]:
            phase, with_event = PHASE_KEYS[k]
            if with_event and not st["event"]:
                st["event"] = L.next_event_id()
            checkpoint()
            p = L.person(st["person"])
            a, b = rng()
            p.phase = paint(p.phase, a, b, phase, st["event"] if with_event else "")
            st["mark"] = None
            commit(f"{p.person_id} {phase}{' ' + st['event'] if with_event else ''} {a:.2f}-{b:.2f}s")
        elif k in VIS_KEYS and st["person"]:
            checkpoint()
            p = L.person(st["person"])
            a, b = rng()
            p.visibility = paint(p.visibility, a, b, VIS_KEYS[k])
            st["mark"] = None
            commit(f"{p.person_id} {VIS_KEYS[k]} {a:.2f}-{b:.2f}s")
        elif k in (ord("f"), ord("y")) and st["person"]:
            p = L.person(st["person"])
            if not p.span:
                continue
            checkpoint()
            t0, t1 = p.span[0], p.span[1] + step
            visibility = "visible" if k == ord("f") else "hands_occluded"
            for name, default in (("phase", "normal"), ("visibility", "visible")):
                if name == "visibility":
                    default = visibility
                segs = getattr(p, name)
                cursor, filled = t0, list(segs)
                for s in sorted(segs, key=lambda s: s.start_s) + [None]:
                    end = t1 if s is None else s.start_s
                    if end - cursor > 1e-6:
                        filled = paint(filled, cursor, end, default)
                    cursor = max(cursor, s.end_s) if s else cursor
                setattr(p, name, filled)
            commit(f"{p.person_id} gaps filled with normal/{visibility}")
        elif k == ord("m"):
            a, b = rng()
            checkpoint()
            ev = st["event"] or L.next_event_id()
            L.missed_events.append(MissedEvent(ev, a, b, note=input(f"Note for missed event {ev}: ").strip()))
            st["mark"] = None
            commit(f"missed event {ev} {a:.2f}-{b:.2f}s")
        elif k == ord("c"):
            # Check as if complete: lint only reports unlabelled time once complete is set.
            errs, warns = completion_check(L)
            if errs:
                print("\nCannot mark complete:\n  " + "\n  ".join(errs))
            L.complete = not errs
            if L.complete:
                commit("video marked COMPLETE")
                if warns:
                    print("\nWarnings:\n  " + "\n  ".join(warns))
            else:
                commit("labels saved (video incomplete)")
            labeled_video = export_labeled_video()
            st["msg"] += f" | video saved: {labeled_video.name}"
    save(L, label_path)
    cv2.destroyAllWindows()
    errs, warns = lint(L)
    print(f"Saved {label_path} - {len(L.persons)} persons, {len(errs)} errors, {len(warns)} warnings")
    for e in errs:
        print("  ERROR", e)


if __name__ == "__main__":
    main()