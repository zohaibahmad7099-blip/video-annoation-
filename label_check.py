"""Quality check over every label file before a dataset build.

  python -m shoplift.label_check
  python -m shoplift.label_check --gallery outputs/actor_gallery.jpg

Per file: name vs video_key, registry match, reference tracks still current,
lint errors/warnings, suspicious durations, event-linked picks. Across files:
the actor roster. --gallery renders one crop per actor per video so a reviewer
can confirm that an actor_id names the same individual everywhere.
"""
import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from shoplift.annotation import index_label_files, lint, load
from shoplift.registry import load_registry

_HERE = Path(__file__).resolve().parent
ROOT = _HERE if (_HERE / "labels").is_dir() else _HERE.parent   # works at repo root or in shoplift/


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check(labels_dir: Path):
    reg = load_registry(labels_dir / "videos.json")
    files = index_label_files(labels_dir)
    roster, blocking = defaultdict(list), 0
    for key, f in sorted(files.items(), key=lambda kv: kv[1].name):
        L = load(f)
        e = reg.get(key)
        notes = []
        if f.stem != key:
            notes.append(f"file named {f.name}; canonical name is {key}.json (still found by content)")
        if e is None:
            notes.append("ERROR video not in registry")
        elif e.sha256 != L.sha256:
            notes.append("ERROR video hash differs from registry")
        ref = ROOT / L.reference_tracks.replace("\\", "/")      # written on Windows, read on Linux/Colab
        if not ref.is_file() or _sha(ref) != L.reference_tracks_sha256:
            notes.append("reference tracks differ from the current extraction (person boxes are used, track ids may not match)")
        if not L.reviewer or L.reviewer.lower() in ("yourname", "your name"):
            notes.append("reviewer not set")
        errs, warns = lint(L)
        picks_unlinked = sum(1 for p in L.persons for s in p.phase if s.value == "pick" and not s.event_id
                             and any(x.value == "suspicious" for x in p.phase))
        if picks_unlinked:
            warns.append(f"{picks_unlinked} pick(s) by a person who later conceals have no event_id (key 4)")
        blocking += len(errs) + sum(n.startswith("ERROR") for n in notes)
        name = e.name if e else L.video_name
        print(f"\n{name} [{f.name}] complete={L.complete} reviewer={L.reviewer!r} persons={len(L.persons)} "
              f"-> {len(errs)} errors, {len(warns)} warnings")
        for n in notes:
            print("  NOTE ", n)
        for x in errs:
            print("  ERROR", x)
        for x in warns:
            print("  WARN ", x)
        for p in L.persons:
            if p.actor_id:
                roster[p.actor_id].append((name, p.person_id, bool(p.is_staff), p.boxes[len(p.boxes) // 2] if p.boxes else None))
    print("\nActor roster (verify identities with --gallery):")
    for a, who in sorted(roster.items(), key=lambda kv: (len(kv[0]), kv[0])):
        print(f"  {a:5s} {[(v, p) for v, p, _, _ in who]}")
    return blocking, roster, reg, files


def gallery(roster, reg, files, out: Path):
    import cv2
    from shoplift.ingest import FileSource
    videos = sorted({v for who in roster.values() for v, _, _, _ in who})
    actors = sorted(roster, key=lambda a: (len(a), a))
    want = defaultdict(dict)                      # video -> time -> [(actor, box)]
    for a, who in roster.items():
        for v, _, _, b in who:
            if b is not None and a not in {x for t in want[v].values() for x, _ in t}:
                want[v].setdefault(round(b[0], 3), []).append((a, b[1:]))
    by_name = {e.name: e for e in reg.values()}
    cells = {}
    for v in videos:
        ts = sorted(want[v])
        if not ts or v not in by_name:
            continue
        for f in FileSource(by_name[v].paths[0]):
            while ts and f.timestamp_s >= ts[0] - 0.045:
                for a, b in want[v][ts.pop(0)]:
                    x1, y1, x2, y2 = [int(c) for c in b]
                    pad = int(0.12 * (y2 - y1))
                    c = f.image[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
                    c = cv2.resize(c, (int(150 * c.shape[1] / max(c.shape[0], 1)), 150))[:, :110]
                    cells[(a, v)] = cv2.copyMakeBorder(c, 0, 150 - c.shape[0], 0, 110 - c.shape[1], cv2.BORDER_CONSTANT)
            if not ts:
                break
    rows = []
    head = np.zeros((20, 60 + 110 * len(videos), 3), np.uint8)
    for j, v in enumerate(videos):
        cv2.putText(head, v[:14], (62 + 110 * j, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    rows.append(head)
    for a in actors:
        lab = np.zeros((150, 60, 3), np.uint8)
        cv2.putText(lab, a, (4, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        rows.append(np.hstack([lab] + [cells.get((a, v), np.zeros((150, 110, 3), np.uint8)) for v in videos]))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows))
    print(f"\nGallery written: {out} (rows = actor_id, columns = video; each row must show one individual)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=ROOT / "labels")
    ap.add_argument("--gallery", type=Path)
    a = ap.parse_args()
    blocking, roster, reg, files = check(a.labels)
    if a.gallery:
        gallery(roster, reg, files, a.gallery)
    print(f"\n{'READY' if not blocking else f'NOT READY: {blocking} blocking problem(s)'}")
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
