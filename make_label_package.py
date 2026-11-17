"""Build a self-contained labelling package for an external labeller.

  python make_label_package.py --out label_pkg_hadiya --videos "1" "2" "159 Back Cooler_05"
  python make_label_package.py --out label_pkg_all --all --zip

Contains only what the Labelling Studio (and annotate.py) need: the tools, the
shoplift modules they import, the guide, the tutorial recording if present,
a registry with package-relative video paths, the YOLO11 reference tracks, the
videos (unless --no-videos), and any existing labels for those videos. No model
weights, Nx/camera settings, config.yaml, backend or other outputs are included.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from shoplift.registry import find, load_registry, save_registry  # noqa: E402

CODE = ["labelling_studio.py", "annotate.py", "label_check.py", "shoplift/__init__.py", "shoplift/annotation.py",
        "shoplift/ingest.py", "shoplift/registry.py", "shoplift/label_readiness.py", "shoplift/label_session.py",
        "shoplift/label_exchange.py", "shoplift/studio_manual.py", "docs/LABELLING_GUIDE.md"]
OPTIONAL = ["videos/Video annoation.mp4"]    # tutorial recording, copied when present
TRACKS = Path("outputs/extract/yolo11m_10hz")

README = """# Labelling package

1. Install Python 3.10-3.12 (python.org, tick "Add to PATH").
2. In this folder (PowerShell):
     py -m venv .venv
     .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
3. Start the Labelling Studio:
     .\\.venv\\Scripts\\python.exe labelling_studio.py
   New here? Open its Manual tab first (guide, decision helper, practice quiz).
   Videos in this package: {videos}
4. Label one video at a time in the Label tab until the Check tab says READY.
5. Export tab > Export zip. Return ONLY that zip file. Never edit label files by hand.

Confidential client footage: do not copy, upload or share the videos; delete the
package when the work is accepted.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--videos", nargs="+", help="Registered names/keys to include")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--no-videos", action="store_true", help="Labeller already has the videos (use --video-file)")
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()

    reg = load_registry(ROOT / "labels" / "videos.json")
    entries = list(reg.values()) if a.all else [find(reg, v) for v in a.videos]
    out = a.out
    if out.exists():
        sys.exit(f"{out} already exists")
    for f in CODE:
        (out / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, out / f)
    for f in OPTIONAL:
        if (ROOT / f).is_file():
            (out / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / f, out / f)
    (out / "requirements.txt").write_text("numpy>=1.26\npandas>=2.0\nopencv-python>=4.9\nPillow>=10\n",
                                          encoding="utf-8")

    pkg_reg, missing = {}, []
    for e in entries:
        tracks = ROOT / TRACKS / f"{e.name}_tracks.csv"
        if not tracks.is_file():
            missing.append(e.name)
            continue
        (out / TRACKS).mkdir(parents=True, exist_ok=True)
        shutil.copy2(tracks, out / TRACKS / tracks.name)
        summary = ROOT / TRACKS / f"{e.name}_summary.json"     # marks the tracks run as finished
        if summary.is_file():
            shutil.copy2(summary, out / TRACKS / summary.name)
        rel = f"videos/{e.video_key}.mp4"
        if not a.no_videos:
            src = next((Path(p) for p in e.paths if Path(p).is_file()), None)
            if src is None:
                sys.exit(f"No copy of video '{e.name}' found at {e.paths}")
            (out / "videos").mkdir(exist_ok=True)
            shutil.copy2(src, out / rel)
        lab = ROOT / "labels" / f"{e.video_key}.json"
        if lab.is_file():
            (out / "labels").mkdir(exist_ok=True)
            shutil.copy2(lab, out / "labels" / lab.name)
        pkg_reg[e.video_key] = type(e)(**{**e.__dict__, "paths": [rel]})
    if missing:
        sys.exit(f"No YOLO11 10 Hz extraction yet for: {missing}. Run the extraction first.")
    save_registry(out / "labels" / "videos.json", pkg_reg)
    (out / "README_LABELLER.md").write_text(README.format(videos=", ".join(f'"{e.name}"' for e in entries)),
                                            encoding="utf-8")
    manifest = {e.name: {"video_key": e.video_key, "sha256": e.sha256, "holdout": e.holdout} for e in entries}
    (out / "package_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    if a.zip:
        shutil.make_archive(str(out), "zip", out)
    print(f"Package ready: {out} ({len(entries)} videos)" + (" + .zip" if a.zip else ""))


if __name__ == "__main__":
    main()
