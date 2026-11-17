"""Labelling Studio: label, check and export Stage A behaviour labels in one window.

  python labelling_studio.py                 # opens the studio for this folder
  python labelling_studio.py --root D:\\pkg   # a labelling package elsewhere

Tabs
  Label    open a video, build persons from tracks, paint phase / visibility / events
           (same keys as annotate.py; saves after every change; undo with X)
  Check    READY / IN PROGRESS / NOT READY / NOT STARTED for every video, with reasons and fixes
  Export   pack READY videos into one verified zip for the training pipeline
  Manual   guide for new labellers, phase decision helper, error fixer, practice quiz

Label files are the same `labels/<video_key>.json` that annotate.py writes
(shoplift/annotation.py). Requirements: numpy, opencv-python, Pillow (Tkinter
ships with Python). Generating missing tracks also needs ultralytics + torch.
"""
import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import tkinter as tk  # noqa: E402
from tkinter import filedialog, messagebox, simpledialog, ttk  # noqa: E402

HZ = 10.0
STUDIO_VERSION = "1.0"


# ====================================================================== track generation (subprocess)
def generate_tracks_cli(a) -> int:
    """Hidden sub-command run in a child process so the window stays responsive."""
    try:
        from shoplift.pipeline import Pipeline, PipelineConfig
    except ImportError as exc:
        print(f"Track generation needs the full repository (shoplift/pipeline.py) with ultralytics and torch "
              f"installed ({exc}). In a labelling package the tracks are included; ask the project lead for "
              f"the tracks of new videos.", flush=True)
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("_tracks.csv", "_summary.json"):
        (out / f"{a.name}{suffix}").unlink(missing_ok=True)
    meta = {"sha256": a.sha, "name": a.name} if a.sha else {"name": a.name}
    Pipeline(PipelineConfig(pose_weights=a.weights, device=a.device, analysis_hz=a.hz), scorer=None).run(
        a.video, out, run_name=a.name, meta=meta)
    print("[Done]", flush=True)
    return 0


# ====================================================================== GUI
def _gui_imports():
    global np, Image, ImageTk, PHASES, VISIBILITY, value_at
    global READY, NOT_READY, IN_PROGRESS, NOT_STARTED, Workspace, check_dataset, completion_check, file_sha256
    global FrameStore, LabelSession, Cancelled, PHASE_COLORS, VIS_COLORS, person_color, tracks_state, jpeg_to_pil
    global export_labels, verify_package, last_exported, read_log
    global load_registry, save_registry, VideoEntry, manual, issue_help
    import numpy as np  # noqa: F401
    from PIL import Image, ImageTk  # noqa: F401
    from shoplift.annotation import PHASES, VISIBILITY, value_at  # noqa: F401
    from shoplift.label_readiness import (READY, NOT_READY, IN_PROGRESS, NOT_STARTED, Workspace,  # noqa: F401
                                          check_dataset, completion_check, file_sha256)
    from shoplift.label_session import (FrameStore, LabelSession, Cancelled, PHASE_COLORS, VIS_COLORS,  # noqa: F401
                                        person_color, tracks_state, jpeg_to_pil)
    from shoplift.label_exchange import export_labels, verify_package, last_exported, read_log  # noqa: F401
    from shoplift.registry import load_registry, save_registry, VideoEntry  # noqa: F401
    from shoplift import studio_manual as manual  # noqa: F401
    from shoplift.studio_manual import issue_help  # noqa: F401


PHASE_KEYS = {"1": ("normal", False), "2": ("inspect", False), "3": ("pick", False), "4": ("pick", True),
              "5": ("suspicious", True), "6": ("basket", False), "7": ("return", False), "0": ("unknown", False)}
PHASE_BUTTONS = [("1", "normal"), ("2", "inspect"), ("3", "pick"), ("4", "pick+event"), ("5", "suspicious"),
                 ("6", "basket"), ("7", "return"), ("0", "unknown")]
VIS_KEYS = {"v": "visible", "o": "hands_occluded", "b": "body_occluded"}
STATUS_FG = {"READY": "#1b5e20", "NOT READY": "#b71c1c", "IN PROGRESS": "#e65100", "NOT STARTED": "#616161"}
STATUS_BG = {"READY": "#c8e6c9", "NOT READY": "#ffcdd2", "IN PROGRESS": "#ffe0b2", "NOT STARTED": "#eeeeee"}
TEXT_INPUT_CLASSES = {"Entry", "TEntry", "TCombobox", "Text", "Spinbox", "TSpinbox"}
REVIEW_CHECKLIST = [
    "I watched the whole video at least once.",
    "Every person who appears has a person with linked tracks (no grey box left that is a real person).",
    "No track is linked to the wrong person (I checked crossings and ID switches).",
    "Every concealment is suspicious with an event; stolen items' picks use key 4; purchases have no event.",
    "Visibility reflects what this camera sees during each concealment.",
    "actor_ids name real individuals, consistent with other videos; staff are marked.",
]


def contrast(hex_color: str) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#000000" if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else "#ffffff"


class Settings(dict):
    DEFAULTS = {"reviewer": "", "pose_weights": "", "device": "cpu", "display_width": 1600, "manual_checks": {},
                "export_dir": "", "last_video": ""}

    def __init__(self, path: Path):
        super().__init__(self.DEFAULTS)
        self.path = path
        try:
            if path.is_file():
                self.update(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
        if not self["pose_weights"]:
            cands = [ROOT / "yolo11m-pose.pt", ROOT / "models" / "yolo11m-pose.pt", ROOT.parent / "models" /
                     "yolo11m-pose.pt", Path("D:/models/yolo11m-pose.pt"), ROOT.parent.parent.parent / "models" /
                     "yolo11m-pose.pt"]
            self["pose_weights"] = str(next((c for c in cands if c.is_file()), cands[0]))

    def save(self):
        try:
            self.path.write_text(json.dumps(self, indent=1), encoding="utf-8")
        except OSError:
            pass


def btn(parent, text, cmd, **kw):
    b = ttk.Button(parent, text=text, command=cmd, takefocus=0, **kw)
    return b


def register_file(ws, path: Path, name: str = None, expect_sha: str = None):
    """Register a video file (by content hash); returns the entry. Raises ValueError on conflicts."""
    import cv2
    path = Path(path).resolve()
    digest = file_sha256(path)
    if expect_sha and digest != expect_sha:
        raise ValueError(f"{path.name} is not the video these labels were made for (content hash differs)")
    reg = load_registry(ws.registry_path)
    dup = next((e for e in reg.values() if e.sha256 == digest), None)
    if dup is not None:
        if str(path) not in dup.paths:
            dup.paths.append(str(path))
            save_registry(ws.registry_path, reg)
        return dup
    name = name or path.stem
    if name in reg:
        raise ValueError(f"a different video is already registered as '{name}'")
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise IOError(f"cannot open video {path}")
    e = VideoEntry(name, digest, name, [str(path)], int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), round(float(cap.get(cv2.CAP_PROP_FPS) or 0), 3),
                   int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    cap.release()
    reg[name] = e
    save_registry(ws.registry_path, reg)
    return e


# ---------------------------------------------------------------------- app shell
class Studio(tk.Tk):
    def __init__(self, ws):
        super().__init__()
        self.ws = ws
        self.settings = Settings(ws.root / "studio_settings.json")
        self.title(f"Labelling Studio - {ws.root.name}")
        self.geometry("1560x940")
        self.minsize(1180, 720)
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Treeview", rowheight=22)
        style.configure("H.TLabel", font=("Segoe UI", 13, "bold"))
        style.configure("Small.TLabel", font=("Segoe UI", 9), foreground="#555")
        style.configure("TNotebook.Tab", padding=(14, 5))
        self.job = None

        head = ttk.Frame(self, padding=(10, 6, 10, 0))
        head.pack(fill="x")
        ttk.Label(head, text="Labelling Studio", style="H.TLabel").pack(side="left")
        ttk.Label(head, text=f"   workspace: {ws.root}", style="Small.TLabel").pack(side="left")
        btn(head, "Settings", self.open_settings).pack(side="right")

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=6, pady=4)
        self.label_tab = LabelTab(self.nb, self)
        self.check_tab = CheckTab(self.nb, self)
        self.export_tab = ExportTab(self.nb, self)
        self.manual_tab = ManualTab(self.nb, self)
        self.nb.add(self.label_tab, text="  Label  ")
        self.nb.add(self.check_tab, text="  Check  ")
        self.nb.add(self.export_tab, text="  Export  ")
        self.nb.add(self.manual_tab, text="  Manual  ")
        self.nb.bind("<<NotebookTabChanged>>", self._tab_changed)

        bar = ttk.Frame(self, padding=(8, 2))
        bar.pack(fill="x", side="bottom")
        self.status_var = tk.StringVar(value="Ready. New here? Open the Manual tab.")
        ttk.Label(bar, textvariable=self.status_var, anchor="w").pack(side="left", fill="x", expand=True)
        self.job_lbl = ttk.Label(bar, text="")
        self.job_bar = ttk.Progressbar(bar, length=260, mode="determinate", maximum=100)
        self.job_cancel = btn(bar, "Cancel", self._cancel_job)

        self.bind_all("<Key>", self._on_key)
        self.protocol("WM_DELETE_WINDOW", self._close)
        first_time = not self.settings.get("reviewer")
        self.after(200, lambda: self.nb.select(self.manual_tab) if first_time else None)

    # -------- status / jobs
    def status(self, msg: str):
        if msg:
            self.status_var.set(msg)

    def run_bg(self, title, work, done=None, error=None, cancellable=True) -> bool:
        if self.job is not None:
            messagebox.showinfo("Busy", f"Please wait: {self.job['title']}", parent=self)
            return False
        q, cancel = queue.Queue(), threading.Event()

        def progress(i, n, text=""):
            q.put(("p", i, n, text))

        def runner():
            try:
                q.put(("ok", work(progress, cancel.is_set)))
            except Cancelled:
                q.put(("cancel",))
            except Exception as exc:  # noqa: BLE001
                q.put(("err", exc, traceback.format_exc()))

        self.job = {"title": title, "q": q, "cancel": cancel, "done": done, "error": error}
        self.job_lbl.configure(text=title)
        self.job_lbl.pack(side="right", padx=4)
        self.job_cancel.pack(side="right", padx=4)
        self.job_bar.pack(side="right", padx=4)
        self.job_bar["value"] = 0
        if not cancellable:
            self.job_cancel.pack_forget()
        threading.Thread(target=runner, daemon=True).start()
        self.after(100, self._poll_job)
        return True

    def _poll_job(self):
        job = self.job
        if job is None:
            return
        try:
            while True:
                item = job["q"].get_nowait()
                if item[0] == "p":
                    _, i, n, text = item
                    self.job_bar["value"] = 100.0 * i / max(n, 1)
                    self.job_lbl.configure(text=f"{job['title']}  {100.0 * i / max(n, 1):.0f}%")
                    continue
                self.job = None
                for w in (self.job_lbl, self.job_bar, self.job_cancel):
                    w.pack_forget()
                if item[0] == "ok":
                    if job["done"]:
                        job["done"](item[1])
                elif item[0] == "cancel":
                    self.status(f"{job['title']}: cancelled")
                    if job["error"]:
                        job["error"](None)
                else:
                    self.status(f"{job['title']}: failed")
                    if job["error"]:
                        job["error"](item[1])
                    else:
                        messagebox.showerror(job["title"], f"{item[1]}\n\n{item[2][-1500:]}", parent=self)
                return
        except queue.Empty:
            pass
        self.after(100, self._poll_job)

    def _cancel_job(self):
        if self.job:
            self.job["cancel"].set()
            self.job_lbl.configure(text=f"{self.job['title']}  cancelling...")

    # -------- keys / tabs
    def _on_key(self, e):
        if self.nb.select() != str(self.label_tab):
            return
        w = e.widget
        try:
            if isinstance(w, tk.Misc) and w.winfo_class() in TEXT_INPUT_CLASSES:
                return
            if isinstance(w, tk.Misc) and w.winfo_toplevel() is not self:
                return
        except (tk.TclError, KeyError):
            return
        self.label_tab.on_key(e)

    def _tab_changed(self, _=None):
        cur = self.nb.select()
        if cur == str(self.check_tab):
            self.check_tab.refresh()
        elif cur == str(self.export_tab):
            self.export_tab.refresh()
        elif cur == str(self.label_tab):
            self.label_tab.refresh_video_list()
        if cur != str(self.label_tab):
            self.label_tab.pause()

    def goto(self, tab):
        self.nb.select(tab)

    def open_settings(self):
        SettingsDialog(self)

    def _close(self):
        if self.job and not messagebox.askyesno("Quit", f"'{self.job['title']}' is still running. Quit anyway?",
                                                parent=self):
            return
        self.label_tab.pause()
        self.settings.save()
        self.destroy()


class SettingsDialog(tk.Toplevel):
    def __init__(self, app: Studio):
        super().__init__(app)
        self.app = app
        self.title("Settings")
        self.transient(app)
        self.resizable(False, False)
        s = app.settings
        f = ttk.Frame(self, padding=14)
        f.pack(fill="both")
        self.vars = {}
        rows = [("reviewer", "Reviewer name"), ("pose_weights", "Pose weights (yolo11m-pose.pt, only for new videos)"),
                ("device", "Inference device (cpu, or 0 for the first GPU)"),
                ("display_width", "Decode width in pixels (smaller = faster, less detail)")]
        for r, (k, lab) in enumerate(rows):
            ttk.Label(f, text=lab).grid(row=r, column=0, sticky="w", pady=3)
            v = tk.StringVar(value=str(s.get(k, "")))
            self.vars[k] = v
            ttk.Entry(f, textvariable=v, width=60).grid(row=r, column=1, sticky="we", padx=6)
            if k == "pose_weights":
                btn(f, "Browse", self._browse).grid(row=r, column=2)
        ttk.Label(f, text="Tracks are always generated at 10 Hz with YOLO11-M into outputs/extract/yolo11m_10hz "
                          "(the training pipeline's reference source).", style="Small.TLabel",
                  wraplength=560).grid(row=len(rows), column=0, columnspan=3, sticky="w", pady=(8, 4))
        b = ttk.Frame(f)
        b.grid(row=len(rows) + 1, column=0, columnspan=3, sticky="e", pady=(8, 0))
        btn(b, "Save", self._save).pack(side="right", padx=4)
        btn(b, "Cancel", self.destroy).pack(side="right")
        self.grab_set()

    def _browse(self):
        p = filedialog.askopenfilename(parent=self, filetypes=[("PyTorch weights", "*.pt"), ("All", "*.*")])
        if p:
            self.vars["pose_weights"].set(p)

    def _save(self):
        s = self.app.settings
        for k, v in self.vars.items():
            val = v.get().strip()
            if k == "display_width":
                try:
                    val = max(320, min(5120, int(val)))
                except ValueError:
                    messagebox.showerror("Settings", "Decode width must be a number", parent=self)
                    return
            s[k] = val
        s.save()
        self.app.label_tab.reviewer_var.set(s["reviewer"])
        self.app.label_tab.commit_reviewer()
        self.destroy()


# ====================================================================== Label tab
class LabelTab(ttk.Frame):
    def __init__(self, master, app: Studio):
        super().__init__(master, padding=4)
        self.app, self.ws = app, app.ws
        self.session = None
        self.frames = None
        self.i = 0
        self.playing = False
        self.mark = None
        self.track = None
        self.pid = None
        self.event = ""
        self.photo = None
        self.fit = (1.0, 0, 0)
        self.names = []
        self._scale_guard = False
        self._play_job = None
        self.report = None
        self._build()
        self.refresh_video_list()

    # ------------------------------------------------------------ layout
    def _build(self):
        top = ttk.Frame(self)
        top.pack(fill="x", pady=(0, 4))
        ttk.Label(top, text="Video").pack(side="left")
        self.video_cb = ttk.Combobox(top, state="readonly", width=44)
        self.video_cb.pack(side="left", padx=4)
        self.video_cb.bind("<<ComboboxSelected>>", lambda e: self.canvas.focus_set())
        btn(top, "Open", self.open_video).pack(side="left")
        btn(top, "Register new video...", self.register_new).pack(side="left", padx=4)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Label(top, text="Reviewer").pack(side="left")
        self.reviewer_var = tk.StringVar(value=self.app.settings.get("reviewer", ""))
        e = ttk.Entry(top, textvariable=self.reviewer_var, width=16)
        e.pack(side="left", padx=4)
        e.bind("<Return>", lambda _: (self.commit_reviewer(), self.canvas.focus_set()))
        e.bind("<FocusOut>", lambda _: self.commit_reviewer())
        ttk.Label(top, text="Session").pack(side="left", padx=(8, 0))
        self.session_var = tk.StringVar()
        e = ttk.Entry(top, textvariable=self.session_var, width=8)
        e.pack(side="left", padx=4)
        e.bind("<Return>", lambda _: (self.commit_session(), self.canvas.focus_set()))
        e.bind("<FocusOut>", lambda _: self.commit_session())
        self.badge = tk.Label(top, text="no video open", padx=10, pady=2, font=("Segoe UI", 10, "bold"),
                              bg="#eeeeee", fg="#616161")
        self.badge.pack(side="right")
        self.badge.bind("<Button-1>", lambda _: self.side.select(self.issues_tab))

        pw = ttk.PanedWindow(self, orient="horizontal")
        pw.pack(fill="both", expand=True)
        left = ttk.Frame(pw)
        right = ttk.Frame(pw, width=400)
        pw.add(left, weight=4)
        pw.add(right, weight=1)
        self._pw = pw
        self.after(400, lambda: pw.sashpos(0, max(600, pw.winfo_width() - 420)))

        self.canvas = tk.Canvas(left, bg="#101010", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.show_frame())
        self.canvas.bind("<Button-1>", self.on_click)

        tr = ttk.Frame(left)
        tr.pack(fill="x", pady=2)
        for text, cmd in (("|<", lambda: self.seek(0)), ("-10s", lambda: self.step(-100)),
                          ("-2s", lambda: self.step(-20)), ("<", lambda: self.step(-1))):
            btn(tr, text, cmd, width=5).pack(side="left")
        self.play_btn = btn(tr, "Play", self.toggle_play, width=7)
        self.play_btn.pack(side="left", padx=2)
        for text, cmd in ((">", lambda: self.step(1)), ("+2s", lambda: self.step(20)),
                          ("+10s", lambda: self.step(100)), (">|", lambda: self.seek(10 ** 9))):
            btn(tr, text, cmd, width=5).pack(side="left")
        self.time_lbl = ttk.Label(tr, text="t = -", width=32, font=("Consolas", 10))
        self.time_lbl.pack(side="right")
        self.scale = ttk.Scale(tr, from_=0, to=1, orient="horizontal", command=self._on_scale, takefocus=0)
        self.scale.pack(side="left", fill="x", expand=True, padx=8)

        self.tl = tk.Canvas(left, height=120, bg="#1c1c1c", highlightthickness=0)
        self.tl.pack(fill="x", pady=(2, 4))
        self.tl.bind("<Configure>", lambda e: self.draw_timeline())
        self.tl.bind("<Button-1>", self.on_timeline)
        self.tl.bind("<B1-Motion>", self.on_timeline_drag)

        ab = ttk.Frame(left)
        ab.pack(fill="x")
        r1 = ttk.Frame(ab)
        r1.pack(fill="x", pady=1)
        ttk.Label(r1, text="Range", width=10).pack(side="left")
        btn(r1, "Start here (S)", self.set_mark).pack(side="left")
        btn(r1, "Clear (Esc)", self.clear_mark).pack(side="left", padx=2)
        self.range_lbl = ttk.Label(r1, text="range: current frame", width=24)
        self.range_lbl.pack(side="left", padx=6)
        ttk.Label(r1, text="Event").pack(side="left", padx=(4, 4))
        self.event_cb = ttk.Combobox(r1, width=6, state="readonly")
        self.event_cb.pack(side="left")
        self.event_cb.bind("<<ComboboxSelected>>", self._event_picked)
        btn(r1, "New (E)", self.new_event).pack(side="left", padx=2)
        btn(r1, "Next (W)", self.cycle_event).pack(side="left")
        complete_btn = tk.Button(r1, text="Mark complete (C)", bg="#2e7d32", fg="#ffffff", activebackground="#1b5e20",
                                 bd=1, padx=10, takefocus=0, command=self.mark_complete)
        complete_btn.pack(side="right")
        btn(r1, "Undo (X)", self.undo).pack(side="right", padx=4)
        r2 = ttk.Frame(ab)
        r2.pack(fill="x", pady=1)
        ttk.Label(r2, text="Phase", width=10).pack(side="left")
        for key, name in PHASE_BUTTONS:
            col = PHASE_COLORS["pick" if name == "pick+event" else name]
            tk.Button(r2, text=f"{key} {name}", bg=col, fg=contrast(col), activebackground=col, relief="raised",
                      bd=1, padx=6, takefocus=0, command=lambda k=key: self.do_phase(k)).pack(
                side="left", padx=1, fill="x", expand=True)
        r3 = ttk.Frame(ab)
        r3.pack(fill="x", pady=1)
        ttk.Label(r3, text="Visibility", width=10).pack(side="left")
        for key, name in (("v", "visible"), ("o", "hands_occluded"), ("b", "body_occluded")):
            col = VIS_COLORS[name]
            tk.Button(r3, text=f"{key.upper()} {name}", bg=col, fg=contrast(col), activebackground=col, bd=1,
                      padx=6, takefocus=0, command=lambda k=key: self.do_vis(k)).pack(side="left", padx=1, fill="x",
                                                                                         expand=True)
        ttk.Separator(r3, orient="vertical").pack(side="left", fill="y", padx=8)
        btn(r3, "Fill gaps (F)", self.fill_gaps).pack(side="left")
        btn(r3, "Missed event (M)", self.missed_event).pack(side="left", padx=4)

        # right side panels
        self.side = ttk.Notebook(right)
        self.side.pack(fill="both", expand=True)
        self.persons_tab = ttk.Frame(self.side, padding=4)
        self.segments_tab = ttk.Frame(self.side, padding=4)
        self.issues_tab = ttk.Frame(self.side, padding=4)
        keys_tab = ttk.Frame(self.side, padding=4)
        self.side.add(self.persons_tab, text="Persons")
        self.side.add(self.segments_tab, text="Segments")
        self.side.add(self.issues_tab, text="Issues")
        self.side.add(keys_tab, text="Keys")
        self._build_persons()
        self._build_segments()
        self._build_issues()
        kt = tk.Text(keys_tab, wrap="word", font=("Consolas", 9), relief="flat", bg="#fafafa")
        kt.pack(fill="both", expand=True)
        keys = next(b for t, blocks in manual.CHAPTERS for b in blocks if b[0] == "keys")[1]
        kt.insert("end", "\n".join(f"{k:<24}{v}" for k, v in keys))
        kt.configure(state="disabled")

    def _build_persons(self):
        f = self.persons_tab
        cols = ("actor", "staff", "tracks", "phase", "vis")
        self.ptree = ttk.Treeview(f, columns=cols, height=9, selectmode="browse")
        self.ptree.heading("#0", text="person")
        self.ptree.column("#0", width=60)
        for c, w, h in zip(cols, (60, 44, 80, 52, 52), ("actor", "staff", "tracks", "phase%", "vis%")):
            self.ptree.heading(c, text=h)
            self.ptree.column(c, width=w, anchor="center")
        self.ptree.pack(fill="x")
        self.ptree.bind("<<TreeviewSelect>>", self._person_picked)
        self.ptree.bind("<Double-1>", lambda e: self._jump_person_start())
        self.track_lbl = ttk.Label(f, text="selected track: -   (click a box in the video)")
        self.track_lbl.pack(fill="x", pady=(4, 2))
        g = ttk.Frame(f)
        g.pack(fill="x")
        for r, row in enumerate(((("New person from track (N)", self.new_person), ("Link track (G)", self.link_track)),
                                 (("Unlink track (U)", self.unlink_track), ("Delete person", self.delete_person)),
                                 (("Reset person labels (Z)", self.reset_person), ("Re-link tracks", self.relink)))):
            for c, (t, cmd) in enumerate(row):
                b = btn(g, t, cmd)
                b.grid(row=r, column=c, sticky="we", padx=1, pady=1)
                if t == "Re-link tracks":
                    self.relink_btn = b
        g.columnconfigure(0, weight=1)
        g.columnconfigure(1, weight=1)
        a = ttk.LabelFrame(f, text="Selected person", padding=4)
        a.pack(fill="x", pady=6)
        ttk.Label(a, text="Actor id").grid(row=0, column=0, sticky="w")
        self.actor_var = tk.StringVar()
        self.actor_cb = ttk.Combobox(a, textvariable=self.actor_var, width=10)
        self.actor_cb.grid(row=0, column=1, sticky="w", padx=4)
        self.actor_cb.bind("<Return>", lambda _: self.set_actor())
        btn(a, "Set (I)", self.set_actor).grid(row=0, column=2, sticky="w")
        self.staff_var = tk.BooleanVar()
        ttk.Checkbutton(a, text="Staff (T)", variable=self.staff_var, command=self.set_staff,
                        takefocus=0).grid(row=0, column=3, padx=8)
        ttk.Label(a, text="Note").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.note_var = tk.StringVar()
        ne = ttk.Entry(a, textvariable=self.note_var, width=30)
        ne.grid(row=1, column=1, columnspan=2, sticky="we", padx=4, pady=(4, 0))
        ne.bind("<Return>", lambda _: self.set_note())
        btn(a, "Save", self.set_note).grid(row=1, column=3, pady=(4, 0))
        ttk.Label(f, text="actor_id names the real individual: the same person keeps the same id in every video; "
                          "never number them p1=A1, p2=A2.", style="Small.TLabel", wraplength=370).pack(fill="x")
        self.tracks_warn = tk.Label(f, text="", fg="#b71c1c", wraplength=370, justify="left")
        self.tracks_warn.pack(fill="x", pady=4)

    def _build_segments(self):
        f = self.segments_tab
        cols = ("value", "start", "end", "event")
        self.stree = ttk.Treeview(f, columns=cols, height=14, selectmode="browse")
        self.stree.heading("#0", text="row")
        self.stree.column("#0", width=70)
        for c, w in zip(cols, (110, 60, 60, 50)):
            self.stree.heading(c, text=c)
            self.stree.column(c, width=w, anchor="center")
        sb = ttk.Scrollbar(f, orient="vertical", command=self.stree.yview)
        self.stree.configure(yscrollcommand=sb.set)
        top = ttk.Frame(f)
        top.pack(fill="both", expand=True)
        self.stree.pack(in_=top, side="left", fill="both", expand=True)
        sb.pack(in_=top, side="right", fill="y")
        self.stree.bind("<Double-1>", lambda e: self.seg_goto())
        b = ttk.Frame(f)
        b.pack(fill="x", pady=4)
        btn(b, "Go to", self.seg_goto).pack(side="left")
        btn(b, "Select as range", self.seg_range).pack(side="left", padx=2)
        btn(b, "Delete segment", self.seg_delete).pack(side="left")
        ttk.Label(f, text="Missed events (no track)").pack(anchor="w", pady=(8, 0))
        self.mtree = ttk.Treeview(f, columns=("start", "end", "note"), height=4, selectmode="browse")
        self.mtree.heading("#0", text="event")
        self.mtree.column("#0", width=60)
        for c, w in (("start", 60), ("end", 60), ("note", 160)):
            self.mtree.heading(c, text=c)
            self.mtree.column(c, width=w)
        self.mtree.pack(fill="x")
        self.mtree.bind("<Double-1>", lambda e: self._missed_goto())
        btn(f, "Delete missed event", self.missed_delete).pack(anchor="w", pady=4)

    def _build_issues(self):
        f = self.issues_tab
        self.ibadge = tk.Label(f, text="-", font=("Segoe UI", 14, "bold"), pady=6, bg="#eeeeee")
        self.ibadge.pack(fill="x")
        self.isum = ttk.Label(f, text="", wraplength=370)
        self.isum.pack(fill="x", pady=2)
        lf = ttk.Frame(f)
        lf.pack(fill="both", expand=True)
        self.ilist = tk.Listbox(lf, activestyle="none", font=("Segoe UI", 9), height=14)
        sb = ttk.Scrollbar(lf, orient="vertical", command=self.ilist.yview)
        self.ilist.configure(yscrollcommand=sb.set)
        self.ilist.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.ilist.bind("<<ListboxSelect>>", self._issue_picked)
        self.ilist.bind("<Double-1>", lambda e: self._issue_jump())
        self.ihelp = tk.Text(f, height=7, wrap="word", font=("Segoe UI", 9), relief="flat", bg="#f5f5f5")
        self.ihelp.pack(fill="x", pady=4)
        self.ihelp.tag_configure("b", font=("Segoe UI", 9, "bold"))
        self.ihelp.configure(state="disabled")
        b = ttk.Frame(f)
        b.pack(fill="x")
        btn(b, "Mark complete (C)", self.mark_complete).pack(side="left")
        btn(b, "Mark incomplete", self.unmark_complete).pack(side="left", padx=2)
        btn(b, "Review video", self.export_review).pack(side="left")
        self.issues = []

    # ------------------------------------------------------------ video list / open
    def refresh_video_list(self):
        try:
            rep = check_dataset(self.ws)
        except Exception as exc:  # noqa: BLE001
            self.app.status(f"cannot read labels: {exc}")
            return
        reg = load_registry(self.ws.registry_path)
        self.names = sorted(reg, key=str.lower)
        status = {r.key: r.status for r in rep.videos}
        vals = [f"{n}    [{status.get(n, NOT_STARTED)}]" for n in self.names]
        cur = self.session.entry.name if self.session else self.app.settings.get("last_video")
        self.video_cb["values"] = vals
        if cur in self.names:
            self.video_cb.current(self.names.index(cur))
        elif vals and self.video_cb.current() < 0:
            self.video_cb.current(0)

    def selected_name(self):
        i = self.video_cb.current()
        return self.names[i] if 0 <= i < len(self.names) else None

    def open_video(self, name=None):
        name = name or self.selected_name()
        if not name:
            messagebox.showinfo("Open", "Pick a video first (or register a new one).", parent=self)
            return
        reg = load_registry(self.ws.registry_path)
        entry = reg.get(name)
        if entry is None:
            messagebox.showerror("Open", f"'{name}' is not registered.", parent=self)
            return
        if not self.reviewer_var.get().strip():
            r = simpledialog.askstring("Reviewer", "Your name (saved in every label file):", parent=self)
            if not r:
                return
            self.reviewer_var.set(r.strip())
            self.commit_reviewer()
        self.pause()
        path = self.ws.video_file(entry)
        if path is None:
            if not messagebox.askyesno(
                    "Video file not found",
                    f"The file for '{name}' is not at:\n  " + "\n  ".join(entry.paths or ["(no path)"]) +
                    "\n\nLocate it now? (It is checked against the registered content hash.)", parent=self):
                return
            f = filedialog.askopenfilename(parent=self, title=f"Locate video '{name}'",
                                           filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
            if not f:
                return
            self.app.run_bg(f"Checking {Path(f).name}", lambda p, c: register_file(self.ws, Path(f), name,
                                                                                     expect_sha=entry.sha256),
                            done=lambda e: self.open_video(name),
                            error=lambda exc: exc and messagebox.showerror("Wrong file", str(exc), parent=self))
            return
        self._ensure_tracks(entry, path)

    def register_new(self):
        f = filedialog.askopenfilename(parent=self, title="Register a new video",
                                       filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
        if not f:
            return

        def done(e):
            self.refresh_video_list()
            if e.name in self.names:
                self.video_cb.current(self.names.index(e.name))
            self.app.status(f"registered '{e.name}' ({e.width}x{e.height}, {e.frame_count} frames)")
            if messagebox.askyesno("Registered", f"'{e.name}' is registered. Open it now?", parent=self):
                self.open_video(e.name)

        self.app.run_bg(f"Registering {Path(f).name}", lambda p, c: register_file(self.ws, Path(f)), done=done,
                        error=lambda exc: exc and messagebox.showerror("Register", str(exc), parent=self))

    def _ensure_tracks(self, entry, path):
        state = tracks_state(self.ws, entry.name)
        if state == "missing":
            if messagebox.askyesno(
                    "Tracks needed",
                    f"'{entry.name}' has no reference tracks yet.\n\nGenerate them now? Person detection and "
                    f"tracking runs once per video (YOLO11-M, 10 Hz). On a CPU this can take several minutes per "
                    f"10 s of video.", parent=self):
                self._generate_tracks(entry, path)
            return
        if state == "unfinished":
            lp = self.ws.label_path(entry.name)
            uses_it = False
            if lp.is_file():
                try:
                    d = json.loads(lp.read_text(encoding="utf-8"))
                    uses_it = d.get("reference_tracks_sha256") == file_sha256(self.ws.tracks_csv(entry.name))
                except (OSError, ValueError):
                    pass
            if not uses_it:
                ans = messagebox.askyesnocancel(
                    "Tracks may be incomplete",
                    f"The tracks file for '{entry.name}' has no summary, so its generation may have been "
                    f"interrupted.\n\nYes = use it anyway\nNo = generate it again\nCancel = do nothing", parent=self)
                if ans is None:
                    return
                if ans is False:
                    self._generate_tracks(entry, path)
                    return
        self._decode(entry, path)

    def _generate_tracks(self, entry, path):
        s = self.app.settings
        weights = Path(s.get("pose_weights", ""))
        if not weights.is_file():
            messagebox.showinfo("Pose weights", "Locate the YOLO11-M pose weights (yolo11m-pose.pt).", parent=self)
            p = filedialog.askopenfilename(parent=self, filetypes=[("PyTorch weights", "*.pt")])
            if not p:
                return
            weights = Path(p)
            s["pose_weights"] = str(weights)
            s.save()
        cmd = [sys.executable, "-u", str(ROOT / "labelling_studio.py"), "generate-tracks", "--video", str(path),
               "--name", entry.name, "--out", str(self.ws.tracks_dir), "--weights", str(weights),
               "--device", str(s.get("device") or "cpu"), "--sha", entry.sha256]
        csv_path, sum_path = self.ws.tracks_csv(entry.name), self.ws.summary_json(entry.name)

        def work(progress, cancel):
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                    cwd=str(ROOT), creationflags=flags, encoding="utf-8", errors="replace")
            tail = []
            try:
                for line in proc.stdout:
                    tail = (tail + [line.rstrip()])[-20:]
                    m = re.search(r"\[Progress\]\s+([\d.]+)%", line)
                    if m:
                        progress(float(m.group(1)), 100.0)
                    if cancel():
                        proc.terminate()
                        raise Cancelled()
                rc = proc.wait()
            finally:
                if proc.poll() is None:
                    proc.kill()
            if rc != 0 or not sum_path.is_file():
                raise RuntimeError("track generation failed:\n" + "\n".join(tail))
            return True

        def failed(exc):
            if not sum_path.is_file():
                csv_path.unlink(missing_ok=True)
            if exc:
                messagebox.showerror("Track generation failed", str(exc)[-2500:], parent=self)

        self.app.run_bg(f"Generating tracks for {entry.name}", work, done=lambda _: self._decode(entry, path),
                        error=failed)

    def _decode(self, entry, path):
        width = int(self.app.settings.get("display_width") or 1600)
        self.app.run_bg(f"Loading {entry.name}",
                        lambda p, c: FrameStore().decode(path, HZ, width, progress=p, cancel=c),
                        done=lambda fr: self._start_session(entry, fr))

    def _start_session(self, entry, frames):
        try:
            sess = LabelSession(self.ws, entry, reviewer=self.reviewer_var.get().strip(), hz=HZ)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Cannot open labels", str(exc), parent=self)
            return
        self.session, self.frames = sess, frames
        self.i, self.mark, self.track, self.pid, self.event = 0, None, None, None, ""
        self.app.settings["last_video"] = entry.name
        self.app.settings.save()
        if sess.L.persons:
            self.pid = sess.L.persons[0].person_id
        self.session_var.set(entry.session_id)
        self.scale.configure(to=max(1, len(frames) - 1))
        self.refresh_video_list()
        self.refresh_all()
        self.canvas.focus_set()
        msg = f"opened {entry.name}: {len(frames)} frames at {HZ:g} Hz, {len(sess.L.persons)} persons"
        if sess.loaded_from and sess.loaded_from != sess.label_path:
            msg += f" (labels read from {sess.loaded_from.name}, saving as {sess.label_path.name})"
        self.app.status(msg)
        if sess.tracks is None:
            messagebox.showwarning("No tracks", "No reference tracks: you can review labels but not create persons.",
                                   parent=self)
        elif not sess.tracks_match:
            messagebox.showwarning(
                "Tracks changed",
                "The tracks file changed since these labels were made, so track numbers may differ.\n\n"
                "Person boxes are kept and drawn correctly. Before linking or unlinking tracks press "
                "'Re-link tracks' (Persons panel) to re-attach persons to the new track numbers.", parent=self)

    # ------------------------------------------------------------ fields
    def commit_reviewer(self):
        v = self.reviewer_var.get().strip()
        self.app.settings["reviewer"] = v
        self.app.settings.save()
        if self.session and v:
            m = self.session.set_reviewer(v)
            if m:
                self.after_change(m)

    def commit_session(self):
        if not self.session:
            return
        v = self.session_var.get().strip()
        reg = load_registry(self.ws.registry_path)
        e = reg.get(self.session.entry.name)
        if e is None or e.session_id == v:
            return
        e.session_id = v
        save_registry(self.ws.registry_path, reg)
        self.session.entry.session_id = v
        self.after_change(f"session id of {e.name} = {v or '-'}")

    # ------------------------------------------------------------ drawing
    def t(self) -> float:
        return float(self.frames.times[self.i]) if self.frames else 0.0

    def refresh_all(self):
        self.refresh_persons()
        self.refresh_segments()
        self.refresh_issues()
        self.refresh_events()
        self.draw_timeline()
        self.show_frame()

    def after_change(self, msg):
        self.app.status(msg)
        self.refresh_all()

    def show_frame(self):
        c = self.canvas
        c.delete("all")
        cw, ch = max(c.winfo_width(), 50), max(c.winfo_height(), 50)
        if not self.frames:
            c.create_text(cw / 2, ch / 2, fill="#9e9e9e", font=("Segoe UI", 13), justify="center",
                          text="Pick a video above and press Open.\n\nFirst time? The Manual tab explains everything.")
            return
        fr = self.frames
        img = jpeg_to_pil(fr.jpegs[self.i])
        s = min(cw / fr.frame_w, ch / fr.frame_h)
        dw, dh = max(1, int(fr.frame_w * s)), max(1, int(fr.frame_h * s))
        if s < 0.5:
            img.draft("RGB", (dw, dh))
        img = img.convert("RGB").resize((dw, dh), Image.BILINEAR)
        self.photo = ImageTk.PhotoImage(img)
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        c.create_image(ox, oy, anchor="nw", image=self.photo)
        k = s * fr.frame_w / max(fr.src_w, 1)
        self.fit = (k, ox, oy)
        t = self.t()
        persons, free = self.session.visible_at(t)
        for tid, box in free:
            x1, y1, x2, y2 = box * k + [ox, oy, ox, oy]
            sel = tid == self.track
            c.create_rectangle(x1, y1, x2, y2, outline="#ffff00" if sel else "#9e9e9e", width=3 if sel else 1,
                               dash=() if sel else (4, 3))
            self._label(x1, y1, f"#{tid}", "#ffff00" if sel else "#bdbdbd")
        for p, box in persons:
            x1, y1, x2, y2 = box * k + [ox, oy, ox, oy]
            col = person_color(p.person_id)
            sel = p.person_id == self.pid
            c.create_rectangle(x1, y1, x2, y2, outline=col, width=3 if sel else 2)
            ph, vi = value_at(p.phase, t), value_at(p.visibility, t)
            lab = f"{p.person_id}{' ' + p.actor_id if p.actor_id else ''}{' STAFF' if p.is_staff else ''}: " \
                  f"{ph.value if ph else '?'}{'/' + vi.value if vi else '/?'}{' ' + ph.event_id if ph and ph.event_id else ''}"
            self._label(x1, y1, lab, col)
        mark = "-" if self.mark is None else f"{self.mark:.2f}s"
        info = f"t={t:7.2f}s  person={self.pid or '-'}  track=#{self.track if self.track is not None else '-'}  " \
               f"event={self.event or '-'}  range start={mark}"
        tid_ = c.create_text(8, 8, anchor="nw", text=info, fill="#ffffff", font=("Consolas", 10))
        bb = c.bbox(tid_)
        c.tag_lower(c.create_rectangle(bb[0] - 4, bb[1] - 2, bb[2] + 4, bb[3] + 2, fill="#000000", outline=""), tid_)
        self.time_lbl.configure(text=f"t = {t:7.2f} s   frame {self.i + 1}/{len(fr)}")
        self._scale_guard = True
        self.scale.set(self.i)
        self._scale_guard = False
        self._update_range_label()
        self.move_playhead()

    def _label(self, x, y, text, color):
        c = self.canvas
        tid = c.create_text(x + 3, max(y - 2, 2), anchor="sw" if y > 18 else "nw", text=text, fill=color,
                            font=("Segoe UI", 9, "bold"))
        bb = c.bbox(tid)
        c.tag_lower(c.create_rectangle(bb[0] - 2, bb[1], bb[2] + 2, bb[3], fill="#000000", outline=""), tid)

    # timeline geometry
    TL_LEFT, TL_RIGHT = 70, 10
    ROW_SPAN, ROW_PHASE, ROW_VIS, ROW_MISSED, ROW_MARK, ROW_ALL = 16, 26, 46, 61, 72, 84

    def _tx(self, s: float) -> float:
        t0 = float(self.frames.times[0])
        t1 = float(self.frames.times[-1]) + self.session.step
        w = max(self.tl.winfo_width() - self.TL_LEFT - self.TL_RIGHT, 10)
        return self.TL_LEFT + (s - t0) / max(t1 - t0, 1e-6) * w

    def _xt(self, x: float) -> float:
        t0 = float(self.frames.times[0])
        t1 = float(self.frames.times[-1]) + self.session.step
        w = max(self.tl.winfo_width() - self.TL_LEFT - self.TL_RIGHT, 10)
        return t0 + (x - self.TL_LEFT) / w * (t1 - t0)

    def draw_timeline(self):
        tl = self.tl
        tl.delete("all")
        if not self.frames or not self.session:
            return
        L = self.session.L
        n = len(L.persons)
        rh = self._row_h = min(10, max(4, 100 // max(n, 1)))
        height = self.ROW_ALL + rh * max(n, 1) + 6
        if int(tl.cget("height")) != height:
            tl.configure(height=height)
        W = tl.winfo_width()
        t0, t1 = float(self.frames.times[0]), float(self.frames.times[-1]) + self.session.step
        span = t1 - t0
        stepv = next((s for s in (0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300) if span / s <= max(W / 70, 1)), 600) \
            if span > 0 else 1
        s = (t0 // stepv) * stepv
        while s <= t1:
            x = self._tx(s)
            if x >= self.TL_LEFT:
                tl.create_line(x, 12, x, height, fill="#2c2c2c")
                tl.create_text(x, 1, anchor="n", text=f"{s:g}s", fill="#8a8a8a", font=("Segoe UI", 7))
            s += stepv
        for y, txt in ((self.ROW_SPAN + 3, "span"), (self.ROW_PHASE + 8, "phase"), (self.ROW_VIS + 5, "visibility"),
                       (self.ROW_MISSED + 3, "missed"), (self.ROW_MARK + 3, "range")):
            tl.create_text(4, y, anchor="w", text=txt, fill="#9e9e9e", font=("Segoe UI", 7))
        p = self.session.person(self.pid)
        if p is not None:
            if p.span:
                tl.create_rectangle(self._tx(p.span[0]), self.ROW_SPAN, self._tx(p.span[1] + self.session.step),
                                    self.ROW_SPAN + 6, fill=person_color(p.person_id), outline="")
            for sg in p.phase:
                x1, x2 = self._tx(sg.start_s), max(self._tx(sg.end_s), self._tx(sg.start_s) + 1)
                tl.create_rectangle(x1, self.ROW_PHASE, x2, self.ROW_PHASE + 16,
                                    fill=PHASE_COLORS.get(sg.value, "#555"), outline="#1c1c1c")
                if sg.event_id and x2 - x1 > 14:
                    tl.create_text(x1 + 2, self.ROW_PHASE + 8, anchor="w", text=sg.event_id, fill="#ffffff",
                                   font=("Segoe UI", 7, "bold"))
            for sg in p.visibility:
                tl.create_rectangle(self._tx(sg.start_s), self.ROW_VIS,
                                    max(self._tx(sg.end_s), self._tx(sg.start_s) + 1), self.ROW_VIS + 10,
                                    fill=VIS_COLORS.get(sg.value, "#555"), outline="#1c1c1c")
        for m in L.missed_events:
            tl.create_rectangle(self._tx(m.start_s), self.ROW_MISSED, self._tx(m.end_s), self.ROW_MISSED + 5,
                                fill="#ff1744", outline="")
        for k, q in enumerate(L.persons):
            y = self.ROW_ALL + rh * k
            sel = q.person_id == self.pid
            if rh >= 8 or sel:
                tl.create_text(4, y + rh / 2, anchor="w", text=q.person_id + (" *" if sel else ""),
                               fill=person_color(q.person_id), font=("Segoe UI", 7, "bold" if sel else "normal"))
            if q.span:
                tl.create_rectangle(self._tx(q.span[0]), y + 1, self._tx(q.span[1] + self.session.step), y + rh - 1,
                                    fill="#3a3a3a", outline="#ffffff" if sel else "")
            for sg in q.phase:
                tl.create_rectangle(self._tx(sg.start_s), y + 2, max(self._tx(sg.end_s), self._tx(sg.start_s) + 1),
                                    y + rh - 2, fill=PHASE_COLORS.get(sg.value, "#555"), outline="")
        self.move_playhead()

    def move_playhead(self):
        tl = self.tl
        tl.delete("mark")
        tl.delete("playhead")
        if not self.frames or not self.session:
            return
        h = int(tl.cget("height"))
        if self.mark is not None:
            a, b = self.session.range(self.t(), self.mark)
            tl.create_rectangle(self._tx(a), self.ROW_MARK, self._tx(b), self.ROW_MARK + 6, fill="#ffeb3b",
                                outline="", tags="mark")
        x = self._tx(self.t())
        tl.create_line(x, 10, x, h, fill="#ffffff", width=1, tags="playhead")
        tl.create_polygon(x - 4, 10, x + 4, 10, x, 16, fill="#ffffff", tags="playhead")

    def _update_range_label(self):
        if not self.session:
            return
        a, b = self.session.range(self.t(), self.mark)
        self.range_lbl.configure(text=f"range {a:.2f}-{b:.2f}s" + ("" if self.mark is not None else " (frame)"))

    # ------------------------------------------------------------ panels
    def refresh_persons(self):
        tree = self.ptree
        tree.delete(*tree.get_children())
        if not self.session:
            return
        for p in self.session.L.persons:
            ph, vi = self.session.coverage(p)
            tree.insert("", "end", iid=p.person_id, text=p.person_id,
                        values=(p.actor_id or "-", "yes" if p.is_staff else "", ",".join(map(str, p.track_ids)) or "-",
                                f"{100 * ph:.0f}", f"{100 * vi:.0f}"), tags=(p.person_id,))
            tree.tag_configure(p.person_id, foreground="#000000")
        if self.pid and tree.exists(self.pid):
            tree.selection_set(self.pid)
            tree.see(self.pid)
        p = self.session.person(self.pid)
        self.actor_cb["values"] = self.session.known_actor_ids()
        self.actor_var.set(p.actor_id if p else "")
        self.staff_var.set(bool(p.is_staff) if p else False)
        self.note_var.set(p.note if p else "")
        tr = self.track
        owner = self.session.L.person_of_track(tr) if tr is not None else None
        self.track_lbl.configure(text=f"selected track: #{tr}  ({'linked to ' + owner.person_id if owner else 'unlinked'})"
                                 if tr is not None else "selected track: -   (click a box in the video)")
        mismatch = self.session.tracks is not None and not self.session.tracks_match
        self.tracks_warn.configure(text="Tracks changed since labelling: press 'Re-link tracks' before linking."
                                   if mismatch else ("No reference tracks for this video."
                                                     if self.session.tracks is None else ""))

    def refresh_segments(self):
        st = self.stree
        st.delete(*st.get_children())
        self.mtree.delete(*self.mtree.get_children())
        if not self.session:
            return
        p = self.session.person(self.pid)
        if p is not None:
            rows = [("phase", k, s) for k, s in enumerate(p.phase)] + [("visibility", k, s) for k, s in enumerate(p.visibility)]
            for kind, k, s in sorted(rows, key=lambda r: (r[2].start_s, r[0])):
                st.insert("", "end", iid=f"{kind}:{k}", text=kind,
                          values=(s.value, f"{s.start_s:.2f}", f"{s.end_s:.2f}", s.event_id or ""))
        for k, m in enumerate(self.session.L.missed_events):
            self.mtree.insert("", "end", iid=str(k), text=m.event_id,
                              values=(f"{m.start_s:.2f}", f"{m.end_s:.2f}", m.note))

    def refresh_events(self):
        if not self.session:
            return
        ids = self.session.L.event_ids()
        if self.event and self.event not in ids:
            ids = ids + [self.event]
        self.event_cb["values"] = ids
        self.event_cb.set(self.event)

    def refresh_issues(self):
        lb = self.ilist
        lb.delete(0, "end")
        self.issues = []
        if not self.session:
            return
        r = self.session.report()
        self.report = r
        st = r.status
        self.badge.configure(text=f"{r.name}: {st}", bg=STATUS_BG[st], fg=STATUS_FG[st])
        self.ibadge.configure(text=st, bg=STATUS_BG[st], fg=STATUS_FG[st])
        if r.name in self.names:
            vals = list(self.video_cb["values"])
            k = self.names.index(r.name)
            if k < len(vals):
                vals[k] = f"{r.name}    [{st}]"
                self.video_cb["values"] = vals
                self.video_cb.current(k)
        self.isum.configure(text=f"{len(r.blocking)} blocking, {len(r.warnings)} warning(s). "
                                 f"{'Ready for export.' if r.ready else 'Select an item to see how to fix it.'}")
        for kind, items, col in (("BLOCK", r.blocking, "#b71c1c"), ("WARN", r.warnings, "#e65100"),
                                 ("NOTE", r.notes, "#455a64")):
            for x in items:
                lb.insert("end", f"{kind}  {x}")
                lb.itemconfig("end", fg=col)
                self.issues.append(x)
        self.app.check_tab.dirty = True

    def _issue_picked(self, _=None):
        sel = self.ilist.curselection()
        txt = self.issues[sel[0]] if sel else ""
        help_ = issue_help(txt) if txt else None
        t = self.ihelp
        t.configure(state="normal")
        t.delete("1.0", "end")
        if txt:
            t.insert("end", txt + "\n\n")
            if help_:
                t.insert("end", "What it means: ", "b")
                t.insert("end", help_[0] + "\n")
                t.insert("end", "How to fix: ", "b")
                t.insert("end", help_[1] + "\n")
            t.insert("end", "\nDouble-click the item to jump to the person / time.")
        t.configure(state="disabled")

    def _issue_jump(self):
        sel = self.ilist.curselection()
        if not sel or not self.session:
            return
        txt = self.issues[sel[0]]
        m = re.match(r"(p\d+):", txt)
        if m and self.session.person(m.group(1)):
            self.select_person(m.group(1))
        tm = re.search(r"(\d+(?:\.\d+)?)(?:-\d+(?:\.\d+)?)?s\b", txt)
        if tm:
            self.seek_time(float(tm.group(1)))
        elif m:
            self._jump_person_start()

    # ------------------------------------------------------------ navigation
    def seek(self, i):
        if not self.frames:
            return
        self.i = int(min(max(i, 0), len(self.frames) - 1))
        self.show_frame()

    def seek_time(self, t):
        if self.frames:
            self.seek(self.frames.index_of(t))

    def step(self, n):
        self.seek(self.i + n)

    def _on_scale(self, v):
        if self._scale_guard or not self.frames:
            return
        i = int(round(float(v)))
        if i != self.i:
            self.i = i
            self.show_frame()

    def toggle_play(self):
        if not self.frames:
            return
        self.playing = not self.playing
        self.play_btn.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            if self.i >= len(self.frames) - 1:
                self.i = 0
            self._tick()

    def pause(self):
        self.playing = False
        try:
            self.play_btn.configure(text="Play")
        except tk.TclError:
            pass

    def _tick(self):
        if not self.playing or not self.frames:
            return
        t0 = time.perf_counter()
        if self.i >= len(self.frames) - 1:
            self.pause()
            return
        self.i += 1
        self.show_frame()
        spent = int((time.perf_counter() - t0) * 1000)
        self.after(max(1, int(1000 / HZ) - spent), self._tick)

    def on_timeline(self, e):
        if not self.frames or not self.session:
            return
        if e.y >= self.ROW_ALL:
            k = int((e.y - self.ROW_ALL) // getattr(self, "_row_h", 10))
            if 0 <= k < len(self.session.L.persons):
                self.select_person(self.session.L.persons[k].person_id)
        self.seek_time(self._xt(e.x))

    def on_timeline_drag(self, e):
        if self.frames and self.session:
            self.seek_time(self._xt(e.x))

    def on_click(self, e):
        self.canvas.focus_set()
        if not self.session or not self.frames:
            return
        k, ox, oy = self.fit
        x, y = (e.x - ox) / k, (e.y - oy) / k
        t = self.t()
        persons, _ = self.session.visible_at(t)
        tracks = self.session.tracks.at(t, self.session.step / 2) if self.session.tracks else []
        hits = []
        for tid, b in tracks:
            if b[0] <= x <= b[2] and b[1] <= y <= b[3]:
                hits.append(((b[2] - b[0]) * (b[3] - b[1]), "t", tid))
        for p, b in persons:
            if b[0] <= x <= b[2] and b[1] <= y <= b[3]:
                hits.append(((b[2] - b[0]) * (b[3] - b[1]) + 0.5, "p", p.person_id))
        if not hits:
            self.track = None
            self.app.status("nothing there - click inside a box")
            self.refresh_persons()
            self.show_frame()
            return
        _, kind, val = min(hits)
        if kind == "t":
            self.track = val
            owner = self.session.L.person_of_track(val) if self.session.tracks_match else None
            if owner:
                self.pid = owner.person_id
            self.app.status(f"track #{val} " + (f"({owner.person_id})" if owner else
                                                "(unlinked: N = new person, G = link to the selected person)"))
        else:
            self.pid = val
            self.app.status(f"selected {val}")
        self.refresh_persons()
        self.refresh_segments()
        self.draw_timeline()
        self.show_frame()

    def select_person(self, pid):
        self.pid = pid
        self.refresh_persons()
        self.refresh_segments()
        self.draw_timeline()
        self.show_frame()

    def _person_picked(self, _=None):
        sel = self.ptree.selection()
        if sel and sel[0] != self.pid:
            self.select_person(sel[0])

    def _jump_person_start(self):
        p = self.session.person(self.pid) if self.session else None
        if p and p.span:
            self.seek_time(p.span[0])

    # ------------------------------------------------------------ actions
    def _ready(self, person=False, track=False):
        if not self.session:
            self.app.status("open a video first")
            return False
        if person and not self.session.person(self.pid):
            self.app.status("select a person first (click their box or the Persons list)")
            self.bell()
            return False
        if track and self.track is None:
            self.app.status("click a track box in the video first")
            self.bell()
            return False
        return True

    def act(self, fn):
        try:
            msg = fn()
        except ValueError as exc:
            self.app.status(str(exc))
            self.bell()
            return None
        self.after_change(msg if isinstance(msg, str) else msg[1])
        return msg

    def set_mark(self):
        if self._ready():
            self.mark = self.t()
            self.app.status(f"range starts at {self.mark:.2f}s - move to the end and press a phase or visibility key")
            self.show_frame()

    def clear_mark(self):
        self.mark = None
        self.show_frame()

    def do_phase(self, key):
        if not self._ready(person=True):
            return
        phase, with_event = PHASE_KEYS[key]
        if with_event and not self.event:
            self.event = self.session.L.next_event_id()
        a, b = self.session.range(self.t(), self.mark)
        ev = self.event if with_event else ""
        if self.act(lambda: self.session.paint_phase(self.pid, a, b, phase, ev)):
            self.mark = None
            self.show_frame()

    def do_vis(self, key):
        if not self._ready(person=True):
            return
        a, b = self.session.range(self.t(), self.mark)
        if self.act(lambda: self.session.paint_visibility(self.pid, a, b, VIS_KEYS[key])):
            self.mark = None
            self.show_frame()

    def new_event(self):
        if self._ready():
            self.event = self.session.L.next_event_id()
            while self.event in self.event_cb["values"]:
                n = int(self.event[2:]) + 1
                self.event = f"ev{n}"
            self.refresh_events()
            self.app.status(f"current event {self.event} - keys 4 and 5 use it")
            self.show_frame()

    def cycle_event(self):
        if not self._ready():
            return
        ids = self.session.L.event_ids()
        if ids:
            self.event = ids[(ids.index(self.event) + 1) % len(ids)] if self.event in ids else ids[0]
            self.refresh_events()
            self.app.status(f"current event {self.event}")
            self.show_frame()

    def _event_picked(self, _=None):
        self.event = self.event_cb.get()
        self.canvas.focus_set()
        self.show_frame()

    def new_person(self):
        if not self._ready(track=True):
            return
        r = self.act(lambda: self.session.new_person(self.track))
        if r:
            self.pid = r[0]
            self.refresh_all()

    def link_track(self):
        if self._ready(person=True, track=True):
            self.act(lambda: self.session.link(self.pid, self.track))

    def unlink_track(self):
        if self._ready(track=True):
            self.act(lambda: self.session.unlink(self.track))

    def delete_person(self):
        if self._ready(person=True) and messagebox.askyesno(
                "Delete person", f"Delete {self.pid} and all their labels? (Undo with X)", parent=self):
            pid = self.pid
            self.pid = None
            self.act(lambda: self.session.delete_person(pid))

    def reset_person(self):
        if self._ready(person=True) and messagebox.askyesno(
                "Reset person", f"Remove all phase and visibility labels of {self.pid}? (Undo with X)", parent=self):
            self.act(lambda: self.session.reset_person(self.pid))

    def reset_all(self):
        if self._ready() and messagebox.askyesno(
                "Reset ALL labels", "Remove every person, label and missed event of this video? (Undo with X)",
                icon="warning", parent=self):
            self.pid = self.track = None
            self.event = ""
            self.act(self.session.reset_all)

    def relink(self):
        if self._ready():
            self.act(self.session.relink_by_overlap)

    def set_actor(self):
        if self._ready(person=True):
            self.act(lambda: self.session.set_actor(self.pid, self.actor_var.get()))
            self.canvas.focus_set()

    def focus_actor(self):
        if self._ready(person=True):
            self.side.select(self.persons_tab)
            self.actor_cb.focus_set()
            self.actor_cb.select_range(0, "end")
            self.app.status("type the actor id (A1, A2, ...) and press Enter")

    def set_staff(self, toggle=False):
        if not self._ready(person=True):
            self.staff_var.set(False)
            return
        p = self.session.person(self.pid)
        val = (not p.is_staff) if toggle else self.staff_var.get()
        self.act(lambda: self.session.set_staff(self.pid, val))

    def set_note(self):
        if self._ready(person=True):
            self.act(lambda: self.session.set_note(self.pid, self.note_var.get()))
            self.canvas.focus_set()

    def fill_gaps(self):
        if self._ready(person=True):
            self.act(lambda: self.session.fill_gaps(self.pid))

    def missed_event(self):
        if not self._ready():
            return
        a, b = self.session.range(self.t(), self.mark)
        if self.mark is None and not messagebox.askyesno(
                "Missed event", "No range is marked, so the missed event covers only this frame.\n"
                                "Continue? (Normally: S at the start, move to the end, then M.)", parent=self):
            return
        note = simpledialog.askstring("Missed event", f"Describe the concealment the detector missed "
                                                      f"({a:.2f}-{b:.2f}s):", parent=self)
        if note is None:
            return
        ev = self.event or self.session.L.next_event_id()
        if self.act(lambda: self.session.add_missed(a, b, ev, note)):
            self.mark = None
            self.show_frame()

    def undo(self):
        if self._ready():
            self.act(self.session.undo)

    def seg_selected(self):
        sel = self.stree.selection()
        if not sel or not self.session:
            return None
        kind, k = sel[0].split(":")
        p = self.session.person(self.pid)
        segs = p.phase if kind == "phase" else p.visibility
        return kind, int(k), segs[int(k)]

    def seg_goto(self):
        s = self.seg_selected()
        if s:
            self.seek_time(s[2].start_s)

    def seg_range(self):
        s = self.seg_selected()
        if s:
            self.mark = s[2].start_s
            self.seek_time(s[2].end_s - self.session.step)
            self.app.status("segment selected as range - press a key to relabel it")

    def seg_delete(self):
        s = self.seg_selected()
        if s:
            self.act(lambda: self.session.delete_segment(self.pid, s[0], s[1]))

    def _missed_goto(self):
        sel = self.mtree.selection()
        if sel:
            self.seek_time(self.session.L.missed_events[int(sel[0])].start_s)

    def missed_delete(self):
        sel = self.mtree.selection()
        if sel and self.session:
            self.act(lambda: self.session.delete_missed(int(sel[0])))

    def mark_complete(self):
        if not self._ready():
            return
        self.commit_reviewer()
        errs, warns = completion_check(self.session.L)
        if errs:
            self.side.select(self.issues_tab)
            messagebox.showerror("Not complete yet",
                                 f"{len(errs)} problem(s) block completion:\n\n- " + "\n- ".join(errs[:12]) +
                                 ("\n..." if len(errs) > 12 else "") +
                                 "\n\nThe Issues panel shows how to fix each one.", parent=self)
            return
        if ChecklistDialog(self, warns).result:
            ok, errs, _ = self.session.mark_complete()
            self.after_change("video marked COMPLETE" if ok else "could not mark complete")
            r = self.report
            if ok and not r.ready:
                self.side.select(self.issues_tab)
                messagebox.showinfo("Complete, but not READY yet",
                                    "Labels are complete. Still needed before export:\n\n- " + "\n- ".join(r.blocking),
                                    parent=self)
            elif ok:
                self.app.status(f"{r.name} is READY for export")

    def unmark_complete(self):
        if self._ready() and self.session.L.complete:
            self.act(self.session.unmark_complete)

    def export_review(self):
        if not self._ready():
            return
        out = self.ws.labels_dir / f"{self.session.entry.name}_labeled.mp4"
        sess, frames = self.session, self.frames
        self.app.run_bg("Writing review video", lambda p, c: sess.export_review_video(frames, out, progress=p),
                        done=lambda path: self.app.status(f"review video written: {path}"))

    # ------------------------------------------------------------ keys
    def on_key(self, e):
        ks = e.keysym
        ch = e.char.lower() if e.char and e.char.isprintable() else (ks.lower() if len(ks) == 1 else "")
        ctrl = bool(isinstance(e.state, int) and e.state & 0x4)
        if ctrl:
            if ks.lower() == "z":
                self.undo()
            return
        if ks == "space":
            self.toggle_play()
        elif ks == "Right" or ch == "d":
            self.step(1)
        elif ks == "Left" or ch == "a":
            self.step(-1)
        elif ch == "l":
            self.step(int(2 * HZ))
        elif ch == "j":
            self.step(-int(2 * HZ))
        elif ch == "k":
            self.step(int(10 * HZ))
        elif ch == "h":
            self.step(-int(10 * HZ))
        elif ks == "Home":
            self.seek(0)
        elif ks == "End":
            self.seek(10 ** 9)
        elif ks == "Escape":
            self.clear_mark()
        elif ks == "BackSpace" or ch == "x":
            self.undo()
        elif ch == "s":
            self.set_mark()
        elif ch in PHASE_KEYS:
            self.do_phase(ch)
        elif ch in VIS_KEYS:
            self.do_vis(ch)
        elif ch == "n":
            self.new_person()
        elif ch == "g":
            self.link_track()
        elif ch == "u":
            self.unlink_track()
        elif ch == "i":
            self.focus_actor()
        elif ch == "t":
            self.set_staff(toggle=True)
        elif ch == "e":
            self.new_event()
        elif ch == "w":
            self.cycle_event()
        elif ch == "f":
            self.fill_gaps()
        elif ch == "m":
            self.missed_event()
        elif ch == "c":
            self.mark_complete()
        elif ch == "z":
            self.reset_person()
        elif ch == "r":
            self.reset_all()
        else:
            return
        return "break"


class ChecklistDialog(tk.Toplevel):
    """Review checklist (guide section 8) that must be confirmed before marking complete."""

    def __init__(self, master, warnings):
        super().__init__(master)
        self.result = False
        self.title("Mark complete - review checklist")
        self.transient(master)
        f = ttk.Frame(self, padding=14)
        f.pack(fill="both", expand=True)
        ttk.Label(f, text="All checks pass. Confirm each point before marking the video complete:",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.vars = []
        for item in REVIEW_CHECKLIST:
            v = tk.BooleanVar()
            ttk.Checkbutton(f, text=item, variable=v, command=self._update).pack(anchor="w", pady=2)
            self.vars.append(v)
        if warnings:
            ttk.Label(f, text=f"\n{len(warnings)} warning(s) - not blocking, but please look:",
                      foreground="#e65100").pack(anchor="w")
            t = tk.Text(f, height=min(8, len(warnings) + 1), width=100, wrap="word", font=("Segoe UI", 9))
            t.insert("end", "\n".join(f"- {w}" for w in warnings))
            t.configure(state="disabled")
            t.pack(fill="x")
        b = ttk.Frame(f)
        b.pack(fill="x", pady=(10, 0))
        self.ok = ttk.Button(b, text="Mark complete", command=self._ok, state="disabled")
        self.ok.pack(side="right")
        ttk.Button(b, text="Cancel", command=self.destroy).pack(side="right", padx=6)
        self.grab_set()
        self.wait_window()

    def _update(self):
        self.ok.configure(state="normal" if all(v.get() for v in self.vars) else "disabled")

    def _ok(self):
        self.result = True
        self.destroy()


# ====================================================================== Check tab
class CheckTab(ttk.Frame):
    def __init__(self, master, app: Studio):
        super().__init__(master, padding=6)
        self.app, self.ws = app, app.ws
        self.rep = None
        self.dirty = True
        top = ttk.Frame(self)
        top.pack(fill="x")
        btn(top, "Refresh", lambda: self.refresh(force=True)).pack(side="left")
        btn(top, "Actor gallery...", self.gallery).pack(side="left", padx=4)
        btn(top, "Delete stray files", self.clean).pack(side="left")
        self.summary = tk.Label(top, text="", font=("Segoe UI", 11, "bold"), padx=10)
        self.summary.pack(side="left", padx=10)

        pw = ttk.PanedWindow(self, orient="vertical")
        pw.pack(fill="both", expand=True, pady=4)
        upper = ttk.Frame(pw)
        lower = ttk.Frame(pw)
        pw.add(upper, weight=3)
        pw.add(lower, weight=2)
        cols = ("status", "complete", "persons", "events", "blocking", "warnings", "reviewer", "session", "updated")
        self.tree = ttk.Treeview(upper, columns=cols, selectmode="browse")
        self.tree.heading("#0", text="video")
        self.tree.column("#0", width=210)
        for c, w in zip(cols, (110, 70, 64, 64, 70, 70, 120, 80, 170)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="center")
        for st in STATUS_BG:
            self.tree.tag_configure(st, background=STATUS_BG[st], foreground=STATUS_FG[st])
        sb = ttk.Scrollbar(upper, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._picked)
        self.tree.bind("<Double-1>", lambda e: self.open_in_label())

        lp = ttk.PanedWindow(lower, orient="horizontal")
        lp.pack(fill="both", expand=True)
        det = ttk.LabelFrame(lp, text="Selected video", padding=4)
        side = ttk.Frame(lp)
        lp.add(det, weight=3)
        lp.add(side, weight=2)
        self.detail = tk.Text(det, wrap="word", font=("Segoe UI", 9), relief="flat", bg="#fafafa", height=10)
        self.detail.pack(fill="both", expand=True)
        for tag, col in (("BLOCK", "#b71c1c"), ("WARN", "#e65100"), ("NOTE", "#455a64"), ("FIX", "#1b5e20")):
            self.detail.tag_configure(tag, foreground=col)
        self.detail.tag_configure("h", font=("Segoe UI", 10, "bold"))

        vs = ttk.LabelFrame(side, text="Video settings (labels/videos.json)", padding=6)
        vs.pack(fill="x")
        self.v_name = ttk.Label(vs, text="-", font=("Segoe UI", 10, "bold"))
        self.v_name.grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(vs, text="Session id").grid(row=1, column=0, sticky="w")
        self.v_session = tk.StringVar()
        ttk.Entry(vs, textvariable=self.v_session, width=12).grid(row=1, column=1, sticky="w", padx=4)
        self.v_staged = tk.BooleanVar()
        ttk.Checkbutton(vs, text="staged (acted)", variable=self.v_staged).grid(row=1, column=2, sticky="w")
        ttk.Label(vs, text="Notes").grid(row=2, column=0, sticky="w")
        self.v_notes = tk.StringVar()
        ttk.Entry(vs, textvariable=self.v_notes, width=36).grid(row=2, column=1, columnspan=2, sticky="we", padx=4)
        self.v_hold = ttk.Label(vs, text="", style="Small.TLabel")
        self.v_hold.grid(row=3, column=0, columnspan=3, sticky="w")
        b = ttk.Frame(vs)
        b.grid(row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))
        btn(b, "Save settings", self.save_settings).pack(side="left")
        btn(b, "Open in Label tab", self.open_in_label).pack(side="left", padx=4)
        btn(b, "Register video file...", self.register).pack(side="left")
        ds = ttk.LabelFrame(side, text="Whole dataset", padding=4)
        ds.pack(fill="both", expand=True, pady=(6, 0))
        self.ds = tk.Text(ds, wrap="word", font=("Segoe UI", 9), relief="flat", bg="#fafafa", height=8)
        self.ds.pack(fill="both", expand=True)
        self.ds.tag_configure("WARN", foreground="#e65100")
        self.ds.tag_configure("NOTE", foreground="#455a64")

    def refresh(self, force=False):
        if not force and not self.dirty and self.rep is not None:
            return
        try:
            self.rep = check_dataset(self.ws)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Check", f"cannot read labels: {exc}", parent=self)
            return
        self.dirty = False
        sel = self.tree.selection()
        self.tree.delete(*self.tree.get_children())
        for r in self.rep.videos:
            e = r.entry
            self.tree.insert("", "end", iid=r.key, text=r.name, tags=(r.status,), values=(
                r.status, "yes" if r.complete else "", r.persons or "", r.events or "",
                len(r.blocking) if r.status != NOT_STARTED else "", len(r.warnings) or "", r.reviewer,
                (e.session_id or ("holdout" if e.holdout else "-")) if e else "?", r.updated.replace("T", " ")[:19]))
        if sel and self.tree.exists(sel[0]):
            self.tree.selection_set(sel[0])
        started = [r for r in self.rep.videos if r.status != NOT_STARTED]
        n_ready, blocked = len(self.rep.ready), sum(1 for r in started if not r.ready)
        st = READY if started and not blocked else (NOT_READY if blocked else NOT_STARTED)
        self.summary.configure(text=f"{n_ready} of {len(started)} labelled videos READY" +
                               (f"  -  {blocked} need work" if blocked else ""),
                               fg=STATUS_FG[st], bg=STATUS_BG[st])
        t = self.ds
        t.configure(state="normal")
        t.delete("1.0", "end")
        for w in self.rep.warnings:
            t.insert("end", f"WARN  {w}\n", "WARN")
        for n in self.rep.notes:
            t.insert("end", f"NOTE  {n}\n", "NOTE")
        t.insert("end", f"\nIndependent split groups: {self.rep.groups}\nActors:\n")
        for a, who in sorted(self.rep.roster.items(), key=lambda kv: (len(kv[0]), kv[0])):
            t.insert("end", f"  {a}: " + ", ".join(f"{v}/{p}{' (staff)' if s else ''}" for v, p, s, _ in who) + "\n")
        t.configure(state="disabled")
        self._picked()

    def current(self):
        sel = self.tree.selection()
        return self.rep.get(sel[0]) if sel and self.rep else None

    def _picked(self, _=None):
        r = self.current()
        t = self.detail
        t.configure(state="normal")
        t.delete("1.0", "end")
        if r is None:
            t.insert("end", "Select a video above.")
            t.configure(state="disabled")
            self.v_name.configure(text="-")
            return
        t.insert("end", f"{r.name}  -  {r.status}\n", "h")
        if r.label_path:
            t.insert("end", f"file: {r.label_path.name}   persons: {r.persons}   events: {r.events}   "
                            f"reviewer: {r.reviewer or '-'}   updated: {r.updated or '-'}\n\n")
        if r.status == NOT_STARTED:
            t.insert("end", "Not labelled yet. Double-click to open it in the Label tab.\n")
        for kind, items in (("BLOCK", r.blocking if r.status != NOT_STARTED else []), ("WARN", r.warnings),
                            ("NOTE", r.notes)):
            for x in items:
                t.insert("end", f"{kind}  {x}\n", kind)
                h = issue_help(x)
                if h and kind != "NOTE":
                    t.insert("end", f"      fix: {h[1]}\n", "FIX")
        if r.ready:
            t.insert("end", "Nothing blocks: this video will be accepted by the training pipeline as it is.\n", "FIX")
        t.configure(state="disabled")
        e = r.entry
        self.v_name.configure(text=r.name + ("" if e else "  (not registered)"))
        self.v_session.set(e.session_id if e else "")
        self.v_staged.set(bool(e.staged) if e else True)
        self.v_notes.set(e.notes if e else "")
        self.v_hold.configure(text=("HOLDOUT video - final test only; holdout is set by the project lead"
                                    if e and e.holdout else ""))

    def save_settings(self):
        r = self.current()
        if not r or not r.entry:
            messagebox.showinfo("Video settings", "Register the video first.", parent=self)
            return
        reg = load_registry(self.ws.registry_path)
        e = reg[r.key]
        e.session_id, e.staged, e.notes = self.v_session.get().strip(), bool(self.v_staged.get()), self.v_notes.get().strip()
        save_registry(self.ws.registry_path, reg)
        lt = self.app.label_tab
        if lt.session and lt.session.entry.name == r.key:
            lt.session.entry.session_id = e.session_id
            lt.session_var.set(e.session_id)
            lt.refresh_issues()
        self.app.status(f"saved settings of {r.name}")
        self.refresh(force=True)

    def open_in_label(self):
        r = self.current()
        if not r:
            return
        if not r.entry:
            messagebox.showinfo("Open", "This video is not registered. Use 'Register video file...' first.", parent=self)
            return
        lt = self.app.label_tab
        lt.refresh_video_list()
        if r.key in lt.names:
            lt.video_cb.current(lt.names.index(r.key))
        self.app.goto(lt)
        if not (lt.session and lt.session.entry.name == r.key):
            lt.open_video(r.key)

    def register(self):
        r = self.current()
        expect = r.labels.sha256 if r and r.labels is not None and r.entry is None else None
        f = filedialog.askopenfilename(parent=self, title=f"Video file for '{r.name}'" if r else "Register video",
                                       filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
        if not f:
            return
        name = r.key if expect else None
        self.app.run_bg(f"Registering {Path(f).name}",
                        lambda p, c: register_file(self.ws, Path(f), name, expect_sha=expect),
                        done=lambda e: (self.app.status(f"registered '{e.name}'"), self.refresh(force=True)),
                        error=lambda exc: exc and messagebox.showerror("Register", str(exc), parent=self))

    def clean(self):
        from shoplift.label_readiness import scan_label_files
        _, _, junk = scan_label_files(self.ws.labels_dir)
        if not junk:
            messagebox.showinfo("Stray files", "No stray files.", parent=self)
            return
        if messagebox.askyesno("Delete stray files", "Delete these lock/temporary files?\n\n" +
                               "\n".join(j.name for j in junk), parent=self):
            for j in junk:
                try:
                    j.unlink()
                except OSError as exc:
                    messagebox.showerror("Delete", f"{j.name}: {exc}", parent=self)
            self.refresh(force=True)

    def gallery(self):
        if self.rep is None:
            self.refresh(force=True)
        reg = load_registry(self.ws.registry_path)
        import copy as _copy
        usable, files = {}, {}
        for e in reg.values():
            p = self.ws.video_file(e)
            if p is not None:
                e2 = _copy.copy(e)
                e2.paths = [str(p)]
                usable[e.name] = e2
        roster = {a: [w for w in who if w[0] in usable] for a, who in self.rep.roster.items()}
        roster = {a: w for a, w in roster.items() if w}
        if not roster:
            messagebox.showinfo("Actor gallery", "No actor ids on videos whose files are on this machine.", parent=self)
            return
        out = self.ws.root / "outputs" / "actor_gallery.jpg"

        def work(p, c):
            import label_check
            label_check.gallery(roster, usable, files, out)
            return out

        def done(path):
            self.app.status(f"gallery written: {path} (rows = actor id, columns = video)")
            try:
                os.startfile(str(path))  # noqa: S606 - Windows image viewer
            except (AttributeError, OSError):
                messagebox.showinfo("Actor gallery", f"Written to {path}", parent=self)

        self.app.run_bg("Building actor gallery", work, done=done, cancellable=False)


# ====================================================================== Export tab
class ExportTab(ttk.Frame):
    def __init__(self, master, app: Studio):
        super().__init__(master, padding=6)
        self.app, self.ws = app, app.ws
        self.include = set()
        self.rep = None
        ttk.Label(self, text="Export READY videos as one zip for the training pipeline. Only READY videos can be "
                             "ticked; the Check tab explains the others.", wraplength=1100).pack(anchor="w")
        top = ttk.Frame(self)
        top.pack(fill="x", pady=4)
        btn(top, "Select new/changed", self.select_changed).pack(side="left")
        btn(top, "Select all READY", self.select_all).pack(side="left", padx=4)
        btn(top, "Clear", self.clear).pack(side="left")
        self.with_tracks = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="include reference tracks (recommended)", variable=self.with_tracks).pack(
            side="left", padx=14)
        cols = ("include", "status", "change", "persons", "events", "reviewer", "session")
        self.tree = ttk.Treeview(self, columns=cols, selectmode="browse", height=14)
        self.tree.heading("#0", text="video")
        self.tree.column("#0", width=220)
        for c, w in zip(cols, (64, 110, 160, 64, 64, 130, 90)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="center")
        for st in STATUS_BG:
            self.tree.tag_configure(st, foreground=STATUS_FG[st])
        self.tree.tag_configure("on", background="#e3f2fd")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<ButtonRelease-1>", self._toggle)
        self.tree.bind("<space>", self._toggle)

        out = ttk.Frame(self)
        out.pack(fill="x", pady=6)
        ttk.Label(out, text="Save to").pack(side="left")
        self.out_var = tk.StringVar(value=app.settings.get("export_dir") or str(self.ws.exports_dir))
        ttk.Entry(out, textvariable=self.out_var, width=70).pack(side="left", padx=4)
        btn(out, "Browse", self.browse).pack(side="left")
        btn(out, "Open folder", self.open_folder).pack(side="left", padx=4)
        self.export_btn = btn(out, "Export zip", self.export)
        self.export_btn.pack(side="right")
        btn(out, "Verify a zip...", self.verify).pack(side="right", padx=4)
        self.sel_lbl = ttk.Label(out, text="")
        self.sel_lbl.pack(side="right", padx=10)

        lf = ttk.LabelFrame(self, text="Result / history", padding=4)
        lf.pack(fill="both", expand=True)
        self.log = tk.Text(lf, height=10, wrap="word", font=("Consolas", 9), relief="flat", bg="#fafafa")
        self.log.pack(fill="both", expand=True)
        self.log.tag_configure("ok", foreground="#1b5e20")
        self.log.tag_configure("bad", foreground="#b71c1c")

    def refresh(self):
        try:
            self.rep = check_dataset(self.ws)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Export", f"cannot read labels: {exc}", parent=self)
            return
        last = last_exported(self.ws)
        self.changes = {}
        self.tree.delete(*self.tree.get_children())
        ready_keys = {r.key for r in self.rep.ready}
        self.include &= ready_keys
        for r in self.rep.videos:
            if r.status == NOT_STARTED:
                continue
            if r.key not in last:
                ch = "new (never exported)"
            elif last[r.key] != r.content_sha:
                ch = "CHANGED since export"
            else:
                ch = "exported, unchanged"
            self.changes[r.key] = ch
            on = r.key in self.include
            self.tree.insert("", "end", iid=r.key, text=r.name, tags=(r.status,) + (("on",) if on else ()), values=(
                ("[x]" if on else "[ ]") if r.ready else "-", r.status, ch, r.persons, r.events, r.reviewer,
                (r.entry.session_id or ("holdout" if r.entry.holdout else "-")) if r.entry else "?"))
        self._count()
        self.show_history()

    def _count(self):
        self.sel_lbl.configure(text=f"{len(self.include)} selected")

    def _toggle(self, e=None):
        iid = self.tree.focus()
        r = self.rep.get(iid) if self.rep and iid else None
        if not r:
            return
        if not r.ready:
            self.app.status(f"{r.name} is {r.status}: " + "; ".join(r.blocking[:2]))
            return
        self.include ^= {iid}
        self._set_row(iid)

    def _set_row(self, iid):
        on = iid in self.include
        vals = list(self.tree.item(iid, "values"))
        vals[0] = "[x]" if on else "[ ]"
        r = self.rep.get(iid)
        self.tree.item(iid, values=vals, tags=(r.status,) + (("on",) if on else ()))
        self._count()

    def select_changed(self):
        self.include = {r.key for r in self.rep.ready if not self.changes.get(r.key, "").startswith("exported")}
        self.refresh()

    def select_all(self):
        self.include = {r.key for r in self.rep.ready}
        self.refresh()

    def clear(self):
        self.include = set()
        self.refresh()

    def browse(self):
        d = filedialog.askdirectory(parent=self, initialdir=self.out_var.get())
        if d:
            self.out_var.set(d)

    def open_folder(self):
        d = Path(self.out_var.get())
        d.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(d))  # noqa: S606
        except (AttributeError, OSError):
            pass

    def _write(self, lines, tag=None):
        self.log.insert("end", "\n".join(lines) + "\n", tag or ())
        self.log.see("end")

    def export(self):
        if not self.include:
            messagebox.showinfo("Export", "Tick at least one READY video (or use 'Select new/changed').", parent=self)
            return
        keys = sorted(self.include)
        out = Path(self.out_var.get())
        self.app.settings["export_dir"] = str(out)
        self.app.settings.save()
        reviewer = self.app.settings.get("reviewer", "")
        with_tracks = self.with_tracks.get()

        def done(res):
            path, man = res
            self.log.delete("1.0", "end")
            self._write([f"Exported {len(man['videos'])} video(s) to", f"  {path}", ""], "ok")
            self._write([f"  {k}: {v['persons']} persons, events {v['events'] or '-'}, session "
                         f"{v['session_id'] or 'holdout'}" for k, v in man["videos"].items()])
            if man["dataset_warnings"]:
                self._write(["", "Dataset warnings travel with the package (the training team will see them):"] +
                            [f"  - {w}" for w in man["dataset_warnings"]], "bad")
            self._write(["", "Send this one zip file to the training team. On their side:",
                         f"  python -m shoplift.label_exchange import \"{path.name}\""])
            self.include = set()
            self.refresh_keep_log()
            self.app.status(f"export written: {path.name}")

        self.app.run_bg("Exporting labels",
                        lambda p, c: export_labels(self.ws, keys, out, with_tracks, reviewer), done=done,
                        error=lambda exc: exc and messagebox.showerror("Export failed", str(exc), parent=self),
                        cancellable=False)

    def refresh_keep_log(self):
        txt = self.log.get("1.0", "end")
        self.refresh()
        self.log.delete("1.0", "end")
        self.log.insert("end", txt)

    def verify(self):
        f = filedialog.askopenfilename(parent=self, initialdir=self.out_var.get(),
                                       filetypes=[("Label export", "*.zip"), ("All files", "*.*")])
        if not f:
            return
        ok, lines = verify_package(Path(f))
        self.log.delete("1.0", "end")
        self._write([f"Verify {Path(f).name}:"] + lines, "ok" if ok else "bad")

    def show_history(self):
        self.log.delete("1.0", "end")
        hist = read_log(self.ws)
        if not hist:
            self._write(["No exports yet."])
            return
        self._write(["Previous exports (newest first):"])
        for h in reversed(hist[-15:]):
            self._write([f"  {h['created'].replace('T', ' ')[:19]}  {h['file']}  ({len(h['videos'])} videos: "
                         f"{', '.join(sorted(h['videos']))[:120]})"])


# ====================================================================== Manual tab
class ManualTab(ttk.Frame):
    TOOLS = ["* Phase decision helper", "* Visibility decision helper", "* Error & warning fixer", "* Practice quiz",
             "> Tutorial recording (older tool, same keys)"]

    def __init__(self, master, app: Studio):
        super().__init__(master, padding=6)
        self.app = app
        pw = ttk.PanedWindow(self, orient="horizontal")
        pw.pack(fill="both", expand=True)
        left = ttk.Frame(pw, width=290)
        self.body = ttk.Frame(pw)
        pw.add(left, weight=0)
        pw.add(self.body, weight=1)
        ttk.Label(left, text="Manual", style="H.TLabel").pack(anchor="w", pady=(0, 4))
        self.toc = tk.Listbox(left, activestyle="none", font=("Segoe UI", 10), width=36, exportselection=False)
        self.toc.pack(fill="both", expand=True)
        for title, _ in manual.CHAPTERS:
            self.toc.insert("end", title)
        self.toc.insert("end", "")
        for t in self.TOOLS:
            self.toc.insert("end", t)
            self.toc.itemconfig("end", fg="#0d47a1")
        self.toc.bind("<<ListboxSelect>>", self._picked)
        nav = ttk.Frame(left)
        nav.pack(fill="x", pady=4)
        btn(nav, "< Previous", lambda: self._move(-1)).pack(side="left")
        btn(nav, "Next >", lambda: self._move(1)).pack(side="right")
        self.toc.selection_set(0)
        self.show_chapter(0)

    def _move(self, d):
        sel = self.toc.curselection()
        i = (sel[0] if sel else 0) + d
        n = len(manual.CHAPTERS)
        if 0 <= i < n:
            self.toc.selection_clear(0, "end")
            self.toc.selection_set(i)
            self.toc.see(i)
            self.show_chapter(i)

    def _picked(self, _=None):
        sel = self.toc.curselection()
        if not sel:
            return
        i = sel[0]
        n = len(manual.CHAPTERS)
        if i < n:
            self.show_chapter(i)
        elif i > n:
            actions = [lambda: self.show_tree_tool(manual.PHASE_TREE, "Which phase is it?"),
                       lambda: self.show_tree_tool(manual.VIS_TREE, "Which visibility is it?"),
                       self.show_fixer, self.show_quiz, self.open_tutorial]
            actions[i - n - 1]()

    def _clear(self):
        for w in self.body.winfo_children():
            w.destroy()

    # ---------------------------------------------------------- chapters
    def _text(self):
        self._clear()
        fr = ttk.Frame(self.body)
        fr.pack(fill="both", expand=True)
        t = tk.Text(fr, wrap="word", font=("Segoe UI", 10), relief="flat", padx=18, pady=12, bg="#ffffff",
                    spacing1=2, spacing3=4, cursor="arrow")
        sb = ttk.Scrollbar(fr, orient="vertical", command=t.yview)
        t.configure(yscrollcommand=sb.set)
        t.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        t.tag_configure("h1", font=("Segoe UI", 17, "bold"), spacing3=10, foreground="#0d47a1")
        t.tag_configure("h2", font=("Segoe UI", 12, "bold"), spacing1=12, spacing3=4)
        t.tag_configure("b", font=("Segoe UI", 10, "bold"))
        t.tag_configure("code", font=("Consolas", 10), background="#eceff1")
        t.tag_configure("tip", background="#e8f5e9", lmargin1=10, lmargin2=10, rmargin=10, spacing1=6, spacing3=6)
        t.tag_configure("warn", background="#fff3e0", lmargin1=10, lmargin2=10, rmargin=10, spacing1=6, spacing3=6)
        t.tag_configure("li", lmargin1=16, lmargin2=34)
        t.tag_configure("codeblock", font=("Consolas", 10), background="#eceff1", lmargin1=16, lmargin2=16)
        return t

    def _inline(self, t, text, base=()):
        for part in re.split(r"(\*\*.+?\*\*|`.+?`)", text):
            if part.startswith("**") and part.endswith("**"):
                t.insert("end", part[2:-2], base + ("b",))
            elif part.startswith("`") and part.endswith("`"):
                t.insert("end", part[1:-1], base + ("code",))
            elif part:
                t.insert("end", part, base)

    def _grid(self, t, header, rows, swatch=None):
        f = tk.Frame(t, bg="#cfd8dc")
        widths = [18] + [60] * (len(header) - 1) if len(header) == 2 else [10, 14, 40, 24, 24]
        for c, h in enumerate(header):
            tk.Label(f, text=h, bg="#cfd8dc", font=("Segoe UI", 9, "bold"), anchor="w", padx=6, pady=3).grid(
                row=0, column=c, sticky="nsew", padx=1, pady=1)
        for r, row in enumerate(rows, start=1):
            for c, val in enumerate(row):
                bg = "#ffffff"
                fg = "#000000"
                if swatch and c == swatch[0]:
                    bg = swatch[1](row)
                    fg = contrast(bg)
                tk.Label(f, text=val, bg=bg, fg=fg, anchor="nw", justify="left", padx=6, pady=3,
                         wraplength=widths[c] * 7, font=("Segoe UI", 9)).grid(row=r, column=c, sticky="nsew",
                                                                              padx=1, pady=1)
        t.window_create("end", window=f)
        t.insert("end", "\n\n")

    def show_chapter(self, i):
        title, blocks = manual.CHAPTERS[i]
        t = self._text()
        for b in blocks:
            kind = b[0]
            if kind in ("h1", "h2"):
                t.insert("end", b[1] + "\n", kind)
            elif kind == "p":
                self._inline(t, b[1])
                t.insert("end", "\n")
            elif kind in ("tip", "warn"):
                self._inline(t, ("Tip: " if kind == "tip" else "Careful: ") + b[1], (kind,))
                t.insert("end", "\n", (kind,))
                t.insert("end", "\n")
            elif kind == "code":
                t.insert("end", b[1] + "\n", ("codeblock",))
            elif kind in ("bullets", "steps"):
                for k, item in enumerate(b[1], 1):
                    t.insert("end", ("  \u2022  " if kind == "bullets" else f"  {k}.  "), ("li",))
                    self._inline(t, item, ("li",))
                    t.insert("end", "\n", ("li",))
                t.insert("end", "\n")
            elif kind == "table":
                self._grid(t, b[1], b[2])
            elif kind == "phases":
                self._grid(t, ("Key", "Phase", "Definition", "Starts", "Ends"), manual.PHASE_ROWS,
                           swatch=(1, lambda row: PHASE_COLORS["pick" if row[1].startswith("pick") else row[1]]))
            elif kind == "visibility":
                self._grid(t, ("Key", "Value", "Use when"),
                           manual.VIS_ROWS, swatch=(1, lambda row: VIS_COLORS[row[1]]))
            elif kind == "keys":
                self._grid(t, ("Key", "Action"), b[1])
            elif kind == "check":
                self._checklist(t, b[1], b[2])
            elif kind == "button":
                action = {"goto_label": lambda: self.app.goto(self.app.label_tab),
                          "goto_decision": lambda: self.show_tree_tool(manual.PHASE_TREE, "Which phase is it?")}[b[2]]
                t.window_create("end", window=ttk.Button(t, text=b[1], command=action))
                t.insert("end", "\n\n")
        nav = ttk.Frame(t)
        n = len(manual.CHAPTERS)
        if i > 0:
            ttk.Button(nav, text=f"< {manual.CHAPTERS[i - 1][0]}", command=lambda: self._move(-1)).pack(side="left")
        if i < n - 1:
            ttk.Button(nav, text=f"{manual.CHAPTERS[i + 1][0]} >", command=lambda: self._move(1)).pack(side="left",
                                                                                                    padx=8)
        t.insert("end", "\n")
        t.window_create("end", window=nav)
        t.configure(state="disabled")

    def _checklist(self, t, cid, items):
        store = self.app.settings.setdefault("manual_checks", {})
        state = store.get(cid, [False] * len(items))
        state = (state + [False] * len(items))[:len(items)]
        f = tk.Frame(t, bg="#ffffff")
        vars_ = []
        prog = tk.Label(f, bg="#ffffff", font=("Segoe UI", 9, "bold"), fg="#0d47a1")

        def changed():
            store[cid] = [v.get() for v in vars_]
            self.app.settings.save()
            prog.configure(text=f"{sum(store[cid])} of {len(items)} done")

        for k, item in enumerate(items):
            v = tk.BooleanVar(value=state[k])
            vars_.append(v)
            tk.Checkbutton(f, text=f"{k + 1}. " + item.replace("**", ""), variable=v, command=changed, bg="#ffffff",
                           anchor="w", justify="left", wraplength=820, font=("Segoe UI", 10)).pack(anchor="w", fill="x")
        prog.pack(anchor="w", pady=(4, 0))
        tk.Button(f, text="Untick all", command=lambda: ([v.set(False) for v in vars_], changed()),
                  relief="groove").pack(anchor="w", pady=4)
        changed()
        t.window_create("end", window=f)
        t.insert("end", "\n\n")

    # ---------------------------------------------------------- tools
    def show_tree_tool(self, tree, title):
        self._clear()
        f = ttk.Frame(self.body, padding=20)
        f.pack(fill="both", expand=True)
        ttk.Label(f, text=title, font=("Segoe UI", 17, "bold"), foreground="#0d47a1").pack(anchor="w")
        ttk.Label(f, text="Answer for the moment you are looking at. You can go back at any time.",
                  style="Small.TLabel").pack(anchor="w", pady=(0, 12))
        box = ttk.Frame(f)
        box.pack(fill="both", expand=True)
        history = ["start"]

        def render():
            for w in box.winfo_children():
                w.destroy()
            node = tree[history[-1]]
            if node[0] == "=":
                _, key, name, expl = node
                base = name.split()[0]
                col = PHASE_COLORS.get(base) or VIS_COLORS.get(base, "#90a4ae")
                tk.Label(box, text=f"  Press  {key}  ", bg=col, fg=contrast(col),
                         font=("Segoe UI", 22, "bold"), padx=10, pady=8).pack(anchor="w", pady=6)
                ttk.Label(box, text=name, font=("Segoe UI", 14, "bold")).pack(anchor="w")
                ttk.Label(box, text=expl, wraplength=760, font=("Segoe UI", 11)).pack(anchor="w", pady=8)
            else:
                q, yes, no = node
                ttk.Label(box, text=f"Question {len(history)}", style="Small.TLabel").pack(anchor="w")
                ttk.Label(box, text=q, wraplength=760, font=("Segoe UI", 13)).pack(anchor="w", pady=10)
                b = ttk.Frame(box)
                b.pack(anchor="w", pady=6)
                ttk.Button(b, text="Yes", width=12, command=lambda: (history.append(yes), render())).pack(side="left")
                ttk.Button(b, text="No", width=12, command=lambda: (history.append(no), render())).pack(side="left",
                                                                                                    padx=8)
            nav = ttk.Frame(box)
            nav.pack(anchor="w", pady=20)
            if len(history) > 1:
                ttk.Button(nav, text="< Back", command=lambda: (history.pop(), render())).pack(side="left")
            ttk.Button(nav, text="Start again", command=lambda: (history.__delitem__(slice(1, None)), render())).pack(
                side="left", padx=8)

        render()

    def show_fixer(self):
        self._clear()
        f = ttk.Frame(self.body, padding=16)
        f.pack(fill="both", expand=True)
        ttk.Label(f, text="Error & warning fixer", font=("Segoe UI", 17, "bold"), foreground="#0d47a1").pack(anchor="w")
        ttk.Label(f, text="Paste a message from the Issues panel or the Check tab, or pick one from the list.",
                  style="Small.TLabel").pack(anchor="w", pady=(0, 8))
        row = ttk.Frame(f)
        row.pack(fill="x")
        q = tk.StringVar()
        ent = ttk.Entry(row, textvariable=q)
        ent.pack(side="left", fill="x", expand=True)
        out = tk.Text(f, height=7, wrap="word", font=("Segoe UI", 11), relief="flat", bg="#f5f5f5")
        out.tag_configure("b", font=("Segoe UI", 11, "bold"))

        def explain(text):
            out.configure(state="normal")
            out.delete("1.0", "end")
            h = issue_help(text)
            if h:
                out.insert("end", "What it means: ", "b")
                out.insert("end", h[0] + "\n\n")
                out.insert("end", "How to fix: ", "b")
                out.insert("end", h[1])
            else:
                out.insert("end", "No explanation found for this text. Check the spelling, or ask the project lead.")
            out.configure(state="disabled")

        ttk.Button(row, text="Explain", command=lambda: explain(q.get())).pack(side="left", padx=4)
        ent.bind("<Return>", lambda _: explain(q.get()))
        out.pack(fill="x", pady=8)
        lb = tk.Listbox(f, font=("Segoe UI", 10), activestyle="none")
        lb.pack(fill="both", expand=True)
        for pat, meaning, _ in manual.ISSUE_HELP:
            lb.insert("end", meaning)

        def pick(_=None):
            s = lb.curselection()
            if s:
                pat, meaning, fix = manual.ISSUE_HELP[s[0]]
                out.configure(state="normal")
                out.delete("1.0", "end")
                out.insert("end", "What it means: ", "b")
                out.insert("end", meaning + "\n\n")
                out.insert("end", "How to fix: ", "b")
                out.insert("end", fix)
                out.configure(state="disabled")

        lb.bind("<<ListboxSelect>>", pick)

    def show_quiz(self):
        self._clear()
        f = ttk.Frame(self.body, padding=20)
        f.pack(fill="both", expand=True)
        ttk.Label(f, text="Practice quiz", font=("Segoe UI", 17, "bold"), foreground="#0d47a1").pack(anchor="w")
        ttk.Label(f, text=f"{len(manual.QUIZ)} situations from real labelling. Try it before your first video.",
                  style="Small.TLabel").pack(anchor="w", pady=(0, 10))
        box = ttk.Frame(f)
        box.pack(fill="both", expand=True)
        state = {"i": 0, "score": 0, "answered": False}

        def render():
            for w in box.winfo_children():
                w.destroy()
            i = state["i"]
            if i >= len(manual.QUIZ):
                n = len(manual.QUIZ)
                ttk.Label(box, text=f"Score: {state['score']} / {n}", font=("Segoe UI", 20, "bold")).pack(anchor="w")
                msg = ("Excellent - you are ready for your first video." if state["score"] >= n - 1 else
                       "Good. Re-read the chapters on phases and events, then try again." if state["score"] >= n * 0.7
                       else "Please read chapters 4-7 again and retry.")
                ttk.Label(box, text=msg, font=("Segoe UI", 12)).pack(anchor="w", pady=10)
                ttk.Button(box, text="Try again", command=lambda: (state.update(i=0, score=0, answered=False),
                                                                    render())).pack(anchor="w")
                return
            q, options, correct, why = manual.QUIZ[i]
            ttk.Label(box, text=f"Question {i + 1} of {len(manual.QUIZ)}     score {state['score']}",
                      style="Small.TLabel").pack(anchor="w")
            ttk.Label(box, text=q, wraplength=780, font=("Segoe UI", 13)).pack(anchor="w", pady=10)
            choice = tk.IntVar(value=-1)
            for k, o in enumerate(options):
                ttk.Radiobutton(box, text=o, variable=choice, value=k).pack(anchor="w", pady=3)
            fb = tk.Label(box, text="", wraplength=780, justify="left", font=("Segoe UI", 11), padx=8, pady=6)
            nav = ttk.Frame(box)

            def check():
                if choice.get() < 0 or state["answered"]:
                    return
                state["answered"] = True
                good = choice.get() == correct
                state["score"] += int(good)
                fb.configure(text=("Correct. " if good else f"Not quite - the answer is: {options[correct]}. ") + why,
                             bg="#e8f5e9" if good else "#ffebee")
                fb.pack(anchor="w", fill="x", pady=8)

            def nxt():
                state["i"] += 1
                state["answered"] = False
                render()

            nav.pack(anchor="w", pady=10)
            ttk.Button(nav, text="Check answer", command=check).pack(side="left")
            ttk.Button(nav, text="Next question >", command=nxt).pack(side="left", padx=8)

        render()

    def open_tutorial(self):
        vids = sorted((ROOT / "videos").glob("*annoation*.mp4")) + sorted((ROOT / "videos").glob("*annotation*.mp4"))
        self._clear()
        f = ttk.Frame(self.body, padding=20)
        f.pack(fill="both", expand=True)
        ttk.Label(f, text="Tutorial recording", font=("Segoe UI", 17, "bold"), foreground="#0d47a1").pack(anchor="w")
        if not vids:
            ttk.Label(f, text="No tutorial recording in the videos folder.").pack(anchor="w", pady=10)
            return
        ttk.Label(f, text="A screen recording of the older annotate.py window. The keys are the same in the Studio; "
                          "the Studio adds buttons, the Persons/Segments/Issues panels and the checks.",
                  wraplength=760).pack(anchor="w", pady=10)

        def play():
            try:
                os.startfile(str(vids[0]))  # noqa: S606
            except (AttributeError, OSError) as exc:
                messagebox.showerror("Tutorial", str(exc), parent=self)

        ttk.Button(f, text=f"Play {vids[0].name}", command=play).pack(anchor="w")


# ====================================================================== entry point
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT, help="Workspace folder (contains labels/ and outputs/)")
    sub = ap.add_subparsers(dest="cmd")
    g = sub.add_parser("generate-tracks", help=argparse.SUPPRESS)
    g.add_argument("--video", required=True)
    g.add_argument("--name", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--weights", required=True)
    g.add_argument("--device", default="cpu")
    g.add_argument("--hz", type=float, default=HZ)
    g.add_argument("--sha", default="")
    a = ap.parse_args()
    if a.cmd == "generate-tracks":
        return generate_tracks_cli(a)
    _gui_imports()
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    ws = Workspace(a.root)
    ws.labels_dir.mkdir(parents=True, exist_ok=True)
    Studio(ws).mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
