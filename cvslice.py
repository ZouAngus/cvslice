"""
cvslice - Multi-View Action Clip Annotator & Exporter
Usage:  python cvslice.py

Features:
 - Load Excel with multiple scene sheets + Data root folder + Calibration folder
 - Switch scenes via combo box (auto-matches sheet -> data subfolder -> CSV)
 - Multi-camera switching with 3D skeleton overlay
 - Per-scene and per-action offset with auto-save/load to JSON
 - Loop playback, single/multi-cam export
 - Handles missing videos (virtual black background) and NaN in CSVs
"""

import sys, os, json, copy, re
import cv2
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QFileDialog, QSlider, QSpinBox, QComboBox, QCheckBox,
    QProgressDialog, QMessageBox, QGroupBox, QFormLayout, QSizePolicy,
    QInputDialog, QMenu, QAction,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, QCoreApplication, QElapsedTimer

CAMERA_NAMES = [
    "topleft", "topcenter", "topright",
    "bottomleft", "bottomcenter", "bottomright",
    "diagonal",
]
DEFAULT_POINTS_FPS = 60.0

JOINT_PAIRS_24 = [
    (6,9),(12,9),(12,15),(20,18),(18,16),(16,13),(13,6),(14,6),(14,17),
    (17,19),(19,21),(3,6),(0,3),(1,0),(2,0),(10,7),(7,4),(4,1),(2,5),(5,8),(11,8),
]
JOINT_PAIRS_17 = [
    (0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(0,7),(7,8),(8,9),(9,10),
    (8,11),(11,12),(12,13),(8,14),(14,15),(15,16),
]
JOINT_PAIRS_MAP = {17: JOINT_PAIRS_17, 24: JOINT_PAIRS_24}
PT_COLOR = (0, 0, 255)


# ---------------------------------------------------------------------------
#  Helpers (parsing, calibration, projection, drawing)
# ---------------------------------------------------------------------------

def parse_excel_actions(xlsx_path, sheet_name):
    """Parse action rows. Handles variable column layouts and multi-repetition columns."""
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    hdr = {str(c).strip().lower(): i for i, c in enumerate(df.columns)}
    action_col = hdr.get("action")
    no_col = hdr.get("no.")
    if action_col is None:
        best = (-1, -1)
        for i in range(len(df.columns)):
            n = int(df.iloc[:, i].apply(lambda x: isinstance(x, str)).sum())
            if n > best[0]: best = (n, i)
        action_col = best[1]
    num_cols = []
    for i in range(len(df.columns)):
        vals = pd.to_numeric(df.iloc[:, i], errors="coerce").dropna()
        if len(vals) >= 2 and float(vals.mean()) > 10: num_cols.append(i)
    if len(num_cols) < 2: return []
    best_pair = (num_cols[-2], num_cols[-1])
    best_score = -1
    for pi in range(len(num_cols) - 1):
        ci, cj = num_cols[pi], num_cols[pi + 1]
        count = 0
        for idx in range(len(df)):
            sv = df.iloc[idx, ci]; ev = df.iloc[idx, cj]
            try:
                s = int(float(sv)); e = int(float(ev))
                if s > 0 and e > s: count += 1
            except (TypeError, ValueError): pass
        if count < 2: continue
        vi = pd.to_numeric(df.iloc[:, ci], errors="coerce").dropna()
        vj = pd.to_numeric(df.iloc[:, cj], errors="coerce").dropna()
        score = count * 100000 + float(vi.mean()) + float(vj.mean())
        if score > best_score:
            best_score = score; best_pair = (ci, cj)
    start_col, end_col = best_pair
    extra_pairs = []
    remaining = [c for c in num_cols if c > end_col]
    for ri in range(0, len(remaining) - 1, 2):
        extra_pairs.append((remaining[ri], remaining[ri + 1]))
    variant_col = None
    cand = action_col + 1
    if cand < len(df.columns) and cand not in (start_col, end_col): variant_col = cand
    act_series = df.iloc[:, action_col].copy().ffill()
    rows = []
    for idx in range(len(df)):
        aname = str(act_series.iloc[idx]).strip() if pd.notna(act_series.iloc[idx]) else "?"
        variant = ""
        if variant_col is not None:
            v = df.iloc[idx, variant_col]
            if pd.notna(v): variant = str(v).strip()
        no_val = None
        if no_col is not None:
            nv = df.iloc[idx, no_col]
            if pd.notna(nv):
                try: no_val = int(float(nv))
                except Exception: pass
        try:
            sf = int(float(df.iloc[idx, start_col]))
            ef = int(float(df.iloc[idx, end_col]))
        except (TypeError, ValueError): sf = ef = 0
        if sf > 0 and ef > sf:
            a = dict(no=no_val, action=aname, variant=variant, start=sf, end=ef)
            a["label"] = make_label(a)
            rows.append(a)
        for rep_i, (rc_s, rc_e) in enumerate(extra_pairs):
            try:
                rs = int(float(df.iloc[idx, rc_s]))
                re_ = int(float(df.iloc[idx, rc_e]))
            except (TypeError, ValueError): continue
            if rs > 0 and re_ > rs:
                a2 = dict(no=no_val, action=aname, variant=variant,
                          start=rs, end=re_, rep=f"rep{rep_i + 2}")
                a2["label"] = make_label(a2)
                rows.append(a2)
    return rows


def load_calibration(cal_dir, cam_name):
    intr = extr = None
    for fn in os.listdir(cal_dir):
        fl = fn.lower()
        if cam_name not in fl: continue
        fp = os.path.join(cal_dir, fn)
        try:
            with open(fp) as f: d = json.load(f)
        except Exception: continue
        if "intrinsic" in fl: intr = d
        elif "extrinsic" in fl: extr = d
    return intr, extr


def _rvec_tvec(extr):
    ext = None
    for k in ("best_extrinsic", "extrinsic", "extrinsics"):
        if k not in extr: continue
        v = extr[k]
        if k == "extrinsics" and isinstance(v, list) and v: v = v[0]
        ext = np.array(v, dtype=float); break
    if ext is None: return None, None
    if ext.shape == (4, 4): ext = ext[:3, :]
    if ext.shape != (3, 4): return None, None
    R = ext[:, :3]; t = ext[:, 3].reshape(3, 1)
    rv, _ = cv2.Rodrigues(R)
    return rv, t


def project_pts(pts3d, intr, extr, flip_x=False, flip_y=False, flip_z=False,
                _cache={}):
    """Project 3D points with cached rvec/tvec/camera_matrix per extrinsic."""
    pts = pts3d.copy()
    if flip_x: pts[:, 0] *= -1
    if flip_y: pts[:, 1] *= -1
    if flip_z: pts[:, 2] *= -1
    # Cache key: id of extr dict (same object = same camera)
    cache_key = id(extr)
    if cache_key in _cache:
        rv, tv, cm, dc = _cache[cache_key]
    else:
        rv, tv = _rvec_tvec(extr)
        if rv is None: return None
        cm = np.array(intr["camera_matrix"], dtype=np.float64)
        dc_raw = intr.get("dist_coeffs") or extr.get("dist_coeffs")
        dc = np.array(dc_raw, dtype=np.float64).reshape(-1) if dc_raw is not None else np.zeros(5, dtype=np.float64)
        _cache[cache_key] = (rv, tv, cm, dc)
    proj, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), rv, tv, cm, dc)
    return proj.squeeze().astype(np.int32)


def draw_skel(frame, proj, color=PT_COLOR):
    h, w = frame.shape[:2]; n = len(proj)
    bc = tuple(int(c * 0.7) for c in color)
    for pt in proj:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < w and 0 <= y < h: cv2.circle(frame, (x, y), 4, color, -1)
    for i, j in JOINT_PAIRS_MAP.get(n, []):
        if i < n and j < n:
            x1, y1 = int(proj[i][0]), int(proj[i][1])
            x2, y2 = int(proj[j][0]), int(proj[j][1])
            if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                cv2.line(frame, (x1, y1), (x2, y2), bc, 2)


def v2p(vf, vfps, pfps, ptot, off=0):
    if vfps <= 0: vfps = 30.0
    if pfps <= 0: pfps = vfps
    idx = int(round((vf + off) * (pfps / vfps)))
    return max(0, min(ptot - 1, idx))


def fmt_time(sec):
    return f"{int(sec//3600):02d}:{int((sec%3600)//60):02d}:{int(sec%60):02d}"


def make_label(a, ov=None):
    if ov is None: ov = {}
    s = ov.get("start", a["start"]); e = ov.get("end", a["end"])
    rep = a.get("rep")
    lbl = (f"#{a['no']} " if a.get("no") else "") + a["action"]
    if a.get("variant"): lbl += f" [{a['variant']}]"
    if rep: lbl += f" {rep}"
    lbl += f"  ({s}-{e})"
    off = ov.get("offset", 0)
    if off != 0: lbl += f" off={off}"
    return lbl


# ---------------------------------------------------------------------------
#  Scene / data discovery helpers
# ---------------------------------------------------------------------------

def _normalize_scene_key(name):
    return re.sub(r'[^a-z0-9]', '', name.lower())


def _find_data_subfolder(data_root, sheet_name):
    if not data_root or not os.path.isdir(data_root):
        return None
    key = _normalize_scene_key(sheet_name)
    for entry in sorted(os.listdir(data_root)):
        full = os.path.join(data_root, entry)
        if os.path.isdir(full) and _normalize_scene_key(entry) == key:
            return full
    for entry in sorted(os.listdir(data_root)):
        full = os.path.join(data_root, entry)
        if os.path.isdir(full):
            ek = _normalize_scene_key(entry)
            if key in ek or ek in key:
                return full
    return None


def _find_csv_in_folder(folder):
    if not folder or not os.path.isdir(folder):
        return None
    for fn in sorted(os.listdir(folder)):
        if fn.lower().startswith("extracted") and fn.lower().endswith(".csv"):
            return os.path.join(folder, fn)
    return None


def _find_csv_for_scene(data_root, sheet_name):
    """Find CSV + video folder for a scene. Returns (csv_path|None, video_folder|None)."""
    subfolder = _find_data_subfolder(data_root, sheet_name)
    # 1. CSV inside subfolder
    if subfolder:
        csv_path = _find_csv_in_folder(subfolder)
        if csv_path:
            return csv_path, subfolder
    # 2. CSV in data root matching scene name
    csv_path = None
    if data_root and os.path.isdir(data_root):
        key = _normalize_scene_key(sheet_name)
        for fn in sorted(os.listdir(data_root)):
            if not fn.lower().endswith(".csv"): continue
            if not fn.lower().startswith("extracted"): continue
            fk = _normalize_scene_key(
                os.path.splitext(fn)[0].replace("extracted", "").strip("_"))
            if key in fk or fk in key:
                csv_path = os.path.join(data_root, fn)
                break
    return csv_path, subfolder


def _find_cameras_in_folder(folder):
    if not folder or not os.path.isdir(folder):
        return []
    cams = []
    for cn in CAMERA_NAMES:
        for fn in os.listdir(folder):
            if cn in fn.lower() and fn.lower().endswith(".mp4"):
                cams.append(cn); break
    return cams


def _load_csv_as_pts3d(csv_path):
    """Load extracted CSV -> (T, J, 3). NaN filled with 0.
    Returns (pts3d_array, valid_mask) where valid_mask[i] is True if frame i has data."""
    df = pd.read_csv(csv_path)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(0.0)
    nc = df.shape[1]
    if nc % 3 != 0:
        return None, None
    pts = df.values.reshape(-1, nc // 3, 3)
    # Pre-compute boolean mask: True = frame has at least one non-zero joint
    valid = np.any(pts != 0, axis=(1, 2))
    return pts, valid


# ---------------------------------------------------------------------------
#  Annotations persistence
# ---------------------------------------------------------------------------

def _annotations_path(xlsx_path):
    base = os.path.splitext(xlsx_path)[0]
    return base + "_annotations.json"


def _load_annotations(xlsx_path):
    p = _annotations_path(xlsx_path)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_annotations(xlsx_path, data):
    p = _annotations_path(xlsx_path)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: failed to save annotations: {e}")


# ---------------------------------------------------------------------------
#  Main Window
# ---------------------------------------------------------------------------

class ClipAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CVSlice - Multi-View Action Clip Tool")
        self.setGeometry(60, 60, 1440, 840)
        self.setFocusPolicy(Qt.StrongFocus)

        # Sources
        self.xlsx_path = None
        self.sheet_names = []       # all sheets in the Excel
        self.data_root = None       # top-level data folder
        self.cal_folder = None

        # Current scene state
        self.cur_scene = None       # current sheet name
        self.actions = []
        self.cur_act = -1
        self.avail_cams = []
        self.active_cam = None
        self.video_folder = None    # folder containing videos for current scene
        self.cap = None
        self.vfps = 30.0
        self.vtotal = 0
        self.pts3d = None
        self.pts3d_valid = None     # boolean mask: True = frame has non-zero data
        self.pfps = DEFAULT_POINTS_FPS
        self.calibs = {}
        self.scene_offset = 0
        self.overrides = {}         # {action_index: {start, end, offset}}
        self.cur_frame = 0
        self.clip_start = self.clip_end = 0
        self.playing = False
        self.show_skel = True
        self.flip = [False, False, False]
        self._suppress_spin = False
        self._cached_frame_idx = -1
        self._cached_frame = None
        self.loop_playback = True    # auto-replay each clip

        # Annotations (persistent)
        self._annotations = {}      # full annotations dict

        self._build_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        # Debounced auto-save: avoids writing JSON on every W/S keystroke
        self._save_timer = QTimer()
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)  # 500ms debounce
        self._save_timer.timeout.connect(self._do_save)

    # =======================================================================
    #  UI construction
    # =======================================================================
    def _build_ui(self):
        mb = self.menuBar()
        fm = mb.addMenu("File")
        fm.addAction("Load Excel...", self._load_xlsx)
        fm.addAction("Load Data Root Folder...", self._load_data_root)
        fm.addAction("Load Calibration Folder...", self._load_cal)
        fm.addSeparator()
        fm.addAction("Save Offsets", self._save_current_annotations)
        em = mb.addMenu("Export")
        em.addAction("Export Current Clip (all cams)...", lambda: self._export(False, single_cam=False))
        em.addAction("Export ALL Clips (all cams)...", lambda: self._export(True, single_cam=False))
        em.addSeparator()
        em.addAction("Export Current Clip (current cam only)...", lambda: self._export(False, single_cam=True))
        em.addAction("Export ALL Clips (current cam only)...", lambda: self._export(True, single_cam=True))

        root = QWidget(); self.setCentralWidget(root)
        hl = QHBoxLayout(root); hl.setContentsMargins(4,4,4,4)

        # ---- LEFT panel ----
        left = QWidget(); left.setFixedWidth(310)
        lv = QVBoxLayout(left); lv.setContentsMargins(0,0,0,0)

        sg = QGroupBox("Sources"); sf = QFormLayout(sg)
        self.lbl_xlsx = QLabel("-"); self.lbl_data = QLabel("-"); self.lbl_cal = QLabel("-")
        sf.addRow("Excel:", self.lbl_xlsx)
        sf.addRow("Data:", self.lbl_data)
        sf.addRow("Calib:", self.lbl_cal)
        lv.addWidget(sg)

        # Scene selector
        scg = QGroupBox("Scene"); scf = QFormLayout(scg)
        self.scene_combo = QComboBox()
        self.scene_combo.currentTextChanged.connect(self._on_scene_changed)
        scf.addRow("Scene:", self.scene_combo)
        self.lbl_scene_info = QLabel("-")
        self.lbl_scene_info.setWordWrap(True)
        self.lbl_scene_info.setStyleSheet("font-size:11px; color:#555;")
        scf.addRow(self.lbl_scene_info)
        lv.addWidget(scg)

        lv.addWidget(QLabel("Actions (right-click to add repetition):"))
        self.act_list = QListWidget()
        self.act_list.currentRowChanged.connect(self._on_act_sel)
        self.act_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.act_list.customContextMenuRequested.connect(self._act_context_menu)
        lv.addWidget(self.act_list)
        hl.addWidget(left)

        # ---- CENTER panel ----
        center = QWidget(); cvl = QVBoxLayout(center); cvl.setContentsMargins(0,0,0,0)
        self.vid_lbl = QLabel()
        self.vid_lbl.setAlignment(Qt.AlignCenter)
        self.vid_lbl.setMinimumSize(640, 400)
        self.vid_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.vid_lbl.setStyleSheet("background:black;")
        cvl.addWidget(self.vid_lbl)
        self.info_lbl = QLabel("Frame: - / -   Time: - / -")
        self.info_lbl.setStyleSheet("font-size:13px; padding:2px;")
        cvl.addWidget(self.info_lbl)
        self.slider = QSlider(Qt.Horizontal); self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._on_slider)
        cvl.addWidget(self.slider)
        br = QHBoxLayout()
        for txt, fn in [("<< -1s", lambda: self._jump(-1)), ("< Prev", self._prev),
                         ("Play / Pause", self._toggle_play), ("Next >", self._nxt),
                         ("+1s >>", lambda: self._jump(1))]:
            b = QPushButton(txt); b.clicked.connect(fn); br.addWidget(b)
        cvl.addLayout(br)
        hl.addWidget(center, stretch=1)

        # ---- RIGHT panel ----
        right = QWidget(); right.setFixedWidth(260)
        rv = QVBoxLayout(right); rv.setContentsMargins(0,0,0,0)
        cg = QGroupBox("Camera"); cf = QFormLayout(cg)
        self.cam_combo = QComboBox()
        self.cam_combo.currentTextChanged.connect(self._on_cam)
        cf.addRow("View:", self.cam_combo)
        rv.addWidget(cg)

        og = QGroupBox("Sync (3D offset)"); of2 = QFormLayout(og)
        self.scene_off_spin = QSpinBox(); self.scene_off_spin.setRange(-50000, 50000)
        self.scene_off_spin.setInputMethodHints(Qt.ImhFormattedNumbersOnly)
        self.scene_off_spin.valueChanged.connect(self._on_scene_off)
        of2.addRow("Scene:", self.scene_off_spin)
        self.act_off_spin = QSpinBox(); self.act_off_spin.setRange(-50000, 50000)
        self.act_off_spin.setInputMethodHints(Qt.ImhFormattedNumbersOnly)
        self.act_off_spin.valueChanged.connect(self._on_act_off)
        of2.addRow("Action:", self.act_off_spin)
        rv.addWidget(og)

        ag = QGroupBox("Action Override"); af = QFormLayout(ag)
        self.start_spin = QSpinBox(); self.start_spin.setRange(0, 9999999)
        self.start_spin.setInputMethodHints(Qt.ImhDigitsOnly)
        self.start_spin.valueChanged.connect(self._on_start_ov)
        af.addRow("Start:", self.start_spin)
        self.end_spin = QSpinBox(); self.end_spin.setRange(0, 9999999)
        self.end_spin.setInputMethodHints(Qt.ImhDigitsOnly)
        self.end_spin.valueChanged.connect(self._on_end_ov)
        af.addRow("End:", self.end_spin)
        rv.addWidget(ag)

        fg = QGroupBox("Display"); ff = QFormLayout(fg)
        self.skel_cb = QCheckBox("Show Skeleton"); self.skel_cb.setChecked(True)
        self.skel_cb.stateChanged.connect(lambda s: setattr(self, "show_skel", s == Qt.Checked) or self._show_frame())
        ff.addRow(self.skel_cb)
        self.loop_cb = QCheckBox("Loop Playback"); self.loop_cb.setChecked(True)
        self.loop_cb.stateChanged.connect(lambda s: setattr(self, "loop_playback", s == Qt.Checked))
        ff.addRow(self.loop_cb)
        for axis, idx in [("Flip X", 0), ("Flip Y", 1), ("Flip Z", 2)]:
            cb = QCheckBox(axis)
            cb.stateChanged.connect(lambda s, i=idx: self._set_flip(i, s == Qt.Checked))
            ff.addRow(cb)
        rv.addWidget(fg)
        rv.addStretch()
        hl.addWidget(right)

        self.statusBar().showMessage(
            "Space=Play  A/D=Prev/Next  Q/E=-/+1s  W/S=SceneOffset  Up/Down=Action")

    def _set_flip(self, idx, val):
        self.flip[idx] = val; self._show_frame()

    # =======================================================================
    #  File loaders
    # =======================================================================
    def _load_xlsx(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Excel", "", "Excel (*.xlsx *.xls)")
        if not path: return
        try:
            xl = pd.ExcelFile(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e)); return
        self.xlsx_path = path
        self.sheet_names = xl.sheet_names
        self._annotations = _load_annotations(path)
        self.lbl_xlsx.setText(os.path.basename(path))

        # Populate scene combo
        self.scene_combo.blockSignals(True)
        self.scene_combo.clear()
        self.scene_combo.addItems(self.sheet_names)
        self.scene_combo.blockSignals(False)

        # Auto-switch to first scene
        if self.sheet_names:
            self.scene_combo.setCurrentIndex(0)
            self._on_scene_changed(self.sheet_names[0])

        self.statusBar().showMessage(
            f"Loaded Excel: {len(self.sheet_names)} sheets. "
            f"Annotations: {'loaded' if self._annotations else 'none'}.")

    def _load_data_root(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Data Root Folder")
        if not folder: return
        self.data_root = folder
        self.lbl_data.setText(os.path.basename(folder))
        # Re-apply current scene if one is selected
        if self.cur_scene:
            self._apply_scene(self.cur_scene)
        self.statusBar().showMessage(f"Data root: {folder}")

    def _load_cal(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Calibration Folder")
        if not folder: return
        self.cal_folder = folder
        self.lbl_cal.setText(os.path.basename(folder))
        self.calibs = {}
        for cn in CAMERA_NAMES:
            intr, extr = load_calibration(folder, cn)
            if intr and extr: self.calibs[cn] = (intr, extr)
        self.statusBar().showMessage(
            f"Loaded calibration for {len(self.calibs)} cameras")
        self._show_frame()

    # =======================================================================
    #  Scene switching
    # =======================================================================
    def _on_scene_changed(self, scene_name):
        if not scene_name: return
        # Save current scene's state before switching
        self._save_scene_state()
        self._apply_scene(scene_name)

    def _apply_scene(self, scene_name):
        """Apply a scene: load actions from Excel, find CSV + videos, restore offsets."""
        self.cur_scene = scene_name

        # --- Parse actions from Excel ---
        if self.xlsx_path:
            self.actions = parse_excel_actions(self.xlsx_path, scene_name)
        else:
            self.actions = []

        # --- Find CSV and video folder ---
        csv_path = None
        self.video_folder = None
        self.pts3d = None
        self.pts3d_valid = None
        self.avail_cams = []

        if self.data_root:
            csv_path, subfolder = _find_csv_for_scene(self.data_root, scene_name)
            if subfolder:
                self.video_folder = subfolder
                self.avail_cams = _find_cameras_in_folder(subfolder)
            # Also check data root itself for videos
            if not self.avail_cams:
                root_cams = _find_cameras_in_folder(self.data_root)
                if root_cams:
                    self.video_folder = self.data_root
                    self.avail_cams = root_cams

        if csv_path:
            result = _load_csv_as_pts3d(csv_path)
            if result[0] is not None:
                self.pts3d, self.pts3d_valid = result

        # Update camera combo
        self.cam_combo.blockSignals(True)
        self.cam_combo.clear()
        if self.avail_cams:
            self.cam_combo.addItems(self.avail_cams)
        else:
            self.cam_combo.addItem("(no cameras)")
        self.cam_combo.blockSignals(False)

        # --- Restore saved state for this scene ---
        self.overrides = {}
        self.scene_offset = 0
        saved = self._annotations.get(scene_name, {})
        if saved:
            self.scene_offset = saved.get("scene_offset", 0)
            saved_ov = saved.get("overrides", {})
            # JSON keys are strings, convert back to int
            self.overrides = {int(k): v for k, v in saved_ov.items()}

        self._suppress_spin = True
        self.scene_off_spin.setValue(self.scene_offset)
        self._suppress_spin = False

        # --- Update info label ---
        info_parts = []
        if csv_path:
            info_parts.append(f"CSV: {os.path.basename(csv_path)}")
            if self.pts3d is not None:
                info_parts.append(f"({self.pts3d.shape[0]} frames, {self.pts3d.shape[1]} joints)")
        else:
            info_parts.append("CSV: not found")
        info_parts.append(f"Cameras: {len(self.avail_cams)}")
        info_parts.append(f"Actions: {len(self.actions)}")
        self.lbl_scene_info.setText("  |  ".join(info_parts))

        # --- Refresh action list ---
        self._refresh_act_list()

        # --- Open first camera ---
        if self.cap:
            self.cap.release(); self.cap = None
        self._cached_frame_idx = -1
        self._cached_frame = None
        self.active_cam = None

        if self.avail_cams:
            self._switch_cam(self.avail_cams[0])
        else:
            # No video: create virtual black frame if we have 3D data
            self.vfps = 30.0
            if self.pts3d is not None:
                self.vtotal = int(self.pts3d.shape[0] * (30.0 / DEFAULT_POINTS_FPS))
            else:
                self.vtotal = 0
            self._cached_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            self._cached_frame_idx = 0

        self._estimate_pfps()

        # Select first action
        if self.actions:
            self.act_list.setCurrentRow(0)
        else:
            self.cur_act = -1
            self._show_frame()

        self.statusBar().showMessage(
            f"Scene: {scene_name} — {len(self.actions)} actions, "
            f"{len(self.avail_cams)} cameras, "
            f"CSV={'yes' if csv_path else 'no'}")

    # =======================================================================
    #  Annotations save/load
    # =======================================================================
    def _save_scene_state(self):
        """Save current scene's offsets into _annotations dict and persist."""
        if not self.cur_scene or not self.xlsx_path:
            return
        scene_data = {
            "scene_offset": self.scene_offset,
            "overrides": {str(k): v for k, v in self.overrides.items()},
        }
        self._annotations[self.cur_scene] = scene_data
        _save_annotations(self.xlsx_path, self._annotations)

    def _save_current_annotations(self):
        """Explicit save (also triggered by menu: Save Offsets)."""
        if self._save_timer.isActive():
            self._save_timer.stop()
        self._save_scene_state()
        if self.xlsx_path:
            QMessageBox.information(
                self, "Saved",
                f"Offsets saved to:\n{_annotations_path(self.xlsx_path)}")

    def _estimate_pfps(self):
        if self.pts3d is not None and self.vtotal > 0 and self.vfps > 0:
            dur = self.vtotal / self.vfps
            if dur > 0:
                self.pfps = self.pts3d.shape[0] / dur
                return
        self.pfps = DEFAULT_POINTS_FPS

    def _switch_cam(self, cam_name):
        if self.cap: self.cap.release(); self.cap = None
        self.active_cam = cam_name
        self._cached_frame_idx = -1
        self._cached_frame = None
        if not self.video_folder: return
        for fn in os.listdir(self.video_folder):
            if cam_name in fn.lower() and fn.lower().endswith(".mp4"):
                vpath = os.path.join(self.video_folder, fn)
                self.cap = cv2.VideoCapture(vpath)
                if self.cap.isOpened():
                    self.vfps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
                    self.vtotal = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self._estimate_pfps()
                break
        self._read_frame(self.cur_frame)
        self._show_frame()

    # =======================================================================
    #  Frame cache
    # =======================================================================
    def _read_frame(self, frame_idx):
        if frame_idx == self._cached_frame_idx and self._cached_frame is not None:
            return self._cached_frame
        if not self.cap or not self.cap.isOpened():
            # Virtual black frame for no-video scenes
            if self._cached_frame is None:
                self._cached_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            self._cached_frame_idx = frame_idx
            return self._cached_frame
        # Sequential read: if next frame, just read (no seek)
        if frame_idx == self._cached_frame_idx + 1:
            ret, frame = self.cap.read()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
        if ret:
            self._cached_frame_idx = frame_idx
            self._cached_frame = frame  # no .copy() — cv2.read() returns a new array
            return self._cached_frame
        return None

    # =======================================================================
    #  Action list helpers
    # =======================================================================
    def _refresh_act_list(self):
        self.act_list.clear()
        for i, a in enumerate(self.actions):
            ov = self.overrides.get(i, {})
            self.act_list.addItem(make_label(a, ov))

    def _act_context_menu(self, pos):
        item = self.act_list.itemAt(pos)
        if item is None: return
        row = self.act_list.row(item)
        menu = QMenu(self)
        add_rep = menu.addAction("Add repetition (duplicate with new start/end)...")
        delete_act = menu.addAction("Delete this entry")
        chosen = menu.exec_(self.act_list.viewport().mapToGlobal(pos))
        if chosen == add_rep:
            self._add_repetition(row)
        elif chosen == delete_act:
            self._delete_action(row)

    def _add_repetition(self, src_row):
        if src_row < 0 or src_row >= len(self.actions): return
        a = self.actions[src_row]
        new_start, ok1 = QInputDialog.getInt(self, "New repetition",
            f"Start frame for new repetition of \"{a['action']}\":",
            value=a["end"] + 1, min=0, max=9999999)
        if not ok1: return
        new_end, ok2 = QInputDialog.getInt(self, "New repetition",
            f"End frame:", value=new_start + (a["end"] - a["start"]),
            min=new_start + 1, max=9999999)
        if not ok2: return
        base_action = a["action"]
        base_variant = a.get("variant", "")
        rep_count = sum(1 for aa in self.actions
                        if aa["action"] == base_action
                        and aa.get("variant", "") == base_variant)
        new_a = dict(no=a["no"], action=a["action"], variant=a.get("variant", ""),
                     start=new_start, end=new_end, rep=f"rep{rep_count + 1}", label="")
        new_a["label"] = make_label(new_a)
        insert_idx = src_row + 1
        self.actions.insert(insert_idx, new_a)
        new_ov = {}
        for k, v in self.overrides.items():
            if k >= insert_idx: new_ov[k + 1] = v
            else: new_ov[k] = v
        self.overrides = new_ov
        self._refresh_act_list()
        self.act_list.setCurrentRow(insert_idx)
        self._auto_save()
        self.statusBar().showMessage(f"Added repetition: {new_a['label']}")

    def _delete_action(self, row):
        if row < 0 or row >= len(self.actions): return
        a = self.actions[row]
        reply = QMessageBox.question(self, "Confirm",
            f"Delete \"{make_label(a)}\"?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes: return
        self.actions.pop(row)
        self.overrides.pop(row, None)
        new_ov = {}
        for k, v in self.overrides.items():
            if k > row: new_ov[k - 1] = v
            elif k < row: new_ov[k] = v
        self.overrides = new_ov
        self._refresh_act_list()
        if self.actions:
            self.act_list.setCurrentRow(min(row, len(self.actions) - 1))
        self._auto_save()

    def _auto_save(self):
        """Schedule a debounced save (500ms). Multiple rapid calls reset the timer."""
        self._save_timer.start()

    def _do_save(self):
        """Actually write annotations to disk."""
        self._save_scene_state()

    # =======================================================================
    #  Action selection & overrides
    # =======================================================================
    def _get_effective_act_offset(self, row):
        ov = self.overrides.get(row, {})
        if "offset" in ov:
            return ov["offset"]
        for r in range(row - 1, -1, -1):
            prev_ov = self.overrides.get(r, {})
            if "offset" in prev_ov:
                return prev_ov["offset"]
        return 0

    def _on_act_sel(self, row):
        if row < 0 or row >= len(self.actions): return
        self.cur_act = row
        a = self.actions[row]
        ov = self.overrides.get(row, {})
        s = ov.get("start", a["start"])
        e = ov.get("end", a["end"])
        self.clip_start = s; self.clip_end = e
        self._suppress_spin = True
        self.start_spin.setValue(s)
        self.end_spin.setValue(e)
        self.act_off_spin.setValue(self._get_effective_act_offset(row))
        self._suppress_spin = False
        self.slider.setRange(s, e)
        self.cur_frame = s
        self.slider.setValue(s)
        self._read_frame(self.cur_frame)
        self._show_frame()

    def _on_start_ov(self, val):
        if self._suppress_spin or self.cur_act < 0: return
        ov = self.overrides.setdefault(self.cur_act, {})
        ov["start"] = val
        self.clip_start = val
        self.slider.setRange(val, self.clip_end)
        self._update_act_label()
        self._auto_save()

    def _on_end_ov(self, val):
        if self._suppress_spin or self.cur_act < 0: return
        ov = self.overrides.setdefault(self.cur_act, {})
        ov["end"] = val
        self.clip_end = val
        self.slider.setRange(self.clip_start, val)
        self._update_act_label()
        self._auto_save()

    def _on_act_off(self, val):
        if self._suppress_spin or self.cur_act < 0: return
        self.overrides.setdefault(self.cur_act, {})["offset"] = val
        self._show_frame()
        self._auto_save()

    def _on_scene_off(self, val):
        self.scene_offset = val
        self._show_frame()
        self._auto_save()

    def _on_cam(self, text):
        if text and text != self.active_cam and text != "(no cameras)":
            self._switch_cam(text)

    def _update_act_label(self):
        if self.cur_act < 0: return
        a = self.actions[self.cur_act]
        ov = self.overrides.get(self.cur_act, {})
        item = self.act_list.item(self.cur_act)
        if item: item.setText(make_label(a, ov))

    # =======================================================================
    #  Playback
    # =======================================================================
    def _on_slider(self, val):
        self.cur_frame = val
        self._read_frame(val)
        self._show_frame()

    def _toggle_play(self):
        if self.playing:
            self.playing = False; self.timer.stop()
        else:
            if not self.cap and self.pts3d is None: return
            self.playing = True
            if self.cap:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cur_frame)
            self.timer.start(int(1000 / self.vfps))

    def _tick(self):
        if self.cur_frame >= self.clip_end:
            if self.loop_playback:
                # Loop: jump back to clip start
                self.cur_frame = self.clip_start
                if self.cap and self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cur_frame)
                self._read_frame(self.cur_frame)
                self.slider.blockSignals(True)
                self.slider.setValue(self.cur_frame)
                self.slider.blockSignals(False)
                self._show_frame()
                return
            else:
                self.playing = False; self.timer.stop(); return
        self.cur_frame += 1
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self._cached_frame_idx = self.cur_frame
                self._cached_frame = frame
        else:
            self._cached_frame_idx = self.cur_frame
        self.slider.blockSignals(True)
        self.slider.setValue(self.cur_frame)
        self.slider.blockSignals(False)
        self._show_frame()

    def _prev(self):
        if self.playing: self._toggle_play()
        if self.cur_frame > self.clip_start:
            self.cur_frame -= 1
            self._read_frame(self.cur_frame)
            self.slider.setValue(self.cur_frame)

    def _nxt(self):
        if self.playing: self._toggle_play()
        if self.cur_frame < self.clip_end:
            self.cur_frame += 1
            self._read_frame(self.cur_frame)
            self.slider.setValue(self.cur_frame)

    def _jump(self, secs):
        if self.playing: self._toggle_play()
        delta = int(secs * self.vfps)
        nf = max(self.clip_start, min(self.clip_end, self.cur_frame + delta))
        self.cur_frame = nf
        self._read_frame(nf)
        self.slider.setValue(nf)

    # =======================================================================
    #  Rendering
    # =======================================================================
    def _show_frame(self):
        raw = self._cached_frame
        if raw is None: return
        need_skel = (self.show_skel and self.pts3d is not None
                     and self.active_cam and self.active_cam in self.calibs)
        # Only copy if we'll draw on the frame (avoids ~3ms copy per frame)
        frame = raw.copy() if need_skel else raw
        # project 3D skeleton
        if need_skel:
            intr, extr = self.calibs[self.active_cam]
            total_off = self.scene_offset + self._get_effective_act_offset(self.cur_act)
            pidx = v2p(self.cur_frame, self.vfps, self.pfps,
                       self.pts3d.shape[0], total_off)
            pts = self.pts3d[pidx]
            # Skip if frame is all zeros (precomputed mask)
            if self.pts3d_valid is not None and self.pts3d_valid[pidx]:
                proj = project_pts(pts, intr, extr,
                                   self.flip[0], self.flip[1], self.flip[2])
                if proj is not None:
                    draw_skel(frame, proj)
        # convert + display
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bpl = ch * w  # bytes per line
        qimg = QImage(rgb.data, w, h, bpl, QImage.Format_RGB888)
        # Use fast scaling during playback, smooth when paused
        transform = Qt.FastTransformation if self.playing else Qt.SmoothTransformation
        pix = QPixmap.fromImage(qimg).scaled(
            self.vid_lbl.size(), Qt.KeepAspectRatio, transform)
        self.vid_lbl.setPixmap(pix)
        # info
        vf = self.vfps if self.vfps > 0 else 30.0
        t_cur = self.cur_frame / vf
        t_tot = self.vtotal / vf if self.vtotal > 0 else 0
        t_cs = self.clip_start / vf
        t_ce = self.clip_end / vf
        scene_tag = f"[{self.cur_scene}]  " if self.cur_scene else ""
        self.info_lbl.setText(
            f"{scene_tag}Frame: {self.cur_frame} / {self.vtotal}   "
            f"Time: {fmt_time(t_cur)} / {fmt_time(t_tot)}   "
            f"Clip: {self.clip_start}-{self.clip_end} "
            f"({fmt_time(t_cs)}-{fmt_time(t_ce)})"
        )

    # =======================================================================
    #  Keyboard
    # =======================================================================
    def keyPressEvent(self, event):
        k = event.key()
        if k == Qt.Key_Space: self._toggle_play()
        elif k == Qt.Key_A: self._prev()
        elif k == Qt.Key_D: self._nxt()
        elif k == Qt.Key_Q: self._jump(-1)
        elif k == Qt.Key_E: self._jump(1)
        elif k == Qt.Key_W: self.scene_off_spin.setValue(self.scene_off_spin.value() + 1)
        elif k == Qt.Key_S: self.scene_off_spin.setValue(self.scene_off_spin.value() - 1)
        elif k == Qt.Key_Up:
            r = max(0, self.act_list.currentRow() - 1)
            self.act_list.setCurrentRow(r)
        elif k == Qt.Key_Down:
            r = min(len(self.actions) - 1, self.act_list.currentRow() + 1)
            self.act_list.setCurrentRow(r)
        else: super().keyPressEvent(event)

    # =======================================================================
    #  Export
    # =======================================================================
    def _export(self, all_actions, single_cam=False):
        if not self.actions:
            QMessageBox.warning(self, "Warning", "No actions loaded."); return
        if not self.video_folder and self.pts3d is None:
            QMessageBox.warning(self, "Warning", "No data loaded."); return
        if single_cam and (not self.active_cam or self.active_cam == "(no cameras)"):
            QMessageBox.warning(self, "Warning", "No camera selected for single-cam export."); return
        out_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not out_dir: return
        indices = list(range(len(self.actions))) if all_actions else (
            [self.cur_act] if self.cur_act >= 0 else [])
        if not indices:
            QMessageBox.warning(self, "Warning", "No action selected."); return

        # Determine cameras to export
        if single_cam:
            export_cams = [self.active_cam]
        else:
            export_cams = self.avail_cams if self.avail_cams else ["virtual"]
        total_ops = len(indices) * len(export_cams)
        prog = QProgressDialog("Exporting...", "Cancel", 0, total_ops, self)
        prog.setWindowModality(Qt.WindowModal); prog.setMinimumDuration(0)
        op = 0
        for ai in indices:
            a = self.actions[ai]
            ov = self.overrides.get(ai, {})
            sf = ov.get("start", a["start"])
            ef = ov.get("end", a["end"])
            total_off = self.scene_offset + self._get_effective_act_offset(ai)
            safe = f"{ai:03d}_{a['action']}"
            if a.get("variant"): safe += f"_{a['variant']}"
            if a.get("rep"): safe += f"_{a['rep']}"
            safe = safe.replace(" ", "_").replace("/", "_")
            act_dir = os.path.join(out_dir, safe)
            os.makedirs(act_dir, exist_ok=True)
            # points csv
            if self.pts3d is not None:
                pi_s = v2p(sf, self.vfps, self.pfps, self.pts3d.shape[0], total_off)
                pi_e = v2p(ef, self.vfps, self.pfps, self.pts3d.shape[0], total_off)
                sl = self.pts3d[pi_s:pi_e+1]
                nj = sl.shape[1]
                cols = []
                for j in range(nj): cols.extend([f"{j}_x", f"{j}_y", f"{j}_z"])
                pd.DataFrame(sl.reshape(sl.shape[0], -1), columns=cols).to_csv(
                    os.path.join(act_dir, "points3d.csv"), index=False)
            # video per camera
            for cn in export_cams:
                if prog.wasCanceled(): break
                if cn == "virtual":
                    # Create a virtual black-bg export
                    w, h = 1280, 720
                    fps = self.vfps or 30.0
                    out_path = os.path.join(act_dir, "virtual.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                    for fi in range(sf, ef + 1):
                        frm = np.zeros((h, w, 3), dtype=np.uint8)
                        if self.pts3d is not None and self.calibs:
                            # Use first available calibration
                            for cal_cam, (intr, extr) in self.calibs.items():
                                pidx = v2p(fi, fps, self.pfps,
                                           self.pts3d.shape[0], total_off)
                                pts = self.pts3d[pidx]
                                if self.pts3d_valid is not None and self.pts3d_valid[pidx]:
                                    proj = project_pts(pts, intr, extr,
                                                       self.flip[0], self.flip[1], self.flip[2])
                                    if proj is not None and self.show_skel:
                                        draw_skel(frm, proj)
                                break
                        t = fi / fps
                        cv2.putText(frm, f"{fmt_time(t)} F:{fi}", (15, 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                        writer.write(frm)
                    writer.release()
                    op += 1; prog.setValue(op); continue

                vpath = None
                if self.video_folder:
                    for fn in os.listdir(self.video_folder):
                        if cn in fn.lower() and fn.lower().endswith(".mp4"):
                            vpath = os.path.join(self.video_folder, fn); break
                if not vpath: op += 1; prog.setValue(op); continue
                cap2 = cv2.VideoCapture(vpath)
                if not cap2.isOpened(): op += 1; prog.setValue(op); continue
                fps = cap2.get(cv2.CAP_PROP_FPS) or 30.0
                w = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out_path = os.path.join(act_dir, f"{cn}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                ie = self.calibs.get(cn)
                cap2.set(cv2.CAP_PROP_POS_FRAMES, sf)
                for fi in range(sf, ef + 1):
                    ret, frm = cap2.read()
                    if not ret: break
                    if self.pts3d is not None and ie:
                        intr, extr = ie
                        pidx = v2p(fi, fps, self.pfps,
                                   self.pts3d.shape[0], total_off)
                        pts = self.pts3d[pidx]
                        if self.pts3d_valid is not None and self.pts3d_valid[pidx]:
                            proj = project_pts(pts, intr, extr,
                                               self.flip[0], self.flip[1], self.flip[2])
                            if proj is not None and self.show_skel:
                                draw_skel(frm, proj)
                    t = fi / fps
                    cv2.putText(frm, f"{fmt_time(t)} F:{fi}", (15, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                    writer.write(frm)
                writer.release(); cap2.release()
                op += 1; prog.setValue(op)
                QCoreApplication.processEvents()
        prog.close()
        QMessageBox.information(self, "Done", f"Exported to {out_dir}")

    def closeEvent(self, event):
        # Flush any pending debounced save
        if self._save_timer.isActive():
            self._save_timer.stop()
        self._save_scene_state()
        if self.cap: self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ClipAnnotator()
    win.show()
    sys.exit(app.exec_())
