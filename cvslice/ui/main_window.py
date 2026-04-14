"""Main application window for CVSlice."""
import os
import cv2
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QListWidget,
    QFileDialog, QSlider, QSpinBox, QComboBox, QCheckBox,
    QProgressDialog, QMessageBox, QGroupBox, QFormLayout, QSizePolicy,
    QInputDialog, QMenu,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, QCoreApplication

from ..core.constants import CAMERA_NAMES, DEFAULT_POINTS_FPS
from ..core.utils import fmt_time, v2p, make_label
from ..io import (
    parse_excel_actions, load_all_calibrations,
    find_csv_for_scene, find_cameras_in_folder, load_csv_as_pts3d,
    annotations_path, load_annotations, save_annotations,
)
from ..vision import (
    project_pts, draw_skel, draw_skel_with_confidence, clear_projection_cache,
    unproject_2d_to_3d, get_camera_depth, extract_R_t, find_nearest_joint,
    AnchorSet, interpolate_anchors, apply_bulk_offset,
    compute_ray, triangulate_two_rays,
)
from .video_label import VideoLabel


class ClipAnnotator(QMainWindow):
    """Multi-view action clip annotator & exporter."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CVSlice - Multi-View Action Clip Tool")
        self.setGeometry(60, 60, 1440, 840)
        self.setFocusPolicy(Qt.StrongFocus)

        # Sources
        self.xlsx_path = None
        self.sheet_names: list[str] = []
        self.data_root = None
        self.cal_folder = None

        # Current scene state
        self.cur_scene = None
        self.actions: list[dict] = []
        self.cur_act = -1
        self.avail_cams: list[str] = []
        self.active_cam = None
        self.video_folder = None
        self.cap = None
        self.vfps = 30.0
        self.vtotal = 0
        self.pts3d = None
        self.pts3d_valid = None
        self.pts3d_was_nan = None   # (T, J) bool: True = originally NaN, now interpolated
        self.pfps = DEFAULT_POINTS_FPS
        self.calibs: dict = {}
        self.scene_offset = 0
        self.overrides: dict = {}
        self.cur_frame = 0
        self.clip_start = self.clip_end = 0
        self.playing = False
        self.show_skel = True
        self.flip = [False, False, False]
        self._suppress_spin = False
        self._cached_frame_idx = -1
        self._cached_frame = None
        self.loop_playback = True
        self._annotations: dict = {}
        self._pts3d_dirty = False   # True if user has manually edited 3D points

        # Joint drag state
        self._drag_joint = None     # index of joint being dragged
        self._drag_proj = None      # current 2D projections (J, 2) for hit-testing
        self._drag_pidx = None      # pts3d frame index of the dragged frame
        self._undo_stack: list[tuple[int, int, np.ndarray]] = []  # (pidx, joint, old_xyz)

        # Propagation state
        self._anchors = AnchorSet()
        self._selected_joint: int | None = None  # joint selected for propagation
        self._last_drag_delta: np.ndarray | None = None  # delta from last drag
        self._last_drag_joint: int | None = None  # joint of last drag

        # Two-view triangulation state
        self._pending_ray: dict | None = None  # {joint, pidx, cam, origin, direction, old_xyz}

        self._build_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self._save_timer = QTimer()
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)
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
        fm.addAction("Save Edited 3D Points...", self._save_edited_csv)
        edm = mb.addMenu("Edit")
        edm.addAction("Undo Joint Move (Ctrl+Z)", self._undo_joint_edit)
        em = mb.addMenu("Export")
        em.addAction("Export Current Clip (all cams)...",
                     lambda: self._export(False, single_cam=False))
        em.addAction("Export ALL Clips (all cams)...",
                     lambda: self._export(True, single_cam=False))
        em.addSeparator()
        em.addAction("Export Current Clip (current cam only)...",
                     lambda: self._export(False, single_cam=True))
        em.addAction("Export ALL Clips (current cam only)...",
                     lambda: self._export(True, single_cam=True))

        root = QWidget(); self.setCentralWidget(root)
        hl = QHBoxLayout(root); hl.setContentsMargins(4, 4, 4, 4)

        # ---- LEFT panel ----
        left = QWidget(); left.setFixedWidth(310)
        lv = QVBoxLayout(left); lv.setContentsMargins(0, 0, 0, 0)

        sg = QGroupBox("Sources"); sf = QFormLayout(sg)
        self.lbl_xlsx = QLabel("-"); self.lbl_data = QLabel("-"); self.lbl_cal = QLabel("-")
        sf.addRow("Excel:", self.lbl_xlsx)
        sf.addRow("Data:", self.lbl_data)
        sf.addRow("Calib:", self.lbl_cal)
        lv.addWidget(sg)

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
        center = QWidget(); cvl = QVBoxLayout(center); cvl.setContentsMargins(0, 0, 0, 0)
        self.vid_lbl = VideoLabel()
        self.vid_lbl.setAlignment(Qt.AlignCenter)
        self.vid_lbl.setMinimumSize(640, 400)
        self.vid_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.vid_lbl.setStyleSheet("background:black;")
        self.vid_lbl.setMouseTracking(True)
        self.vid_lbl.mouse_pressed.connect(self._on_mouse_press)
        self.vid_lbl.mouse_moved.connect(self._on_mouse_move)
        self.vid_lbl.mouse_released.connect(self._on_mouse_release)
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
        right = QWidget(); right.setFixedWidth(280)
        rv = QVBoxLayout(right); rv.setContentsMargins(0, 0, 0, 0)

        cg = QGroupBox("Camera"); cf = QFormLayout(cg)
        self.cam_combo = QComboBox()
        self.cam_combo.currentTextChanged.connect(self._on_cam)
        cf.addRow("View:", self.cam_combo)
        rv.addWidget(cg)

        og = QGroupBox("Sync (3D offset)"); of2 = QFormLayout(og)
        self.scene_off_spin = QSpinBox(); self.scene_off_spin.setRange(-50000, 50000)
        self.scene_off_spin.valueChanged.connect(self._on_scene_off)
        of2.addRow("Scene:", self.scene_off_spin)
        self.act_off_spin = QSpinBox(); self.act_off_spin.setRange(-50000, 50000)
        self.act_off_spin.valueChanged.connect(self._on_act_off)
        of2.addRow("Action:", self.act_off_spin)
        rv.addWidget(og)

        ag = QGroupBox("Action Override"); af = QFormLayout(ag)
        self.start_spin = QSpinBox(); self.start_spin.setRange(0, 9999999)
        self.start_spin.valueChanged.connect(self._on_start_ov)
        af.addRow("Start:", self.start_spin)
        self.end_spin = QSpinBox(); self.end_spin.setRange(0, 9999999)
        self.end_spin.valueChanged.connect(self._on_end_ov)
        af.addRow("End:", self.end_spin)
        rv.addWidget(ag)

        fg = QGroupBox("Display"); ff = QFormLayout(fg)
        self.skel_cb = QCheckBox("Show Skeleton"); self.skel_cb.setChecked(True)
        self.skel_cb.stateChanged.connect(
            lambda s: setattr(self, "show_skel", s == Qt.Checked) or self._show_frame())
        ff.addRow(self.skel_cb)
        self.loop_cb = QCheckBox("Loop Playback"); self.loop_cb.setChecked(True)
        self.loop_cb.stateChanged.connect(
            lambda s: setattr(self, "loop_playback", s == Qt.Checked))
        ff.addRow(self.loop_cb)
        self.edit_cb = QCheckBox("Edit Mode (drag joints)")
        self.edit_cb.setChecked(False)
        self.edit_cb.setToolTip(
            "Enable to click and drag skeleton joints.\n"
            "Edits modify the 3D points in memory.\n"
            "Use Ctrl+Z to undo, File > Save Edited 3D Points to export.")
        ff.addRow(self.edit_cb)
        self.tri_cb = QCheckBox("Triangulate (2-view)")
        self.tri_cb.setChecked(False)
        self.tri_cb.setToolTip(
            "Two-view triangulation for accurate 3D editing:\n"
            "1. Drag joint to desired 2D position in camera A\n"
            "2. Switch to camera B, drag same joint again\n"
            "3. The true 3D position is computed from both views\n\n"
            "This corrects depth (Z) which single-view drag cannot.")
        self.tri_cb.stateChanged.connect(self._on_tri_toggled)
        ff.addRow(self.tri_cb)
        self.tri_status_lbl = QLabel("")
        self.tri_status_lbl.setStyleSheet("font-size:11px; color:#888;")
        self.tri_status_lbl.setWordWrap(True)
        ff.addRow(self.tri_status_lbl)
        for axis, idx in [("Flip X", 0), ("Flip Y", 1), ("Flip Z", 2)]:
            cb = QCheckBox(axis)
            cb.stateChanged.connect(lambda s, i=idx: self._set_flip(i, s == Qt.Checked))
            ff.addRow(cb)
        rv.addWidget(fg)

        # ---- Propagation panel ----
        pg = QGroupBox("Propagate Edits")
        pfl = QVBoxLayout(pg)

        # Joint selector — click-to-select (shown as read-only label)
        jrow = QHBoxLayout()
        jrow.addWidget(QLabel("Joint:"))
        self.joint_lbl = QLabel("(click to select)")
        self.joint_lbl.setStyleSheet(
            "font-weight:bold; padding:2px 6px; "
            "background:#2a2a2a; border:1px solid #555; border-radius:3px;")
        self.joint_lbl.setToolTip("Click a joint in the view to select it for propagation")
        jrow.addWidget(self.joint_lbl)
        pfl.addLayout(jrow)

        # Anchor controls
        arow = QHBoxLayout()
        self.set_anchor_btn = QPushButton("Set Anchor")
        self.set_anchor_btn.setToolTip(
            "Mark current frame as anchor for the selected joint.\n"
            "Drag the joint first, then click Set Anchor.")
        self.set_anchor_btn.clicked.connect(self._set_anchor)
        arow.addWidget(self.set_anchor_btn)
        self.del_anchor_btn = QPushButton("Del Anchor")
        self.del_anchor_btn.setToolTip("Remove anchor at current frame")
        self.del_anchor_btn.clicked.connect(self._del_anchor)
        arow.addWidget(self.del_anchor_btn)
        pfl.addLayout(arow)

        # Anchor list
        self.anchor_list = QListWidget()
        self.anchor_list.setMaximumHeight(100)
        self.anchor_list.setToolTip("Anchors: joint@frame")
        self.anchor_list.currentRowChanged.connect(self._on_anchor_selected)
        pfl.addWidget(self.anchor_list)

        # Range for interpolation
        rform = QFormLayout()
        self.prop_start_spin = QSpinBox()
        self.prop_start_spin.setRange(0, 9999999)
        rform.addRow("From:", self.prop_start_spin)
        self.prop_end_spin = QSpinBox()
        self.prop_end_spin.setRange(0, 9999999)
        rform.addRow("To:", self.prop_end_spin)
        pfl.addLayout(rform)

        # Method selector
        mrow = QHBoxLayout()
        mrow.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["spline", "linear"])
        mrow.addWidget(self.method_combo)
        pfl.addLayout(mrow)

        # Apply buttons
        self.apply_interp_btn = QPushButton("Apply Interpolation")
        self.apply_interp_btn.setToolTip(
            "Interpolate between anchors across the frame range")
        self.apply_interp_btn.clicked.connect(self._apply_interpolation)
        pfl.addWidget(self.apply_interp_btn)

        # Bulk offset
        orow = QHBoxLayout()
        orow.addWidget(QLabel("Taper:"))
        self.taper_combo = QComboBox()
        self.taper_combo.addItems(["none", "linear", "cosine"])
        orow.addWidget(self.taper_combo)
        pfl.addLayout(orow)
        self.apply_offset_btn = QPushButton("Apply Last Drag as Offset")
        self.apply_offset_btn.setToolTip(
            "Take the delta from the last single-frame drag and apply it\n"
            "across the frame range with optional tapering.")
        self.apply_offset_btn.clicked.connect(self._apply_bulk_offset)
        pfl.addWidget(self.apply_offset_btn)

        # Clear
        self.clear_anchors_btn = QPushButton("Clear All Anchors")
        self.clear_anchors_btn.clicked.connect(self._clear_anchors)
        pfl.addWidget(self.clear_anchors_btn)

        rv.addWidget(pg)

        # ---- Help button ----
        self.help_btn = QPushButton("❓ Editing Guide")
        self.help_btn.setToolTip("Show how to use the 3D joint editing tools")
        self.help_btn.clicked.connect(self._show_editing_help)
        rv.addWidget(self.help_btn)

        rv.addStretch()
        hl.addWidget(right)
        self.statusBar().showMessage(
            "Space=Play  A/D=Prev/Next  Q/E=-/+1s  W/S=SceneOffset  Up/Down=Action")

    def _set_flip(self, idx, val):
        self.flip[idx] = val; self._show_frame()

    def _on_tri_toggled(self, state):
        if state != Qt.Checked:
            self._pending_ray = None
            self.tri_status_lbl.setText("")

    def _show_editing_help(self):
        """Show a dialog with editing instructions."""
        text = (
            "<h3>单帧拖拽编辑</h3>"
            "<ol>"
            "<li>勾选 <b>Edit Mode</b></li>"
            "<li>点击关节并拖拽到目标位置（深度保持不变）</li>"
            "<li>Ctrl+Z 撤销</li>"
            "</ol>"
            "<h3>双视角三角化（修正深度）</h3>"
            "<ol>"
            "<li>勾选 <b>Edit Mode</b> + <b>Triangulate (2-view)</b></li>"
            "<li>在相机 A 拖拽关节到正确的 2D 位置 → 记录射线 1（青色 T1 标记）</li>"
            "<li>切换到相机 B，拖拽同一关节 → 自动三角化得到精确 3D</li>"
            "<li>提示：选择夹角大的两个相机效果最好</li>"
            "</ol>"
            "<h3>锚点插值（修正连续偏移）</h3>"
            "<ol>"
            "<li>跳到偏移起始帧，拖拽关节到正确位置 → 点 <b>Set Anchor</b></li>"
            "<li>跳到偏移结束帧，拖拽同一关节 → 点 <b>Set Anchor</b></li>"
            "<li>可以设置多个锚点，插值会依次经过每个锚点</li>"
            "<li>选择 Method (spline/linear) → 点 <b>Apply Interpolation</b></li>"
            "</ol>"
            "<h3>批量偏移（整段平移）</h3>"
            "<ol>"
            "<li>在任意一帧拖拽关节到正确位置（记录 delta）</li>"
            "<li>设置 From/To 帧范围</li>"
            "<li>选择 Taper: none(均匀) / linear(线性衰减) / cosine(余弦衰减)</li>"
            "<li>点 <b>Apply Last Drag as Offset</b></li>"
            "</ol>"
            "<h3>快捷键</h3>"
            "<ul>"
            "<li><b>Space</b> — 播放/暂停</li>"
            "<li><b>A/D</b> — 前/后一帧</li>"
            "<li><b>Q/E</b> — 前/后 1 秒</li>"
            "<li><b>Ctrl+Z</b> — 撤销</li>"
            "</ul>"
        )
        QMessageBox.information(self, "Editing Guide", text)

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
        self.sheet_names = [s for s in xl.sheet_names if s != "1A2B"]
        self._annotations = load_annotations(path)
        self.lbl_xlsx.setText(os.path.basename(path))
        self.scene_combo.blockSignals(True)
        self.scene_combo.clear()
        self.scene_combo.addItems(self.sheet_names)
        self.scene_combo.blockSignals(False)
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
        if self.cur_scene:
            self._apply_scene(self.cur_scene)
        self.statusBar().showMessage(f"Data root: {folder}")

    def _load_cal(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Calibration Folder")
        if not folder: return
        self.cal_folder = folder
        self.lbl_cal.setText(os.path.basename(folder))
        self.calibs = load_all_calibrations(folder)
        clear_projection_cache()
        self.statusBar().showMessage(
            f"Loaded calibration for {len(self.calibs)} cameras")
        self._show_frame()

    # =======================================================================
    #  Scene switching
    # =======================================================================
    def _on_scene_changed(self, scene_name):
        if not scene_name: return
        self._save_scene_state()
        self._apply_scene(scene_name)

    def _apply_scene(self, scene_name):
        self.cur_scene = scene_name
        if self.xlsx_path:
            self.actions = parse_excel_actions(self.xlsx_path, scene_name)
        else:
            self.actions = []

        csv_path = None
        self.video_folder = None
        self.pts3d = None
        self.pts3d_valid = None
        self.pts3d_was_nan = None
        self.avail_cams = []
        self._anchors.clear_all()
        self._last_drag_delta = None
        self._last_drag_joint = None
        self._pending_ray = None

        if self.data_root:
            csv_path, subfolder = find_csv_for_scene(self.data_root, scene_name)
            if subfolder:
                self.video_folder = subfolder
                self.avail_cams = find_cameras_in_folder(subfolder)
            if not self.avail_cams:
                root_cams = find_cameras_in_folder(self.data_root)
                if root_cams:
                    self.video_folder = self.data_root
                    self.avail_cams = root_cams

        if csv_path:
            result = load_csv_as_pts3d(csv_path)
            if result[0] is not None:
                self.pts3d, self.pts3d_valid, self.pts3d_was_nan = result

        self.cam_combo.blockSignals(True)
        self.cam_combo.clear()
        if self.avail_cams:
            self.cam_combo.addItems(self.avail_cams)
        else:
            self.cam_combo.addItem("(no cameras)")
        self.cam_combo.blockSignals(False)

        self.overrides = {}
        self.scene_offset = 0
        saved = self._annotations.get(scene_name, {})
        if saved:
            self.scene_offset = saved.get("scene_offset", 0)
            saved_ov = saved.get("overrides", {})
            self.overrides = {int(k): v for k, v in saved_ov.items()}

        self._suppress_spin = True
        self.scene_off_spin.setValue(self.scene_offset)
        self._suppress_spin = False

        info_parts = []
        if csv_path:
            info_parts.append(f"CSV: {os.path.basename(csv_path)}")
            if self.pts3d is not None:
                info_parts.append(
                    f"({self.pts3d.shape[0]} frames, {self.pts3d.shape[1]} joints)")
        else:
            info_parts.append("CSV: not found")
        info_parts.append(f"Cameras: {len(self.avail_cams)}")
        info_parts.append(f"Actions: {len(self.actions)}")
        self.lbl_scene_info.setText("  |  ".join(info_parts))

        self._refresh_act_list()
        self._refresh_anchor_list()
        if self.cap:
            self.cap.release(); self.cap = None
        self._cached_frame_idx = -1
        self._cached_frame = None
        self.active_cam = None

        if self.avail_cams:
            self._switch_cam(self.avail_cams[0])
        else:
            self.vfps = 30.0
            if self.pts3d is not None:
                self.vtotal = int(self.pts3d.shape[0] * (30.0 / DEFAULT_POINTS_FPS))
            else:
                self.vtotal = 0
            self._cached_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            self._cached_frame_idx = 0

        self._estimate_pfps()
        if self.actions:
            self.act_list.setCurrentRow(0)
        else:
            self.cur_act = -1
            self._show_frame()

        self.statusBar().showMessage(
            f"Scene: {scene_name} -- {len(self.actions)} actions, "
            f"{len(self.avail_cams)} cameras, "
            f"CSV={'yes' if csv_path else 'no'}")

    # =======================================================================
    #  Annotations save/load
    # =======================================================================
    def _save_scene_state(self):
        if not self.cur_scene or not self.xlsx_path: return
        scene_data = {
            "scene_offset": self.scene_offset,
            "overrides": {str(k): v for k, v in self.overrides.items()},
        }
        self._annotations[self.cur_scene] = scene_data
        save_annotations(self.xlsx_path, self._annotations)

    def _save_current_annotations(self):
        if self._save_timer.isActive():
            self._save_timer.stop()
        self._save_scene_state()
        if self.xlsx_path:
            QMessageBox.information(
                self, "Saved",
                f"Offsets saved to:\n{annotations_path(self.xlsx_path)}")

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
            if self._cached_frame is None:
                self._cached_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            self._cached_frame_idx = frame_idx
            return self._cached_frame
        if frame_idx == self._cached_frame_idx + 1:
            ret, frame = self.cap.read()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
        if ret:
            self._cached_frame_idx = frame_idx
            self._cached_frame = frame
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
        new_start, ok1 = QInputDialog.getInt(
            self, "New repetition",
            f"Start frame for new repetition of \"{a['action']}\":",
            value=a["end"] + 1, min=0, max=9999999)
        if not ok1: return
        new_end, ok2 = QInputDialog.getInt(
            self, "New repetition", "End frame:",
            value=new_start + (a["end"] - a["start"]),
            min=new_start + 1, max=9999999)
        if not ok2: return
        rep_count = sum(1 for aa in self.actions
                        if aa["action"] == a["action"]
                        and aa.get("variant", "") == a.get("variant", ""))
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
        reply = QMessageBox.question(
            self, "Confirm", f"Delete \"{make_label(a)}\"?",
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
        self._save_timer.start()

    def _do_save(self):
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
        ov["start"] = val; self.clip_start = val
        self.slider.setRange(val, self.clip_end)
        self._update_act_label(); self._auto_save()

    def _on_end_ov(self, val):
        if self._suppress_spin or self.cur_act < 0: return
        ov = self.overrides.setdefault(self.cur_act, {})
        ov["end"] = val; self.clip_end = val
        self.slider.setRange(self.clip_start, val)
        self._update_act_label(); self._auto_save()

    def _on_act_off(self, val):
        if self._suppress_spin or self.cur_act < 0: return
        self.overrides.setdefault(self.cur_act, {})["offset"] = val
        self._show_frame(); self._auto_save()

    def _on_scene_off(self, val):
        self.scene_offset = val
        self._show_frame(); self._auto_save()

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
        frame = raw.copy() if need_skel else raw
        self._drag_proj = None  # reset projection cache
        self._drag_pidx = None
        if need_skel:
            intr, extr = self.calibs[self.active_cam]
            total_off = self.scene_offset + self._get_effective_act_offset(self.cur_act)
            pidx = v2p(self.cur_frame, self.vfps, self.pfps,
                       self.pts3d.shape[0], total_off)
            pts = self.pts3d[pidx]
            if self.pts3d_valid is not None and self.pts3d_valid[pidx]:
                proj = project_pts(pts, intr, extr,
                                   self.flip[0], self.flip[1], self.flip[2])
                if proj is not None:
                    nan_mask = None
                    if self.pts3d_was_nan is not None:
                        nan_mask = self.pts3d_was_nan[pidx]
                    draw_skel_with_confidence(frame, proj, nan_mask)
                    # Highlight dragged joint
                    if self._drag_joint is not None and self._drag_joint < len(proj):
                        jx, jy = int(proj[self._drag_joint][0]), int(proj[self._drag_joint][1])
                        cv2.circle(frame, (jx, jy), 8, (0, 255, 0), 2)
                    # Highlight pending triangulation joint (cyan + "T1")
                    if (self._pending_ray is not None
                            and self._pending_ray["pidx"] == pidx
                            and self._pending_ray["joint"] < len(proj)):
                        tj = self._pending_ray["joint"]
                        tx, ty = int(proj[tj][0]), int(proj[tj][1])
                        cv2.circle(frame, (tx, ty), 12, (255, 255, 0), 2)  # cyan in BGR
                        cv2.putText(frame, "T1", (tx + 14, ty - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 0), 2)
                    # Highlight anchor joints at this frame
                    for aj in self._anchors.all_joints():
                        if pidx in self._anchors.get_anchors(aj) and aj < len(proj):
                            ax, ay = int(proj[aj][0]), int(proj[aj][1])
                            cv2.circle(frame, (ax, ay), 10, (255, 0, 255), 2)  # magenta
                            cv2.putText(frame, "A", (ax + 12, ay - 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (255, 0, 255), 1)
                    self._drag_proj = proj
                    self._drag_pidx = pidx
        # Update video label frame size for coordinate mapping
        h_f, w_f = (frame.shape[0], frame.shape[1])
        self.vid_lbl.set_frame_size(w_f, h_f)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bpl = ch * w
        qimg = QImage(rgb.data, w, h, bpl, QImage.Format_RGB888)
        transform = Qt.FastTransformation if self.playing else Qt.SmoothTransformation
        pix = QPixmap.fromImage(qimg).scaled(
            self.vid_lbl.size(), Qt.KeepAspectRatio, transform)
        self.vid_lbl.setPixmap(pix)
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
            f"({fmt_time(t_cs)}-{fmt_time(t_ce)})")

    # =======================================================================
    #  Joint dragging (Edit Mode)
    # =======================================================================
    def _on_mouse_press(self, fx, fy):
        """Mouse pressed on video at frame coords (fx, fy)."""
        if not self.edit_cb.isChecked():
            return
        if self._drag_proj is None or self._drag_pidx is None:
            return
        if self.playing:
            self._toggle_play()  # pause during editing
        joint = find_nearest_joint(fx, fy, self._drag_proj)
        if joint is not None:
            # Always update selected joint for propagation panel
            self._selected_joint = joint
            self.joint_lbl.setText(f"Joint {joint}")
            self._drag_joint = joint
            # Save old position for undo BEFORE any move
            pidx = self._drag_pidx
            self._undo_stack.append(
                (pidx, joint, self.pts3d[pidx, joint].copy()))
            self.statusBar().showMessage(f"Dragging joint {joint}...")
            self._show_frame()  # highlight selected joint

    def _on_mouse_move(self, fx, fy):
        """Mouse moved during drag — live preview of joint position."""
        if self._drag_joint is None or not self.edit_cb.isChecked():
            return
        if self._drag_pidx is None or self.pts3d is None:
            return
        if not self.active_cam or self.active_cam not in self.calibs:
            return
        intr, extr = self.calibs[self.active_cam]
        Rt = extract_R_t(extr)
        if Rt is None:
            return
        R, t = Rt
        K = np.array(intr["camera_matrix"], dtype=np.float64)

        pidx = self._drag_pidx
        j = self._drag_joint
        old_pt = self.pts3d[pidx, j].copy()

        # Apply flip to get the point as it was projected
        pt_flip = old_pt.copy()
        if self.flip[0]: pt_flip[0] *= -1
        if self.flip[1]: pt_flip[1] *= -1
        if self.flip[2]: pt_flip[2] *= -1

        z_cam = get_camera_depth(pt_flip, R, t)
        if z_cam <= 0:
            return  # point is behind camera

        new_pt_flip = unproject_2d_to_3d(fx, fy, z_cam, K, R, t)

        # Un-flip to get back to storage space
        new_pt = new_pt_flip.copy()
        if self.flip[0]: new_pt[0] *= -1
        if self.flip[1]: new_pt[1] *= -1
        if self.flip[2]: new_pt[2] *= -1

        self.pts3d[pidx, j] = new_pt
        self._show_frame()

    def _on_mouse_release(self, fx, fy):
        """Mouse released — finalize the drag and record delta."""
        if self._drag_joint is None:
            return
        if self._drag_pidx is not None and self.pts3d is not None:
            joint = self._drag_joint
            pidx = self._drag_pidx
            new_xyz = self.pts3d[pidx, joint].copy()

            # Compute and store delta for bulk offset
            if self._undo_stack:
                last_entry = self._undo_stack[-1]
                if last_entry[0] != 'range':
                    old_xyz = last_entry[2]
                    self._last_drag_delta = new_xyz - old_xyz
                    self._last_drag_joint = joint

            # --- Triangulation mode ---
            if self.tri_cb.isChecked() and self.edit_cb.isChecked():
                cam = self.active_cam
                if cam and cam in self.calibs:
                    intr, extr = self.calibs[cam]
                    Rt = extract_R_t(extr)
                    if Rt is not None:
                        R, t = Rt
                        K = np.array(intr["camera_matrix"], dtype=np.float64)
                        origin, direction = compute_ray(float(fx), float(fy), K, R, t)

                        if (self._pending_ray is not None
                                and self._pending_ray["joint"] == joint
                                and self._pending_ray["pidx"] == pidx
                                and self._pending_ray["cam"] != cam):
                            # Second view — triangulate!
                            r1 = self._pending_ray
                            pt3d = triangulate_two_rays(
                                r1["origin"], r1["direction"],
                                origin, direction)

                            # Apply flip inversion if needed
                            if self.flip[0]: pt3d[0] *= -1
                            if self.flip[1]: pt3d[1] *= -1
                            if self.flip[2]: pt3d[2] *= -1

                            # Revert the single-view edit, replace with triangulated
                            self.pts3d[pidx, joint] = pt3d
                            # Update undo: replace last entry's "new" with triangulated
                            # (old_xyz from first ray is the true original)
                            if self._undo_stack:
                                self._undo_stack.pop()  # remove view-2 undo
                            # Keep view-1 undo entry (has original old_xyz)
                            # but update: the undo should restore to original
                            # The first ray's undo entry already has the right old_xyz

                            self._pending_ray = None
                            self.tri_status_lbl.setText("")
                            self._pts3d_dirty = True
                            self.statusBar().showMessage(
                                f"Triangulated joint {joint} at frame {pidx} "
                                f"from {r1['cam']} + {cam}. Ctrl+Z to undo.")
                            self._drag_joint = None
                            self._show_frame()
                            return
                        else:
                            # First view — record ray, revert 3D edit
                            # Restore original position (don't apply single-view edit)
                            if self._undo_stack:
                                entry = self._undo_stack[-1]
                                if entry[0] != 'range':
                                    orig_xyz = entry[2]
                                    self.pts3d[pidx, joint] = orig_xyz.copy()

                            self._pending_ray = {
                                "joint": joint, "pidx": pidx, "cam": cam,
                                "origin": origin, "direction": direction,
                                "old_xyz": self._undo_stack[-1][2].copy() if self._undo_stack else new_xyz,
                            }
                            self.tri_status_lbl.setText(
                                f"✔ Ray 1: J{joint} @ {cam}\n"
                                f"Switch camera, drag same joint")
                            self.statusBar().showMessage(
                                f"Ray 1 recorded for joint {joint} from {cam}. "
                                f"Now switch to another camera and drag the same joint.")
                            self._drag_joint = None
                            self._show_frame()
                            return

            # --- Normal (single-view) mode ---
            self._pts3d_dirty = True
            self.statusBar().showMessage(
                f"Joint {joint} moved at pts3d frame {pidx}. "
                f"Ctrl+Z to undo. {len(self._undo_stack)} edits in stack.")
        self._drag_joint = None
        self._show_frame()

    def _undo_joint_edit(self):
        """Undo the last joint edit (single point or range)."""
        if not self._undo_stack:
            self.statusBar().showMessage("Nothing to undo.")
            return
        entry = self._undo_stack.pop()
        if entry[0] == 'range':
            # Range undo: ('range', joint, fs, fe, old_data)
            _, joint, fs, fe, old_data = entry
            if self.pts3d is not None:
                self.pts3d[fs:fe + 1, joint] = old_data
                self._show_frame()
                self.statusBar().showMessage(
                    f"Undone: joint {joint} range [{fs}-{fe}] restored. "
                    f"{len(self._undo_stack)} edits remaining.")
        else:
            # Single point undo: (pidx, joint, old_xyz)
            pidx, joint, old_xyz = entry
            if self.pts3d is not None and pidx < self.pts3d.shape[0]:
                self.pts3d[pidx, joint] = old_xyz
                self._show_frame()
                self.statusBar().showMessage(
                    f"Undone: joint {joint} at frame {pidx} restored. "
                    f"{len(self._undo_stack)} edits remaining.")

    def _save_edited_csv(self):
        """Save the (possibly edited) 3D points to a new CSV file."""
        if self.pts3d is None:
            QMessageBox.warning(self, "Warning", "No 3D point data loaded.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Edited 3D Points", "", "CSV Files (*.csv)")
        if not path:
            return
        T, J, _ = self.pts3d.shape
        cols = []
        for j in range(J):
            cols.extend([f"{j}_x", f"{j}_y", f"{j}_z"])
        flat = self.pts3d.reshape(T, -1)
        pd.DataFrame(flat, columns=cols).to_csv(path, index=False)
        self._pts3d_dirty = False
        QMessageBox.information(self, "Saved", f"3D points saved to:\n{path}")

    # =======================================================================
    #  Propagation (multi-frame editing)
    # =======================================================================
    def _refresh_anchor_list(self):
        """Refresh the anchor list widget."""
        self.anchor_list.clear()
        for joint, frame, xyz in self._anchors.summary():
            self.anchor_list.addItem(
                f"J{joint} @ F{frame}  ({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f})")

    def _set_anchor(self):
        """Set an anchor at the current frame for the selected joint."""
        if self.pts3d is None:
            QMessageBox.warning(self, "Warning", "No 3D data loaded."); return
        if self._drag_pidx is None:
            # Compute pidx from current frame
            total_off = self.scene_offset + self._get_effective_act_offset(self.cur_act)
            pidx = v2p(self.cur_frame, self.vfps, self.pfps,
                       self.pts3d.shape[0], total_off)
        else:
            pidx = self._drag_pidx
        joint = self._selected_joint
        if joint is None:
            QMessageBox.warning(self, "Warning",
                                "No joint selected. Click a joint in the view first.")
            return
        if joint >= self.pts3d.shape[1]:
            QMessageBox.warning(self, "Warning",
                                f"Joint {joint} out of range (max {self.pts3d.shape[1]-1}).")
            return
        xyz = self.pts3d[pidx, joint].copy()
        self._anchors.set_anchor(joint, pidx, xyz)
        self._refresh_anchor_list()
        # Auto-update range to cover all anchors for this joint
        anchors = self._anchors.get_anchors(joint)
        if anchors:
            frames = sorted(anchors.keys())
            self.prop_start_spin.setValue(max(0, frames[0]))
            self.prop_end_spin.setValue(min(self.pts3d.shape[0] - 1, frames[-1]))
        self.statusBar().showMessage(
            f"Anchor set: joint {joint} at pts3d frame {pidx}")
        self._show_frame()

    def _del_anchor(self):
        """Remove anchor at current frame for selected joint."""
        if self.pts3d is None: return
        total_off = self.scene_offset + self._get_effective_act_offset(self.cur_act)
        pidx = v2p(self.cur_frame, self.vfps, self.pfps,
                   self.pts3d.shape[0], total_off)
        joint = self._selected_joint
        if joint is None:
            QMessageBox.warning(self, "Warning",
                                "No joint selected. Click a joint in the view first.")
            return
        self._anchors.remove_anchor(joint, pidx)
        self._refresh_anchor_list()
        self.statusBar().showMessage(
            f"Anchor removed: joint {joint} at frame {pidx}")

    def _on_anchor_selected(self, row):
        """Jump to the frame of the selected anchor."""
        summary = self._anchors.summary()
        if row < 0 or row >= len(summary): return
        joint, frame, _ = summary[row]
        self._selected_joint = joint
        self.joint_lbl.setText(f"Joint {joint}")
        # Convert pts3d frame back to video frame (approximate inverse of v2p)
        if self.pts3d is not None and self.pfps > 0 and self.vfps > 0:
            total_off = self.scene_offset + self._get_effective_act_offset(self.cur_act)
            vframe = int(round(frame * (self.vfps / self.pfps) - total_off))
            vframe = max(self.clip_start, min(self.clip_end, vframe))
            self.cur_frame = vframe
            self._read_frame(vframe)
            self.slider.setValue(vframe)

    def _apply_interpolation(self):
        """Apply anchor-based interpolation across the frame range."""
        if self.pts3d is None:
            QMessageBox.warning(self, "Warning", "No 3D data loaded."); return
        joint = self._selected_joint
        if joint is None:
            QMessageBox.warning(self, "Warning",
                                "No joint selected. Click a joint in the view first.")
            return
        anchors = self._anchors.get_anchors(joint)
        if not anchors:
            QMessageBox.warning(self, "Warning",
                                f"No anchors set for joint {joint}.\n"
                                "Drag the joint to the desired position, then click Set Anchor.")
            return
        fs = self.prop_start_spin.value()
        fe = self.prop_end_spin.value()
        if fs >= fe:
            QMessageBox.warning(self, "Warning", "Start frame must be < end frame."); return
        method = self.method_combo.currentText()

        # Save undo for the entire range
        old_data = self.pts3d[fs:fe + 1, joint].copy()
        self._undo_stack.append(('range', joint, fs, fe, old_data))

        new_positions = interpolate_anchors(
            self.pts3d, joint, anchors, fs, fe, method=method)
        self.pts3d[fs:fe + 1, joint] = new_positions
        self._pts3d_dirty = True
        n_frames = fe - fs + 1
        self.statusBar().showMessage(
            f"Interpolated joint {joint} across {n_frames} frames "
            f"({len(anchors)} anchors, {method}). Ctrl+Z to undo.")
        self._show_frame()

    def _apply_bulk_offset(self):
        """Apply the last drag delta across the frame range."""
        if self.pts3d is None:
            QMessageBox.warning(self, "Warning", "No 3D data loaded."); return
        if self._last_drag_delta is None:
            QMessageBox.warning(self, "Warning",
                                "No drag recorded yet.\n"
                                "Drag a joint first, then use this button.")
            return
        joint = self._selected_joint
        if joint is None:
            QMessageBox.warning(self, "Warning",
                                "No joint selected. Click a joint in the view first.")
            return
        if self._last_drag_joint is not None and joint != self._last_drag_joint:
            reply = QMessageBox.question(
                self, "Joint Mismatch",
                f"Last drag was on joint {self._last_drag_joint}, "
                f"but propagation target is joint {joint}.\n"
                "Apply anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
        fs = self.prop_start_spin.value()
        fe = self.prop_end_spin.value()
        if fs >= fe:
            QMessageBox.warning(self, "Warning", "Start frame must be < end frame."); return
        taper = self.taper_combo.currentText()

        # Save undo
        old_data = self.pts3d[fs:fe + 1, joint].copy()
        self._undo_stack.append(('range', joint, fs, fe, old_data))

        new_positions = apply_bulk_offset(
            self.pts3d, joint, fs, fe, self._last_drag_delta, taper=taper)
        self.pts3d[fs:fe + 1, joint] = new_positions
        self._pts3d_dirty = True
        n_frames = fe - fs + 1
        d = self._last_drag_delta
        self.statusBar().showMessage(
            f"Offset applied to joint {joint} across {n_frames} frames "
            f"(delta=[{d[0]:.1f},{d[1]:.1f},{d[2]:.1f}], taper={taper}). Ctrl+Z to undo.")
        self._show_frame()

    def _clear_anchors(self):
        """Clear all anchors."""
        self._anchors.clear_all()
        self._refresh_anchor_list()
        self.statusBar().showMessage("All anchors cleared.")

    # =======================================================================
    #  Keyboard
    # =======================================================================
    def keyPressEvent(self, event):
        k = event.key()
        mods = event.modifiers()
        if mods & Qt.ControlModifier and k == Qt.Key_Z:
            self._undo_joint_edit()
        elif k == Qt.Key_Space: self._toggle_play()
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

    def _make_action_tag(self, action_dict: dict) -> str:
        """Merge action + variant into a single tag: e.g. 'walking_clockwise'.

        Cleans up:
        - '/' treated as no variant
        - spaces → underscores, lowercase
        - If an action has variants elsewhere but this row is empty,
          we don't append anything (the rep counter handles uniqueness).
        """
        act = action_dict["action"].strip().lower().replace(" ", "_").replace("/", "_")
        var = (action_dict.get("variant") or "").strip()
        # Treat "/" or empty as no variant
        if var in ("", "/"):
            var = ""
        else:
            var = var.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        if var:
            return f"{act}_{var}"
        return act

    def _assign_reps(self, indices: list[int]) -> dict[int, int]:
        """Assign rep numbers per action tag, starting from 1.

        Returns {action_index: rep_number}.
        """
        counts: dict[str, int] = {}  # action_tag -> next rep
        result: dict[int, int] = {}
        for ai in indices:
            tag = self._make_action_tag(self.actions[ai])
            rep = counts.get(tag, 1)
            result[ai] = rep
            counts[tag] = rep + 1
        return result

    def _build_export_stem(self, actor_id: int, scene: str,
                           cam: str, action_dict: dict, rep: int) -> str:
        """Build filename stem: 15-boss-topcenter-walk-rep1."""
        scene_s = scene.lower().replace(" ", "_").replace("/", "_") if scene else "unknown"
        cam_s = cam.lower().replace(" ", "_")
        act_tag = self._make_action_tag(action_dict)
        return f"{actor_id:02d}-{scene_s}-{cam_s}-{act_tag}-rep{rep}"

    def _build_csv_stem(self, actor_id: int, scene: str,
                        action_dict: dict, rep: int) -> str:
        """Build CSV filename stem: 15-boss-walk-rep1."""
        scene_s = scene.lower().replace(" ", "_").replace("/", "_") if scene else "unknown"
        act_tag = self._make_action_tag(action_dict)
        return f"{actor_id:02d}-{scene_s}-{act_tag}-rep{rep}"

    def _build_export_dir_name(self, actor_id: int, scene: str) -> str:
        """Build directory name: 15-boss (actor_id fixed for all actions)."""
        scene_s = scene.lower().replace(" ", "_").replace("/", "_") if scene else "unknown"
        return f"{actor_id:02d}-{scene_s}"

    def _guess_actor_id(self) -> int:
        """Try to auto-detect actor/session number from data paths.

        Checks (in order):
          1. Excel filename: DataCollection_15.xlsx → 15
          2. CSV filename: extracted_boss_01.csv → 01
          3. Data folder name containing digits
        Falls back to 1.
        """
        import re
        # From Excel path
        xlsx = getattr(self, "xlsx_path", None)
        if xlsx:
            base = os.path.splitext(os.path.basename(xlsx))[0]
            m = re.search(r'(\d+)', base)
            if m:
                return int(m.group(1))
        # From CSV path
        csv_path = getattr(self, "_csv_path", None)
        if csv_path:
            base = os.path.splitext(os.path.basename(csv_path))[0]
            m = re.search(r'(\d+)', base)
            if m:
                return int(m.group(1))
        # From data folder
        vf = getattr(self, "video_folder", None)
        if vf:
            m = re.search(r'(\d+)', os.path.basename(vf))
            if m:
                return int(m.group(1))
        return 1

    def _preview_export_tree(self, indices: list[int],
                             export_cams: list[str],
                             actor_id: int) -> list[str]:
        """Generate preview lines showing the full export tree."""
        reps = self._assign_reps(indices)
        lines = []
        dir_name = self._build_export_dir_name(actor_id, self.cur_scene)
        lines.append(f"{dir_name}/")
        for ai in indices:
            a = self.actions[ai]
            rep = reps[ai]
            for cn in export_cams:
                stem = self._build_export_stem(actor_id, self.cur_scene, cn, a, rep)
                lines.append(f"  {stem}.mp4")
            if self.pts3d is not None:
                csv_stem = self._build_csv_stem(actor_id, self.cur_scene, a, rep)
                lines.append(f"  {csv_stem}.csv")
        lines.append(f"  offsets.json")
        lines.append(f"  calibration/")
        return lines

    def _export(self, all_actions, single_cam=False):
        if not self.actions:
            QMessageBox.warning(self, "Warning", "No actions loaded."); return
        if not self.video_folder and self.pts3d is None:
            QMessageBox.warning(self, "Warning", "No data loaded."); return
        if single_cam and (not self.active_cam or self.active_cam == "(no cameras)"):
            QMessageBox.warning(self, "Warning",
                                "No camera selected for single-cam export."); return
        out_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not out_dir: return
        indices = list(range(len(self.actions))) if all_actions else (
            [self.cur_act] if self.cur_act >= 0 else [])
        if not indices:
            QMessageBox.warning(self, "Warning", "No action selected."); return

        if single_cam:
            export_cams = [self.active_cam]
        else:
            export_cams = self.avail_cams if self.avail_cams else ["virtual"]

        # --- Preview dialog ---
        guessed_id = self._guess_actor_id()
        preview_lines = self._preview_export_tree(indices, export_cams, guessed_id)

        from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QTextEdit
        dlg = QDialog(self)
        dlg.setWindowTitle("Export Preview")
        dlg.setMinimumSize(620, 450)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("Export preview — all actions share the same actor ID:"))
        seq_row = QHBoxLayout()
        seq_row.addWidget(QLabel("Actor / session ID:"))
        seq_spin = QSpinBox()
        seq_spin.setRange(0, 999)
        seq_spin.setValue(guessed_id)
        seq_spin.setToolTip(
            f"Auto-detected: {guessed_id}\n"
            "Change this to set the actor/session number for all exported files.")
        seq_row.addWidget(seq_spin)
        seq_row.addStretch()
        lay.addLayout(seq_row)
        te = QTextEdit()
        te.setReadOnly(True)
        te.setPlainText("\n".join(preview_lines))
        lay.addWidget(te)

        def _refresh():
            lines = self._preview_export_tree(
                indices, export_cams, seq_spin.value())
            te.setPlainText("\n".join(lines))

        seq_spin.valueChanged.connect(_refresh)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        if dlg.exec_() != QDialog.Accepted:
            return

        actor_id = seq_spin.value()
        reps = self._assign_reps(indices)

        # Single shared directory for all actions
        dir_name = self._build_export_dir_name(actor_id, self.cur_scene)
        act_dir = os.path.join(out_dir, dir_name)
        os.makedirs(act_dir, exist_ok=True)

        # --- Save offsets.json (all actions in one file) ---
        import json as _json
        all_offsets = []
        for ai in indices:
            a = self.actions[ai]
            rep = reps[ai]
            ov = self.overrides.get(ai, {})
            sf = ov.get("start", a["start"])
            ef = ov.get("end", a["end"])
            total_off = self.scene_offset + self._get_effective_act_offset(ai)
            all_offsets.append({
                "action": self._make_action_tag(a),
                "rep": rep,
                "start_frame": sf,
                "end_frame": ef,
                "scene_offset": self.scene_offset,
                "effective_offset": total_off,
            })
        offset_doc = {
            "actor_id": actor_id,
            "scene": self.cur_scene,
            "actions": all_offsets,
        }
        with open(os.path.join(act_dir, "offsets.json"), "w") as f:
            _json.dump(offset_doc, f, indent=2, ensure_ascii=False)

        # --- Copy calibration files ---
        if self.calibs:
            cal_dst = os.path.join(act_dir, "calibration")
            os.makedirs(cal_dst, exist_ok=True)
            cal_folder = getattr(self, "cal_folder", None)
            if cal_folder and os.path.isdir(cal_folder):
                import shutil
                for fn in os.listdir(cal_folder):
                    if fn.lower().endswith(".json"):
                        src = os.path.join(cal_folder, fn)
                        shutil.copy2(src, os.path.join(cal_dst, fn))

        total_ops = len(indices) * len(export_cams)
        prog = QProgressDialog("Exporting...", "Cancel", 0, total_ops, self)
        prog.setWindowModality(Qt.WindowModal); prog.setMinimumDuration(0)
        op = 0

        for ai in indices:
            a = self.actions[ai]
            rep = reps[ai]
            ov = self.overrides.get(ai, {})
            sf = ov.get("start", a["start"])
            ef = ov.get("end", a["end"])
            total_off = self.scene_offset + self._get_effective_act_offset(ai)
            act_tag = self._make_action_tag(a)

            # --- Export 3D points CSV ---
            if self.pts3d is not None:
                pi_s = v2p(sf, self.vfps, self.pfps, self.pts3d.shape[0], total_off)
                pi_e = v2p(ef, self.vfps, self.pfps, self.pts3d.shape[0], total_off)
                sl = self.pts3d[pi_s:pi_e + 1]
                nj = sl.shape[1]
                cols = []
                for j in range(nj):
                    cols.extend([f"{j}_x", f"{j}_y", f"{j}_z"])
                csv_name = self._build_csv_stem(actor_id, self.cur_scene, a, rep) + ".csv"
                pd.DataFrame(sl.reshape(sl.shape[0], -1), columns=cols).to_csv(
                    os.path.join(act_dir, csv_name), index=False)

            # --- Export video per camera ---
            for cn in export_cams:
                if prog.wasCanceled(): break
                stem = self._build_export_stem(actor_id, self.cur_scene, cn, a, rep)

                if cn == "virtual":
                    self._export_virtual_to(act_dir, stem, sf, ef, total_off)
                    op += 1; prog.setValue(op); continue

                vpath = None
                if self.video_folder:
                    for fn in os.listdir(self.video_folder):
                        if cn in fn.lower() and fn.lower().endswith(".mp4"):
                            vpath = os.path.join(self.video_folder, fn); break
                if not vpath:
                    op += 1; prog.setValue(op); continue
                cap2 = cv2.VideoCapture(vpath)
                if not cap2.isOpened():
                    op += 1; prog.setValue(op); continue
                fps = cap2.get(cv2.CAP_PROP_FPS) or 30.0
                w = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out_path = os.path.join(act_dir, f"{stem}.mp4")
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
                                nan_mask = None
                                if self.pts3d_was_nan is not None:
                                    nan_mask = self.pts3d_was_nan[pidx]
                                draw_skel_with_confidence(frm, proj, nan_mask)
                    t = fi / fps
                    cv2.putText(frm, f"{fmt_time(t)} F:{fi}", (15, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    writer.write(frm)
                writer.release(); cap2.release()
                op += 1; prog.setValue(op)
                QCoreApplication.processEvents()
        prog.close()
        QMessageBox.information(self, "Done",
            f"Exported {len(indices)} actions to:\n{act_dir}")

    def _export_virtual_to(self, act_dir, stem, sf, ef, total_off):
        """Export a virtual (black background + skeleton) video clip."""
        w, h = 1280, 720
        fps = self.vfps or 30.0
        out_path = os.path.join(act_dir, f"{stem}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        for fi in range(sf, ef + 1):
            frm = np.zeros((h, w, 3), dtype=np.uint8)
            if self.pts3d is not None and self.calibs:
                for cal_cam, (intr, extr) in self.calibs.items():
                    pidx = v2p(fi, fps, self.pfps,
                               self.pts3d.shape[0], total_off)
                    pts = self.pts3d[pidx]
                    if self.pts3d_valid is not None and self.pts3d_valid[pidx]:
                        proj = project_pts(pts, intr, extr,
                                           self.flip[0], self.flip[1], self.flip[2])
                        if proj is not None and self.show_skel:
                            nan_mask = None
                            if self.pts3d_was_nan is not None:
                                nan_mask = self.pts3d_was_nan[pidx]
                            draw_skel_with_confidence(frm, proj, nan_mask)
                    break
            t = fi / fps
            cv2.putText(frm, f"{fmt_time(t)} F:{fi}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            writer.write(frm)
        writer.release()

    def closeEvent(self, event):
        if self._pts3d_dirty:
            reply = QMessageBox.question(
                self, "Unsaved Edits",
                "You have unsaved 3D point edits. Close without saving?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                event.ignore()
                return
        if self._save_timer.isActive():
            self._save_timer.stop()
        self._save_scene_state()
        if self.cap: self.cap.release()
        event.accept()
