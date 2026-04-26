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
    QInputDialog, QMenu, QToolTip, QScrollArea, QSplitter,
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
from ..io.discovery import scene_name_matches
from ..vision import (
    project_pts, draw_skel, draw_skel_with_confidence, clear_projection_cache,
    unproject_2d_to_3d, get_camera_depth, extract_R_t, find_nearest_joint,
    AnchorSet, interpolate_anchors, apply_bulk_offset,
    interpolate_all_joints, apply_bulk_offset_all_joints,
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
        # _pending_rays: {joint_idx: {vframe, cam, origin, direction, old_xyz, pidx_a}}
        self._pending_rays: dict[int, dict] = {}

        # Per-camera, per-action view offset:
        #   {scene_name: {cam_name: {action_idx_str: int}}}
        # Unset actions inherit from the previous action (like act_offset).
        # Legacy format {scene: {cam: int}} is auto-migrated on load.
        self._view_offsets: dict[str, dict[str, dict[str, int]]] = {}
        self._skeleton_offset: dict[str, int] = {}  # {scene_name: int}

        # All-joints keyframe state
        self._keyframes: list[int] = []  # sorted pts3d frame indices for keyframes

        # Auto-checkpoint for 3D edits
        self._checkpoint_dir: str | None = None
        self._edits_since_checkpoint = 0
        _CHECKPOINT_INTERVAL = 50  # auto-save every N edits

        # Undo stack cap
        self._UNDO_MAX = 200

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

        splitter = QSplitter(Qt.Horizontal)

        # ---- LEFT panel ----
        left = QWidget()
        left.setMinimumWidth(250); left.setMaximumWidth(400)
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
        splitter.addWidget(left)

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
        self.slider.setTracking(True)
        self.slider.installEventFilter(self)
        cvl.addWidget(self.slider)
        br = QHBoxLayout()
        for txt, fn in [("<< -1s", lambda: self._jump(-1)), ("< Prev", self._prev),
                         ("Play / Pause", self._toggle_play), ("Next >", self._nxt),
                         ("+1s >>", lambda: self._jump(1))]:
            b = QPushButton(txt); b.clicked.connect(fn); br.addWidget(b)
        self.zoom_lbl = QLabel("")
        self.zoom_lbl.setStyleSheet("font-size:11px; color:#888; padding:0 4px;")
        br.addWidget(self.zoom_lbl)
        cvl.addLayout(br)
        splitter.addWidget(center)

        # ---- RIGHT panel (scrollable) ----
        right_inner = QWidget()
        rv = QVBoxLayout(right_inner); rv.setContentsMargins(0, 0, 0, 0)
        right_scroll = QScrollArea()
        right_scroll.setWidget(right_inner)
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(250); right_scroll.setMaximumWidth(400)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        cg = QGroupBox("Camera"); cf = QFormLayout(cg)
        self.cam_combo = QComboBox()
        self.cam_combo.currentTextChanged.connect(self._on_cam)
        cf.addRow("View:", self.cam_combo)
        rv.addWidget(cg)

        og = QGroupBox("Video Offset"); of2 = QFormLayout(og)
        self.scene_off_spin = QSpinBox(); self.scene_off_spin.setRange(-50000, 50000)
        self.scene_off_spin.setAttribute(Qt.WA_InputMethodEnabled, False)
        self.scene_off_spin.setInputMethodHints(Qt.ImhFormattedNumbersOnly)
        self.scene_off_spin.valueChanged.connect(self._on_scene_off)
        of2.addRow("Scene:", self.scene_off_spin)
        self.act_off_spin = QSpinBox(); self.act_off_spin.setRange(-50000, 50000)
        self.act_off_spin.setAttribute(Qt.WA_InputMethodEnabled, False)
        self.act_off_spin.setInputMethodHints(Qt.ImhFormattedNumbersOnly)
        self.act_off_spin.valueChanged.connect(self._on_act_off)
        of2.addRow("Action:", self.act_off_spin)
        self.view_off_spin = QSpinBox(); self.view_off_spin.setRange(-50000, 50000)
        self.view_off_spin.setAttribute(Qt.WA_InputMethodEnabled, False)
        self.view_off_spin.setInputMethodHints(Qt.ImhFormattedNumbersOnly)
        self.view_off_spin.valueChanged.connect(self._on_view_off)
        of2.addRow("View:", self.view_off_spin)
        sync_help_btn = QPushButton("❓")
        sync_help_btn.setFixedWidth(32)
        sync_help_btn.setToolTip("Offset 说明")
        sync_help_btn.clicked.connect(self._show_sync_help)
        of2.addRow("", sync_help_btn)
        rv.addWidget(og)

        sg = QGroupBox("Skeleton Offset"); sf2 = QFormLayout(sg)
        self.skel_off_spin = QSpinBox(); self.skel_off_spin.setRange(-50000, 50000)
        self.skel_off_spin.setAttribute(Qt.WA_InputMethodEnabled, False)
        self.skel_off_spin.setInputMethodHints(Qt.ImhFormattedNumbersOnly)
        self.skel_off_spin.valueChanged.connect(self._on_skel_off)
        sf2.addRow("Offset:", self.skel_off_spin)
        rv.addWidget(sg)

        ag = QGroupBox("Action Override"); af = QFormLayout(ag)
        self.start_spin = QSpinBox(); self.start_spin.setRange(0, 9999999)
        self.start_spin.setAttribute(Qt.WA_InputMethodEnabled, False)
        self.start_spin.setInputMethodHints(Qt.ImhDigitsOnly)
        self.start_spin.valueChanged.connect(self._on_start_ov)
        af.addRow("Start:", self.start_spin)
        self.end_spin = QSpinBox(); self.end_spin.setRange(0, 9999999)
        self.end_spin.setAttribute(Qt.WA_InputMethodEnabled, False)
        self.end_spin.setInputMethodHints(Qt.ImhDigitsOnly)
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
        self.show_jid_cb = QCheckBox("Show Joint IDs")
        self.show_jid_cb.setChecked(False)
        self.show_jid_cb.setToolTip("Display joint index numbers on the skeleton")
        self.show_jid_cb.stateChanged.connect(lambda s: self._show_frame())
        ff.addRow(self.show_jid_cb)
        for axis, idx in [("Flip X", 0), ("Flip Y", 1), ("Flip Z", 2)]:
            cb = QCheckBox(axis)
            cb.stateChanged.connect(lambda s, i=idx: self._set_flip(i, s == Qt.Checked))
            ff.addRow(cb)
        rv.addWidget(fg)

        # ---- Propagation panel ----
        pg = QGroupBox("Propagate Edits")
        pfl = QVBoxLayout(pg)

        # All-joints toggle
        self.all_joints_cb = QCheckBox("All Joints (whole skeleton)")
        self.all_joints_cb.setChecked(True)
        self.all_joints_cb.setToolTip(
            "When checked, interpolation and offset apply to ALL joints.\n"
            "Use 'Add Keyframe' to mark multiple keyframes, then interpolate.\n"
            "When unchecked, operations apply to the clicked joint only.")
        self.all_joints_cb.stateChanged.connect(lambda s: self._refresh_kf_list())
        pfl.addWidget(self.all_joints_cb)

        # Keyframe controls (unified for both modes)
        kfrow = QHBoxLayout()
        self.add_kf_btn = QPushButton("Add Keyframe")
        self.add_kf_btn.setToolTip(
            "All Joints: mark current frame as a whole-skeleton keyframe.\n"
            "Single Joint: mark current frame as a keyframe for the selected joint.")
        self.add_kf_btn.clicked.connect(self._add_keyframe)
        kfrow.addWidget(self.add_kf_btn)
        self.del_kf_btn = QPushButton("Del Keyframe")
        self.del_kf_btn.setToolTip("Remove the selected keyframe from the list")
        self.del_kf_btn.clicked.connect(self._del_keyframe)
        kfrow.addWidget(self.del_kf_btn)
        pfl.addLayout(kfrow)
        self.kf_list = QListWidget()
        self.kf_list.setMaximumHeight(90)
        self.kf_list.setToolTip("Keyframes: click to jump to that frame")
        self.kf_list.currentRowChanged.connect(self._on_kf_selected)
        pfl.addWidget(self.kf_list)
        self.kf_status_lbl = QLabel("")
        self.kf_status_lbl.setStyleSheet("font-size:11px; color:#888;")
        self.kf_status_lbl.setWordWrap(True)
        pfl.addWidget(self.kf_status_lbl)

        # Joint selector — click-to-select (shown as read-only label)
        jrow = QHBoxLayout()
        jrow.addWidget(QLabel("Joint:"))
        self.joint_lbl = QLabel("(click to select)")
        self.joint_lbl.setStyleSheet(
            "font-weight:bold; padding:2px 6px; "
            "background:#2a2a2a; border:1px solid #555; border-radius:3px;")
        self.joint_lbl.setToolTip("Click a joint in the view to select it (single-joint mode)")
        jrow.addWidget(self.joint_lbl)
        pfl.addLayout(jrow)

        # Range for interpolation (pts3d frame indices)
        rform = QFormLayout()
        self.prop_start_spin = QSpinBox()
        self.prop_start_spin.setRange(0, 9999999)
        self.prop_start_spin.setAttribute(Qt.WA_InputMethodEnabled, False)
        self.prop_start_spin.setInputMethodHints(Qt.ImhDigitsOnly)
        rform.addRow("From (3D):", self.prop_start_spin)
        self.prop_end_spin = QSpinBox()
        self.prop_end_spin.setRange(0, 9999999)
        self.prop_end_spin.setAttribute(Qt.WA_InputMethodEnabled, False)
        self.prop_end_spin.setInputMethodHints(Qt.ImhDigitsOnly)
        rform.addRow("To (3D):", self.prop_end_spin)
        pfl.addLayout(rform)
        self.prop_range_hint = QLabel("")
        self.prop_range_hint.setStyleSheet("font-size:11px; color:#888;")
        self.prop_range_hint.setWordWrap(True)
        pfl.addWidget(self.prop_range_hint)
        # Connect spinbox changes to update hint
        self.prop_start_spin.valueChanged.connect(self._update_prop_range_hint)
        self.prop_end_spin.valueChanged.connect(self._update_prop_range_hint)

        # Feedback label for apply operations
        self.prop_feedback_lbl = QLabel("")
        self.prop_feedback_lbl.setStyleSheet("font-size:12px; font-weight:bold; color:#4CAF50;")
        self.prop_feedback_lbl.setWordWrap(True)
        pfl.addWidget(self.prop_feedback_lbl)
        self._feedback_timer = QTimer()
        self._feedback_timer.setSingleShot(True)
        self._feedback_timer.timeout.connect(lambda: self.prop_feedback_lbl.setText(""))

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
        self.clear_anchors_btn = QPushButton("Clear All Keyframes")
        self.clear_anchors_btn.clicked.connect(self._clear_anchors)
        pfl.addWidget(self.clear_anchors_btn)

        rv.addWidget(pg)

        # ---- Help button ----
        self.help_btn = QPushButton("❓ Editing Guide")
        self.help_btn.setToolTip("Show how to use the 3D joint editing tools")
        self.help_btn.clicked.connect(self._show_editing_help)
        rv.addWidget(self.help_btn)

        rv.addStretch()
        splitter.addWidget(right_scroll)

        # Set initial splitter proportions (left:center:right ~ 310:flex:280)
        splitter.setStretchFactor(0, 0)  # left: don't stretch
        splitter.setStretchFactor(1, 1)  # center: stretch
        splitter.setStretchFactor(2, 0)  # right: don't stretch
        splitter.setSizes([310, 800, 280])
        hl.addWidget(splitter)
        self.statusBar().showMessage(
            "Space=Play  A/D=Prev/Next  Shift+A/D=±5  Q/E=-/+1s  W/S=Offset  "
            "1-7=Camera  Home/End=ClipEdge  Tab=EditMode  K=Keyframe  Ctrl+Scroll=Zoom")

    def _set_flip(self, idx, val):
        self.flip[idx] = val; self._show_frame()

    def _on_tri_toggled(self, state):
        if state != Qt.Checked:
            self._pending_rays.clear()
            self.tri_status_lbl.setText("")

    def _update_tri_status(self):
        """Update the triangulation status label with all pending rays."""
        if not self._pending_rays:
            self.tri_status_lbl.setText("")
            return
        cam = None
        parts = []
        for j, info in sorted(self._pending_rays.items()):
            cam = info["cam"]
            parts.append(f"J{j}")
        joints_str = ", ".join(parts)
        self.tri_status_lbl.setText(
            f"\u2714 Ray 1: {joints_str} @ {cam}\n"
            f"\u5207\u6362\u76f8\u673a\uff0c\u62d6\u62fd\u540c\u4e00\u5173\u8282\u5b8c\u6210\u4e09\u89d2\u5316")

    # Editing guide language preference (persists within session)
    _help_lang = "zh"

    def _show_editing_help(self):
        """Show a dialog with editing instructions, with CN/EN toggle."""
        from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QTextBrowser

        dlg = QDialog(self)
        dlg.setWindowTitle("Editing Guide")
        dlg.setMinimumSize(560, 520)
        lay = QVBoxLayout(dlg)

        lang_btn = QPushButton("Switch to English" if self._help_lang == "zh" else "切换到中文")
        lang_btn.setFixedWidth(160)
        lay.addWidget(lang_btn)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        lay.addWidget(browser)

        btns = QDialogButtonBox(QDialogButtonBox.Ok)
        btns.accepted.connect(dlg.accept)
        lay.addWidget(btns)

        def _set_text():
            browser.setHtml(self._help_text_zh() if self._help_lang == "zh" else self._help_text_en())
            lang_btn.setText("Switch to English" if self._help_lang == "zh" else "切换到中文")

        def _toggle():
            self._help_lang = "en" if self._help_lang == "zh" else "zh"
            _set_text()

        lang_btn.clicked.connect(_toggle)
        _set_text()
        dlg.exec_()

    @staticmethod
    def _help_text_zh() -> str:
        return (
            "<h3>全关节插值（推荐，最省力）</h3>"
            "<ol>"
            "<li>勾选 <b>All Joints</b>（默认已勾选）</li>"
            "<li>跳到偏移起始帧，确认/修正关节位置 → 点 <b>Add Keyframe</b></li>"
            "<li>可以添加多个关键帧（A, B, C, D...），插值会依次经过每个关键帧</li>"
            "<li>选择 Method (spline/linear) → 点 <b>Apply Interpolation</b></li>"
            "<li>关键帧之间所有帧的所有关节自动平滑过渡</li>"
            "</ol>"
            "<h3>单关节插值（精修单个关节）</h3>"
            "<ol>"
            "<li>取消勾选 <b>All Joints</b>，点击画面中的关节选中它</li>"
            "<li>跳到偏移起始帧，拖拽关节到正确位置 → 点 <b>Add Keyframe</b></li>"
            "<li>跳到偏移结束帧，拖拽同一关节 → 点 <b>Add Keyframe</b></li>"
            "<li>选择 Method → 点 <b>Apply Interpolation</b>，只修改该关节</li>"
            "</ol>"
            "<h3>单帧拖拽编辑</h3>"
            "<ol>"
            "<li>勾选 <b>Edit Mode</b></li>"
            "<li>点击关节并拖拽到目标位置（深度保持不变）</li>"
            "<li>Ctrl+Z 撤销</li>"
            "</ol>"
            "<h3>双视角三角化（修正深度）</h3>"
            "<ol>"
            "<li>勾选 <b>Edit Mode</b> + <b>Triangulate (2-view)</b></li>"
            "<li>在相机 A 拖拽关节到正确的 2D 位置 → 记录射线（青色 T1 标记）</li>"
            "<li>切换到相机 B，拖拽同一关节 → 自动三角化得到精确 3D</li>"
            "<li>提示：选择夹角大的两个相机效果最好</li>"
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
            "<li><b>A/D</b> — 前/后一帧，<b>Shift+A/D</b> — ±5 帧</li>"
            "<li><b>Q/E</b> — 前/后 1 秒</li>"
            "<li><b>1-7</b> — 切换相机视角</li>"
            "<li><b>Home/End</b> — 跳到片段起始/结束</li>"
            "<li><b>Tab</b> — 切换 Edit Mode，<b>K</b> — 添加关键帧</li>"
            "<li><b>R</b> — 重置缩放，<b>Ctrl+Z</b> — 撤销</li>"
            "<li><b>Ctrl+滚轮</b> — 缩放，<b>中键拖拽</b> — 平移</li>"
            "</ul>"
        )

    @staticmethod
    def _help_text_en() -> str:
        return (
            "<h3>All-Joint Interpolation (Recommended)</h3>"
            "<ol>"
            "<li>Check <b>All Joints</b> (on by default)</li>"
            "<li>Go to the start frame, adjust joints → click <b>Add Keyframe</b></li>"
            "<li>Add multiple keyframes (A, B, C, D...) — interpolation passes through each</li>"
            "<li>Choose Method (spline/linear) → click <b>Apply Interpolation</b></li>"
            "<li>All joints between keyframes are smoothly interpolated</li>"
            "</ol>"
            "<h3>Single-Joint Interpolation (Precision Fix)</h3>"
            "<ol>"
            "<li>Uncheck <b>All Joints</b>, click a joint in the view to select it</li>"
            "<li>Go to drift start, drag joint to correct position → click <b>Add Keyframe</b></li>"
            "<li>Go to drift end, drag same joint → click <b>Add Keyframe</b></li>"
            "<li>Choose Method → click <b>Apply Interpolation</b> — only that joint is modified</li>"
            "</ol>"
            "<h3>Single-Frame Drag Edit</h3>"
            "<ol>"
            "<li>Check <b>Edit Mode</b></li>"
            "<li>Click and drag a joint to the target position (depth is preserved)</li>"
            "<li>Ctrl+Z to undo</li>"
            "</ol>"
            "<h3>Two-View Triangulation (Fix Depth)</h3>"
            "<ol>"
            "<li>Check <b>Edit Mode</b> + <b>Triangulate (2-view)</b></li>"
            "<li>In camera A, drag joint(s) to correct 2D position → ray recorded (cyan T1 marker)</li>"
            "<li>Switch to camera B, drag the same joint → auto-triangulates to precise 3D</li>"
            "<li>Tip: choose two cameras with a large angle between them</li>"
            "</ol>"
            "<h3>Bulk Offset (Shift Entire Segment)</h3>"
            "<ol>"
            "<li>Drag a joint to the correct position on any frame (records delta)</li>"
            "<li>Set From/To frame range</li>"
            "<li>Choose Taper: none (uniform) / linear (fade) / cosine (smooth fade)</li>"
            "<li>Click <b>Apply Last Drag as Offset</b></li>"
            "</ol>"
            "<h3>Keyboard Shortcuts</h3>"
            "<ul>"
            "<li><b>Space</b> — Play / Pause</li>"
            "<li><b>A/D</b> — ±1 frame, <b>Shift+A/D</b> — ±5 frames</li>"
            "<li><b>Q/E</b> — ±1 second</li>"
            "<li><b>1-7</b> — Switch camera view</li>"
            "<li><b>Home/End</b> — Jump to clip start / end</li>"
            "<li><b>Tab</b> — Toggle Edit Mode, <b>K</b> — Add keyframe</li>"
            "<li><b>R</b> — Reset zoom, <b>Ctrl+Z</b> — Undo</li>"
            "<li><b>Ctrl+Scroll</b> — Zoom, <b>Middle-drag</b> — Pan</li>"
            "</ul>"
        )

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
        self.sheet_names = [s for s in xl.sheet_names if s not in ("1A2B", "A1A2B1")]
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
        self._pending_rays.clear()
        self._keyframes.clear()

        if self.data_root:
            csv_path, subfolder = find_csv_for_scene(self.data_root, scene_name)
            if subfolder:
                self.video_folder = subfolder
                self.avail_cams = find_cameras_in_folder(subfolder, scene_name)
            if not self.avail_cams:
                root_cams = find_cameras_in_folder(self.data_root, scene_name)
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
            saved_vo = saved.get("view_offsets", {})
            if saved_vo:
                # Auto-migrate legacy format {cam: int} → {cam: {"0": int}}
                migrated = {}
                for cam_key, cam_val in saved_vo.items():
                    if isinstance(cam_val, int):
                        migrated[cam_key] = {"0": cam_val}
                    elif isinstance(cam_val, dict):
                        migrated[cam_key] = {str(k): v for k, v in cam_val.items()}
                    else:
                        migrated[cam_key] = {}
                self._view_offsets[scene_name] = migrated
            # Load skeleton offset (default 0)
            self._skeleton_offset[scene_name] = saved.get("skeleton_offset", 0)

        self._suppress_spin = True
        self.scene_off_spin.setValue(self.scene_offset)
        self.view_off_spin.setValue(0)
        self.skel_off_spin.setValue(self._skeleton_offset.get(scene_name, 0))
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
        self._refresh_kf_list()
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
            "view_offsets": self._view_offsets.get(self.cur_scene, {}),
            "skeleton_offset": self._skeleton_offset.get(self.cur_scene, 0),
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

    def _find_video_for_cam(self, cam_name: str) -> str | None:
        """Find the video file for *cam_name* in video_folder, preferring
        files that also match the current scene name or a known alias."""
        if not self.video_folder or not os.path.isdir(self.video_folder):
            return None
        # First pass: match both scene (with aliases) and camera
        if self.cur_scene:
            for fn in sorted(os.listdir(self.video_folder)):
                fl = fn.lower()
                if fl.endswith(".mp4") and cam_name in fl and scene_name_matches(fl, self.cur_scene):
                    return os.path.join(self.video_folder, fn)
        # Fallback: match camera only (single-scene folders)
        for fn in sorted(os.listdir(self.video_folder)):
            fl = fn.lower()
            if fl.endswith(".mp4") and cam_name in fl:
                return os.path.join(self.video_folder, fn)
        return None

    def _switch_cam(self, cam_name):
        if self.cap: self.cap.release(); self.cap = None
        self.active_cam = cam_name
        self._cached_frame_idx = -1
        self._cached_frame = None
        if not self.video_folder: return
        vpath = self._find_video_for_cam(cam_name)
        if vpath:
            self.cap = cv2.VideoCapture(vpath)
            if self.cap.isOpened():
                self.vfps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
                self.vtotal = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self._estimate_pfps()
        # Load view offset for this camera. Camera switches should not move
        # the raw clip window; view offset only changes skeleton mapping.
        self._suppress_spin = True
        self.view_off_spin.setValue(self._get_view_offset())
        self._suppress_spin = False

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

    def _on_act_sel(self, row, *, offset_delta=0):
        if row < 0 or row >= len(self.actions): return
        self.cur_act = row
        a = self.actions[row]
        ov = self.overrides.get(row, {})
        raw_s = ov.get("start", a["start"])
        raw_e = ov.get("end", a["end"])
        # Clip boundaries stay in raw video-frame coordinates.
        # Video offsets only affect video↔skeleton mapping, not which
        # original frames belong to the action clip.
        s = max(0, raw_s)
        if self.vtotal > 0:
            e = min(self.vtotal - 1, raw_e)
        else:
            e = raw_e
        e = max(s, e)
        self.clip_start = s; self.clip_end = e
        self._suppress_spin = True
        self.start_spin.setValue(raw_s)
        self.end_spin.setValue(raw_e)
        self.act_off_spin.setValue(self._get_effective_act_offset(row))
        self.view_off_spin.setValue(self._get_view_offset(row))
        self._suppress_spin = False
        self.slider.setRange(s, e)
        if offset_delta:
            # Offset changes should not move the clip window. Keep the current
            # raw video frame if possible, only clamp back into the clip range.
            self.cur_frame = max(s, min(e, self.cur_frame))
        else:
            self.cur_frame = s
        self.slider.setValue(self.cur_frame)
        self._read_frame(self.cur_frame)
        self._show_frame()

    def _on_start_ov(self, val):
        if self._suppress_spin or self.cur_act < 0: return
        ov = self.overrides.setdefault(self.cur_act, {})
        ov["start"] = val
        self._update_act_label()
        self._on_act_sel(self.cur_act)
        self._auto_save()

    def _on_end_ov(self, val):
        if self._suppress_spin or self.cur_act < 0: return
        ov = self.overrides.setdefault(self.cur_act, {})
        ov["end"] = val
        self._update_act_label()
        self._on_act_sel(self.cur_act)
        self._auto_save()

    def _on_act_off(self, val):
        if self._suppress_spin or self.cur_act < 0: return
        old_off = self._get_total_video_off()
        self.overrides.setdefault(self.cur_act, {})["offset"] = val
        new_off = self._get_total_video_off()
        self._on_act_sel(self.cur_act, offset_delta=new_off - old_off)
        self._auto_save()

    def _on_scene_off(self, val):
        old_off = self._get_total_video_off() if self.cur_act >= 0 else 0
        self.scene_offset = val
        if self.cur_act >= 0:
            new_off = self._get_total_video_off()
            self._on_act_sel(self.cur_act, offset_delta=new_off - old_off)
        else:
            self._show_frame()
        self._auto_save()

    def _on_view_off(self, val):
        """Store per-camera, per-action view offset."""
        if self._suppress_spin: return
        if not self.cur_scene or not self.active_cam: return
        if self.cur_act < 0: return
        old_off = self._get_total_video_off()
        cam_dict = self._view_offsets.setdefault(
            self.cur_scene, {}).setdefault(self.active_cam, {})
        cam_dict[str(self.cur_act)] = val
        new_off = self._get_total_video_off()
        self._on_act_sel(self.cur_act, offset_delta=new_off - old_off)
        self._auto_save()

    def _get_view_offset(self, action_row: int | None = None) -> int:
        """Get the view offset for current scene + camera + action.

        Inherits from the previous action if not explicitly set
        (same logic as _get_effective_act_offset).
        """
        if not self.cur_scene or not self.active_cam:
            return 0
        cam_dict = self._view_offsets.get(
            self.cur_scene, {}).get(self.active_cam, {})
        if not cam_dict:
            return 0
        row = action_row if action_row is not None else self.cur_act
        # Walk backwards to find the nearest set value
        for r in range(row, -1, -1):
            if str(r) in cam_dict:
                return cam_dict[str(r)]
        return 0

    def _get_view_offset_for(self, scene: str, cam: str,
                             action_row: int) -> int:
        """Get view offset for an arbitrary scene/cam/action (for export)."""
        cam_dict = self._view_offsets.get(scene, {}).get(cam, {})
        if not cam_dict:
            return 0
        for r in range(action_row, -1, -1):
            if str(r) in cam_dict:
                return cam_dict[str(r)]
        return 0

    def _get_total_video_off(self, action_row: int | None = None) -> int:
        """Total video offset = scene + action + view.  Used to recover the
        'raw' video frame from cur_frame so skeleton mapping stays independent."""
        row = action_row if action_row is not None else self.cur_act
        if row < 0:
            return self.scene_offset
        return (self.scene_offset
                + self._get_effective_act_offset(row)
                + self._get_view_offset(row))

    def _on_skel_off(self, val):
        """Update per-scene skeleton offset."""
        if self._suppress_spin: return
        if not self.cur_scene: return
        self._skeleton_offset[self.cur_scene] = val
        self._show_frame()
        self._auto_save()

    def _show_sync_help(self):
        """Show explanation of the offset types."""
        text = (
            "<h3>Offset 说明</h3>"
            "<p><b>Video Offset</b> — 调整视频帧的裁剪范围：</p>"
            "<ul>"
            "<li><b>Scene</b> — 场景级别的全局偏移，影响该场景下所有 action 和所有相机视角。</li>"
            "<li><b>Action</b> — 单个 action 的偏移。"
            "未设置的 action 会继承前一个 action 的值。</li>"
            "<li><b>View</b> — 当前相机视角 + 当前 action 的视频帧偏移。"
            "每个 action 和相机组合可以有独立的 View offset。"
            "当不同相机的视频录制起始时间不同时使用。</li>"
            "</ul>"
            "<p><b>视频总偏移 = Scene + Action + View</b></p>"
            "<p><b>Skeleton Offset</b> — 调整骨架数据的读取帧偏移。"
            "通常不需要调整，仅在 3D 数据与视频帧率不匹配时使用。</p>"
            "<p>调整顺序建议：先调 Scene 对齐大部分视角，"
            "再用 View 微调个别视角的偏差。</p>"
        )
        QMessageBox.information(self, "Offset 说明", text)

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
            # Strip video offset so skeleton mapping is independent
            raw_vf = self.cur_frame - self._get_total_video_off()
            skel_off = self._skeleton_offset.get(self.cur_scene, 0)
            pidx = v2p(raw_vf, self.vfps, self.pfps,
                       self.pts3d.shape[0], skel_off)
            pts = self.pts3d[pidx]
            if self.pts3d_valid is not None and self.pts3d_valid[pidx]:
                proj = project_pts(pts, intr, extr,
                                   self.flip[0], self.flip[1], self.flip[2])
                if proj is not None:
                    nan_mask = None
                    if self.pts3d_was_nan is not None:
                        nan_mask = self.pts3d_was_nan[pidx]
                    draw_skel_with_confidence(frame, proj, nan_mask)
                    # Joint ID overlay
                    if self.show_jid_cb.isChecked():
                        h_f, w_f = frame.shape[:2]
                        for ji in range(len(proj)):
                            jx, jy = int(proj[ji][0]), int(proj[ji][1])
                            if 0 <= jx < w_f and 0 <= jy < h_f:
                                cv2.putText(frame, str(ji), (jx + 6, jy - 6),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                            (200, 200, 200), 1, cv2.LINE_AA)
                    # Highlight dragged joint
                    if self._drag_joint is not None and self._drag_joint < len(proj):
                        jx, jy = int(proj[self._drag_joint][0]), int(proj[self._drag_joint][1])
                        cv2.circle(frame, (jx, jy), 8, (0, 255, 0), 2)
                    # Highlight pending triangulation joints (cyan + "T1")
                    for tj, ray_info in self._pending_rays.items():
                        if ray_info["vframe"] == self.cur_frame and tj < len(proj):
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
                    # Keyframe indicator overlay
                    if pidx in self._keyframes:
                        kf_idx = self._keyframes.index(pidx) + 1
                        cv2.putText(frame, f"KF{kf_idx}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 200, 255), 2)  # orange
        # Update video label frame size for coordinate mapping
        h_f, w_f = (frame.shape[0], frame.shape[1])
        self.vid_lbl.set_frame_size(w_f, h_f)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bpl = ch * w
        qimg = QImage(rgb.data, w, h, bpl, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.vid_lbl.setPixmap(pix)
        # Update zoom indicator
        z = self.vid_lbl.zoom_level
        self.zoom_lbl.setText(f"{z:.0%}" if z != 1.0 else "")
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
    def _push_undo(self, entry):
        """Push to undo stack with cap and auto-checkpoint."""
        self._undo_stack.append(entry)
        if len(self._undo_stack) > self._UNDO_MAX:
            self._undo_stack = self._undo_stack[-self._UNDO_MAX:]
        self._edits_since_checkpoint += 1
        if self._edits_since_checkpoint >= 50:
            self._auto_checkpoint()

    def _auto_checkpoint(self):
        """Auto-save 3D points to a checkpoint file."""
        if self.pts3d is None:
            return
        import tempfile, os
        if self._checkpoint_dir is None:
            self._checkpoint_dir = tempfile.mkdtemp(prefix="cvslice_ckpt_")
        path = os.path.join(self._checkpoint_dir, "pts3d_checkpoint.npy")
        try:
            np.save(path, self.pts3d)
            self._edits_since_checkpoint = 0
            self.statusBar().showMessage(
                f"Auto-checkpoint saved ({self.pts3d.shape[0]} frames) → {path}")
        except Exception as e:
            self.statusBar().showMessage(f"Checkpoint failed: {e}")

    def eventFilter(self, obj, event):
        """Show frame number tooltip when hovering over the slider."""
        if obj is self.slider:
            from PyQt5.QtCore import QEvent
            if event.type() in (QEvent.MouseMove, QEvent.HoverMove, QEvent.MouseButtonPress):
                val = self.slider.value()
                vf = self.vfps if self.vfps > 0 else 30.0
                t = val / vf
                tip = f"Frame {val}  ({fmt_time(t)})"
                pos = obj.mapToGlobal(event.pos())
                QToolTip.showText(pos, tip, self.slider)
        return super().eventFilter(obj, event)

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
            self._push_undo(
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
        dc_raw = intr.get("dist_coeffs") or extr.get("dist_coeffs")
        dc = (np.array(dc_raw, dtype=np.float64).reshape(-1)
              if dc_raw is not None else None)

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

        new_pt_flip = unproject_2d_to_3d(fx, fy, z_cam, K, R, t, dist_coeffs=dc)

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
                        dc_raw = intr.get("dist_coeffs") or extr.get("dist_coeffs")
                        dc = (np.array(dc_raw, dtype=np.float64).reshape(-1)
                              if dc_raw is not None else None)
                        origin, direction = compute_ray(float(fx), float(fy), K, R, t, dist_coeffs=dc)
                        vframe = self.cur_frame  # use video frame for cross-camera matching

                        if (joint in self._pending_rays
                                and self._pending_rays[joint]["vframe"] == vframe
                                and self._pending_rays[joint]["cam"] != cam):
                            # Second view — triangulate this joint!
                            r1 = self._pending_rays[joint]
                            pt3d = triangulate_two_rays(
                                r1["origin"], r1["direction"],
                                origin, direction)

                            # Un-flip: rays were computed in flipped space
                            if self.flip[0]: pt3d[0] *= -1
                            if self.flip[1]: pt3d[1] *= -1
                            if self.flip[2]: pt3d[2] *= -1

                            # Write triangulated result to BOTH pidx_a and pidx_b
                            # (they may differ due to view offset)
                            pidx_a = r1["pidx_a"]
                            pidx_b = pidx  # current camera's pidx
                            self.pts3d[pidx_b, joint] = pt3d
                            if pidx_a != pidx_b:
                                self.pts3d[pidx_a, joint] = pt3d

                            # Undo: remove view-2 entry, keep view-1 (has original old_xyz)
                            if self._undo_stack:
                                self._undo_stack.pop()  # remove view-2 undo

                            del self._pending_rays[joint]
                            self._pts3d_dirty = True
                            remaining = len(self._pending_rays)
                            if remaining == 0:
                                self.tri_status_lbl.setText("")
                                self.statusBar().showMessage(
                                    f"Triangulated J{joint} from {r1['cam']}+{cam}. "
                                    f"All done! Ctrl+Z to undo.")
                            else:
                                self._update_tri_status()
                                self.statusBar().showMessage(
                                    f"Triangulated J{joint} from {r1['cam']}+{cam}. "
                                    f"{remaining} joints remaining.")
                            self._drag_joint = None
                            self._show_frame()
                            return
                        else:
                            # First view — record ray, revert 3D edit
                            if self._undo_stack:
                                entry = self._undo_stack[-1]
                                if entry[0] != 'range':
                                    orig_xyz = entry[2]
                                    self.pts3d[pidx, joint] = orig_xyz.copy()

                            self._pending_rays[joint] = {
                                "vframe": vframe, "cam": cam,
                                "origin": origin, "direction": direction,
                                "old_xyz": self._undo_stack[-1][2].copy() if self._undo_stack else new_xyz,
                                "pidx_a": pidx,
                            }
                            self._update_tri_status()
                            n = len(self._pending_rays)
                            self.statusBar().showMessage(
                                f"Ray 1 recorded: J{joint} @ {cam}. "
                                f"{n} joint(s) pending. Switch camera & drag to triangulate.")
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
        """Undo the last joint edit (single point, range, or range_all)."""
        if not self._undo_stack:
            self.statusBar().showMessage("Nothing to undo.")
            return
        entry = self._undo_stack.pop()
        if entry[0] == 'range_all':
            # All-joints range undo: ('range_all', fs, fe, old_data)
            _, fs, fe, old_data = entry
            if self.pts3d is not None:
                self.pts3d[fs:fe + 1] = old_data
                self._show_frame()
                self.statusBar().showMessage(
                    f"Undone: ALL joints range [{fs}-{fe}] restored. "
                    f"{len(self._undo_stack)} edits remaining.")
        elif entry[0] == 'range':
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
    def _get_current_pidx(self) -> int | None:
        """Get the pts3d frame index for the current video frame."""
        if self.pts3d is None:
            return None
        raw_vf = self.cur_frame - self._get_total_video_off()
        skel_off = self._skeleton_offset.get(self.cur_scene, 0)
        return v2p(raw_vf, self.vfps, self.pfps,
                   self.pts3d.shape[0], skel_off)

    def _add_keyframe(self):
        """Add current frame as a keyframe (all-joints or single-joint)."""
        pidx = self._get_current_pidx()
        if pidx is None:
            QMessageBox.warning(self, "Warning", "No 3D data loaded."); return
        if self.all_joints_cb.isChecked():
            # All-joints mode
            if pidx not in self._keyframes:
                self._keyframes.append(pidx)
                self._keyframes.sort()
            self._refresh_kf_list()
            if self._keyframes:
                self.prop_start_spin.setValue(self._keyframes[0])
                self.prop_end_spin.setValue(self._keyframes[-1])
            self.statusBar().showMessage(
                f"Keyframe added at 3D frame {pidx} ({len(self._keyframes)} total)")
        else:
            # Single-joint mode: store as anchor internally
            joint = self._selected_joint
            if joint is None:
                QMessageBox.warning(self, "Warning",
                                    "No joint selected. Click a joint in the view first.")
                return
            if joint >= self.pts3d.shape[1]:
                QMessageBox.warning(self, "Warning",
                                    f"Joint {joint} out of range."); return
            xyz = self.pts3d[pidx, joint].copy()
            self._anchors.set_anchor(joint, pidx, xyz)
            self._refresh_kf_list()
            # Auto-update range
            anchors = self._anchors.get_anchors(joint)
            if anchors:
                frames = sorted(anchors.keys())
                self.prop_start_spin.setValue(max(0, frames[0]))
                self.prop_end_spin.setValue(min(self.pts3d.shape[0] - 1, frames[-1]))
            self.statusBar().showMessage(
                f"Keyframe set: joint {joint} at 3D frame {pidx}")
        self._show_frame()

    def _del_keyframe(self):
        """Remove the selected keyframe."""
        row = self.kf_list.currentRow()
        if row < 0:
            QMessageBox.warning(self, "Warning", "No keyframe selected."); return
        if self.all_joints_cb.isChecked():
            if row >= len(self._keyframes): return
            removed = self._keyframes.pop(row)
            self._refresh_kf_list()
            if self._keyframes:
                self.prop_start_spin.setValue(self._keyframes[0])
                self.prop_end_spin.setValue(self._keyframes[-1])
            self.statusBar().showMessage(
                f"Keyframe F{removed} removed ({len(self._keyframes)} remaining)")
        else:
            # Single-joint: remove from anchors
            joint = self._selected_joint
            if joint is None: return
            anchors = self._anchors.get_anchors(joint)
            if not anchors: return
            frames = sorted(anchors.keys())
            if row >= len(frames): return
            pidx = frames[row]
            self._anchors.remove_anchor(joint, pidx)
            self._refresh_kf_list()
            self.statusBar().showMessage(
                f"Keyframe removed: joint {joint} at frame {pidx}")
        self._show_frame()

    def _refresh_kf_list(self):
        """Refresh the keyframe list widget and status label."""
        self.kf_list.clear()
        ratio = self.vfps / self.pfps if (self.pfps > 0 and self.vfps > 0) else 1.0
        skel_off = self._skeleton_offset.get(self.cur_scene, 0)
        vid_off = self._get_total_video_off()
        if self.all_joints_cb.isChecked():
            for i, pidx in enumerate(self._keyframes):
                vf = int(round(pidx * ratio - skel_off)) + vid_off
                self.kf_list.addItem(f"KF{i+1}: 3D F{pidx} (\u2248video {vf})")
            n = len(self._keyframes)
            self.kf_status_lbl.setText(f"{n} keyframe(s) set" if n else "")
        else:
            # Show per-joint keyframes (anchors) for selected joint
            joint = self._selected_joint
            if joint is not None:
                anchors = self._anchors.get_anchors(joint)
                if anchors:
                    for i, (pidx, xyz) in enumerate(sorted(anchors.items())):
                        vf = int(round(pidx * ratio - skel_off)) + vid_off
                        self.kf_list.addItem(
                            f"J{joint} KF{i+1}: 3D F{pidx} (\u2248video {vf})")
                n = len(anchors) if anchors else 0
                self.kf_status_lbl.setText(
                    f"Joint {joint}: {n} keyframe(s)" if n else "")
            else:
                # Show all anchors across all joints
                for joint_a, frame, xyz in self._anchors.summary():
                    vf = int(round(frame * ratio - skel_off)) + vid_off
                    self.kf_list.addItem(
                        f"J{joint_a} @ 3D F{frame} (\u2248video {vf})")
                total = len(self._anchors.summary())
                self.kf_status_lbl.setText(
                    f"{total} joint keyframe(s)" if total else "")

    def _on_kf_selected(self, row):
        """Jump to the frame of the selected keyframe."""
        if row < 0: return
        pidx = None
        if self.all_joints_cb.isChecked():
            if row >= len(self._keyframes): return
            pidx = self._keyframes[row]
        else:
            joint = self._selected_joint
            if joint is not None:
                anchors = self._anchors.get_anchors(joint)
                if anchors:
                    frames = sorted(anchors.keys())
                    if row < len(frames):
                        pidx = frames[row]
            else:
                summary = self._anchors.summary()
                if row < len(summary):
                    joint_a, frame, _ = summary[row]
                    self._selected_joint = joint_a
                    self.joint_lbl.setText(f"Joint {joint_a}")
                    pidx = frame
        if pidx is not None and self.pts3d is not None and self.pfps > 0 and self.vfps > 0:
            skel_off = self._skeleton_offset.get(self.cur_scene, 0)
            raw_vf = int(round(pidx * (self.vfps / self.pfps) - skel_off))
            # Add video offset to get actual frame position in clip range
            vframe = raw_vf + self._get_total_video_off()
            vframe = max(self.clip_start, min(self.clip_end, vframe))
            self.cur_frame = vframe
            self._read_frame(vframe)
            self.slider.setValue(vframe)

    def _update_prop_range_hint(self):
        """Show video frame equivalents for the pts3d From/To range."""
        if self.pts3d is None or self.pfps <= 0 or self.vfps <= 0:
            self.prop_range_hint.setText("")
            return
        fs = self.prop_start_spin.value()
        fe = self.prop_end_spin.value()
        skel_off = self._skeleton_offset.get(self.cur_scene, 0)
        vid_off = self._get_total_video_off()
        # Approximate inverse: video_frame = pts3d_frame * (vfps/pfps) - skel_off + vid_off
        ratio = self.vfps / self.pfps
        vf_s = int(round(fs * ratio - skel_off)) + vid_off
        vf_e = int(round(fe * ratio - skel_off)) + vid_off
        self.prop_range_hint.setText(
            f"≈ 视频帧 {vf_s}–{vf_e}  (共 {fe - fs + 1} 个 3D 帧)")

    def _show_prop_feedback(self, text: str, duration_ms: int = 3000):
        """Show a brief feedback message in the propagation panel."""
        self.prop_feedback_lbl.setText(text)
        self._feedback_timer.start(duration_ms)

    def _apply_interpolation(self):
        """Apply interpolation across the frame range (all joints or single joint)."""
        if self.pts3d is None:
            QMessageBox.warning(self, "Warning", "No 3D data loaded."); return
        fs = self.prop_start_spin.value()
        fe = self.prop_end_spin.value()
        if fs >= fe:
            QMessageBox.warning(self, "Warning", "Start frame must be < end frame."); return
        method = self.method_combo.currentText()

        if self.all_joints_cb.isChecked():
            # All-joints mode: interpolate every joint using keyframes as anchors
            old_data = self.pts3d[fs:fe + 1].copy()  # (N, J, 3)
            self._push_undo(('range_all', fs, fe, old_data))
            # Pass intermediate keyframes (excluding fs/fe which are boundaries)
            kf_list = [k for k in self._keyframes if fs < k < fe]
            new_positions = interpolate_all_joints(
                self.pts3d, fs, fe, method=method, keyframes=kf_list if kf_list else None)
            self.pts3d[fs:fe + 1] = new_positions
            self._pts3d_dirty = True
            n_frames = fe - fs + 1
            n_kf = len(kf_list) + 2  # +2 for start/end
            self.statusBar().showMessage(
                f"Interpolated ALL joints across {n_frames} frames "
                f"({n_kf} keyframes, {method}). Ctrl+Z to undo.")
            self._show_prop_feedback(f"\u2714 已插值 {n_frames} 帧, {n_kf} 个关键帧 ({method})")
        else:
            # Single-joint mode
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
            old_data = self.pts3d[fs:fe + 1, joint].copy()
            self._push_undo(('range', joint, fs, fe, old_data))
            new_positions = interpolate_anchors(
                self.pts3d, joint, anchors, fs, fe, method=method)
            self.pts3d[fs:fe + 1, joint] = new_positions
            self._pts3d_dirty = True
            n_frames = fe - fs + 1
            self.statusBar().showMessage(
                f"Interpolated joint {joint} across {n_frames} frames "
                f"({len(anchors)} anchors, {method}). Ctrl+Z to undo.")
            self._show_prop_feedback(f"\u2714 已插值 J{joint}, {n_frames} 帧 ({method})")
        self._show_frame()

    def _apply_bulk_offset(self):
        """Apply the last drag delta across the frame range (all joints or single)."""
        if self.pts3d is None:
            QMessageBox.warning(self, "Warning", "No 3D data loaded."); return
        if self._last_drag_delta is None:
            QMessageBox.warning(self, "Warning",
                                "No drag recorded yet.\n"
                                "Drag a joint first, then use this button.")
            return
        fs = self.prop_start_spin.value()
        fe = self.prop_end_spin.value()
        if fs >= fe:
            QMessageBox.warning(self, "Warning", "Start frame must be < end frame."); return
        taper = self.taper_combo.currentText()
        d = self._last_drag_delta

        if self.all_joints_cb.isChecked():
            # All-joints mode: apply same delta to every joint
            J = self.pts3d.shape[1]
            deltas = np.tile(d, (J, 1))  # (J, 3)
            old_data = self.pts3d[fs:fe + 1].copy()
            self._push_undo(('range_all', fs, fe, old_data))
            new_positions = apply_bulk_offset_all_joints(
                self.pts3d, fs, fe, deltas, taper=taper)
            self.pts3d[fs:fe + 1] = new_positions
            self._pts3d_dirty = True
            n_frames = fe - fs + 1
            self.statusBar().showMessage(
                f"Offset applied to ALL joints across {n_frames} frames "
                f"(delta=[{d[0]:.1f},{d[1]:.1f},{d[2]:.1f}], taper={taper}). Ctrl+Z to undo.")
            self._show_prop_feedback(f"\u2714 已偏移 {n_frames} 帧 (ALL joints, {taper})")
        else:
            # Single-joint mode
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
            old_data = self.pts3d[fs:fe + 1, joint].copy()
            self._push_undo(('range', joint, fs, fe, old_data))
            new_positions = apply_bulk_offset(
                self.pts3d, joint, fs, fe, d, taper=taper)
            self.pts3d[fs:fe + 1, joint] = new_positions
            self._pts3d_dirty = True
            n_frames = fe - fs + 1
            self.statusBar().showMessage(
                f"Offset applied to joint {joint} across {n_frames} frames "
                f"(delta=[{d[0]:.1f},{d[1]:.1f},{d[2]:.1f}], taper={taper}). Ctrl+Z to undo.")
            self._show_prop_feedback(f"\u2714 已偏移 J{joint}, {n_frames} 帧 ({taper})")
        self._show_frame()

    def _clear_anchors(self):
        """Clear all anchors and keyframes."""
        self._anchors.clear_all()
        self._keyframes.clear()
        self._refresh_kf_list()
        self.statusBar().showMessage("All keyframes cleared.")

    # =======================================================================
    #  Keyboard
    # =======================================================================
    def keyPressEvent(self, event):
        k = event.key()
        mods = event.modifiers()
        if mods & Qt.ControlModifier and k == Qt.Key_Z:
            self._undo_joint_edit()
        elif k == Qt.Key_Space: self._toggle_play()
        # Shift+A/D = ±5 frames, plain A/D = ±1
        elif k == Qt.Key_A:
            if mods & Qt.ShiftModifier:
                self._step_frames(-5)
            else:
                self._prev()
        elif k == Qt.Key_D:
            if mods & Qt.ShiftModifier:
                self._step_frames(5)
            else:
                self._nxt()
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
        # 1-7: switch camera
        elif k in (Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4,
                   Qt.Key_5, Qt.Key_6, Qt.Key_7):
            idx = k - Qt.Key_1
            if idx < self.cam_combo.count():
                self.cam_combo.setCurrentIndex(idx)
        # Home/End: jump to clip start/end
        elif k == Qt.Key_Home:
            self.cur_frame = self.clip_start
            self._read_frame(self.cur_frame)
            self.slider.setValue(self.cur_frame)
        elif k == Qt.Key_End:
            self.cur_frame = self.clip_end
            self._read_frame(self.cur_frame)
            self.slider.setValue(self.cur_frame)
        # Tab: toggle Edit Mode
        elif k == Qt.Key_Tab:
            self.edit_cb.setChecked(not self.edit_cb.isChecked())
        # K: add keyframe
        elif k == Qt.Key_K:
            self._add_keyframe()
        # R: reset zoom/pan
        elif k == Qt.Key_R and not (mods & Qt.ControlModifier):
            self.vid_lbl.reset_view()
            self._show_frame()
        else: super().keyPressEvent(event)

    def _step_frames(self, n: int):
        """Step forward/backward by n frames."""
        if self.playing:
            self._toggle_play()
        nf = max(self.clip_start, min(self.clip_end, self.cur_frame + n))
        self.cur_frame = nf
        self._read_frame(nf)
        self.slider.setValue(nf)

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
        seq_spin.setAttribute(Qt.WA_InputMethodEnabled, False)
        seq_spin.setInputMethodHints(Qt.ImhDigitsOnly)
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
                "raw_clip_start": sf,
                "raw_clip_end": ef,
                "view_offsets": {cn: self._get_view_offset_for(
                    self.cur_scene, cn, ai) for cn in self.avail_cams},
            })
        offset_doc = {
            "actor_id": actor_id,
            "scene": self.cur_scene,
            "skeleton_offset": self._skeleton_offset.get(self.cur_scene, 0),
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
            raw_sf = ov.get("start", a["start"])
            raw_ef = ov.get("end", a["end"])
            total_off = self.scene_offset + self._get_effective_act_offset(ai)
            act_tag = self._make_action_tag(a)
            # Export the same raw video clip window regardless of video offset.
            sf = max(0, raw_sf)
            if self.vtotal > 0:
                ef = min(self.vtotal - 1, raw_ef)
            else:
                ef = raw_ef
            ef = max(sf, ef)
            export_skel_off = self._skeleton_offset.get(self.cur_scene, 0)

            # --- Export 3D points CSV ---
            # Use raw (un-offset) frame range for skeleton mapping
            if self.pts3d is not None:
                pi_s = v2p(raw_sf, self.vfps, self.pfps, self.pts3d.shape[0], export_skel_off)
                pi_e = v2p(raw_ef, self.vfps, self.pfps, self.pts3d.shape[0], export_skel_off)
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

                cam_view_off = self._get_view_offset_for(self.cur_scene, cn, ai)

                # Keep the raw video clip window fixed. View offset only changes
                # which skeleton frame is mapped onto each video frame.
                cam_sf = sf
                cam_ef = ef
                cam_total_vid_off = total_off + cam_view_off
                cam_skel_off = export_skel_off

                if cn == "virtual":
                    self._export_virtual_to(act_dir, stem, cam_sf, cam_ef,
                                            cam_skel_off, cam_total_vid_off)
                    op += 1; prog.setValue(op); continue

                vpath = self._find_video_for_cam(cn) if self.video_folder else None
                if not vpath:
                    op += 1; prog.setValue(op); continue
                cap2 = cv2.VideoCapture(vpath)
                if not cap2.isOpened():
                    op += 1; prog.setValue(op); continue
                cam_vtotal = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap2.get(cv2.CAP_PROP_FPS) or 30.0
                w = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
                # Clamp to valid video range
                cam_sf = max(0, cam_sf)
                if cam_vtotal > 0:
                    cam_ef = min(cam_vtotal - 1, cam_ef)
                cam_ef = max(cam_sf, cam_ef)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                # --- Raw video (no overlays) ---
                out_path = os.path.join(act_dir, f"{stem}.mp4")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

                # --- Check video (with skeleton + frame info overlay) ---
                check_dir = os.path.join(act_dir, "check")
                os.makedirs(check_dir, exist_ok=True)
                chk_path = os.path.join(check_dir, f"{stem}.mp4")
                chk_writer = cv2.VideoWriter(chk_path, fourcc, fps, (w, h))

                ie = self.calibs.get(cn)
                cap2.set(cv2.CAP_PROP_POS_FRAMES, cam_sf)
                for fi in range(cam_sf, cam_ef + 1):
                    ret, frm = cap2.read()
                    if not ret: break
                    # Write raw frame
                    writer.write(frm)
                    # Build overlay copy for check video
                    chk = frm.copy()
                    if self.pts3d is not None and ie:
                        intr, extr = ie
                        raw_fi = fi - cam_total_vid_off
                        pidx = v2p(raw_fi, fps, self.pfps,
                                   self.pts3d.shape[0], cam_skel_off)
                        pts = self.pts3d[pidx]
                        if self.pts3d_valid is not None and self.pts3d_valid[pidx]:
                            proj = project_pts(pts, intr, extr,
                                               self.flip[0], self.flip[1], self.flip[2])
                            if proj is not None:
                                nan_mask = None
                                if self.pts3d_was_nan is not None:
                                    nan_mask = self.pts3d_was_nan[pidx]
                                draw_skel_with_confidence(chk, proj, nan_mask)
                    t = fi / fps
                    raw_fi_info = fi - cam_total_vid_off
                    cv2.putText(chk, f"{fmt_time(t)} F:{fi} P:{v2p(raw_fi_info, fps, self.pfps, self.pts3d.shape[0], cam_skel_off) if self.pts3d is not None else '?'}",
                                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (255, 255, 255), 2)
                    chk_writer.write(chk)
                writer.release(); chk_writer.release(); cap2.release()
                op += 1; prog.setValue(op)
                QCoreApplication.processEvents()
        prog.close()
        QMessageBox.information(self, "Done",
            f"Exported {len(indices)} actions to:\n{act_dir}")

    def _export_virtual_to(self, act_dir, stem, sf, ef, skel_off, vid_off=0):
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
                    raw_fi = fi - vid_off
                    pidx = v2p(raw_fi, fps, self.pfps,
                               self.pts3d.shape[0], skel_off)
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
            ckpt_msg = ""
            if self._checkpoint_dir:
                ckpt_msg = f"\n\nAuto-checkpoint available at:\n{self._checkpoint_dir}"
            reply = QMessageBox.question(
                self, "Unsaved Edits",
                f"You have unsaved 3D point edits. Close without saving?{ckpt_msg}",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                event.ignore()
                return
        if self._save_timer.isActive():
            self._save_timer.stop()
        self._save_scene_state()
        if self.cap: self.cap.release()
        event.accept()
