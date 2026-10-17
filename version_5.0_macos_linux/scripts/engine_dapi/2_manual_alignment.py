# ============================================================
# 2_manual_alignment.py - Stage 2: Manual initial alignment
#
# Interactive DAPI->H&E overlay (affine / perspective handles) to
# produce an initial homography, saved to
# initial_alignment.json along with overlay previews.
# Usage: python 2_manual_alignment.py <RUN_DIR>
# ============================================================
import math
import numpy as np
import cv2
from datetime import datetime
import json
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment_geometry import compose_with_orientation, lift_to_level0
from run_artifacts import INITIAL_ALIGNMENT, INITIAL_OVERLAY
import cli_stage2_orb as stage2_matching
import run_roles
from PyQt5.QtCore import Qt, QPointF, QRectF, QEvent, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QTransform, QColor, QPainterPath, QImage
from PyQt5.QtWidgets import (
    QApplication, QGraphicsView, QGraphicsScene,
    QGraphicsItem, QGraphicsRectItem, QGraphicsPixmapItem,
    QVBoxLayout, QWidget, QPushButton, QHBoxLayout,
    QMessageBox, QFileDialog
)
from my_utils import mask_to_rgba, warp_mask, overlay_rgba_on_bgr
print("[Stage2] imports done", flush=True)


# -----------------------------
# Utilities
# -----------------------------
def log_event(run_dir, event_name, stage="stage2", **extra):
    out_json = Path(run_dir) / "pipeline_times.json"
    now_str = datetime.now().isoformat(timespec="seconds")

    # -------- load existing --------
    if out_json.exists():
        try:
            with open(out_json, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}

    # -------- stage key --------
    stage_key = f"{stage}_events"
    if stage_key not in data or not isinstance(data[stage_key], list):
        data[stage_key] = []

    # -------- record --------
    rec = {
        "event": event_name,
        "time": now_str,
    }
    rec.update(extra)
    data[stage_key].append(rec)

    # -------- write --------
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # debug print
    print(f"[LOG] {event_name} -> {out_json}", flush=True)

def cvH_to_qtransform(H: np.ndarray) -> QTransform:
    H = np.asarray(H, dtype=np.float64)
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return QTransform(
        float(H[0, 0]), float(H[1, 0]), float(H[2, 0]),
        float(H[0, 1]), float(H[1, 1]), float(H[2, 1]),
        float(H[0, 2]), float(H[1, 2]), float(H[2, 2])
    )

def qtransform_to_cvH(t: QTransform) -> np.ndarray:
    """Read Qt's full projective transform without sampling float32 corners."""
    return np.array([
        [t.m11(), t.m21(), t.m31()],
        [t.m12(), t.m22(), t.m32()],
        [t.m13(), t.m23(), t.m33()],
    ], dtype=np.float64)

def set_override_cursor(cursor_shape):
    """Avoid stacking override cursors on every hoverMove."""
    if QApplication.overrideCursor() is None:
        QApplication.setOverrideCursor(cursor_shape)
    else:
        QApplication.changeOverrideCursor(cursor_shape)

def clear_override_cursor():
    """Clear ALL stacked override cursors (in case some got stacked)."""
    while QApplication.overrideCursor() is not None:
        QApplication.restoreOverrideCursor()

def mask_to_colored_pixmap(path, fg_rgba=(255, 0, 0, 255), bg_rgba=(0, 0, 0, 0)):
    """
    Fast colorize a binary-ish mask image using vectorized numpy ops.
    - pixels > 0  -> fg_rgba
    - pixels == 0 -> bg_rgba

    Returns: QPixmap
    """
    img = QImage(path).convertToFormat(QImage.Format_ARGB32)
    w, h = img.width(), img.height()

    # QImage ARGB32 memory layout is BGRA on little-endian platforms.
    fr, fg, fb, fa = fg_rgba
    br, bg, bb, ba = bg_rgba

    # Get raw buffer
    ptr = img.bits()
    ptr.setsize(img.byteCount())

    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 4))

    # Build mask: treat any non-zero in RGB as foreground
    mask = (arr[..., 2] > 0) | (arr[..., 1] > 0) | (arr[..., 0] > 0)

    # Write BGRA (not RGBA)
    arr[mask] = np.array([fb, fg, fr, fa], dtype=np.uint8)
    arr[~mask] = np.array([bb, bg, br, ba], dtype=np.uint8)

    return QPixmap.fromImage(img)

# -----------------------------
# Low-level graphics items
# -----------------------------
class ResizeHandle(QGraphicsRectItem):
    VISUAL_SIZE = 10
    HIT_SIZE = 28

    def __init__(self, idx: int, overlay):
        s = float(self.VISUAL_SIZE)
        super().__init__(-s/2, -s/2, s, s)
        self.idx = idx
        self.overlay = overlay

        self.setBrush(QColor(100, 200, 255))
        self.setPen(QColor(0, 0, 0, 0))
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setZValue(10_000)

        if idx in (0, 2):
            self._hover_cursor = Qt.SizeFDiagCursor
        else:
            self._hover_cursor = Qt.SizeBDiagCursor

        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.LeftButton)

    def hoverEnterEvent(self, event):
        set_override_cursor(self._hover_cursor)
        event.accept()

    def hoverMoveEvent(self, event):
        set_override_cursor(self._hover_cursor)
        event.accept()

    def hoverLeaveEvent(self, event):
        clear_override_cursor()
        event.accept()

    def shape(self):
        path = QPainterPath()
        hs = self.HIT_SIZE / 2
        path.addRect(-hs, -hs, self.HIT_SIZE, self.HIT_SIZE)
        return path

    def boundingRect(self):
        hs = self.HIT_SIZE / 2
        return QRectF(-hs, -hs, self.HIT_SIZE, self.HIT_SIZE)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.overlay.start_handle_drag(self.idx, event.scenePos())
            set_override_cursor(self._hover_cursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.overlay.drag_handle_to(event.scenePos())
        event.accept()

    def mouseReleaseEvent(self, event):
        clear_override_cursor()
        self.overlay.end_drag()
        event.accept()

class DapiOverlayItem(QGraphicsPixmapItem):
    MODE_AFFINE = "affine"
    MODE_PERSPECTIVE = "perspective"
    def __init__(self, pixmap):
        super().__init__(pixmap)
        self.setOpacity(1.0)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)

        self.mode = self.MODE_AFFINE
        # perspective mode state
        self._src_corners_local = None   # (4,2) local pixel corners (TL,TR,BR,BL)
        self._dst_corners_scene = None   # (4,2) scene corners (same order)

        self.handles = []
        self._dragging_overlay = False
        self._dragging_handle = False
        self._active_handle_idx = None
        self._press_pos = None
        self._orig_transform = None

    def reset_perspective_cache(self):
        self._src_corners_local = None
        self._dst_corners_scene = None

    def _init_src_corners_local(self):
        r = self.boundingRect()
        # TL, TR, BR, BL  (统一这个顺序!)
        self._src_corners_local = np.array([
            [r.left(), r.top()],
            [r.right(), r.top()],
            [r.right(), r.bottom()],
            [r.left(), r.bottom()],
        ], dtype=np.float32)

    def enter_perspective_mode(self):
        """Freeze current pose into 4 scene corners, then drive future edits by corners."""
        self.mode = self.MODE_PERSPECTIVE
        if self._src_corners_local is None:
            self._init_src_corners_local()

        pts_scene = [self.mapToScene(QPointF(x, y)) for x, y in self._src_corners_local]
        self._dst_corners_scene = np.array([[p.x(), p.y()] for p in pts_scene], dtype=np.float32)
        self._apply_h_from_corners()

    def enter_affine_mode(self):
        self.mode = self.MODE_AFFINE
        self.reset_perspective_cache()

    def _apply_h_from_corners(self):
        if self._src_corners_local is None or self._dst_corners_scene is None:
            return

        H = cv2.getPerspectiveTransform(
            self._src_corners_local.astype(np.float32),
            self._dst_corners_scene.astype(np.float32),
        )
        qtT = cvH_to_qtransform(H)

        self.setRotation(0)
        self.setPos(0, 0)
        self.setTransform(qtT, combine=False)


    def update_handles(self):
        rect = self.boundingRect()
        corners_local = [
            QPointF(rect.left(), rect.top()),  # 0 TL
            QPointF(rect.right(), rect.top()),  # 1 TR
            QPointF(rect.right(), rect.bottom()),  # 2 BR
            QPointF(rect.left(), rect.bottom()),  # 3 BL
        ]
        for idx, p_local in enumerate(corners_local):
            p_scene = self.mapToScene(p_local)
            self.handles[idx].setPos(p_scene)

    def hoverEnterEvent(self, event):
        hits = self.scene().items(event.scenePos())
        if any(isinstance(it, ResizeHandle) for it in hits):
            event.accept()
            return

        if not self._dragging_handle:
            set_override_cursor(Qt.OpenHandCursor)
        event.accept()

    def hoverMoveEvent(self, event):
        hits = self.scene().items(event.scenePos())
        if any(isinstance(it, ResizeHandle) for it in hits):
            event.accept()
            return

        if not self._dragging_handle:
            set_override_cursor(Qt.OpenHandCursor)
        event.accept()

    def hoverLeaveEvent(self, event):
        if not self._dragging_handle:
            clear_override_cursor()
        event.accept()

    def mousePressEvent(self, event):
        scene_pt = event.scenePos()
        hits = self.scene().items(scene_pt)

        for it in hits:
            if isinstance(it, ResizeHandle):
                event.ignore()
                return

        if event.button() == Qt.LeftButton:
            self._dragging_overlay = True
            self._press_pos = scene_pt
            self._orig_transform = self.transform()
            set_override_cursor(Qt.ClosedHandCursor)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self._dragging_handle = False
        self._dragging_overlay = False
        set_override_cursor(Qt.OpenHandCursor)
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging_overlay:
            current_pos = event.scenePos()
            delta = current_pos - self._press_pos

            if self.mode == self.MODE_PERSPECTIVE:
                if self._dst_corners_scene is None:
                    self.enter_perspective_mode()
                self._dst_corners_scene[:, 0] += float(delta.x())
                self._dst_corners_scene[:, 1] += float(delta.y())
                self._press_pos = current_pos
                self._apply_h_from_corners()
                self.update_handles()
                event.accept()
                return

            ## affine mode:
            T = QTransform(self._orig_transform)
            T.translate(delta.x(), delta.y())
            self.setTransform(T)
            self.update_handles()
            event.accept()
            return

        super().mouseMoveEvent(event)

    def attach_handles(self, scene):
        self.handles = [ResizeHandle(i, self) for i in range(4)]
        for h in self.handles:
            scene.addItem(h)
        self.update_handles()

    def start_handle_drag(self, handle_idx: int, scene_pt: QPointF):
        self._dragging_handle = True
        self._active_handle_idx = handle_idx
        self._press_pos = scene_pt
        self._orig_transform = self.transform()

    def drag_handle_to(self, scene_pt: QPointF):
        if not self._dragging_handle:
            return

        # -------------------------
        # PERSPECTIVE MODE
        # -------------------------
        if self.mode == self.MODE_PERSPECTIVE:
            if self._dst_corners_scene is None:
                self.enter_perspective_mode()

            i = int(self._active_handle_idx)
            self._dst_corners_scene[i, 0] = float(scene_pt.x())
            self._dst_corners_scene[i, 1] = float(scene_pt.y())

            self._apply_h_from_corners()
            self.update_handles()
            return

        # -------------------------
        # AFFINE MODE (keep your old logic)
        # -------------------------
        current_pos = scene_pt
        opp_idx = (self._active_handle_idx + 2) % 4
        fixed_pt = self.handles[opp_idx].scenePos()

        v0 = self._press_pos - fixed_pt
        v1 = current_pos - fixed_pt
        if abs(v0.x()) < 1e-6 and abs(v0.y()) < 1e-6:
            return

        theta = math.radians(self.rotation())
        ux = QPointF(math.cos(theta), math.sin(theta))
        uy = QPointF(-math.sin(theta), math.cos(theta))

        def dot(a: QPointF, b: QPointF) -> float:
            return a.x() * b.x() + a.y() * b.y()

        v0x, v0y = dot(v0, ux), dot(v0, uy)
        v1x, v1y = dot(v1, ux), dot(v1, uy)

        eps = 1e-6
        sx = v1x / (v0x if abs(v0x) > eps else (eps if v0x >= 0 else -eps))
        sy = v1y / (v0y if abs(v0y) > eps else (eps if v0y >= 0 else -eps))

        sx = max(sx, 0.05)
        sy = max(sy, 0.05)

        if QApplication.keyboardModifiers() & Qt.ShiftModifier:
            s = sx if abs(sx - 1.0) >= abs(sy - 1.0) else sy
            s = max(s, 0.05)
            sx = sy = s

        T = QTransform()
        T.translate(fixed_pt.x(), fixed_pt.y())
        T.rotate(self.rotation())
        T.scale(sx, sy)
        T.rotate(-self.rotation())
        T.translate(-fixed_pt.x(), -fixed_pt.y())

        self.setTransform(T * self._orig_transform)
        self.update_handles()

    def end_drag(self):
        self._dragging_handle = False
        self._dragging_overlay = False
        self._active_handle_idx = None

    def set_visible_all(self, visible: bool):
        """Show/hide overlay pixmap AND its resize handles."""
        self.setVisible(visible)
        if self.handles:
            for h in self.handles:
                h.setVisible(visible)

# -----------------------------
# View class
# -----------------------------
class ManualAlignView(QGraphicsView):
    """
    Supports two background/overlay modes:
      - mask mode: he_mask_pix / dapi_mask_pix
      - original mode: he_orig_pix / dapi_orig_pix

    When switching modes, the current overlay pose is preserved
    and applied to the other pixmap set.
    """
    MODE_MASK = "mask"
    MODE_ORIG = "original"
    # -------------------------
    # init / view setup
    # -------------------------
    def __init__(self, he_mask_path, dapi_mask_path, he_orig_path, dapi_orig_path):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        # ---- load pixmaps ----
        self.he_mask_pix = mask_to_colored_pixmap(
            he_mask_path, fg_rgba=(255, 0, 0, 255), bg_rgba=(0, 0, 0, 255)
        )
        self.dapi_mask_pix = mask_to_colored_pixmap(
            dapi_mask_path,
            fg_rgba=(50, 120, 255, 120),
            bg_rgba=(0, 0, 0, 0),
        )

        self.he_orig_pix = QPixmap(he_orig_path)
        self.dapi_orig_pix = QPixmap(dapi_orig_path)

        # ---- items ----
        self.he_item = QGraphicsPixmapItem(self.he_orig_pix)
        self.scene.addItem(self.he_item)
        self.overlay = DapiOverlayItem(self.dapi_orig_pix)
        self.scene.addItem(self.overlay)
        self.overlay.attach_handles(self.scene)
        self.mode = self.MODE_ORIG
        self.overlay.setOpacity(0.5)

        self.view_pad = 0.95
        self.view_pad_min = 0.2
        self.view_pad_max = 1.8
        self.view_pad_step = 0.05

        self.fit_bg_to_view()
        self.init_overlay_pose(scale=0.85)

        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

        self._overlay_hidden_by_key = False
        self.installEventFilter(self)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

    def fit_bg_to_view(self):
        bg_rect = self.he_item.boundingRect()

        pad_x = bg_rect.width() * 0.2
        pad_y = bg_rect.height() * 0.2

        scene_rect = QRectF(
            bg_rect.left() - pad_x,
            bg_rect.top() - pad_y,
            bg_rect.width() + 2 * pad_x,
            bg_rect.height() + 2 * pad_y,
        )
        self.setSceneRect(scene_rect)

        self.resetTransform()
        self.fitInView(scene_rect, Qt.KeepAspectRatio)

        pad = getattr(self, "view_pad", 0.95)
        self.scale(pad, pad)

    def adjust_view_pad(self, delta):
        old_pad = self.view_pad
        self.view_pad = max(self.view_pad_min, min(self.view_pad_max, self.view_pad + delta))
        print(f"[zoom] view_pad: {old_pad:.2f} -> {self.view_pad:.2f}", flush=True)
        self.fit_bg_to_view()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_bg_to_view()

    def wheelEvent(self, event):
        event.ignore()

    def keyPressEvent(self, event):
        mods = event.modifiers()

        # macOS Command key = MetaModifier
        is_cmd = bool(mods & Qt.MetaModifier)

        if is_cmd and event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self.adjust_view_pad(+self.view_pad_step)
            event.accept()
            return

        if is_cmd and event.key() in (Qt.Key_Minus, Qt.Key_Underscore):
            self.adjust_view_pad(-self.view_pad_step)
            event.accept()
            return

        if event.key() == Qt.Key_Q:
            T = QTransform(self.overlay.transform())
            T.rotate(-5)
            self.overlay.setTransform(T)
            self.overlay.update_handles()
            event.accept()
            return

        if event.key() == Qt.Key_E:
            T = QTransform(self.overlay.transform())
            T.rotate(5)
            self.overlay.setTransform(T)
            self.overlay.update_handles()
            event.accept()
            return

        super().keyPressEvent(event)

    def leaveEvent(self, event):
        clear_override_cursor()
        if self._overlay_hidden_by_key:
            self._set_overlay_hidden(False)
        super().leaveEvent(event)

    # -------------------------
    # overlay pose helpers
    # -------------------------
    def init_overlay_pose(self, scale=0.35):
        self.overlay.enter_affine_mode()

        bg_rect = self.he_item.sceneBoundingRect()
        bg_center = bg_rect.center()

        self.overlay.setRotation(0)
        self.overlay.setTransform(QTransform())
        self.overlay.setPos(0, 0)

        ov_rect = self.overlay.boundingRect()
        ov_center = ov_rect.center()

        self.overlay.setPos(bg_center - ov_center)

        T = QTransform()
        T.translate(ov_center.x(), ov_center.y())
        T.scale(scale, scale)
        T.translate(-ov_center.x(), -ov_center.y())
        self.overlay.setTransform(T)

        self.overlay.update_handles()

    def get_overlay_pose(self):
        """Return the current overlay pose (must include pos/rotation/transform)."""
        return {
            "pos": QPointF(self.overlay.pos()),
            "rotation": float(self.overlay.rotation()),
            "transform": QTransform(self.overlay.transform()),
        }

    def apply_overlay_pose_keep_scene_center(self, pose):
        """
        Apply a pose to the overlay while keeping its scene center unchanged,
        preventing drift after swapping pixmaps.
        """
        # old scene center (based on current pixmap)
        old_center_scene = self.overlay.mapToScene(self.overlay.boundingRect().center())

        # apply pose
        self.overlay.setRotation(pose["rotation"])
        self.overlay.setTransform(pose["transform"])
        self.overlay.setPos(pose["pos"])

        # compute new center and compensate shift
        new_center_scene = self.overlay.mapToScene(self.overlay.boundingRect().center())
        delta = old_center_scene - new_center_scene
        self.overlay.setPos(self.overlay.pos() + delta)

        self.overlay.update_handles()

    def swap_mode(self):
        """
        Core logic when swapping modes:
          1) Record current overlay pose
          2) Switch background + overlay pixmap
          3) Apply pose to the new pixmap (with center compensation)
        """
        clear_override_cursor()

        pose = self.get_overlay_pose()

        if self.mode == self.MODE_MASK:
            # switch to original
            self.mode = self.MODE_ORIG
            self.overlay.setOpacity(0.5)
            self.he_item.setPixmap(self.he_orig_pix)
            self.overlay.setPixmap(self.dapi_orig_pix)
            self.apply_overlay_pose_keep_scene_center(pose)
            self.overlay.reset_perspective_cache()
        else:
            # switch to mask
            self.mode = self.MODE_MASK
            self.he_item.setPixmap(self.he_mask_pix)
            self.overlay.setOpacity(1)
            self.overlay.setPixmap(self.dapi_mask_pix)
            self.apply_overlay_pose_keep_scene_center(pose)
            self.overlay.reset_perspective_cache()

        self.fit_bg_to_view()
        self.overlay.update_handles()

    def get_homography_dapi_to_he(self):
        """Current overlay-local -> fixed-image transform, including its pose."""
        transform, valid = self.overlay.itemTransform(self.he_item)
        if not valid:
            raise ValueError("Cannot map the overlay to the fixed image")
        return qtransform_to_cvH(transform)

    def get_corners_dapi_to_he(self):
        """
        Return 4 corner correspondences:
          src = DAPI local (pixel) corners
          dst = HE pixel corners (scene coordinates)
        Order: TL, TR, BR, BL
        """
        rect = self.overlay.boundingRect()  # DAPI local coordinate rectangle

        src_qt = [rect.topLeft(), rect.topRight(), rect.bottomRight(), rect.bottomLeft()]
        dst_qt = [self.overlay.mapToScene(p) for p in src_qt]

        src = np.array([[p.x(), p.y()] for p in src_qt], dtype=np.float32)
        dst = np.array([[p.x(), p.y()] for p in dst_qt], dtype=np.float32)
        return src, dst

    def apply_homography_to_overlay(self, H_gui2he: np.ndarray):
        """
        Apply a 3x3 homography mapping overlay-local coordinates (GUI DAPI pixels)
        directly to scene coordinates (HE pixels).

        NOTE: This sets the overlay pose immediately (rotation=0, pos=(0,0), transform=H).
        """
        H = np.asarray(H_gui2he, dtype=np.float64)
        if H.shape != (3, 3):
            raise ValueError(f"H_gui2he must be (3,3), got {H.shape}")

        # normalize for numerical stability
        if abs(H[2, 2]) > 1e-12:
            H = H / H[2, 2]

        qtT = QTransform(
            float(H[0, 0]), float(H[1, 0]), float(H[2, 0]),
            float(H[0, 1]), float(H[1, 1]), float(H[2, 1]),
            float(H[0, 2]), float(H[1, 2]), float(H[2, 2])
        )

        # reset pose components to avoid double transformation
        self.overlay.setRotation(0)
        self.overlay.setPos(0, 0)
        self.overlay.setTransform(qtT, combine=False)
        self.overlay.update_handles()
        self.overlay.reset_perspective_cache()

    # -------------------------
    # event filtering / visibility
    # -------------------------
    def _set_overlay_hidden(self, hide: bool):
        """Unified entry to toggle overlay visibility (avoid redundant setVisible calls)."""
        if hide == self._overlay_hidden_by_key:
            return
        self._overlay_hidden_by_key = hide
        self.overlay.set_visible_all(not hide)

    def eventFilter(self, obj, event):
        et = event.type()

        # fallback: sometimes Meta key press/release is unreliable,
        # so also check via modifiers
        def meta_down(e):
            try:
                return bool(e.modifiers() & Qt.MetaModifier)
            except Exception:
                return False

        if et == QEvent.KeyPress:
            # Command pressed (or any key while Meta is held)
            if event.key() == Qt.Key_Meta or meta_down(event):
                self._set_overlay_hidden(True)
                return False  # do not consume (avoid blocking Cmd+Q / Cmd+W)

        elif et == QEvent.KeyRelease:
            # Command released
            if event.key() == Qt.Key_Meta:
                self._set_overlay_hidden(False)
                return False

            # fallback: if other key released but Meta no longer held, restore
            if self._overlay_hidden_by_key and not meta_down(event):
                self._set_overlay_hidden(False)
                return False

        elif et in (QEvent.FocusOut, QEvent.WindowDeactivate):
            # restore if window loses focus (avoid stuck hidden state)
            if self._overlay_hidden_by_key:
                self._set_overlay_hidden(False)

        return super().eventFilter(obj, event)

# -----------------------------
# Background feature alignment: shared CLI matcher, GUI acceptance policy
# -----------------------------
class FeatureAlignWorker(QThread):
    """Match the saved Stage-1 PNGs; the user, not check_fit, judges the result.

    Uses the CLI's parameter reader and feature_homography, hence the same log1p
    DAPI preparation, descriptors, ratio test, RANSAC and physical-grid scaling.
    No private 1600-pixel resize, quality gate, mirror/scale-hypothesis ladder or
    pyramid relevel. Only OME spacing metadata may be read from the source TIFFs.
    The next configured detector runs only if the previous one cannot fit a matrix.
    """
    finished_ok = pyqtSignal(object)   # diagnostics dict including Stage-1 PNG -> fixed H
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, run_dir, info=None, engine_dir=None, parent=None):
        super().__init__(parent)
        self.run_dir = Path(run_dir)
        self.info = info
        self.engine_dir = Path(engine_dir) if engine_dir else Path(__file__).resolve().parent

    def run(self):
        try:
            info = self.info if self.info is not None else json.loads(
                (self.run_dir / "images_info.json").read_text())
            roles = run_roles.read(info)
            mode = roles.pair.name
            moving_path = run_roles.artifact_path(self.run_dir, run_roles.MOVING, "1_{token}_level_image.png")
            fixed_path = run_roles.artifact_path(self.run_dir, run_roles.FIXED, "1_{token}_level_image.png")
            physical_scale = stage2_matching.physical_matching_scale(info, roles)
            methods = stage2_matching.stage2_config(self.engine_dir, mode, "")["methods"]
            failures = []
            for method in methods:
                if self.isInterruptionRequested():
                    return
                self.progress.emit(str(method).upper())
                cfg = stage2_matching.stage2_config(self.engine_dir, mode, method)
                try:
                    fit = stage2_matching.feature_homography(
                        method, moving_path, fixed_path,
                        invert_src=roles.moving.modality != roles.fixed.modality,
                        n_features=int(cfg["n_features"]), ratio=float(cfg["ratio"]),
                        ransac_thresh=float(cfg["ransac_thresh"]),
                        random_seed=int(cfg["random_seed"]),
                        physical_scale=physical_scale, flip=None,
                    )
                except (RuntimeError, ValueError, cv2.error) as exc:
                    failures.append(f"{str(method).upper()}: {exc}")
                    continue
                if not self.isInterruptionRequested():
                    # Deliberately no check_fit: any computed fit is a visual draft.
                    self.finished_ok.emit(fit)
                return
            if not self.isInterruptionRequested():
                self.failed.emit("; ".join(failures) or "No Stage-2 detectors configured")
        except Exception as e:
            if not self.isInterruptionRequested():
                self.failed.emit(str(e))


# -----------------------------
# Window class
# -----------------------------
class ManualAlignWindow(QWidget):
    def __init__(self, run_dir, he_mask_path, dapi_mask_path, he_orig_path, dapi_orig_path,
                 dapi_gui_affine, info=None, auto_orb=True):
        super().__init__()
        self.run_dir = str(run_dir)
        self.info = info or {}
        self.dapi_gui_affine = np.asarray(dapi_gui_affine, dtype=np.float64)

        # Keep the LUT path for display, but never feed its false colors to matching.
        self.he_orig_path = str(he_orig_path)
        self.dapi_orig_path = str(dapi_orig_path)

        self.external_h_mat = None
        self.external_h_source = None
        self.external_h_view = None

        # The legacy --no-auto-orb switch still disables the entire suggestion in
        # fail-safe mode, including SIFT. It does not change the manual workflow.
        self.auto_H = None
        self.auto_fit = None
        self.feature_worker = None
        self.auto_start_view = None
        self.auto_orb = bool(auto_orb)

        self.setWindowTitle(
            "Manual Alignment (ORB/SIFT preview)" if self.auto_orb
            else "Manual Alignment (manual only)"
        )

        self.view = ManualAlignView(he_mask_path, dapi_mask_path, he_orig_path, dapi_orig_path)
        self.resize(1400, 900)

        self.btn_load_h = QPushButton("Load H (.json)")
        self.btn_auto   = QPushButton("Auto-align (ORB/SIFT): computing…")
        self.btn_auto.setToolTip("No quality gate. Inspect the overlay, adjust if needed, and Save to accept.")
        self.btn_reset  = QPushButton("Reset")
        self.btn_save   = QPushButton("Save Alignment")
        self.btn_mode = QPushButton("Mode: Affine")

        self.btn_auto.setEnabled(False)

        self.btn_load_h.clicked.connect(self.on_load_h_json)
        self.btn_auto.clicked.connect(self.on_apply_auto)
        self.btn_reset.clicked.connect(self.on_reset)
        self.btn_save.clicked.connect(self.on_save)
        self.btn_mode.clicked.connect(self.on_toggle_mode)


        hl = QHBoxLayout()
        hl.addWidget(self.btn_load_h)
        if self.auto_orb:
            hl.addWidget(self.btn_auto)
        hl.addWidget(self.btn_mode)
        hl.addWidget(self.btn_reset)
        hl.addWidget(self.btn_save)

        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.addLayout(hl)
        self.setLayout(layout)

        self.pose_mask = None
        self.pose_orig = None

        # Compute an unchecked preview, applying it when ready unless the user has
        # already changed/imported a pose. In that case offer Apply without
        # overwriting their work. Save is always required to accept the alignment.
        # The fail-safe launcher turns this off: a run only reaches manual
        # alignment because ORB and SIFT already rejected the pair, so offering
        # the same fit again suggests a wrong answer rather than no answer.
        if self.auto_orb:
            self._start_auto_align()
        else:
            log_event(self.run_dir, "system_skipped_orb_autoalign")

    # -------------------------
    # Feature auto-alignment (unchecked background preview)
    # -------------------------
    def _start_auto_align(self):
        self.auto_start_view = self.view.get_homography_dapi_to_he().copy()
        self.feature_worker = FeatureAlignWorker(self.run_dir, self.info, parent=self)
        self.feature_worker.finished_ok.connect(self.on_auto_ready)
        self.feature_worker.failed.connect(self.on_auto_failed)
        self.feature_worker.progress.connect(self.on_auto_progress)
        self.btn_save.setEnabled(False)
        self.feature_worker.finished.connect(lambda: self.btn_save.setEnabled(True))
        log_event(self.run_dir, "system_start_feature_autoalign", quality_gate=False, relevel=False)
        self.feature_worker.start()

    def on_auto_progress(self, method):
        self.btn_auto.setText(f"Auto-align ({method}): computing…")

    def on_auto_ready(self, fit):
        self.auto_fit = fit
        self.auto_H = np.asarray(fit["H"], dtype=np.float64)
        method, n_inl = str(fit["method"]).upper(), int(fit["n_inliers"])
        self.btn_auto.setText(f"Apply {method} ({n_inl} inliers, unchecked)")
        self.btn_auto.setEnabled(True)
        print(f"[Stage2][{method}] preview: {n_inl} inliers (no quality gate)", flush=True)
        log_event(self.run_dir, "system_ready_feature_autoalign", method=method,
                  inliers=n_inl, quality_gate=False, relevel=False)
        if (self.external_h_view is None and self.auto_start_view is not None
                and np.array_equal(self.view.get_homography_dapi_to_he(), self.auto_start_view)):
            self.on_apply_auto(automatic=True)

    def on_auto_failed(self, msg):
        self.auto_H = None
        self.auto_fit = None
        self.btn_auto.setText("Auto-align: unavailable — align manually")
        self.btn_auto.setToolTip(msg)
        self.btn_auto.setEnabled(False)
        print(f"[Stage2] feature matching failed: {msg}", flush=True)
        log_event(self.run_dir, "system_failed_feature_autoalign", reason=str(msg), relevel=False)

    def on_apply_auto(self, checked=False, *, automatic=False):
        if self.auto_H is None:
            return
        # The fitted homography maps DAPI-image pixels -> HE-image pixels, i.e. the
        # ORIGINAL-mode pixmaps. Switch to original mode so overlay-local coords
        # match what apply_homography_to_overlay expects.
        if self.view.mode != self.view.MODE_ORIG:
            self.view.swap_mode()
        self.view.apply_homography_to_overlay(self.auto_H)
        # Preserve the exact shared-matcher coefficients through an unchanged Save,
        # just as for imported matrices. Any subsequent manual edit wins instead.
        self.external_h_mat = compose_with_orientation(self.auto_H, self.dapi_gui_affine)
        self.external_h_view = self.view.get_homography_dapi_to_he().copy()
        self.external_h_source = None
        self.pose_orig = self.view.get_overlay_pose()
        method = str(self.auto_fit["method"]).upper()
        self.btn_auto.setText(f"Reapply {method} ({self.auto_fit['n_inliers']} inliers, unchecked)")
        log_event(self.run_dir, "system_apply_feature_preview" if automatic else "user_click_apply_feature_autoalign",
                  method=method, quality_gate=False)

    def closeEvent(self, event):
        # Do not destroy a live QThread if the window is closed during matching.
        if self.feature_worker is not None and self.feature_worker.isRunning():
            self.feature_worker.requestInterruption()
            self.feature_worker.finished.connect(self.close)
            event.ignore()
            return
        super().closeEvent(event)

    def on_toggle_mode(self):
        ov = self.view.overlay
        if ov.mode == ov.MODE_AFFINE:
            ov.enter_perspective_mode()
            self.btn_mode.setText("Mode: Perspective")
        else:
            ov.enter_affine_mode()
            self.btn_mode.setText("Mode: Affine")
        ov.update_handles()

    def on_reset(self):
        self.auto_start_view = None  # A pending preview must not undo an explicit Reset.
        clear_override_cursor()
        self.view.init_overlay_pose(scale=0.85)
        self.btn_mode.setText("Mode: Affine")

    def on_swap(self):
        cur_pose = self.view.get_overlay_pose()
        if self.view.mode == self.view.MODE_MASK:
            self.pose_mask = cur_pose
        else:
            self.pose_orig = cur_pose

        self.view.swap_mode()

    def on_load_h_json(self):
        """
        Load an original-moving -> fixed level-image matrix and apply it.
        Retain its exact coefficients for an unchanged load/save round trip.
        """
        start_dir = self.run_dir if os.path.isdir(self.run_dir) else str(Path.cwd())

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select alignment json (contains H_mat / H_homo)",
            start_dir,
            "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)

            # try multiple keys (your file might use H_mat or H_homo)
            H = None
            key_used = None
            for k in ("H_mat", "H_homo", "H", "h_mat", "H_matrix", "homography_3x3"):
                if k in data and data[k] is not None:
                    H = np.asarray(data[k], dtype=np.float64)
                    key_used = k
                    break

            if H is None:
                raise KeyError("No H_mat / H_homo found in the selected json.")

            # normalize to 3x3 if possible
            if H.shape == (2, 3):
                H = np.vstack([H, [0, 0, 1]])
            if H.shape != (3, 3):
                raise ValueError(f"Expected H as (3,3) (or (2,3) convertible), got {H.shape}")

            if not np.isfinite(H).all() or np.linalg.matrix_rank(H) != 3:
                raise ValueError("H must be finite and invertible")

            # A: orig -> GUI (from images_info.json)
            A = np.asarray(self.dapi_gui_affine, dtype=np.float64)
            if A.shape == (2, 3):
                A = np.vstack([A, [0, 0, 1]]).astype(np.float64)
            if A.shape != (3, 3):
                raise ValueError(f"dapi_gui_affine must be (2,3) or (3,3), got {A.shape}")

            A_inv = np.linalg.inv(A)

            # "subtract" the gui affine: H_gui2he = H_total * inv(A)
            H_gui2he = H @ A_inv

            # apply immediately to overlay pixmap (four-corner positioning)
            self.view.apply_homography_to_overlay(H_gui2he)
            self.external_h_mat = H.copy()
            self.external_h_source = path
            self.external_h_view = self.view.get_homography_dapi_to_he().copy()
            log_event(self.run_dir, "user_load_alignment", source=path, key=key_used)

            # optional: also stash the current pose as pose_orig/mask so Save writes consistent state
            cur_pose = self.view.get_overlay_pose()
            if self.view.mode == self.view.MODE_MASK:
                self.pose_mask = cur_pose
            else:
                self.pose_orig = cur_pose

            QMessageBox.information(
                self,
                "Loaded H",
                f"Loaded {key_used} from:\n{path}\n\n"
                f"H shape: {H.shape}\n"
                f"The homography has been applied to the current overlay."
            )
        except Exception as e:
            QMessageBox.warning(self, "Load failed", f"Failed to load H from json:\n{path}\n\nError: {e}")

    def _pose_to_jsonable(self, pose):
        t = pose["transform"]
        p = pose["pos"]
        return {
            "pos_x": p.x(),
            "pos_y": p.y(),
            "rotation_deg": pose["rotation"],
            "m11": t.m11(), "m12": t.m12(),
            "m21": t.m21(), "m22": t.m22(),
            "dx":  t.dx(),  "dy":  t.dy(),
        }

    def alignment_for_save(self):
        H_gui = self.view.get_homography_dapi_to_he()
        # Even double-precision orientation round trips can perturb coefficients.
        # Reuse only if the full current mapping is EXACTLY the loaded mapping;
        # any translation, rotation or perspective edit must be exported instead.
        if (self.external_h_view is not None and
                np.array_equal(H_gui, self.external_h_view)):
            return self.external_h_mat.copy()
        return compose_with_orientation(H_gui, self.dapi_gui_affine)

    def on_save(self):
        log_event(self.run_dir, "user_click_save_alignment")
        cur_pose = self.view.get_overlay_pose()
        if self.view.mode == self.view.MODE_MASK:
            self.pose_mask = cur_pose
        else:
            self.pose_orig = cur_pose

        H_homo = self.alignment_for_save()

        roles = run_roles.read(self.info)
        Ld = int(roles.moving.level)
        Lh = int(roles.fixed.level)
        H_mat_level_0 = lift_to_level0(H_homo, Ld, Lh)
        data = {
            "active_mode": self.view.mode,
            "mask_pose": self._pose_to_jsonable(self.pose_mask) if self.pose_mask else None,
            "original_pose": self._pose_to_jsonable(self.pose_orig) if self.pose_orig else None,
            "H_mat": H_homo.tolist(),
            "H_mat_level_0": H_mat_level_0.tolist()
        }
        with open(os.path.join(self.run_dir, INITIAL_ALIGNMENT), "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {INITIAL_ALIGNMENT}")

        # --------------------------------
        # save overlay images
        # --------------------------------
        try:
            run_dir = Path(self.run_dir)

            # Original-space overlay preview (the mask overlay was removed).
            H_gui2he = self.view.get_homography_dapi_to_he()
            moving_path, fixed_path = display_image_paths(run_dir)
            he_orig_bgr = cv2.imread(str(fixed_path), cv2.IMREAD_COLOR)
            dapi_raw_bgr = cv2.imread(str(moving_path), cv2.IMREAD_COLOR)
            if he_orig_bgr is None or dapi_raw_bgr is None:
                raise RuntimeError("Failed to read original images for overlay saving.")
            warped = cv2.warpPerspective(
                dapi_raw_bgr, H_gui2he,
                (he_orig_bgr.shape[1], he_orig_bgr.shape[0]),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            )
            out = cv2.addWeighted(he_orig_bgr, 0.7, warped, 0.8, 0)
            cv2.imwrite(str(run_dir / INITIAL_OVERLAY), out)
            print("[INFO] Saved overlay:", str(run_dir / INITIAL_OVERLAY), flush=True)

        except Exception as e:
            print(f"[WARN] overlay save failed: {e}", flush=True)

        log_event(self.run_dir, "system_ready_save_alignment")
        QMessageBox.information(self, "Step 2 Saved",
                                "Alignment saved successfully.\n\n"
                                "Return to the pipeline and run Step 3.")
        log_event(self.run_dir, "user_click_exit")
        log_event(self.run_dir, "system_ready_exit")
        QApplication.quit()


def display_image_paths(run_dir):
    roles = run_roles.read(run_dir)
    paths = []
    for role, side in ((run_roles.MOVING, roles.moving), (run_roles.FIXED, roles.fixed)):
        template = "1_{token}_lut.png" if side.modality == "dapi" else "1_{token}_level_image.png"
        path = run_roles.artifact_path(run_dir, role, template)
        if not path.exists():
            path = run_roles.artifact_path(run_dir, role, "1_{token}_level_image.png")
        paths.append(path)
    return tuple(paths)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    auto_orb = "--no-auto-orb" not in arguments
    positional = [argument for argument in arguments if not argument.startswith("-")]
    if not positional:
        print("Usage: python 2_manual_alignment.py <RUN_DIR> [--no-auto-orb]")
        sys.exit(2)

    run_dir = Path(positional[0]).resolve()
    if not run_dir.exists():
        print(f"[ERROR] RUN_DIR not found: {run_dir}")
        sys.exit(2)

    info_path = run_dir / "images_info.json"
    if not info_path.exists():
        print(f"[ERROR] images_info.json not found in RUN_DIR: {info_path}")
        sys.exit(2)

    with open(info_path, "r") as f:
        info = json.load(f)

    roles = run_roles.read(info)
    dapi_orig_path, he_orig_path = display_image_paths(run_dir)
    he_mask_path = he_orig_path
    dapi_mask_path = dapi_orig_path

    # The mask overlay view was removed; only the original level images are
    # required. Fall back to the originals for the (unused) mask pixmaps so the
    # view still constructs if the old mask files are absent.
    missing = [p for p in [he_orig_path, dapi_orig_path] if not p.exists()]
    if missing:
        print("[ERROR] Missing required Step 1 outputs:")
        for p in missing:
            print("  -", p)
        sys.exit(2)
    if roles.moving.gui_affine is None:
        print("[ERROR] moving gui_affine missing in images_info.json")
        sys.exit(2)
    dapi_gui_affine = np.array(roles.moving.gui_affine, dtype=np.float64)

    app = QApplication(sys.argv)
    window = ManualAlignWindow(
        run_dir=str(run_dir),
        he_mask_path=str(he_mask_path),
        dapi_mask_path=str(dapi_mask_path),
        he_orig_path=str(he_orig_path),
        dapi_orig_path=str(dapi_orig_path),
        dapi_gui_affine=dapi_gui_affine,
        info=info,
        auto_orb=auto_orb,
    )
    window.show()
    log_event(run_dir, "system_ready_initial_start")
    sys.exit(app.exec_())
