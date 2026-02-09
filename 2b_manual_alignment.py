import math
import numpy as np
import cv2
import os, sys, json
from pathlib import Path
from PyQt5.QtCore import Qt, QPointF, QRectF, QProcess, QEvent
from PyQt5.QtGui import QPixmap, QTransform, QColor, QPainterPath, QImage
from PyQt5.QtWidgets import (
    QApplication, QGraphicsView, QGraphicsScene,
    QGraphicsItem, QGraphicsRectItem, QGraphicsPixmapItem,
    QVBoxLayout, QWidget, QPushButton, QHBoxLayout,
    QMessageBox, QDialog, QLabel, QProgressBar
)
from my_utils import mask_to_rgba, warp_mask, overlay_rgba_on_bgr

def estimate_H_from_4corners(src, dst):
    # src,dst: (4,2) float32, order must match
    H = cv2.getPerspectiveTransform(src, dst)  # 3x3
    return H

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
    # We'll write channels accordingly.
    fr, fg, fb, fa = fg_rgba
    br, bg, bb, ba = bg_rgba

    # Get raw buffer
    ptr = img.bits()
    ptr.setsize(img.byteCount())

    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 4))

    # Build mask: treat any non-zero in RGB as foreground
    # (For grayscale masks, R==G==B, so this is fine.)
    mask = (arr[..., 2] > 0) | (arr[..., 1] > 0) | (arr[..., 0] > 0)

    # Write BGRA (not RGBA)
    arr[mask] = np.array([fb, fg, fr, fa], dtype=np.uint8)
    arr[~mask] = np.array([bb, bg, br, ba], dtype=np.uint8)

    return QPixmap.fromImage(img)

def qpixmap_to_bgr(pix: QPixmap) -> np.ndarray:
    """QPixmap -> BGR uint8 (H,W,3)."""
    img = pix.toImage().convertToFormat(QImage.Format_ARGB32)
    w, h = img.width(), img.height()

    ptr = img.bits()
    ptr.setsize(img.byteCount())
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 4))  # BGRA

    bgr = arr[..., :3].copy()  # BG R
    return bgr

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
    """
    Overlay item that supports:
      - drag translate (stored in transform.translate)
      - corner resize (non-uniform, Shift for uniform)
      - Q/E rotation (rotation property)
    """

    def __init__(self, pixmap):
        super().__init__(pixmap)
        self.setOpacity(1.0)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)

        self.handles = []
        self._dragging_overlay = False
        self._dragging_handle = False
        self._active_handle_idx = None
        self._press_pos = None
        self._orig_transform = None

    def update_handles(self):
        rect = self.boundingRect()
        corners_local = [
            QPointF(rect.left(), rect.top()),
            QPointF(rect.right(), rect.top()),
            QPointF(rect.right(), rect.bottom()),
            QPointF(rect.left(), rect.bottom())
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
                self._dragging_handle = True
                self._active_handle_idx = it.idx
                self._press_pos = scene_pt
                self._orig_transform = self.transform()
                event.accept()
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
        if self._dragging_handle:
            current_pos = event.scenePos()
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

            if event.modifiers() & Qt.ShiftModifier:
                s = sx if abs(sx - 1.0) >= abs(sy - 1.0) else sy
                s = max(s, 0.05)
                sx = sy = s

            T = QTransform()
            T.translate(fixed_pt.x(), fixed_pt.y())
            T.rotate(self.rotation())
            T.scale(sx, sy)
            T.rotate(-self.rotation())
            T.translate(-fixed_pt.x(), -fixed_pt.y())

            newT = T * self._orig_transform
            self.setTransform(newT)
            self.update_handles()
            event.accept()
            return

        if self._dragging_overlay:
            current_pos = event.scenePos()
            delta = current_pos - self._press_pos

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

class ManualAlignView(QGraphicsView):
    """
    支持两套底图/overlay：
      - mask 模式：he_mask_pix / dapi_mask_pix
      - original 模式：he_orig_pix / dapi_orig_pix

    swap 时记录当前 overlay pose，并把 pose 应用到另一套 pixmap 上。
    """
    MODE_MASK = "mask"
    MODE_ORIG = "original"
    def __init__(self, he_mask_path, dapi_mask_path, he_orig_path, dapi_orig_path, case_id):
        super().__init__()
        self.case_id = int(case_id)   # ✅ 保存到 view
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

        self.fit_bg_to_view()
        self.init_overlay_pose(scale=0.85)

        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

        self._overlay_hidden_by_key = False
        self.installEventFilter(self)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

    def leaveEvent(self, event):
        clear_override_cursor()
        if self._overlay_hidden_by_key:
            self._set_overlay_hidden(False)
        super().leaveEvent(event)

    def fit_bg_to_view(self):
        self.setSceneRect(self.he_item.boundingRect())
        self.fitInView(self.he_item, Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_bg_to_view()

    def wheelEvent(self, event):
        event.ignore()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            T = QTransform(self.overlay.transform())
            T.rotate(-5)
            self.overlay.setTransform(T)
            self.overlay.update_handles()
        elif event.key() == Qt.Key_E:
            T = QTransform(self.overlay.transform())
            T.rotate(5)
            self.overlay.setTransform(T)
            self.overlay.update_handles()
        else:
            super().keyPressEvent(event)

    def init_overlay_pose(self, scale=0.35):
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

    # -------------------------
    # Pose record / apply
    # -------------------------
    def get_overlay_pose(self):
        """返回 overlay 当前 pose（必须包含 pos/rotation/transform）。"""
        return {
            "pos": QPointF(self.overlay.pos()),
            "rotation": float(self.overlay.rotation()),
            "transform": QTransform(self.overlay.transform()),
        }

    def apply_overlay_pose_keep_scene_center(self, pose):
        """
        把 pose 应用到 overlay，同时保持“overlay 的 scene center 不变”，避免换 pixmap 后中心漂移。
        """
        # old scene center (based on current pixmap)
        old_center_scene = self.overlay.mapToScene(self.overlay.boundingRect().center())

        # apply pose
        self.overlay.setRotation(pose["rotation"])
        self.overlay.setTransform(pose["transform"])
        self.overlay.setPos(pose["pos"])

        # after applying, compute new center and compensate by shifting pos
        new_center_scene = self.overlay.mapToScene(self.overlay.boundingRect().center())
        delta = old_center_scene - new_center_scene
        self.overlay.setPos(self.overlay.pos() + delta)

        self.overlay.update_handles()

    def swap_mode(self):
        """
        核心：按 swap 的时候
          1) 记录当前 overlay pose
          2) 切换底图 + overlay pixmap
          3) 把 pose 应用到新 pixmap 上（并用 center 补偿）
        """
        clear_override_cursor()

        pose = self.get_overlay_pose()

        if self.mode == self.MODE_MASK:
            # switch to original
            self.mode = self.MODE_ORIG
            self.he_item.setPixmap(self.he_orig_pix)
            self.overlay.setOpacity(0.5)
            self.overlay.setPixmap(self.dapi_orig_pix)
            self.apply_overlay_pose_keep_scene_center(pose)

        else:
            # switch to mask
            self.mode = self.MODE_MASK
            self.he_item.setPixmap(self.he_mask_pix)

            self.overlay.setOpacity(1)
            self.overlay.setPixmap(self.dapi_mask_pix)
            self.apply_overlay_pose_keep_scene_center(pose)

        self.fit_bg_to_view()
        self.overlay.update_handles()

    def get_corners_dapi_to_he(self):
        """
        返回 4 对点：
          src = DAPI local(pixel) corners
          dst = HE pixel corners (scene coords)
        顺序: TL, TR, BL, BR
        """
        rect = self.overlay.boundingRect()  # DAPI local 坐标系里的矩形

        # DAPI local corners (pixel coords)
        src_qt = [
            rect.topLeft(),
            rect.topRight(),
            rect.bottomLeft(),
            rect.bottomRight(),
        ]

        # Map to HE(scene) coords
        dst_qt = [self.overlay.mapToScene(p) for p in src_qt]

        src = np.array([[p.x(), p.y()] for p in src_qt], dtype=np.float32)
        dst = np.array([[p.x(), p.y()] for p in dst_qt], dtype=np.float32)
        return src, dst

    def _set_overlay_hidden(self, hide: bool):
        """统一入口，避免重复 setVisible。"""
        if hide == self._overlay_hidden_by_key:
            return
        self._overlay_hidden_by_key = hide
        self.overlay.set_visible_all(not hide)

    def eventFilter(self, obj, event):
        et = event.type()

        # 兜底：有些情况下 Meta 的 press/release 不稳定，
        # 用 modifiers 也能判断当前是否按着 Command。
        def meta_down(e):
            try:
                return bool(e.modifiers() & Qt.MetaModifier)
            except Exception:
                return False

        if et == QEvent.KeyPress:
            # Command 按下（或任何按键但当前 meta 处于按下状态）
            if event.key() == Qt.Key_Meta or meta_down(event):
                self._set_overlay_hidden(True)
                return False  # 不要 consume，避免吃掉 Cmd+Q / Cmd+W 等

        elif et == QEvent.KeyRelease:
            # Command 松开
            if event.key() == Qt.Key_Meta:
                self._set_overlay_hidden(False)
                return False

            # 兜底：如果松开的是别的键，但此时 meta 已不再按下，也恢复
            if self._overlay_hidden_by_key and not meta_down(event):
                self._set_overlay_hidden(False)
                return False

        elif et in (QEvent.FocusOut, QEvent.WindowDeactivate):
            # 切走窗口/失去焦点时，强制恢复，避免“卡在隐藏状态”
            if self._overlay_hidden_by_key:
                self._set_overlay_hidden(False)

        return super().eventFilter(obj, event)

class ManualAlignWindow(QWidget):
    def __init__(self, run_dir, he_mask_path, dapi_mask_path, he_orig_path, dapi_orig_path, dapi_gui_affine, case_id=0):
        super().__init__()
        self.run_dir = str(run_dir)
        self.dapi_gui_affine = np.asarray(dapi_gui_affine, dtype=np.float32)
        self.setWindowTitle("Manual Alignment + Swap Mask/Original")

        self.view = ManualAlignView(he_mask_path, dapi_mask_path, he_orig_path, dapi_orig_path, case_id=case_id)
        self.resize(1400, 900)

        self.btn_reset = QPushButton("Reset")
        self.btn_swap = QPushButton("Swap Mask/Original")
        self.btn_save = QPushButton("Save Alignment")

        self.btn_reset.clicked.connect(self.on_reset)
        self.btn_swap.clicked.connect(self.on_swap)
        self.btn_save.clicked.connect(self.on_save)

        hl = QHBoxLayout()
        hl.addWidget(self.btn_reset)
        hl.addWidget(self.btn_swap)
        hl.addWidget(self.btn_save)

        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.addLayout(hl)
        self.setLayout(layout)

        # 可选：分别缓存两套模式的 pose（这样你 save 时可以同时写出）
        self.pose_mask = None
        self.pose_orig = None

    def on_reset(self):
        clear_override_cursor()
        self.view.init_overlay_pose(scale=0.85)

    def on_swap(self):
        # swap 前先把当前 pose 存到对应模式
        cur_pose = self.view.get_overlay_pose()
        if self.view.mode == self.view.MODE_MASK:
            self.pose_mask = cur_pose
        else:
            self.pose_orig = cur_pose

        self.view.swap_mode()

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

    def on_save(self):
        # 先更新当前模式 pose
        cur_pose = self.view.get_overlay_pose()
        if self.view.mode == self.view.MODE_MASK:
            self.pose_mask = cur_pose
        else:
            self.pose_orig = cur_pose

        # 1) 四角：GUI DAPI -> HE
        src_gui, dst_he = self.view.get_corners_dapi_to_he()  # (4,2) float32
        # 2) GUI -> original（和 blob matching 一样）
        T_gui = np.array(self.dapi_gui_affine, dtype=np.float32)  # original -> GUI
        if T_gui.shape == (2, 3):
            T_gui = np.vstack([T_gui, [0, 0, 1]]).astype(np.float32)
        T_gui_inv = np.linalg.inv(T_gui)  # GUI -> original
        src_gui_h = np.hstack([src_gui, np.ones((len(src_gui), 1), dtype=np.float32)])  # (4,3)
        src_orig = (T_gui_inv @ src_gui_h.T).T[:, :2].astype(np.float32)  # (4,2)
        # 3) homography：ORIG DAPI -> HE
        if src_orig.shape[0] != 4 or dst_he.shape[0] != 4:
            raise ValueError("Homography needs exactly 4 corner correspondences")
        H_homo = cv2.getPerspectiveTransform(src_orig.astype(np.float32),
                                             dst_he.astype(np.float32))  # (3,3)
        data = {
            "active_mode": self.view.mode,
            "mask_pose": self._pose_to_jsonable(self.pose_mask) if self.pose_mask else None,
            "original_pose": self._pose_to_jsonable(self.pose_orig) if self.pose_orig else None,
            "H_mat": H_homo.tolist(),
        }
        with open(os.path.join(self.run_dir, "manual_initial_alignment.json"), "w") as f:
            json.dump(data, f, indent=2)
        print("Saved manual_alignment.json")

        # --------------------------------
        # save overlay images
        # --------------------------------
        try:
            run_dir = Path(self.run_dir)

            # ----- helpers -----
            def _as_3x3(M):
                M = np.asarray(M, dtype=np.float32)
                if M.shape == (2, 3):
                    M = np.vstack([M, [0, 0, 1]]).astype(np.float32)
                if M.shape != (3, 3):
                    raise ValueError(f"Expected (2,3) or (3,3), got {M.shape}")
                return M

            def _compute_H_gui2he():
                # Uses CURRENT view.mode + CURRENT overlay pose + CURRENT pixmaps
                src_gui, dst_he = self.view.get_corners_dapi_to_he()   # (4,2)
                src_gui = np.asarray(src_gui, dtype=np.float32)
                dst_he  = np.asarray(dst_he,  dtype=np.float32)
                H = cv2.getPerspectiveTransform(src_gui, dst_he)       # 3x3
                return H.astype(np.float32)

            def _warp_overlay(bgr_bg, bgr_fg, H_3x3, out_path, alpha_bg=0.7, alpha_fg=0.8, interp=cv2.INTER_LINEAR):
                H_3x3 = _as_3x3(H_3x3)
                warped = cv2.warpPerspective(
                    bgr_fg,
                    H_3x3,
                    (bgr_bg.shape[1], bgr_bg.shape[0]),
                    flags=interp,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                out = cv2.addWeighted(bgr_bg, float(alpha_bg), warped, float(alpha_fg), 0)
                cv2.imwrite(str(out_path), out)

            # ----- stash current GUI state so we can restore -----
            mode0 = self.view.mode
            pose0 = self.view.get_overlay_pose()

            # Make sure we have both poses recorded (so both overlays reflect what you saw)
            if self.pose_mask is None:
                # we at least have current pose for current mode already
                if self.view.mode == self.view.MODE_MASK:
                    self.pose_mask = self.view.get_overlay_pose()
            if self.pose_orig is None:
                if self.view.mode == self.view.MODE_ORIG:
                    self.pose_orig = self.view.get_overlay_pose()

            # =====================================================
            # 1) Save MASK overlay  (mask images are already in GUI space)
            # =====================================================
            if self.view.mode != self.view.MODE_MASK:
                self.view.swap_mode()
            if self.pose_mask is not None:
                self.view.apply_overlay_pose_keep_scene_center(self.pose_mask)
            H_gui2he_mask = _compute_H_gui2he()
            he_mask_bgr = cv2.imread(str(run_dir / "1_confirmed_he_dense_mask.png"))
            dapi_mask_bgr = cv2.imread(str(run_dir / "1_confirmed_dapi_mask.png"))
            warped_dapi_mask = warp_mask(
                dapi_mask_bgr,
                H_gui2he_mask,
                (he_mask_bgr.shape[1], he_mask_bgr.shape[0])
            )
            he_rgba = mask_to_rgba(he_mask_bgr, color_rgb=(255, 0, 0), alpha=0.5)  # 红
            dapi_rgba = mask_to_rgba(warped_dapi_mask, color_rgb=(0, 0, 255), alpha=0.5)  # 蓝
            out = overlay_rgba_on_bgr(he_mask_bgr, he_rgba)
            out = overlay_rgba_on_bgr(out, dapi_rgba)
            cv2.imwrite(str(run_dir / "2_manual_overlay_mask.png"), out)
            # =====================================================
            # 2) Save ORIGINAL overlay (mask images are already in GUI space)
            # =====================================================
            if self.view.mode != self.view.MODE_ORIG:
                self.view.swap_mode()
            if self.pose_orig is not None:
                self.view.apply_overlay_pose_keep_scene_center(self.pose_orig)
            H_gui2he_orig = _compute_H_gui2he()
            he_orig_bgr = cv2.imread(str(run_dir / "1_he_level_image.png"), cv2.IMREAD_COLOR)
            dapi_raw_bgr = cv2.imread(str(run_dir / "1_dapi_lut.png"), cv2.IMREAD_COLOR)
            if he_orig_bgr is None or dapi_raw_bgr is None:
                raise RuntimeError("Failed to read original images for overlay saving.")
            _warp_overlay(
                he_orig_bgr,
                dapi_raw_bgr,
                H_gui2he_orig,
                out_path=(run_dir / "2_manual_overlay_original.png"),
                alpha_bg=0.7, alpha_fg=0.8,
                interp=cv2.INTER_LINEAR,
            )
            print("[INFO] Saved overlays:",
                  str(run_dir / "2_manual_overlay_original.png"),
                  str(run_dir / "2_manual_overlay_mask.png"),
                  flush=True)
            if self.view.mode != mode0:
                self.view.swap_mode()
            self.view.apply_overlay_pose_keep_scene_center(pose0)

        except Exception as e:
            print(f"[WARN] overlay save failed: {e}", flush=True)

        QMessageBox.information(self, "Step 2 Saved",
                                "Alignment saved successfully.\n\n"
                                "Return to the pipeline and run Step 3.")
        QApplication.quit()



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 2_manual_alignment.py <RUN_DIR>")
        sys.exit(2)

    run_dir = Path(sys.argv[1]).resolve()
    if not run_dir.exists():
        print(f"[ERROR] RUN_DIR not found: {run_dir}")
        sys.exit(2)

    info_path = run_dir / "images_info.json"
    if not info_path.exists():
        print(f"[ERROR] images_info.json not found in RUN_DIR: {info_path}")
        sys.exit(2)

    with open(info_path, "r") as f:
        info = json.load(f)

    RUN_ID = info.get("RUN_ID", info.get("run_id", run_dir.name.replace("runs_", "", 1)))
    he_mask_path   = run_dir / "1_confirmed_he_dense_mask.png"
    dapi_mask_path = run_dir / "1_confirmed_dapi_mask.png"
    he_orig_path   = run_dir / "1_he_level_image.png"
    dapi_orig_path = run_dir / "1_dapi_lut.png"

    missing = [p for p in [he_mask_path, dapi_mask_path, he_orig_path, dapi_orig_path] if not p.exists()]
    if missing:
        print("[ERROR] Missing required Step 1 outputs:")
        for p in missing:
            print("  -", p)
        sys.exit(2)

    if "DAPI_gui_affine" not in info:
        print("[ERROR] DAPI_gui_affine missing in images_info.json")
        sys.exit(2)
    dapi_gui_affine = np.array(info["DAPI_gui_affine"], dtype=np.float32)

    app = QApplication(sys.argv)
    case_id = int(info.get("DAPI_orientation_case", 0))
    window = ManualAlignWindow(
        run_dir=str(run_dir),
        he_mask_path=str(he_mask_path),
        dapi_mask_path=str(dapi_mask_path),
        he_orig_path=str(he_orig_path),
        dapi_orig_path=str(dapi_orig_path),
        dapi_gui_affine=dapi_gui_affine,
        case_id=case_id,
    )
    window.show()
    sys.exit(app.exec_())
