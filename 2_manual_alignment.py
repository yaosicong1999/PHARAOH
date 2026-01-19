import sys
import json
import math
from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QPixmap, QTransform, QColor, QPainterPath, QImage
from PyQt5.QtWidgets import (
    QApplication, QGraphicsView, QGraphicsScene,
    QGraphicsItem, QGraphicsRectItem, QGraphicsPixmapItem,
    QVBoxLayout, QWidget, QPushButton, QHBoxLayout
)

def lut_pixmap_black_to_transparent(
    path: str,
    black_thresh: int = 10,
    keep_alpha: int = 160
) -> QPixmap:
    """
    Load a LUT/RGB image and make near-black pixels transparent.

    Args:
        path: input image path (png/jpg/tif -> QImage readable)
        black_thresh: treat pixels with max(R,G,B) <= black_thresh as black
        keep_alpha: alpha to assign for non-black pixels (0-255)

    Returns:
        QPixmap with black background transparent.
    """
    img = QImage(path).convertToFormat(QImage.Format_ARGB32)
    w, h = img.width(), img.height()

    for y in range(h):
        for x in range(w):
            c = QColor(img.pixel(x, y))
            r, g, b = c.red(), c.green(), c.blue()

            # "black" if all channels are very small
            if max(r, g, b) <= black_thresh:
                img.setPixelColor(x, y, QColor(r, g, b, 0))
            else:
                img.setPixelColor(x, y, QColor(r, g, b, keep_alpha))

    return QPixmap.fromImage(img)

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

def mask_to_colored_pixmap(path, fg_rgba=(255,0,0,255), bg_rgba=(0,0,0,0)):
    """
    Read a binary-ish mask image and colorize:
    - pixels > 0  -> fg_rgba
    - pixels == 0 -> bg_rgba
    """
    img = QImage(path).convertToFormat(QImage.Format_ARGB32)
    w, h = img.width(), img.height()

    fr, fg, fb, fa = fg_rgba
    br, bg, bb, ba = bg_rgba

    for y in range(h):
        for x in range(w):
            c = QColor(img.pixel(x, y))
            v = c.red()  # grayscale assumption; ok for mask png
            if v > 0:
                img.setPixelColor(x, y, QColor(fr, fg, fb, fa))
            else:
                img.setPixelColor(x, y, QColor(br, bg, bb, ba))

    return QPixmap.fromImage(img)


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


class ManualAlignView(QGraphicsView):
    """
    支持两套底图/overlay：
      - mask 模式：he_mask_pix / dapi_mask_pix
      - original 模式：he_orig_pix / dapi_orig_pix

    swap 时记录当前 overlay pose，并把 pose 应用到另一套 pixmap 上。
    """

    MODE_MASK = "mask"
    MODE_ORIG = "original"

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

        # originals: 你可以按需要在这里做 normalize / 伪彩 / alpha
        self.he_orig_pix = QPixmap(he_orig_path)
        self.dapi_orig_pix = lut_pixmap_black_to_transparent(
            dapi_orig_path,
            black_thresh=10,  # 黑色阈值：越大越“更容易变透明”
            keep_alpha=160  # 非黑像素透明度：越小越透明
        )
        # ---- items ----
        self.he_item = QGraphicsPixmapItem(self.he_mask_pix)
        self.scene.addItem(self.he_item)

        self.overlay = DapiOverlayItem(self.dapi_mask_pix)
        self.scene.addItem(self.overlay)
        self.overlay.attach_handles(self.scene)

        self.mode = self.MODE_MASK

        self.fit_bg_to_view()
        self.init_overlay_pose(scale=0.85)

        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

    def leaveEvent(self, event):
        clear_override_cursor()
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

            # overlay: change pixmap first, then apply pose with center compensation
            self.overlay.setPixmap(self.dapi_orig_pix)
            self.apply_overlay_pose_keep_scene_center(pose)

        else:
            # switch to mask
            self.mode = self.MODE_MASK
            self.he_item.setPixmap(self.he_mask_pix)

            self.overlay.setPixmap(self.dapi_mask_pix)
            self.apply_overlay_pose_keep_scene_center(pose)

        self.fit_bg_to_view()
        self.overlay.update_handles()


class ManualAlignWindow(QWidget):
    def __init__(self, he_mask_path, dapi_mask_path, he_orig_path, dapi_orig_path):
        super().__init__()
        self.setWindowTitle("Manual Alignment + Swap Mask/Original")

        self.view = ManualAlignView(he_mask_path, dapi_mask_path, he_orig_path, dapi_orig_path)
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

        data = {
            "active_mode": self.view.mode,
            "mask_pose": self._pose_to_jsonable(self.pose_mask) if self.pose_mask else None,
            "original_pose": self._pose_to_jsonable(self.pose_orig) if self.pose_orig else None,
        }

        with open("manual_alignment.json", "w") as f:
            json.dump(data, f, indent=2)

        print("Saved manual_alignment.json")
        QApplication.quit()


if __name__ == "__main__":
    # 你需要把 original 路径补上
    he_mask_path = "/Users/sicongy/Documents/GitHub/Project2/runs_202601191418/1_confirmed_he_dense_mask.png"
    dapi_mask_path = "/Users/sicongy/Documents/GitHub/Project2/runs_202601191418/1_confirmed_dapi_mask.png"

    he_orig_path = "/Users/sicongy/Documents/GitHub/Project2/runs_202601191418/1_he_level_image.png"
    dapi_orig_path = "/Users/sicongy/Documents/GitHub/Project2/runs_202601191418/1_dapi_lut.png"

    app = QApplication(sys.argv)
    window = ManualAlignWindow(he_mask_path, dapi_mask_path, he_orig_path, dapi_orig_path)
    window.show()
    sys.exit(app.exec_())