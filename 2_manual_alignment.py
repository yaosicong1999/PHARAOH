import sys
import json
import math
from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QPixmap, QTransform, QColor, QPen
from PyQt5.QtWidgets import (
    QApplication, QGraphicsView, QGraphicsScene,
    QGraphicsItem, QGraphicsRectItem, QGraphicsPixmapItem,
    QVBoxLayout, QWidget, QPushButton, QHBoxLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QGraphicsItem, QGraphicsRectItem
from PyQt5.QtGui import QPainterPath
from PyQt5.QtGui import QImage


from PyQt5.QtWidgets import QApplication

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

    # bigger hit area
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

    def __init__(self, pixmap):
        super().__init__(pixmap)
        self.setOpacity(1.0)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)

        self.setAcceptHoverEvents(True)

        self.handles = []  # will be created in attach_handles(scene)
        self._dragging_overlay = False
        self._dragging_handle = False
        self._active_handle_idx = None
        self._press_pos = None
        self._orig_rect = None
        self._orig_transform = None

    def bounding_rect_in_scene(self):
        return self.mapToScene(self.boundingRect()).boundingRect()

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
        if not self._dragging_handle:
            set_override_cursor(Qt.OpenHandCursor)
        event.accept()

    def hoverMoveEvent(self, event):
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
        set_override_cursor(Qt.OpenHandCursor)  # 如果鼠标还在overlay内会是open hand
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging_handle:
            current_pos = event.scenePos()
            opp_idx = (self._active_handle_idx + 2) % 4

            # fixed point in scene coords (center of opposite handle square)
            fixed_pt = self.handles[opp_idx].scenePos()

            # vectors from fixed point
            v0 = self._press_pos - fixed_pt
            v1 = current_pos - fixed_pt

            # avoid degenerate
            if abs(v0.x()) < 1e-6 and abs(v0.y()) < 1e-6:
                return

            # Decompose in OVERLAY local axes (taking current rotation into account)
            theta = math.radians(self.rotation())  # Qt rotation is degrees
            ux = QPointF(math.cos(theta), math.sin(theta))  # local +x axis in scene
            uy = QPointF(-math.sin(theta), math.cos(theta))  # local +y axis in scene

            def dot(a: QPointF, b: QPointF) -> float:
                return a.x() * b.x() + a.y() * b.y()

            # signed components along local axes
            v0x, v0y = dot(v0, ux), dot(v0, uy)
            v1x, v1y = dot(v1, ux), dot(v1, uy)

            # compute non-uniform scale (independent)
            eps = 1e-6
            sx = v1x / (v0x if abs(v0x) > eps else (eps if v0x >= 0 else -eps))
            sy = v1y / (v0y if abs(v0y) > eps else (eps if v0y >= 0 else -eps))

            # optional: prevent flipping through zero (keep positive scale)
            sx = max(sx, 0.05)
            sy = max(sy, 0.05)

            if event.modifiers() & Qt.ShiftModifier:
                # 用“变化更大”的那个来主导，保证手感更跟手
                s = sx if abs(sx - 1.0) >= abs(sy - 1.0) else sy
                s = max(s, 0.05)
                sx = sy = s

            # apply scale around fixed point (scene coords)
            T = QTransform()
            T.translate(fixed_pt.x(), fixed_pt.y())
            T.rotate(self.rotation())  # keep current rotation consistent
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
        fixed_pt = self.handles[opp_idx].scenePos()  # handle center (because we centered it)

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
    def __init__(self, he_path, dapi_path):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        he_pix = mask_to_colored_pixmap(he_path, fg_rgba=(255,0,0,255), bg_rgba=(0, 0, 0, 255))
        dapi_pix = mask_to_colored_pixmap(
            dapi_path,
            fg_rgba=(50, 120, 255, 120),  # 半透明蓝（你可以调 alpha）
            bg_rgba=(0, 0, 0, 0)  # 背景透明
        )
        self.he_item = QGraphicsPixmapItem(he_pix)
        self.scene.addItem(self.he_item)

        self.overlay = DapiOverlayItem(dapi_pix)
        self.scene.addItem(self.overlay)

        self.overlay.attach_handles(self.scene)
        self.fit_bg_to_view()
        self.init_overlay_pose(scale=0.35)  # 0.25~0.5 自己调

        self.setDragMode(QGraphicsView.NoDrag)

    def leaveEvent(self, event):
        clear_override_cursor()
        super().leaveEvent(event)

    def fit_bg_to_view(self):
        # 让视图自动缩放到刚好容纳底图
        self.setSceneRect(self.he_item.boundingRect())
        self.fitInView(self.he_item, Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_bg_to_view()
        # 注意：resize 时不要反复重置 overlay，否则用户对齐会被打断

    def wheelEvent(self, event):
        event.ignore()

    def keyPressEvent(self, event):
        if event.key()==Qt.Key_Q:
            T = QTransform(self.overlay.transform())
            T.rotate(-5)
            self.overlay.setTransform(T)
            self.overlay.update_handles()
        elif event.key()==Qt.Key_E:
            T = QTransform(self.overlay.transform())
            T.rotate(5)
            self.overlay.setTransform(T)
            self.overlay.update_handles()
        else:
            super().keyPressEvent(event)
    def init_overlay_pose(self, scale=0.35):
        """
        scale: overlay 相对原尺寸的缩放倍数（初始）
        """
        # 1) 取底图中心（scene coords）
        bg_rect = self.he_item.sceneBoundingRect()
        bg_center = bg_rect.center()

        # 2) overlay 原始中心（local -> scene），先把 overlay 放到(0,0)的默认状态
        self.overlay.setTransform(QTransform())
        self.overlay.setRotation(0)

        ov_rect = self.overlay.boundingRect()
        ov_center_local = ov_rect.center()

        # 3) 先把 overlay 的 local center 移到 scene 的 bg_center（平移）
        #    即：overlay.mapToScene(ov_center_local) == bg_center
        #    在 transform 里做 translate：
        T = QTransform()
        T.translate(bg_center.x() - ov_center_local.x(), bg_center.y() - ov_center_local.y())
        self.overlay.setTransform(T)

        # 4) 再以 bg_center 为中心做缩放
        S = QTransform()
        S.translate(bg_center.x(), bg_center.y())
        S.scale(scale, scale)
        S.translate(-bg_center.x(), -bg_center.y())

        self.overlay.setTransform(S * self.overlay.transform())
        self.overlay.update_handles()


class ManualAlignWindow(QWidget):
    def __init__(self, he_path, dapi_path):
        super().__init__()
        self.setWindowTitle("Manual Alignment Fixed-Corner Scale")

        self.view = ManualAlignView(he_path, dapi_path)
        self.resize(1400, 900)  # 你想要多大都行
        self.btn_reset = QPushButton("Reset")
        self.btn_save  = QPushButton("Save Alignment")

        self.btn_reset.clicked.connect(self.on_reset)
        self.btn_save.clicked.connect(self.on_save)

        hl = QHBoxLayout()
        hl.addWidget(self.btn_reset)
        hl.addWidget(self.btn_save)

        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.addLayout(hl)
        self.setLayout(layout)

    def on_reset(self):
        self.view.overlay.setTransform(QTransform())
        self.view.overlay.update_handles()

    def on_save(self):
        t = self.view.overlay.transform()
        data = {
            "m11": t.m11(), "m12": t.m12(),
            "m21": t.m21(), "m22": t.m22(),
            "dx":  t.dx(),  "dy":  t.dy()
        }
        with open("manual_alignment.json", "w") as f:
            json.dump(data, f, indent=2)
        print("Saved manual_alignment.json")
        QApplication.quit()



if __name__ == "__main__":
    # if len(sys.argv) < 3:
    #     print("Usage: python 2_manual_alignment.py he_mask.png dapi_mask.png")
    #     sys.exit(1)

    he_mask_path = "/Users/sicongy/Documents/GitHub/Project2/runs_202601161626/1_confirmed_he_dense_mask.png"
    dapi_mask_path ="/Users/sicongy/Documents/GitHub/Project2/runs_202601161626/1_confirmed_dapi_mask.png"

    app = QApplication(sys.argv)
    window = ManualAlignWindow(he_mask_path, dapi_mask_path)
    window.show()
    sys.exit(app.exec_())