"""Custom video label with mouse event forwarding for joint dragging."""
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, pyqtSignal, QPoint


class VideoLabel(QLabel):
    """QLabel subclass that emits mouse signals with frame-space coordinates.

    The label displays a scaled pixmap. Mouse coordinates are mapped back
    to the original (unscaled) frame coordinate space.
    """

    # Signals: (frame_x, frame_y)
    mouse_pressed = pyqtSignal(int, int)
    mouse_moved = pyqtSignal(int, int)
    mouse_released = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame_w = 0   # original frame width
        self._frame_h = 0   # original frame height

    def set_frame_size(self, w: int, h: int):
        """Set the original frame dimensions for coordinate mapping."""
        self._frame_w = w
        self._frame_h = h

    def _to_frame_coords(self, pos: QPoint) -> tuple[int, int] | None:
        """Map widget pixel position to original frame coordinates."""
        pix = self.pixmap()
        if pix is None or pix.isNull():
            return None
        # The pixmap is centered in the label with aspect ratio preserved
        pw, ph = pix.width(), pix.height()
        lw, lh = self.width(), self.height()
        # Offset of pixmap within label
        ox = (lw - pw) // 2
        oy = (lh - ph) // 2
        # Position relative to pixmap
        rx = pos.x() - ox
        ry = pos.y() - oy
        if rx < 0 or ry < 0 or rx >= pw or ry >= ph:
            return None
        # Scale to original frame coords
        if self._frame_w <= 0 or self._frame_h <= 0:
            return None
        fx = int(rx * self._frame_w / pw)
        fy = int(ry * self._frame_h / ph)
        return fx, fy

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            coords = self._to_frame_coords(event.pos())
            if coords:
                self.mouse_pressed.emit(*coords)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        coords = self._to_frame_coords(event.pos())
        if coords:
            self.mouse_moved.emit(*coords)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            coords = self._to_frame_coords(event.pos())
            if coords:
                self.mouse_released.emit(*coords)
        super().mouseReleaseEvent(event)
