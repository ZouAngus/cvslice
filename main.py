"""CVSlice entry point."""
import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from cvslice.ui import ClipAnnotator


def main():
    # High-DPI support
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    # Set a cross-platform font that handles CJK + digits reliably
    font = QFont()
    families = ["Microsoft YaHei UI", "Segoe UI", "PingFang SC",
                "Noto Sans CJK SC", "Helvetica Neue", "Arial"]
    font.setFamilies(families) if hasattr(font, 'setFamilies') else font.setFamily("Segoe UI")
    font.setPointSize(9)
    app.setFont(font)

    win = ClipAnnotator()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
