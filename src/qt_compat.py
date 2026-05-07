"""
Qt 兼容层：优先使用 PySide6，回退到 PyQt5。
所有模块统一从这里导入 Qt 符号，避免硬编码绑定。
"""
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QTextEdit, QTableWidget, QTableWidgetItem,
        QComboBox, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
        QCheckBox, QFileDialog, QMessageBox, QProgressBar, QSplitter,
        QTabWidget, QScrollArea, QFrame, QHeaderView, QDialog,
        QListWidget, QListWidgetItem, QTextBrowser, QGridLayout,
        QSizePolicy, QLineEdit, QSlider, QRadioButton, QButtonGroup,
        QStackedWidget, QToolButton, QMenu, QAbstractItemView,
        QDateEdit, QCompleter,
    )
    from PySide6.QtCore import (
        Qt, QTimer, QThread, QObject, QPoint, QSize, QRect,
        QPropertyAnimation, QEasingCurve, QDate, QDateTime,
        Signal, Slot,
    )
    from PySide6.QtGui import (
        QFont, QColor, QMouseEvent, QPainter, QPen, QBrush,
        QPainterPath, QPixmap, QIcon, QTextCursor, QFontMetrics,
        QLinearGradient, QAction,
    )
    pyqtSignal = Signal
    pyqtSlot = Slot
    _QT_BINDING = "PySide6"

except ImportError:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QTextEdit, QTableWidget, QTableWidgetItem,
        QComboBox, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
        QCheckBox, QFileDialog, QMessageBox, QProgressBar, QSplitter,
        QTabWidget, QScrollArea, QFrame, QHeaderView, QDialog,
        QListWidget, QListWidgetItem, QTextBrowser, QGridLayout,
        QSizePolicy, QLineEdit, QSlider, QRadioButton, QButtonGroup,
        QStackedWidget, QToolButton, QMenu, QAction, QAbstractItemView,
        QDateEdit, QCompleter,
    )
    from PyQt5.QtCore import (
        Qt, QTimer, QThread, QObject, QPoint, QSize, QRect,
        QPropertyAnimation, QEasingCurve, QDate, QDateTime,
        pyqtSignal as Signal, pyqtSlot as Slot,
    )
    from PyQt5.QtGui import (
        QFont, QColor, QMouseEvent, QPainter, QPen, QBrush,
        QPainterPath, QPixmap, QIcon, QTextCursor, QFontMetrics,
        QLinearGradient,
    )
    try:
        from PyQt5.QtWidgets import QAction
    except ImportError:
        from PyQt5.QtGui import QAction
    pyqtSignal = Signal
    pyqtSlot = Slot
    _QT_BINDING = "PyQt5"

__all__ = [
    "QApplication", "QMainWindow", "QWidget", "QVBoxLayout", "QHBoxLayout",
    "QPushButton", "QLabel", "QTextEdit", "QTableWidget", "QTableWidgetItem",
    "QComboBox", "QGroupBox", "QFormLayout", "QSpinBox", "QDoubleSpinBox",
    "QCheckBox", "QFileDialog", "QMessageBox", "QProgressBar", "QSplitter",
    "QTabWidget", "QScrollArea", "QFrame", "QHeaderView", "QDialog",
    "QListWidget", "QListWidgetItem", "QTextBrowser", "QGridLayout",
    "QSizePolicy", "QLineEdit", "QSlider", "QRadioButton", "QButtonGroup",
    "QStackedWidget", "QToolButton", "QMenu", "QAction", "QAbstractItemView",
    "Qt", "QTimer", "QThread", "QObject", "QPoint", "QSize", "QRect",
    "QPropertyAnimation", "QEasingCurve", "QDate", "QDateTime",
    "Signal", "Slot", "pyqtSignal", "pyqtSlot",
    "QFont", "QColor", "QMouseEvent", "QPainter", "QPen", "QBrush",
    "QPainterPath", "QPixmap", "QIcon", "QTextCursor", "QFontMetrics",
    "QLinearGradient",
    "_QT_BINDING",
]
