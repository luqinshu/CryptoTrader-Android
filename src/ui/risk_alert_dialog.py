"""
风险预警弹窗：非阻塞式弹出警报，支持多级别颜色、自动关闭、操作建议。
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, Callable, Dict, List, Optional
from src.qt_compat import QApplication, QColor, QEasingCurve, QFont, QFrame, QHBoxLayout, QLabel, QObject, QPainter, QPainterPath, QPixmap, QPropertyAnimation, QPushButton, QRect, QSizePolicy, QTextEdit, QTimer, QVBoxLayout, QWidget, Qt, Signal, Slot



# ─── 级别颜色配置 ──────────────────────────────────────────────────────────────
_LEVEL_CFG = {
    "INFO":     {"bg": "#1a2940", "border": "#3498db", "icon": "ℹ️", "title_color": "#5dade2"},
    "WARNING":  {"bg": "#2d2000", "border": "#f39c12", "icon": "⚠️", "title_color": "#f5b041"},
    "CRITICAL": {"bg": "#2d0a00", "border": "#e74c3c", "icon": "🚨", "title_color": "#ec7063"},
    "DANGER":   {"bg": "#1a0020", "border": "#8e44ad", "icon": "💀", "title_color": "#a569bd"},
}
_DEFAULT_CFG = {"bg": "#1a1a2e", "border": "#555", "icon": "📢", "title_color": "#aaa"}


class AlertCard(QFrame):
    """单条风险预警卡片，支持滑入动画 + 自动关闭"""

    closed = Signal(object)   # emits self

    def __init__(self, alert, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._alert = alert
        self._auto_timer: Optional[QTimer] = None
        self._setup_ui()
        self._apply_style()

    def _setup_ui(self):
        cfg = _LEVEL_CFG.get(getattr(self._alert, "level", "INFO"), _DEFAULT_CFG)
        self.setFixedWidth(380)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)

        # ── 标题行 ────────────────────────────────────────────────────────────
        header = QHBoxLayout()
        header.setSpacing(6)

        icon_label = QLabel(cfg["icon"])
        icon_label.setFont(QFont("Segoe UI Emoji", 16))
        icon_label.setFixedWidth(24)
        header.addWidget(icon_label)

        title_label = QLabel(getattr(self._alert, "title", "风险预警"))
        title_label.setFont(QFont("", 13, QFont.Bold))
        title_label.setStyleSheet(f"color: {cfg['title_color']}; background: transparent;")
        title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        header.addWidget(title_label)

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(20, 20)
        close_btn.setStyleSheet(
            "QPushButton { color: #888; background: transparent; border: none; font-size: 12px; }"
            "QPushButton:hover { color: #fff; }"
        )
        close_btn.clicked.connect(self._dismiss)
        header.addWidget(close_btn)

        layout.addLayout(header)

        # ── 消息体 ────────────────────────────────────────────────────────────
        msg = getattr(self._alert, "message", "")
        if msg:
            msg_label = QLabel(msg)
            msg_label.setWordWrap(True)
            msg_label.setStyleSheet("color: #ccc; font-size: 12px; background: transparent;")
            layout.addWidget(msg_label)

        # ── 详情 ──────────────────────────────────────────────────────────────
        detail = getattr(self._alert, "detail", "")
        if detail:
            detail_label = QLabel(detail)
            detail_label.setWordWrap(True)
            detail_label.setStyleSheet("color: #999; font-size: 11px; background: transparent;")
            layout.addWidget(detail_label)

        # ── 建议操作 ──────────────────────────────────────────────────────────
        action = getattr(self._alert, "suggested_action", "")
        if action:
            action_label = QLabel(f"💡 {action}")
            action_label.setWordWrap(True)
            action_label.setStyleSheet(
                f"color: {cfg['title_color']}; font-size: 11px; "
                "background: rgba(255,255,255,0.05); border-radius: 4px; padding: 4px;"
            )
            layout.addWidget(action_label)

        # ── 时间戳 ────────────────────────────────────────────────────────────
        ts = getattr(self._alert, "timestamp", "")
        cat = getattr(self._alert, "category", "")
        meta = QLabel(f"{cat}  {ts[-8:] if ts else ''}")
        meta.setStyleSheet("color: #555; font-size: 10px; background: transparent;")
        meta.setAlignment(Qt.AlignRight)
        layout.addWidget(meta)

        # ── 自动关闭 ──────────────────────────────────────────────────────────
        auto_sec = getattr(self._alert, "auto_close_sec", 0)
        if auto_sec and auto_sec > 0:
            self._auto_timer = QTimer(self)
            self._auto_timer.setSingleShot(True)
            self._auto_timer.timeout.connect(self._dismiss)
            self._auto_timer.start(auto_sec * 1000)

    def _apply_style(self):
        cfg = _LEVEL_CFG.get(getattr(self._alert, "level", "INFO"), _DEFAULT_CFG)
        self.setStyleSheet(
            f"AlertCard {{"
            f"  background-color: {cfg['bg']};"
            f"  border: 1px solid {cfg['border']};"
            f"  border-left: 4px solid {cfg['border']};"
            f"  border-radius: 8px;"
            f"}}"
        )

    @Slot()
    def _dismiss(self):
        if self._auto_timer:
            self._auto_timer.stop()
        self.closed.emit(self)


class AlertOverlay(QWidget):
    """
    屏幕右下角的预警叠加层，管理多张 AlertCard 的堆叠显示。
    必须在主线程创建；跨线程 emit alert_queued 信号即可添加预警。
    """

    _alert_queued = Signal(object)   # 内部信号，供跨线程安全添加

    def __init__(self, parent: Optional[QWidget] = None, max_cards: int = 5):
        super().__init__(parent, Qt.Window | Qt.FramelessWindowHint |
                         Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setFixedWidth(400)

        self._max_cards = max_cards
        self._cards: List[AlertCard] = []
        self._pending: deque = deque()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignBottom)
        self._layout = layout

        self._alert_queued.connect(self._on_alert_queued)
        self._reposition()

    # ── 公开接口 ─────────────────────────────────────────────────────────────

    def push_alert(self, alert):
        """线程安全：从任意线程添加预警"""
        self._alert_queued.emit(alert)

    # ── 内部槽 ───────────────────────────────────────────────────────────────

    @Slot(object)
    def _on_alert_queued(self, alert):
        if len(self._cards) >= self._max_cards:
            self._pending.append(alert)
        else:
            self._add_card(alert)

    def _add_card(self, alert):
        card = AlertCard(alert, self)
        card.closed.connect(self._remove_card)
        self._cards.append(card)
        self._layout.addWidget(card)
        self._reposition()
        self.show()

        # 滑入动画
        anim = QPropertyAnimation(card, b"maximumHeight", self)
        anim.setDuration(250)
        anim.setStartValue(0)
        anim.setEndValue(card.sizeHint().height() + 20)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.start()

    @Slot(object)
    def _remove_card(self, card: AlertCard):
        if card in self._cards:
            self._cards.remove(card)
            self._layout.removeWidget(card)
            card.deleteLater()
            self._reposition()

        if not self._cards and not self._pending:
            self.hide()
        elif self._pending and len(self._cards) < self._max_cards:
            self._add_card(self._pending.popleft())

    def _reposition(self):
        """定位到屏幕右下角"""
        screen = QApplication.primaryScreen()
        if not screen:
            return
        geo = screen.availableGeometry()
        w = self.width()
        h = max(self.sizeHint().height(), 50)
        self.setGeometry(geo.right() - w - 10, geo.bottom() - h - 10, w, h)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition()


# ─── 全局单例管理 ─────────────────────────────────────────────────────────────

_overlay: Optional[AlertOverlay] = None
_overlay_lock = threading.Lock()


def get_alert_overlay() -> AlertOverlay:
    """获取/创建全局 AlertOverlay 单例（主线程调用）"""
    global _overlay
    with _overlay_lock:
        if _overlay is None:
            _overlay = AlertOverlay()
        return _overlay


def show_alert(alert):
    """
    从任意线程安全地弹出风险预警。
    首次调用时需确保 QApplication 已创建，且该函数首次在主线程调用过。
    """
    overlay = get_alert_overlay()
    overlay.push_alert(alert)


def show_simple_alert(
    title: str,
    message: str,
    level: str = "WARNING",
    detail: str = "",
    category: str = "SYSTEM",
    suggested_action: str = "",
    auto_close_sec: int = 0,
):
    """便捷函数：用字典模拟 RiskAlert dataclass"""
    class _FakeAlert:
        pass

    a = _FakeAlert()
    a.title = title
    a.message = message
    a.level = level
    a.detail = detail
    a.category = category
    a.suggested_action = suggested_action
    a.auto_close_sec = auto_close_sec
    from datetime import datetime
    a.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    show_alert(a)


# ─── 风险历史面板（嵌入 UI 的小组件）────────────────────────────────────────

class AlertHistoryWidget(QWidget):
    """在 AI 顾问页面嵌入显示最近的预警历史"""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setStyleSheet(
            "QTextEdit { background: #0d1117; color: #cdd9e5; "
            "border: 1px solid #30363d; border-radius: 6px; font-size: 12px; }"
        )
        layout.addWidget(self._text)

        self._records: List[Dict[str, Any]] = []

    def add_alert(self, alert):
        """追加一条预警记录"""
        cfg = _LEVEL_CFG.get(getattr(alert, "level", "INFO"), _DEFAULT_CFG)
        color = cfg["title_color"]
        icon = cfg["icon"]
        ts = getattr(alert, "timestamp", "")
        title = getattr(alert, "title", "")
        msg = getattr(alert, "message", "")
        cat = getattr(alert, "category", "")

        html = (
            f'<div style="margin-bottom:6px; padding:6px; '
            f'border-left:3px solid {cfg["border"]}; background:{cfg["bg"]}; border-radius:4px;">'
            f'<span style="color:{color}; font-weight:bold;">{icon} {title}</span>'
            f'<span style="color:#666; font-size:10px;"> [{cat}] {ts[-8:] if ts else ""}</span>'
            f'<br><span style="color:#bbb;">{msg}</span>'
            f'</div>'
        )
        current = self._text.toHtml()
        self._text.setHtml(html + (current or ""))

        self._records.append({
            "level": getattr(alert, "level", ""),
            "title": title,
            "message": msg,
            "category": cat,
            "timestamp": ts,
        })
        if len(self._records) > 200:
            self._records = self._records[-200:]

    def clear(self):
        self._text.clear()
        self._records.clear()

    def get_records(self) -> List[Dict[str, Any]]:
        return list(self._records)
