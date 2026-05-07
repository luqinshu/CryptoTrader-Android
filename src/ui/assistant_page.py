"""
AI 智能交易助理页面  v2.0
量化专家视角重设计：实时市场情绪 + 组合风控仪表盘 + 智能快捷操作 + 绩效追踪
"""
from __future__ import annotations

import json
import math
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

from src.qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QDialog,
    QPushButton, QLabel, QTextEdit, QCheckBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QFrame, QSplitter, QScrollArea, QComboBox, QLineEdit,
    QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QProgressBar, QTabWidget,
    Qt, QTimer, Signal,
    QFont, QColor, QSizePolicy,
)
from src.ai_agent.trading_assistant import TradingAssistant, Permission, AgentAction, AgentCycle
from src.ai_agent.llm_client import LLMClient
from src.rl_optimizer.tracker import SignalTracker
from src.rl_optimizer.timeframe_tracker import MultiTimeframeTracker


# ════════════════════════════════════════════════════════════════
# 颜色主题
# ════════════════════════════════════════════════════════════════
C = {
    "bg":      "#0d1117",
    "panel":   "#161b22",
    "border":  "#30363d",
    "text":    "#c9d1d9",
    "dim":     "#8b949e",
    "green":   "#3fb950",
    "red":     "#f85149",
    "yellow":  "#e3b341",
    "orange":  "#f0883e",
    "blue":    "#58a6ff",
    "purple":  "#bc8cff",
    "cyan":    "#00ccff",
    "dark_red":"#da3633",
}

ADVICE_COLORS = {
    "加仓": C["green"],  "买入": C["green"],  "做多": C["green"],
    "观望": C["yellow"], "持有": C["blue"],
    "减仓": C["orange"], "部分减仓": C["orange"],
    "卖出": C["red"],    "做空": C["red"],    "止损": C["dark_red"],
    "立即止损": C["dark_red"], "止盈出场": C["yellow"],
}


# ════════════════════════════════════════════════════════════════
# 通用 UI 组件
# ════════════════════════════════════════════════════════════════

def _card(title: str) -> QGroupBox:
    g = QGroupBox(title)
    g.setStyleSheet(f"""
        QGroupBox {{background:{C['panel']}; border:1px solid {C['border']}; border-radius:8px;
                    margin-top:8px; padding-top:4px; color:{C['text']}; font-weight:bold; font-size:12px;}}
        QGroupBox::title {{subcontrol-origin:margin; left:10px; color:{C['dim']};}}
    """)
    return g


def _btn(text: str, bg: str, hover: str = "", h: int = 34) -> QPushButton:
    b = QPushButton(text)
    b.setMinimumHeight(h)
    hc = hover or bg
    b.setStyleSheet(f"""
        QPushButton {{background:{bg}; color:white; border:none; border-radius:6px;
                      font-weight:bold; font-size:12px; padding:2px 8px;}}
        QPushButton:hover {{background:{hc};}}
        QPushButton:disabled {{background:#21262d; color:#484f58;}}
    """)
    return b


def _label(text: str, color: str = None, size: int = 11, bold: bool = False) -> QLabel:
    lbl = QLabel(text)
    c   = color or C["text"]
    w   = "bold" if bold else "normal"
    lbl.setStyleSheet(f"color:{c};font-size:{size}px;font-weight:{w};")
    return lbl


class RiskGauge(QFrame):
    """单项风控指标条（标签 + 进度条 + 数值）"""
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"QFrame{{background:{C['bg']};border-radius:6px;padding:2px;}}")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 3, 6, 3)
        lay.setSpacing(6)
        self._lbl = QLabel(label)
        self._lbl.setFixedWidth(90)
        self._lbl.setStyleSheet(f"color:{C['dim']};font-size:10px;")
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setFixedHeight(8)
        self._bar.setTextVisible(False)
        self._bar.setStyleSheet(
            f"QProgressBar{{background:#21262d;border-radius:4px;}}"
            f"QProgressBar::chunk{{background:{C['green']};border-radius:4px;}}")
        self._val = QLabel("--")
        self._val.setFixedWidth(60)
        self._val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._val.setStyleSheet(f"color:{C['text']};font-size:11px;font-weight:bold;")
        lay.addWidget(self._lbl)
        lay.addWidget(self._bar, 1)
        lay.addWidget(self._val)

    def update(self, pct: float, value_str: str):
        pct = max(0.0, min(100.0, pct))
        self._bar.setValue(int(pct))
        self._val.setText(value_str)
        if pct >= 80:
            chunk_color = C["red"]
        elif pct >= 50:
            chunk_color = C["yellow"]
        else:
            chunk_color = C["green"]
        self._bar.setStyleSheet(
            f"QProgressBar{{background:#21262d;border-radius:4px;}}"
            f"QProgressBar::chunk{{background:{chunk_color};border-radius:4px;}}")
        self._val.setStyleSheet(
            f"color:{chunk_color};font-size:11px;font-weight:bold;")


class MarketTickerWidget(QFrame):
    """顶部市场行情小组件（单个交易对）"""
    def __init__(self, symbol_short: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"QFrame{{background:{C['panel']};border:1px solid {C['border']};"
            f"border-radius:6px;padding:2px 6px;}}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(0)
        self._sym = QLabel(symbol_short)
        self._sym.setStyleSheet(f"color:{C['dim']};font-size:9px;")
        self._price = QLabel("--")
        self._price.setStyleSheet(f"color:{C['text']};font-size:14px;font-weight:bold;")
        self._chg = QLabel("")
        self._chg.setStyleSheet(f"color:{C['dim']};font-size:10px;")
        lay.addWidget(self._sym)
        lay.addWidget(self._price)
        lay.addWidget(self._chg)
        self.setFixedWidth(110)

    def update(self, price: float, chg_pct: float):
        self._price.setText(f"{price:,.2f}" if price > 100 else f"{price:.4f}")
        color = C["green"] if chg_pct >= 0 else C["red"]
        sign  = "+" if chg_pct >= 0 else ""
        self._chg.setText(f"{sign}{chg_pct:.2f}%")
        self._chg.setStyleSheet(f"color:{color};font-size:10px;font-weight:bold;")
        self._price.setStyleSheet(f"color:{color};font-size:14px;font-weight:bold;")


class ActionFeed(QTextEdit):
    COLORS = {"INFO": C["dim"],  "ANALYSIS": C["blue"], "TRADE": C["cyan"],
              "SUCCESS": C["green"], "WARNING": C["yellow"], "ERROR": C["red"],
              "DEBUG": "#484f58", "RISK": C["orange"]}
    ICONS  = {"ANALYSIS": "🔵", "TRADE": "⚡", "SUCCESS": "✅",
              "WARNING": "⚠️", "ERROR": "❌", "RISK": "🛡️", "INFO": "·"}

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setStyleSheet(
            f"QTextEdit{{background:{C['bg']};color:{C['text']};"
            f"font-family:monospace;font-size:11px;"
            f"border:1px solid #21262d;border-radius:6px;}}")

    def push(self, msg: str, level: str = "INFO"):
        color = self.COLORS.get(level, C["dim"])
        icon  = self.ICONS.get(level, "·")
        ts    = datetime.now().strftime("%H:%M:%S")
        self.append(
            f'<span style="color:#484f58">[{ts}]</span> '
            f'<span style="color:{color}">{icon} {msg}</span>')
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())


# ════════════════════════════════════════════════════════════════
# 主页面
# ════════════════════════════════════════════════════════════════

class AssistantPage(QWidget):

    # ── 内部信号（线程安全传递数据到 UI 线程）────────────────────
    _sig_pos_data = Signal(list)   # [{inst_id, side, upl_ratio, upl, avg_px, size_usdt, entry_ts}]
    _sig_mkt_data = Signal(dict)   # {btc_px, btc_chg, eth_px, eth_chg, regime}

    def __init__(self, okx_client=None, trade_executor=None,
                 scanner_page=None, parent=None):
        super().__init__(parent)
        self.okx_client     = okx_client
        self.trade_executor = trade_executor
        self.scanner_page   = scanner_page
        self._agent: Optional[TradingAssistant] = None
        self._running        = False
        self._last_results: List[dict] = []
        self._pos_refreshing = False
        self._mkt_fetching   = False
        self._open_symbol_dialogs: List[QDialog] = []

        # 持仓分区：AI管控 vs 手动操作（inst_id 集合）
        self._ai_managed_ids: set = set()   # 归属 AI 管控区的持仓
        # 记录最新持仓行数据，用于右键菜单/双击时定位
        self._pos_rows: List[dict] = []     # 全量最新持仓列表

        # 绩效追踪
        self._session_cycles    = 0
        self._session_actions   = 0
        self._session_alerts    = 0
        self._peak_balance      = 0.0
        self._session_start     = datetime.now()
        self._pos_entry_ts: Dict[str, float] = {}  # inst_id -> 首次发现时间戳

        # 信号追踪器
        self._signal_tracker = SignalTracker()
        self._tf_tracker = MultiTimeframeTracker()

        self._init_ui()
        self._load_config()
        # 不在启动时加载 AI 管理仓位 ID，每次重启所有仓位默认归手动区，规避意外风险
        self._sig_pos_data.connect(self._on_pos_data)
        self._sig_mkt_data.connect(self._on_mkt_data)

        # 持仓刷新定时器 30s
        self._pos_timer = QTimer(self)
        self._pos_timer.setInterval(30_000)
        self._pos_timer.timeout.connect(self._refresh_positions_async)
        self._pos_timer.start()

        # 市场行情刷新定时器 60s
        self._mkt_timer = QTimer(self)
        self._mkt_timer.setInterval(60_000)
        self._mkt_timer.timeout.connect(self._fetch_market_async)
        self._mkt_timer.start()

        # 信号验证定时器：每 30 分钟验证一次已记录信号的实际走势
        self._validate_timer = QTimer(self)
        self._validate_timer.setInterval(30 * 60 * 1000)
        self._validate_timer.timeout.connect(self._validate_signals_async)
        self._validate_timer.start()

        QTimer.singleShot(1_500, self._refresh_positions_async)
        QTimer.singleShot(3_000, self._fetch_market_async)
        QTimer.singleShot(90_000, self._validate_signals_async)  # 启动 90s 后首次验证

    # ════════════════════════════════════════════════════════════
    # UI 构建
    # ════════════════════════════════════════════════════════════

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        root.addWidget(self._build_header())
        root.addWidget(self._build_market_bar())

        body = QSplitter(Qt.Horizontal)
        body.addWidget(self._build_config_panel())
        body.addWidget(self._build_center_panel())
        body.addWidget(self._build_monitor_panel())
        body.setStretchFactor(0, 2)
        body.setStretchFactor(1, 5)
        body.setStretchFactor(2, 3)
        root.addWidget(body, 1)

    # ── 顶部状态栏 ───────────────────────────────────────────────
    def _build_header(self) -> QFrame:
        f = QFrame()
        f.setFixedHeight(50)
        f.setStyleSheet(
            f"QFrame{{background:{C['panel']};border:1px solid {C['border']};border-radius:8px;}}")
        lay = QHBoxLayout(f)
        lay.setContentsMargins(14, 6, 14, 6)

        self._dot = QLabel("●")
        self._dot.setStyleSheet(f"color:#21262d;font-size:18px;")
        lay.addWidget(self._dot)

        title = _label("🧠  AI 智能交易助理  v2", C["text"], 14, bold=True)
        lay.addWidget(title)

        self._status_lbl = _label("就绪 — 等待扫描结果", C["dim"], 11)
        self._status_lbl.setContentsMargins(10, 0, 0, 0)
        lay.addWidget(self._status_lbl)

        lay.addStretch()

        self._cycle_lbl = _label("轮次: 0 | 决策: 0 | 预警: 0", C["dim"], 10)
        self._cycle_lbl.setContentsMargins(0, 0, 14, 0)
        lay.addWidget(self._cycle_lbl)

        for text, bg, hover, slot in [
            ("🔍 立即分析",  C["blue"],     "#388bfd", self._trigger_analysis_now),
            ("🚀 启动助理",  "#238636",     "#2ea043", self._toggle_agent),
            ("⛔ 紧急停止",  C["dark_red"], C["red"],  self._emergency_stop),
        ]:
            btn = _btn(text, bg, hover)
            btn.setFixedWidth(110)
            btn.clicked.connect(slot)
            if text.startswith("🚀"):
                self._toggle_btn = btn
            elif text.startswith("🔍"):
                self._analyze_now_btn = btn
            lay.addWidget(btn)
        return f

    # ── 市场情绪条 ───────────────────────────────────────────────
    def _build_market_bar(self) -> QFrame:
        f = QFrame()
        f.setFixedHeight(56)
        f.setStyleSheet(
            f"QFrame{{background:{C['panel']};border:1px solid {C['border']};border-radius:8px;}}")
        lay = QHBoxLayout(f)
        lay.setContentsMargins(14, 4, 14, 4)
        lay.setSpacing(12)

        lay.addWidget(_label("市场行情", C["dim"], 10))

        self._btc_ticker = MarketTickerWidget("BTC/USDT")
        self._eth_ticker = MarketTickerWidget("ETH/USDT")
        lay.addWidget(self._btc_ticker)
        lay.addWidget(self._eth_ticker)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f"color:{C['border']};")
        lay.addWidget(sep)

        # 市场情绪
        lay.addWidget(_label("市场情绪:", C["dim"], 10))
        self._regime_lbl = _label("-- 加载中 --", C["dim"], 12, bold=True)
        lay.addWidget(self._regime_lbl)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet(f"color:{C['border']};")
        lay.addWidget(sep2)

        # 资金费率区域
        lay.addWidget(_label("BTC资金费率:", C["dim"], 10))
        self._funding_lbl = _label("--", C["dim"], 11, bold=True)
        lay.addWidget(self._funding_lbl)

        lay.addStretch()

        self._mkt_update_lbl = _label("行情: --", C["dim"], 9)
        lay.addWidget(self._mkt_update_lbl)
        return f

    # ── 左：配置 + 风控 + 快捷操作 ──────────────────────────────
    def _build_config_panel(self) -> QWidget:
        w = QWidget()
        w.setMaximumWidth(300)
        lay = QVBoxLayout(w)
        lay.setSpacing(6)
        lay.setContentsMargins(0, 0, 4, 0)

        lay.addWidget(self._build_llm_card())
        lay.addWidget(self._build_permission_card())
        lay.addWidget(self._build_risk_breakers_card())
        lay.addWidget(self._build_smart_actions_card())
        lay.addStretch()
        return w

    def _build_llm_card(self) -> QGroupBox:
        g = _card("🔑 LLM 配置")
        gl = QGridLayout(g)
        gl.setSpacing(4)

        gl.addWidget(_label("API 地址:", C["dim"], 10), 0, 0)
        self._url_edit = self._le("https://api.deepseek.com/v1")
        gl.addWidget(self._url_edit, 0, 1)

        gl.addWidget(_label("API Key:", C["dim"], 10), 1, 0)
        self._key_edit = self._le("")
        self._key_edit.setEchoMode(QLineEdit.Password)
        gl.addWidget(self._key_edit, 1, 1)

        gl.addWidget(_label("模型:", C["dim"], 10), 2, 0)
        self._model_combo = QComboBox()
        self._model_combo.addItems([
            "deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner",
            "gpt-4o", "gpt-4o-mini",
        ])
        self._model_combo.setEditable(True)
        self._model_combo.setStyleSheet(
            f"QComboBox{{background:{C['bg']};color:{C['text']};"
            f"border:1px solid {C['border']};border-radius:4px;padding:3px;}}")
        gl.addWidget(self._model_combo, 2, 1)

        test_btn = _btn("🔗 测试连接", "#21262d", C["border"], h=26)
        test_btn.clicked.connect(self._test_llm)
        gl.addWidget(test_btn, 3, 0, 1, 2)
        return g

    def _build_permission_card(self) -> QGroupBox:
        g = _card("🔒 交易权限")
        pl = QVBoxLayout(g)
        pl.setSpacing(3)
        chk_style = (f"QCheckBox{{color:{C['text']};font-size:11px;}}"
                     f"QCheckBox::indicator{{width:13px;height:13px;}}")

        # 每次程序启动全部默认关闭，需人工手动开启，规避意外交易风险
        self._chk_trade   = QCheckBox("允许买入 / 做空下单")
        self._chk_close   = QCheckBox("允许平仓")
        self._chk_scan    = QCheckBox("允许触发新一轮扫描")
        self._chk_mod     = QCheckBox("允许修改策略参数")
        self._chk_confirm = QCheckBox("高风险操作须人工确认")
        for c in [self._chk_trade, self._chk_close, self._chk_scan,
                  self._chk_mod, self._chk_confirm]:
            c.setStyleSheet(chk_style)
            pl.addWidget(c)
        return g

    def _build_risk_breakers_card(self) -> QGroupBox:
        g = _card("⚡ 风控熔断器")
        rl = QGridLayout(g)
        rl.setSpacing(4)
        ls = f"color:{C['dim']};font-size:10px;"

        params = [
            ("单笔上限 USDT:", "_max_usdt",     (10, 100000), 100,  ""),
            ("日亏损止损 %:",  "_max_loss",     (0.1, 50),    5.0,  "%"),
            ("单仓止损 %:",    "_pos_loss_pct", (0.5, 30),    5.0,  "%"),
            ("最大持仓数:",    "_max_positions",(1, 20),      5,    ""),
            ("分析间隔 s:",    "_interval",     (30, 3600),   120,  "s"),
            ("交易冷却 s:",    "_cooldown",     (60, 3600),   300,  "s"),
        ]
        for row, (lbl, attr, rng, val, suffix) in enumerate(params):
            l = QLabel(lbl); l.setStyleSheet(ls)
            rl.addWidget(l, row, 0)
            sp = QDoubleSpinBox() if isinstance(val, float) else QSpinBox()
            sp.setRange(*rng)
            sp.setValue(val)
            if suffix:
                sp.setSuffix(suffix)
            sp.setStyleSheet(
                f"QSpinBox,QDoubleSpinBox{{background:{C['bg']};color:{C['text']};"
                f"border:1px solid {C['border']};border-radius:4px;padding:2px;}}")
            setattr(self, attr, sp)
            rl.addWidget(sp, row, 1)

        # 分析选项
        chk_style = (f"QCheckBox{{color:{C['dim']};font-size:10px;}}"
                     f"QCheckBox::indicator{{width:12px;height:12px;}}")
        self._chk_auto_analyze = QCheckBox("扫描完成后自动 AI 分析")
        self._chk_auto_analyze.setChecked(True)
        self._chk_auto_analyze.setStyleSheet(chk_style)
        self._chk_top_only = QCheckBox("只分析 Top 30 结果")
        self._chk_top_only.setChecked(True)
        self._chk_top_only.setStyleSheet(chk_style)
        row_n = len(params)
        rl.addWidget(self._chk_auto_analyze, row_n,   0, 1, 2)
        rl.addWidget(self._chk_top_only,     row_n+1, 0, 1, 2)
        return g

    def _build_smart_actions_card(self) -> QGroupBox:
        g = _card("🎯 智能快捷操作")
        gl = QGridLayout(g)
        gl.setSpacing(5)
        actions = [
            ("🛡️ 风险评估",  C["blue"],    "#388bfd",    self._quick_risk_assess),
            ("⚡ 一键半仓",  C["orange"],  "#f0883e",    self._emergency_half_reduce),
            ("🔒 锁定浮盈",  "#238636",    "#2ea043",    self._lock_profits),
            ("📊 压力测试",  C["purple"],  "#bc8cff",    self._stress_test),
            ("🔄 触发扫描",  "#1f6feb",    "#388bfd",    self._trigger_scan_action),
            ("🔁 刷新持仓",  "#21262d",    C["border"],  self._refresh_positions_async),
        ]
        for i, (text, bg, hov, slot) in enumerate(actions):
            b = _btn(text, bg, hov, h=30)
            b.setFont(QFont("", 10))
            b.clicked.connect(slot)
            gl.addWidget(b, i // 2, i % 2)
        return g

    # ── 中：AI 评估 / 信号追踪 / 策略绩效（三标签页）────────────
    def _build_center_panel(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(6)

        tab_style = f"""
            QTabWidget::pane  {{border:1px solid {C['border']};background:{C['bg']};border-radius:5px;}}
            QTabBar::tab      {{background:{C['panel']};color:{C['dim']};padding:5px 14px;
                               border:1px solid {C['border']};border-bottom:none;
                               border-top-left-radius:5px;border-top-right-radius:5px;font-size:11px;}}
            QTabBar::tab:selected {{background:{C['bg']};color:{C['text']};font-weight:bold;}}
        """
        self._center_tabs = QTabWidget()
        self._center_tabs.setStyleSheet(tab_style)
        self._center_tabs.addTab(self._build_ai_analysis_tab(),     "🤖  AI 评估")
        self._center_tabs.addTab(self._build_signal_tracking_tab(), "📍  信号追踪")
        self._center_tabs.addTab(self._build_strategy_perf_tab(),   "📊  策略绩效")
        lay.addWidget(self._center_tabs, 3)

        # AI 综合总结（固定在所有标签页下方）
        summary_grp = _card("🧠 AI 市场综合判断")
        sl = QVBoxLayout(summary_grp)
        self._summary_text = QTextEdit()
        self._summary_text.setReadOnly(True)
        self._summary_text.setFixedHeight(120)
        self._summary_text.setStyleSheet(
            f"QTextEdit{{background:{C['bg']};color:{C['text']};"
            f"font-size:12px;border:none;line-height:1.6;}}")
        sl.addWidget(self._summary_text)
        lay.addWidget(summary_grp)
        return w

    def _build_ai_analysis_tab(self) -> QWidget:
        """标签页 1：扫描结果 + AI 评估"""
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(6)

        # 状态栏
        bar = QHBoxLayout()
        self._scan_count_lbl = _label("扫描结果: 0 条", C["dim"], 11)
        self._scan_time_lbl  = _label("最后更新: --", C["dim"], 10)
        bar.addWidget(self._scan_count_lbl)
        bar.addWidget(self._scan_time_lbl)
        bar.addStretch()
        self._analyze_btn2 = _btn("🤖 AI 分析全部结果", C["blue"], "#388bfd")
        self._analyze_btn2.clicked.connect(self._trigger_analysis_now)
        bar.addWidget(self._analyze_btn2)
        lay.addLayout(bar)

        # 扫描结果表格
        tbl_grp = _card("📋 扫描结果 · AI 综合评估")
        tbl_lay = QVBoxLayout(tbl_grp)

        self._result_table = QTableWidget(0, 9)
        self._result_table.setHorizontalHeaderLabels([
            "扫描时间", "交易对", "机会类型", "建议方向", "评分", "价格", "AI 建议", "置信度", "操作理由"
        ])
        hdr = self._result_table.horizontalHeader()
        for i in range(8):
            hdr.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(8, QHeaderView.Stretch)
        self._result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._result_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._result_table.setAlternatingRowColors(True)
        self._result_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self._result_table.customContextMenuRequested.connect(self._on_table_context_menu)
        self._result_table.doubleClicked.connect(self._on_table_double_click)
        self._result_table.setStyleSheet(f"""
            QTableWidget {{background:{C['bg']}; color:{C['text']};
                           gridline-color:#21262d; font-size:11px;
                           alternate-background-color:#111820;}}
            QHeaderView::section {{background:{C['panel']}; color:{C['dim']};
                                   border:none; padding:4px; font-weight:bold;}}
            QTableWidget::item:selected {{background:#1f6feb40;}}
        """)
        tbl_lay.addWidget(self._result_table)

        self._analysis_progress = _label("", C["blue"], 11)
        self._analysis_progress.setContentsMargins(4, 2, 0, 0)
        tbl_lay.addWidget(self._analysis_progress)
        lay.addWidget(tbl_grp, 1)
        return w

    def _build_signal_tracking_tab(self) -> QWidget:
        """标签页 2：信号追踪 — 记录每条扫描信号并对比后续走势"""
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(5)

        cb_style = (f"QComboBox{{background:{C['bg']};color:{C['text']};"
                    f"border:1px solid {C['border']};border-radius:4px;padding:3px 6px;}}")

        # 工具栏
        bar = QHBoxLayout()
        self._track_total_lbl = _label("已记录: 0 条", C["dim"], 11)
        bar.addWidget(self._track_total_lbl)

        bar.addWidget(_label("策略:", C["dim"], 10))
        self._track_strategy_combo = QComboBox()
        self._track_strategy_combo.addItem("全部策略")
        self._track_strategy_combo.setMinimumWidth(160)
        self._track_strategy_combo.setStyleSheet(cb_style)
        self._track_strategy_combo.currentTextChanged.connect(self._refresh_signal_tracking_tab)
        bar.addWidget(self._track_strategy_combo)

        bar.addWidget(_label("状态:", C["dim"], 10))
        self._track_filter_combo = QComboBox()
        self._track_filter_combo.addItems(["全部", "待验证", "盈利", "亏损", "中性"])
        self._track_filter_combo.setStyleSheet(cb_style)
        self._track_filter_combo.currentTextChanged.connect(self._refresh_signal_tracking_tab)
        bar.addWidget(self._track_filter_combo)

        bar.addStretch()
        self._track_stats_lbl = _label("", C["dim"], 10)
        bar.addWidget(self._track_stats_lbl)

        validate_btn = _btn("🔄 立即验证", C["blue"], "#388bfd", h=28)
        validate_btn.clicked.connect(self._validate_signals_async)
        bar.addWidget(validate_btn)
        refresh_btn = _btn("↺ 刷新", "#21262d", C["border"], h=28)
        refresh_btn.clicked.connect(self._refresh_signal_tracking_tab)
        bar.addWidget(refresh_btn)
        lay.addLayout(bar)

        # 追踪表格：时间/交易对/策略/方向/评分/入场价/综合/2h/6h/24h/72h
        self._track_table = QTableWidget(0, 11)
        self._track_table.setHorizontalHeaderLabels([
            "时间", "交易对", "策略", "方向", "评分", "入场价",
            "综合", "2h", "6h", "24h", "72h"
        ])
        th = self._track_table.horizontalHeader()
        th.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        th.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        th.setSectionResizeMode(2, QHeaderView.Stretch)
        for i in range(3, 11):
            th.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self._track_table.verticalHeader().setVisible(False)
        self._track_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._track_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._track_table.setAlternatingRowColors(True)
        self._track_table.setStyleSheet(f"""
            QTableWidget {{background:{C['bg']};color:{C['text']};
                           gridline-color:#21262d;font-size:11px;
                           alternate-background-color:#111820;}}
            QHeaderView::section {{background:{C['panel']};color:{C['dim']};
                                   border:none;padding:4px;font-weight:bold;}}
            QTableWidget::item:selected {{background:#1f6feb40;}}
        """)
        lay.addWidget(self._track_table, 1)
        return w

    def _build_strategy_perf_tab(self) -> QWidget:
        """标签页 3：策略绩效统计 + 参数优化建议"""
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(5)

        cb_style = (f"QComboBox{{background:{C['bg']};color:{C['text']};"
                    f"border:1px solid {C['border']};border-radius:4px;padding:3px 6px;}}")

        bar = QHBoxLayout()
        bar.addWidget(_label("策略绩效统计", C["text"], 12, bold=True))
        bar.addStretch()
        bar.addWidget(_label("周期:", C["dim"], 10))
        self._perf_days_combo = QComboBox()
        self._perf_days_combo.addItems(["近7天", "近14天", "近30天"])
        self._perf_days_combo.setCurrentIndex(2)
        self._perf_days_combo.setStyleSheet(cb_style)
        self._perf_days_combo.currentTextChanged.connect(self._refresh_strategy_perf_tab)
        bar.addWidget(self._perf_days_combo)
        refresh_btn = _btn("↺ 刷新", "#21262d", C["border"], h=28)
        refresh_btn.clicked.connect(self._refresh_strategy_perf_tab)
        bar.addWidget(refresh_btn)
        lay.addLayout(bar)

        # 绩效表格
        self._perf_table = QTableWidget(0, 8)
        self._perf_table.setHorizontalHeaderLabels([
            "策略名称", "信号数", "已验证", "贝叶斯胜率", "均盈%", "均亏%", "盈亏比", "夏普"
        ])
        ph = self._perf_table.horizontalHeader()
        ph.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, 8):
            ph.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self._perf_table.verticalHeader().setVisible(False)
        self._perf_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._perf_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._perf_table.setAlternatingRowColors(True)
        self._perf_table.setStyleSheet(f"""
            QTableWidget {{background:{C['bg']};color:{C['text']};
                           gridline-color:#21262d;font-size:11px;
                           alternate-background-color:#111820;}}
            QHeaderView::section {{background:{C['panel']};color:{C['dim']};
                                   border:none;padding:4px;font-weight:bold;}}
            QTableWidget::item:selected {{background:#1f6feb40;}}
        """)
        self._perf_table.selectionModel().selectionChanged.connect(
            self._on_perf_strategy_selected)
        lay.addWidget(self._perf_table)

        # 参数建议区
        rec_grp = _card("💡 参数优化建议")
        rec_lay = QVBoxLayout(rec_grp)
        rec_lay.setSpacing(4)

        self._rec_text = QTextEdit()
        self._rec_text.setReadOnly(True)
        self._rec_text.setFixedHeight(110)
        self._rec_text.setStyleSheet(
            f"QTextEdit{{background:{C['bg']};color:{C['text']};"
            f"font-size:11px;border:none;line-height:1.5;}}")
        self._rec_text.setPlaceholderText("点击绩效表中某策略后，此处将显示规则建议；点击「AI 分析参数」获取 AI 深度建议。")
        rec_lay.addWidget(self._rec_text)

        btn_row = QHBoxLayout()
        self._ai_advice_btn = _btn("🧠 AI 分析参数", C["purple"], "#bc8cff", h=30)
        self._ai_advice_btn.clicked.connect(self._request_ai_param_advice)
        btn_row.addWidget(self._ai_advice_btn)
        self._apply_advice_btn = _btn("✏️ 应用建议参数", C["orange"], "#f0883e", h=30)
        self._apply_advice_btn.setEnabled(False)
        self._apply_advice_btn.setToolTip("需先勾选「允许修改策略参数」权限")
        self._apply_advice_btn.clicked.connect(self._apply_param_advice)
        btn_row.addWidget(self._apply_advice_btn)
        btn_row.addStretch()
        rec_lay.addLayout(btn_row)
        lay.addWidget(rec_grp)

        self._pending_param_advice: dict = {}  # 最新 AI 建议的参数修改
        return w

    # ── 右：组合监控 + 风控仪表 + 持仓 + 绩效 + 日志 ────────────
    def _build_monitor_panel(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(6)
        lay.setContentsMargins(4, 0, 0, 0)

        lay.addWidget(self._build_metrics_row())
        lay.addWidget(self._build_risk_gauge_card())
        lay.addWidget(self._build_position_card(), 2)
        lay.addWidget(self._build_performance_card())
        lay.addWidget(self._build_log_card(), 2)
        return w

    def _build_metrics_row(self) -> QWidget:
        w   = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(5)

        def _mc(label, attr):
            f = QFrame()
            f.setStyleSheet(
                f"QFrame{{background:{C['bg']};border:1px solid #21262d;border-radius:7px;}}")
            fl = QVBoxLayout(f)
            fl.setContentsMargins(8, 4, 8, 4)
            fl.setSpacing(0)
            lb = _label(label, C["dim"], 9)
            vl = _label("--", C["text"], 15, bold=True)
            fl.addWidget(lb); fl.addWidget(vl)
            setattr(self, attr + "_lbl", vl)
            return f

        lay.addWidget(_mc("余额 USDT",  "_m_bal"))
        lay.addWidget(_mc("日内盈亏",   "_m_pnl"))
        lay.addWidget(_mc("持仓数",     "_m_pos"))
        lay.addWidget(_mc("扫描信号",   "_m_scn"))
        return w

    def _build_risk_gauge_card(self) -> QGroupBox:
        g = _card("🛡️ 风控仪表盘")
        lay = QVBoxLayout(g)
        lay.setSpacing(4)

        self._gauge_pos   = RiskGauge("持仓占用")
        self._gauge_loss  = RiskGauge("日内亏损")
        self._gauge_conc  = RiskGauge("最大集中度")
        self._gauge_dd    = RiskGauge("历史回撤")
        for gauge in [self._gauge_pos, self._gauge_loss,
                      self._gauge_conc, self._gauge_dd]:
            lay.addWidget(gauge)

        # 风险总评
        self._risk_summary_lbl = _label("▶  风险: 待评估", C["dim"], 11, bold=True)
        self._risk_summary_lbl.setContentsMargins(4, 2, 0, 2)
        lay.addWidget(self._risk_summary_lbl)
        return g

    def _build_position_card(self) -> QGroupBox:
        g = _card("📊 持仓快照")
        pl = QVBoxLayout(g)
        pl.setSpacing(4)

        # ── Tab 容器 ──────────────────────────────────────
        tabs = QTabWidget()
        tabs.setStyleSheet(f"""
            QTabWidget::pane  {{border:1px solid {C['border']};background:{C['bg']};border-radius:5px;}}
            QTabBar::tab      {{background:{C['panel']};color:{C['dim']};padding:5px 14px;
                               border:1px solid {C['border']};border-bottom:none;
                               border-top-left-radius:5px;border-top-right-radius:5px;font-size:11px;}}
            QTabBar::tab:selected {{background:{C['bg']};color:{C['text']};font-weight:bold;}}
        """)

        tabs.addTab(self._build_ai_pos_tab(),     "🤖  AI 管控仓位")
        tabs.addTab(self._build_manual_pos_tab(),  "👤  手动操作仓位")
        pl.addWidget(tabs, 1)

        # ── 底部状态 ──────────────────────────────────────
        foot = QHBoxLayout()
        self._pos_update_lbl = _label("刷新: --", C["dim"], 9)
        foot.addWidget(self._pos_update_lbl)
        foot.addStretch()
        # AI管控区锁定提示
        lock_lbl = _label("🔒 手动区：AI 助理禁止自动操作", C["orange"], 9, bold=False)
        foot.addWidget(lock_lbl)
        pl.addLayout(foot)
        return g

    def _make_pos_table(self) -> QTableWidget:
        """创建统一样式的持仓表格（6 列）。"""
        t = QTableWidget(0, 6)
        t.setHorizontalHeaderLabels(["交易对", "方向", "均价", "浮盈%", "浮盈U", "时长"])
        h = t.horizontalHeader()
        for i in range(5):
            h.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(5, QHeaderView.Stretch)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.setContextMenuPolicy(Qt.CustomContextMenu)
        t.setStyleSheet(
            f"QTableWidget{{background:{C['bg']};color:{C['text']};"
            f"gridline-color:#21262d;font-size:11px;}}"
            f"QHeaderView::section{{background:{C['panel']};color:{C['dim']};"
            f"border:none;padding:3px;}}")
        return t

    def _build_ai_pos_tab(self) -> QWidget:
        """AI 管控仓位标签页：AI 助理可自动操作。"""
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        # 说明横幅
        banner = QFrame()
        banner.setStyleSheet(
            f"QFrame{{background:#0d2137;border:1px solid {C['blue']};"
            f"border-radius:5px;padding:2px;}}")
        blay = QHBoxLayout(banner)
        blay.setContentsMargins(8, 3, 8, 3)
        blay.addWidget(_label(
            "🤖  AI 助理可对此区域仓位执行自动操作（减仓 / 止损 / 加仓建议）",
            C["blue"], 10))
        blay.addStretch()
        blay.addWidget(_label("右键 → 移至手动区 可撤销授权", C["dim"], 9))
        lay.addWidget(banner)

        self._ai_pos_table = self._make_pos_table()
        self._ai_pos_table.customContextMenuRequested.connect(
            self._on_ai_pos_context_menu)
        self._ai_pos_table.doubleClicked.connect(
            lambda idx: self._on_pos_double_click_table(idx, self._ai_pos_table))
        lay.addWidget(self._ai_pos_table, 1)

        # 快捷按钮行
        btn_row = QHBoxLayout()
        for text, bg, hov, slot in [
            ("🛡️ AI风控评估",  C["blue"],   "#388bfd", self._quick_risk_assess),
            ("⚡ AI减仓建议",  C["orange"], "#f0883e", self._lock_profits),
            ("🔒 止盈建议",    "#238636",   "#2ea043", self._lock_profits),
        ]:
            b = _btn(text, bg, hov, h=26)
            b.setFont(QFont("", 9))
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        lay.addLayout(btn_row)
        return w

    def _build_manual_pos_tab(self) -> QWidget:
        """手动操作仓位标签页：仅限人工操作，AI 助理不触碰。"""
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        # 说明横幅
        banner = QFrame()
        banner.setStyleSheet(
            f"QFrame{{background:#1a0d00;border:1px solid {C['orange']};"
            f"border-radius:5px;padding:2px;}}")
        blay = QHBoxLayout(banner)
        blay.setContentsMargins(8, 3, 8, 3)
        blay.addWidget(_label(
            "👤  此区域仓位由人工独立管理，AI 助理无权自动操作",
            C["orange"], 10))
        blay.addStretch()
        blay.addWidget(_label("右键 → 移至AI区 可授权托管", C["dim"], 9))
        lay.addWidget(banner)

        self._manual_pos_table = self._make_pos_table()
        self._manual_pos_table.customContextMenuRequested.connect(
            self._on_manual_pos_context_menu)
        self._manual_pos_table.doubleClicked.connect(
            lambda idx: self._on_pos_double_click_table(idx, self._manual_pos_table))
        lay.addWidget(self._manual_pos_table, 1)

        # 只保留只读分析按钮，不含 AI 自动操作
        btn_row = QHBoxLayout()
        for text, bg, hov, slot in [
            ("📊 K线分析",  C["purple"], "#bc8cff", self._manual_kline_analysis),
            ("🔁 刷新持仓", "#21262d",   C["border"], self._refresh_positions_async),
        ]:
            b = _btn(text, bg, hov, h=26)
            b.setFont(QFont("", 9))
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        btn_row.addStretch()
        lay.addLayout(btn_row)
        return w

    def _build_performance_card(self) -> QGroupBox:
        g = _card("📈 本次会话绩效")
        gl = QGridLayout(g)
        gl.setSpacing(4)
        gl.setContentsMargins(8, 4, 8, 6)

        def _stat(label, attr):
            lb = _label(label + ":", C["dim"], 10)
            vl = _label("--", C["text"], 11, bold=True)
            setattr(self, attr, vl)
            return lb, vl

        stats = [
            ("决策轮次",  "_stat_cycles"),
            ("执行动作",  "_stat_actions"),
            ("风险预警",  "_stat_alerts"),
            ("会话时长",  "_stat_duration"),
            ("峰值余额",  "_stat_peak"),
            ("最大回撤",  "_stat_dd"),
        ]
        for i, (lbl, attr) in enumerate(stats):
            lb, vl = _stat(lbl, attr)
            gl.addWidget(lb, i // 2, (i % 2) * 2)
            gl.addWidget(vl, i // 2, (i % 2) * 2 + 1)

        # 绩效更新定时器
        self._perf_timer = QTimer(self)
        self._perf_timer.setInterval(10_000)
        self._perf_timer.timeout.connect(self._update_performance)
        self._perf_timer.start()
        return g

    def _build_log_card(self) -> QGroupBox:
        g = _card("⚡ 助理动作日志")
        ll = QVBoxLayout(g)
        self._feed = ActionFeed()
        ll.addWidget(self._feed)
        clr = _btn("清空日志", "#21262d", C["border"], h=24)
        clr.clicked.connect(self._feed.clear)
        ll.addWidget(clr)
        return g

    # ════════════════════════════════════════════════════════════
    # 工具方法
    # ════════════════════════════════════════════════════════════

    def _le(self, placeholder: str) -> QLineEdit:
        e = QLineEdit()
        e.setPlaceholderText(placeholder)
        e.setStyleSheet(
            f"QLineEdit{{background:{C['bg']};color:{C['text']};"
            f"border:1px solid {C['border']};border-radius:4px;padding:3px 6px;}}")
        return e

    # ════════════════════════════════════════════════════════════
    # 后台数据获取
    # ════════════════════════════════════════════════════════════

    def _refresh_positions_async(self):
        if not self.trade_executor or self._pos_refreshing:
            return
        self._pos_refreshing = True
        threading.Thread(target=self._fetch_positions_bg, daemon=True).start()

    def _fetch_positions_bg(self):
        rows = []
        try:
            positions = self.trade_executor.get_positions()
            now = time.time()
            for inst_id, pos in positions.items():
                if inst_id not in self._pos_entry_ts:
                    # 尝试从 OKX c_time 字段获取开仓时间
                    c_time = getattr(pos, "c_time", "") or ""
                    try:
                        self._pos_entry_ts[inst_id] = (
                            int(c_time) / 1000 if c_time.isdigit() else now)
                    except Exception:
                        self._pos_entry_ts[inst_id] = now

                # PositionInfo 字段：entry_price / unrealized_pnl / pnl_percent / side(enum)
                side_obj   = getattr(pos, "side", None)
                side_str   = side_obj.name if hasattr(side_obj, "name") else str(side_obj)
                avg_px     = float(getattr(pos, "entry_price",    0) or 0)
                cur_px     = float(getattr(pos, "current_price",  0) or 0)
                upl        = float(getattr(pos, "unrealized_pnl", 0) or 0)
                upl_ratio  = float(getattr(pos, "pnl_percent",    0) or 0)  # 已是 %
                size       = float(getattr(pos, "size",           0) or 0)
                notional   = float(getattr(pos, "notional_usd",   0) or 0)

                rows.append({
                    "inst_id":   inst_id,
                    "side":      side_str,
                    "avg_px":    avg_px,
                    "cur_px":    cur_px,
                    "upl_ratio": upl_ratio,
                    "upl":       upl,
                    "size":      size,
                    "notional":  notional,
                    "entry_ts":  self._pos_entry_ts.get(inst_id, now),
                })
        except Exception as e:
            print(f"[AssistantPage] 持仓刷新异常: {e}")
        finally:
            self._pos_refreshing = False
        self._sig_pos_data.emit(rows)

    def _fetch_market_async(self):
        if not self.okx_client or self._mkt_fetching:
            return
        self._mkt_fetching = True
        threading.Thread(target=self._fetch_market_bg, daemon=True).start()

    def _fetch_market_bg(self):
        data = {"btc_px": 0, "btc_chg": 0, "eth_px": 0, "eth_chg": 0,
                "regime": "unknown", "btc_funding": 0}
        try:
            for sym, px_key, chg_key in [
                ("BTC-USDT", "btc_px", "btc_chg"),
                ("ETH-USDT", "eth_px", "eth_chg"),
            ]:
                res = self.okx_client.get_ticker(sym)
                if res.get("code") == "0" and res.get("data"):
                    d = res["data"][0]
                    last  = float(d.get("last", 0) or 0)
                    open8 = float(d.get("sodUtc8", d.get("open24h", last)) or last)
                    chg   = (last - open8) / open8 * 100 if open8 else 0
                    data[px_key]  = last
                    data[chg_key] = chg

            # 资金费率 BTC-USDT-SWAP
            try:
                res = self.okx_client.get_funding_rate("BTC-USDT-SWAP")
                if res.get("code") == "0" and res.get("data"):
                    data["btc_funding"] = float(
                        res["data"][0].get("fundingRate", 0) or 0) * 100
            except Exception:
                pass

            # 市场情绪判断
            btc_chg = data["btc_chg"]
            if btc_chg > 3:
                data["regime"] = "🟢 强势上涨"
            elif btc_chg > 1:
                data["regime"] = "🔵 温和上行"
            elif btc_chg > -1:
                data["regime"] = "⚪ 横盘震荡"
            elif btc_chg > -3:
                data["regime"] = "🟡 弱势下行"
            else:
                data["regime"] = "🔴 急速下跌"
        except Exception:
            pass
        self._mkt_fetching = False
        self._sig_mkt_data.emit(data)

    # ════════════════════════════════════════════════════════════
    # 信号回调（主线程）
    # ════════════════════════════════════════════════════════════

    def _on_pos_data(self, rows: list):
        """更新持仓表格（AI管控区 + 手动操作区）+ 风控仪表盘"""
        now = datetime.now()
        self._pos_rows = rows  # 缓存供右键菜单使用

        ai_rows     = [r for r in rows if r["inst_id"] in self._ai_managed_ids]
        manual_rows = [r for r in rows if r["inst_id"] not in self._ai_managed_ids]

        self._fill_pos_table(self._ai_pos_table,     ai_rows,     zone="ai")
        self._fill_pos_table(self._manual_pos_table, manual_rows, zone="manual")

        ai_cnt     = len(ai_rows)
        manual_cnt = len(manual_rows)
        self._pos_update_lbl.setText(
            f"刷新: {now.strftime('%H:%M:%S')}  "
            f"AI管控 {ai_cnt} | 手动 {manual_cnt}")
        self._m_pos_lbl.setText(str(len(rows)))
        self._update_risk_gauges(rows)

    def _fill_pos_table(self, table: QTableWidget, rows: list, zone: str):
        """将持仓数据填充到指定表格。"""
        table.setRowCount(0)

        def _it(text, fg=None, align=Qt.AlignCenter, bold=False):
            it = QTableWidgetItem(str(text))
            it.setTextAlignment(align)
            if fg:
                it.setForeground(QColor(fg))
            if bold:
                f = QFont(); f.setBold(True); it.setFont(f)
            return it

        for r in rows:
            upl_ratio = r["upl_ratio"]
            upl       = r["upl"]
            side      = r["side"]
            color     = C["green"] if upl >= 0 else C["red"]
            side_c    = (C["green"] if "long" in side.lower() or "buy" in side.lower()
                         else C["red"] if "short" in side.lower() or "sell" in side.lower()
                         else C["dim"])

            held_sec = time.time() - r.get("entry_ts", time.time())
            held_h   = int(held_sec // 3600)
            held_m   = int((held_sec % 3600) // 60)
            held_str = f"{held_h}h{held_m}m" if held_h else f"{held_m}m"

            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, _it(r["inst_id"], align=Qt.AlignLeft | Qt.AlignVCenter))
            table.setItem(row, 1, _it(side, fg=side_c))
            table.setItem(row, 2, _it(
                f"{r['avg_px']:.4f}" if r['avg_px'] else "--", fg=C["dim"]))
            table.setItem(row, 3, _it(f"{upl_ratio:+.2f}%", fg=color, bold=True))
            table.setItem(row, 4, _it(f"{upl:+.2f}", fg=color))
            table.setItem(row, 5, _it(held_str, fg=C["dim"]))

            # 高亏损行背景
            if upl_ratio <= -3:
                for c in range(6):
                    it = table.item(row, c)
                    if it:
                        bg = "#1a0d0d" if zone == "manual" else "#2d1a1a"
                        it.setBackground(QColor(bg))

    def _update_risk_gauges(self, rows: list):
        max_pos = self._max_positions.value() if hasattr(self, "_max_positions") else 5
        max_loss_pct = self._max_loss.value() if hasattr(self, "_max_loss") else 5.0

        pos_count = len(rows)
        pos_pct   = pos_count / max_pos * 100 if max_pos else 0

        daily_loss = sum(r["upl"] for r in rows if r["upl"] < 0)
        # 估算日亏损占比（需要余额，用绝对亏损/持仓估算）
        total_upl = sum(r["upl"] for r in rows)
        worst_upl = min((r["upl_ratio"] for r in rows), default=0)
        loss_pct  = abs(worst_upl) / max_loss_pct * 100 if max_loss_pct else 0

        # 集中度：最大单仓占总仓位的比例
        upls = [abs(r["upl"]) for r in rows]
        total_abs = sum(upls) or 1
        max_conc  = max(upls, default=0) / total_abs * 100 if upls else 0

        # 历史回撤
        dd_pct = 0.0
        if self._peak_balance > 0:
            bal_str = getattr(self, "_m_bal_lbl", None)
            # 从标签里取当前余额
            try:
                cur = float((bal_str.text() if bal_str else "0").replace(",", ""))
                dd_pct = (self._peak_balance - cur) / self._peak_balance * 100
                dd_pct = max(0, dd_pct)
            except Exception:
                pass

        self._gauge_pos.update(pos_pct,  f"{pos_count}/{max_pos}")
        self._gauge_loss.update(loss_pct, f"{worst_upl:.1f}%")
        self._gauge_conc.update(max_conc, f"{max_conc:.0f}%")
        self._gauge_dd.update(min(dd_pct * 5, 100), f"{dd_pct:.1f}%")

        # 综合风险评级
        risk_score = max(pos_pct, loss_pct, max_conc)
        if risk_score >= 80:
            risk_text  = "🔴 HIGH — 建议立即检查仓位"
            risk_color = C["red"]
        elif risk_score >= 50:
            risk_text  = "🟡 MEDIUM — 保持关注"
            risk_color = C["yellow"]
        else:
            risk_text  = "🟢 LOW — 风险可控"
            risk_color = C["green"]
        self._risk_summary_lbl.setText(f"▶  {risk_text}")
        self._risk_summary_lbl.setStyleSheet(
            f"color:{risk_color};font-size:11px;font-weight:bold;")

    def _on_mkt_data(self, data: dict):
        btc_px   = data.get("btc_px", 0)
        btc_chg  = data.get("btc_chg", 0)
        eth_px   = data.get("eth_px", 0)
        eth_chg  = data.get("eth_chg", 0)
        regime   = data.get("regime", "--")
        funding  = data.get("btc_funding", 0)

        if btc_px:
            self._btc_ticker.update(btc_px, btc_chg)
        if eth_px:
            self._eth_ticker.update(eth_px, eth_chg)

        self._regime_lbl.setText(regime)
        regime_color = (C["red"] if "下跌" in regime else
                        C["green"] if "上涨" in regime else
                        C["yellow"] if "弱势" in regime else C["blue"])
        self._regime_lbl.setStyleSheet(
            f"color:{regime_color};font-size:12px;font-weight:bold;")

        fund_color = C["red"] if funding > 0.05 else \
                     C["green"] if funding < -0.01 else C["dim"]
        self._funding_lbl.setText(f"{funding:+.4f}%")
        self._funding_lbl.setStyleSheet(
            f"color:{fund_color};font-size:11px;font-weight:bold;")

        self._mkt_update_lbl.setText(
            f"行情: {datetime.now().strftime('%H:%M:%S')}")

    # ════════════════════════════════════════════════════════════
    # 扫描结果推送 / AI 分析
    # ════════════════════════════════════════════════════════════

    def on_scan_results_received(self, results: list):
        # 按扫描时间由新到旧排序（updated_at 为 ISO 字符串，字典序即时间序）
        def _sort_key(r):
            return str(r.get('updated_at', r.get('time', '')) or '')
        sorted_results = sorted(results, key=_sort_key, reverse=True)
        self._last_results = sorted_results
        self._scan_count_lbl.setText(f"扫描结果: {len(sorted_results)} 条")
        self._scan_time_lbl.setText(f"最后更新: {datetime.now().strftime('%H:%M:%S')}")
        self._feed.push(f"📥 收到扫描结果 {len(sorted_results)} 条", "INFO")
        self._populate_table(sorted_results, ai_advices={})
        self._m_scn_lbl.setText(str(len(sorted_results)))
        # 自动记录信号用于追踪验证
        self._record_signals_for_tracking(results)
        if self._chk_auto_analyze.isChecked():
            QTimer.singleShot(300, self._trigger_analysis_now)

    def _populate_table(self, results: list, ai_advices: dict):
        top = results[:30] if self._chk_top_only.isChecked() else results
        self._result_table.setRowCount(0)
        for item in top:
            sym       = item.get("symbol", item.get("instId", "?"))
            category  = item.get("category", "--")
            direction = str(item.get("side", item.get("direction", "WATCH"))).upper()
            score     = float(item.get("opportunity_score", item.get("score", 0)) or 0)
            price     = float(item.get("last_price", 0) or 0)
            ai        = ai_advices.get(sym, {})
            advice    = ai.get("advice", "--")
            conf      = ai.get("confidence", "--")
            ai_reason = ai.get("reason", "")

            # 扫描时间：优先 updated_at(ISO)，其次 time 字段
            raw_ts = str(item.get('updated_at', item.get('time', '')) or '')
            try:
                # ISO 格式 "2024-01-01T12:34:56.123456" → "12:34:56"
                scan_time = raw_ts[11:19] if 'T' in raw_ts else raw_ts[11:19] if len(raw_ts) >= 19 else raw_ts
            except Exception:
                scan_time = "--"

            row = self._result_table.rowCount()
            self._result_table.insertRow(row)

            dir_color = (C["green"] if "BUY" in direction or "LONG" in direction else
                         C["red"]   if "SELL" in direction or "SHORT" in direction else
                         C["dim"])
            adv_color = ADVICE_COLORS.get(advice, C["dim"])

            def _it(text, fg=None, align=Qt.AlignCenter, bold=False):
                it = QTableWidgetItem(str(text))
                it.setTextAlignment(align)
                if fg: it.setForeground(QColor(fg))
                if bold:
                    f = QFont(); f.setBold(True); it.setFont(f)
                return it

            # col 0: 扫描时间（新增列），col 1-8: 原有列整体右移 1
            self._result_table.setItem(row, 0, _it(scan_time, fg=C["dim"]))
            self._result_table.setItem(row, 1, _it(sym, align=Qt.AlignLeft | Qt.AlignVCenter))
            self._result_table.setItem(row, 2, _it(category))
            self._result_table.setItem(row, 3, _it(direction, fg=dir_color))
            self._result_table.setItem(row, 4, _it(f"{score:.1f}"))
            self._result_table.setItem(row, 5, _it(f"{price:.4f}" if price else "--",
                                                    align=Qt.AlignRight | Qt.AlignVCenter))
            self._result_table.setItem(row, 6, _it(advice, fg=adv_color, bold=True))
            self._result_table.setItem(row, 7, _it(conf))
            self._result_table.setItem(row, 8, _it(ai_reason,
                                                    align=Qt.AlignLeft | Qt.AlignVCenter))

    def _trigger_analysis_now(self):
        if not self._last_results:
            self._feed.push("暂无扫描结果可分析", "WARNING"); return
        client = self._build_llm_client()
        if not client:
            self._feed.push("请先填写 LLM API 地址和 Key", "ERROR"); return
        self._analyze_btn2.setEnabled(False)
        self._analyze_now_btn.setEnabled(False)
        self._analysis_progress.setText("🧠 AI 正在分析扫描结果...")
        self._feed.push(f"开始分析 {len(self._last_results)} 条扫描结果...", "ANALYSIS")
        results = self._last_results[:30] if self._chk_top_only.isChecked() else self._last_results

        def _do():
            advices, summary = self._run_ai_analysis(client, results)
            QTimer.singleShot(0, lambda: self._on_analysis_done(advices, summary, results))
        threading.Thread(target=_do, daemon=True).start()

    def _run_ai_analysis(self, client: LLMClient, results: list) -> tuple:
        rows = []
        for i, item in enumerate(results[:30]):
            sym      = item.get("symbol", item.get("instId", "?"))
            category = item.get("category", "--")
            direction= str(item.get("side", item.get("direction", "WATCH"))).upper()
            score    = float(item.get("opportunity_score", item.get("score", 0)) or 0)
            reason   = item.get("priority_reason", item.get("reason", ""))[:80]
            rows.append(f"{i+1}. {sym} | {category} | {direction} | 评分{score:.1f} | {reason}")

        position_text = "无持仓"
        if self.trade_executor:
            try:
                positions = self.trade_executor.get_positions()
                if positions:
                    p_rows = []
                    for inst_id, pos in positions.items():
                        upl_ratio = float(getattr(pos, "upl_ratio", 0) or 0) * 100
                        p_rows.append(
                            f"  {inst_id} | {getattr(pos,'side','?')} | 盈亏{upl_ratio:+.2f}%")
                    position_text = "\n".join(p_rows)
            except Exception:
                pass

        system_prompt = """你是一位专业的加密货币量化交易分析师。
根据策略扫描系统发现的机会，结合持仓情况，对每个标的给出专业操作建议。

建议类型（选一个）：加仓 / 买入 / 做空 / 观望 / 持有 / 减仓 / 卖出 / 止损
置信度：高 / 中 / 低

输出严格 JSON：
{
  "summary": "市场整体判断（2-3句）",
  "regime": "当前市场机制（趋势/震荡/高波动）",
  "top_picks": ["最佳机会1", "最佳机会2"],
  "risks": "主要风险提示",
  "advices": {
    "BTC-USDT-SWAP": {"advice": "加仓", "confidence": "高", "reason": "理由"}
  }
}"""
        user_msg = (
            f"扫描发现 {len(results)} 个机会（按评分排序）:\n\n"
            + "\n".join(rows)
            + f"\n\n当前持仓:\n{position_text}\n\n"
            "请对每个标的给出建议，并给出市场综合判断。"
        )
        raw = client.chat(
            [{"role": "system", "content": system_prompt},
             {"role": "user",   "content": user_msg}],
            timeout=90,
        )
        if not raw:
            return {}, f"LLM 无响应: {client.last_error}"
        try:
            s = raw.find("{"); e = raw.rfind("}") + 1
            data    = json.loads(raw[s:e])
            advices = data.get("advices", {})
            parts   = []
            if data.get("regime"):
                parts.append(f"<b>📡 市场机制</b>：{data['regime']}")
            if data.get("summary"):
                parts.append(f"<b>📊 综合判断</b><br>{data['summary']}")
            if data.get("top_picks"):
                parts.append(f"<b>⭐ 重点关注</b>：{'、'.join(data['top_picks'])}")
            if data.get("risks"):
                parts.append(f"<b>⚠️ 风险提示</b>：{data['risks']}")
            return advices, "<br><br>".join(parts)
        except Exception as e:
            return {}, f"解析失败: {e}<br>原始: {raw[:300]}"

    def _on_analysis_done(self, advices: dict, summary: str, results: list):
        self._populate_table(results, ai_advices=advices)
        self._summary_text.setHtml(
            f'<div style="color:{C["text"]};font-size:12px;line-height:1.7;">{summary}</div>')
        n = len(advices)
        self._feed.push(f"✅ AI 分析完成，评估 {n} 个标的", "SUCCESS")

        counts: dict = {}
        for v in advices.values():
            a = v.get("advice", "?")
            counts[a] = counts.get(a, 0) + 1
        dist = "  ".join(f"{k}×{v}" for k, v in sorted(counts.items(), key=lambda x: -x[1]))
        if dist:
            self._feed.push(f"建议分布: {dist}", "INFO")

        self._analysis_progress.setText(
            f"✅ {datetime.now().strftime('%H:%M:%S')} · {n} 个标的")
        self._analyze_btn2.setEnabled(True)
        self._analyze_now_btn.setEnabled(True)

        for sym, ai in advices.items():
            if ai.get("advice") in ("加仓", "买入", "做空") and ai.get("confidence") == "高":
                self._feed.push(
                    f"⭐ 高置信: {sym} → {ai['advice']} | {ai.get('reason','')[:60]}",
                    "TRADE")

    # ════════════════════════════════════════════════════════════
    # 智能快捷操作
    # ════════════════════════════════════════════════════════════

    def _quick_risk_assess(self):
        """即时风险评估：基于当前持仓生成文字报告"""
        if not self.trade_executor:
            self._feed.push("未连接交易执行器", "ERROR"); return
        self._feed.push("🛡️ 即时风险评估中...", "RISK")
        threading.Thread(target=self._do_risk_assess, daemon=True).start()

    def _do_risk_assess(self):
        try:
            positions = self.trade_executor.get_positions()
            if not positions:
                QTimer.singleShot(0, lambda: self._feed.push(
                    "当前无持仓，风险等级: 零", "SUCCESS"))
                return
            lines = []
            total_loss = 0.0
            max_loss_r  = 0.0
            for inst_id, pos in positions.items():
                upl_ratio = float(getattr(pos, "pnl_percent", 0) or 0)  # 已是 %
                upl       = float(getattr(pos, "unrealized_pnl", 0) or 0)
                zone_tag  = "🤖" if inst_id in self._ai_managed_ids else "👤"
                if upl < 0:
                    total_loss += abs(upl)
                    max_loss_r  = max(max_loss_r, abs(upl_ratio))
                level = "🔴" if upl_ratio < -3 else "🟡" if upl_ratio < 0 else "🟢"
                lines.append(
                    f"{level}{zone_tag} {inst_id}: 盈亏{upl_ratio:+.2f}% / {upl:+.2f} USDT")

            overall = ("🔴 HIGH" if max_loss_r > 5 or total_loss > 200 else
                       "🟡 MEDIUM" if max_loss_r > 2 or total_loss > 50 else
                       "🟢 LOW")
            report = (f"风险总评: {overall} | "
                      f"总浮亏: {total_loss:.2f} USDT | 最大单仓亏损: {max_loss_r:.1f}%\n"
                      "🤖=AI管控  👤=手动区\n"
                      + "\n".join(lines))
            QTimer.singleShot(0, lambda r=report: self._feed.push(r, "RISK"))
        except Exception as e:
            QTimer.singleShot(0, lambda: self._feed.push(f"风险评估失败: {e}", "ERROR"))

    def _emergency_half_reduce(self):
        """一键半仓减仓：对所有仓位减少 50%，需二次确认"""
        if not self.trade_executor:
            self._feed.push("未连接交易执行器", "ERROR"); return
        reply = QMessageBox.warning(
            self, "⚡ 一键半仓减仓",
            "将对所有当前持仓减仓 50%！\n\n"
            "此操作不可撤销，是否继续？",
            QMessageBox.Yes | QMessageBox.Cancel,
        )
        if reply != QMessageBox.Yes:
            self._feed.push("已取消半仓减仓", "INFO"); return
        self._feed.push("⚡ 执行一键半仓减仓...", "TRADE")
        threading.Thread(target=self._do_half_reduce, daemon=True).start()

    def _do_half_reduce(self):
        # 检查 executor 是否支持减仓接口
        has_reduce = hasattr(self.trade_executor, "execute_reduce")
        has_sell   = hasattr(self.trade_executor, "execute_sell")
        if not has_reduce and not has_sell:
            QTimer.singleShot(0, lambda: self._feed.push(
                "⚠️ 交易执行器不支持减仓操作，请手动操作", "WARNING"))
            return
        try:
            positions = self.trade_executor.get_positions()
            if not positions:
                QTimer.singleShot(0, lambda: self._feed.push("当前无持仓", "INFO"))
                return
            skipped = []
            for inst_id, pos in positions.items():
                # 只操作 AI 管控区的持仓；手动区严格隔离
                if inst_id not in self._ai_managed_ids:
                    skipped.append(inst_id)
                    continue
                try:
                    size = float(getattr(pos, "size", 0) or 0)
                    if size <= 0:
                        continue
                    half = size / 2
                    if has_reduce:
                        result = self.trade_executor.execute_reduce(inst_id, ratio=0.5)
                    else:
                        side = str(getattr(pos, "side", "long")).lower()
                        if "long" in side or "buy" in side:
                            result = self.trade_executor.execute_sell(inst_id, half)
                        elif hasattr(self.trade_executor, "execute_cover"):
                            result = self.trade_executor.execute_cover(inst_id, half)
                        else:
                            result = self.trade_executor.execute_sell(inst_id, half)
                    ok  = getattr(result, "success", False)
                    msg = f"{'✅' if ok else '❌'} {inst_id} 减仓50% (半仓={half:.4f})"
                    lv  = "SUCCESS" if ok else "ERROR"
                    QTimer.singleShot(0, lambda m=msg, l=lv: self._feed.push(m, l))
                except Exception as e:
                    QTimer.singleShot(
                        0, lambda ei=inst_id, err=e:
                        self._feed.push(f"❌ {ei} 减仓失败: {err}", "ERROR"))
            if skipped:
                QTimer.singleShot(0, lambda s=skipped: self._feed.push(
                    f"🔒 手动区仓位已跳过（{len(s)} 个）: {', '.join(s[:5])}", "WARNING"))
        except Exception as e:
            QTimer.singleShot(0, lambda: self._feed.push(f"❌ 半仓减仓异常: {e}", "ERROR"))

    def _lock_profits(self):
        """锁定所有浮盈仓位：触发 AI 为每个盈利仓位推荐止盈价"""
        client = self._build_llm_client()
        if not client:
            self._feed.push("请先配置 LLM", "ERROR"); return
        if not self.trade_executor:
            self._feed.push("未连接交易执行器", "ERROR"); return
        self._feed.push("🔒 正在为盈利仓位生成止盈建议...", "ANALYSIS")
        threading.Thread(target=self._do_lock_profits, args=(client,), daemon=True).start()

    def _do_lock_profits(self, client: LLMClient):
        try:
            positions = self.trade_executor.get_positions()
            profitable = []
            for inst_id, pos in positions.items():
                upl_ratio = float(getattr(pos, "upl_ratio", 0) or 0) * 100
                if upl_ratio > 0:
                    profitable.append({
                        "inst_id":  inst_id,
                        "side":     str(getattr(pos, "side", "?")),
                        "avg_px":   float(getattr(pos, "avg_px", 0) or 0),
                        "upl_ratio":upl_ratio,
                    })
            if not profitable:
                QTimer.singleShot(
                    0, lambda: self._feed.push("当前无盈利仓位", "INFO")); return

            pos_text = "\n".join(
                f"- {p['inst_id']} {p['side']} 均价{p['avg_px']:.4f} 浮盈{p['upl_ratio']:+.1f}%"
                for p in profitable)
            raw = client.chat(
                [{"role": "system", "content":
                    "你是资深量化交易风控专家。根据持仓情况给出具体止盈建议。"
                    "输出JSON: {\"positions\": {\"inst_id\": {\"tp_price\": 0.0, "
                    "\"reason\": \"20字内\", \"action\": \"部分止盈|全部止盈|继续持有\"}}}"},
                 {"role": "user", "content":
                    f"以下盈利仓位请给出止盈建议:\n{pos_text}"}],
                timeout=30,
            )
            if raw:
                s = raw.find("{"); e = raw.rfind("}") + 1
                data = json.loads(raw[s:e])
                for inst_id, rec in data.get("positions", {}).items():
                    tp    = rec.get("tp_price", 0)
                    act   = rec.get("action", "继续持有")
                    rsn   = rec.get("reason", "")
                    msg   = (f"🔒 {inst_id} → {act}"
                             + (f" 止盈价:{tp:.4f}" if tp else "")
                             + (f" | {rsn}" if rsn else ""))
                    QTimer.singleShot(
                        0, lambda m=msg: self._feed.push(m, "TRADE"))
        except Exception as e:
            QTimer.singleShot(
                0, lambda: self._feed.push(f"锁定浮盈失败: {e}", "ERROR"))

    def _stress_test(self):
        """压力测试：模拟 BTC 下跌 20% 对当前组合的影响"""
        if not self.trade_executor:
            self._feed.push("未连接交易执行器", "ERROR"); return
        self._feed.push("📊 压力测试：模拟 BTC -20% 场景...", "ANALYSIS")
        threading.Thread(target=self._do_stress_test, daemon=True).start()

    def _do_stress_test(self):
        try:
            positions = self.trade_executor.get_positions()
            if not positions:
                QTimer.singleShot(
                    0, lambda: self._feed.push("无持仓，压力测试无意义", "INFO")); return
            lines = ["📊 压力测试结果（假设 BTC -20%，山寨币 -30%）:"]
            total_est_loss = 0.0
            for inst_id, pos in positions.items():
                upl     = float(getattr(pos, "upl", 0) or 0)
                side    = str(getattr(pos, "side", "long")).lower()
                is_btc  = "BTC" in inst_id.upper()
                drop    = 0.20 if is_btc else 0.30
                # 多头亏损，空头盈利
                est = upl - abs(upl + float(getattr(pos, "avg_px", 1) or 1)) * drop \
                      if "long" in side or "buy" in side else \
                      upl + abs(upl + float(getattr(pos, "avg_px", 1) or 1)) * drop
                total_est_loss += est
                icon = "🔴" if est < 0 else "🟢"
                lines.append(f"  {icon} {inst_id}: 预估盈亏 {est:+.2f} USDT")
            lines.append(
                f"  总预估盈亏: {total_est_loss:+.2f} USDT  "
                f"{'⚠️ 建议适当减仓' if total_est_loss < -100 else '✅ 风险可承受'}")
            report = "\n".join(lines)
            QTimer.singleShot(0, lambda r=report: self._feed.push(r, "RISK"))
        except Exception as e:
            QTimer.singleShot(
                0, lambda: self._feed.push(f"压力测试失败: {e}", "ERROR"))

    def _trigger_scan_action(self):
        if self.scanner_page and hasattr(self.scanner_page, "start_scan"):
            self._feed.push("🔄 手动触发扫描...", "INFO")
            try:
                self.scanner_page.start_scan()
            except Exception as e:
                self._feed.push(f"触发扫描失败: {e}", "ERROR")
        else:
            self._feed.push("未连接扫描器", "WARNING")

    # ════════════════════════════════════════════════════════════
    # Agent 控制
    # ════════════════════════════════════════════════════════════

    def _toggle_agent(self):
        self._stop_agent() if self._running else self._start_agent()

    def _start_agent(self):
        client = self._build_llm_client()
        if not client:
            self._feed.push("请先填写 LLM API 地址和 Key", "ERROR"); return
        perm = Permission(
            can_trade=self._chk_trade.isChecked(),
            can_close=self._chk_close.isChecked(),
            can_scan=self._chk_scan.isChecked(),
            can_modify_strategy=self._chk_mod.isChecked(),
            require_confirm=self._chk_confirm.isChecked(),
            max_single_usdt=self._max_usdt.value(),
            max_daily_loss_pct=self._max_loss.value(),
        )
        self._agent = TradingAssistant(
            llm_client=client,
            okx_client=self.okx_client,
            trade_executor=self.trade_executor,
            scanner_page=self.scanner_page,
            permission=perm,
            interval=self._interval.value(),
        )
        self._agent.action_log.connect(lambda m, lv: self._feed.push(m, lv))
        self._agent.cycle_done.connect(self._on_cycle_done)
        self._agent.metrics_upd.connect(self._on_metrics)
        self._agent.confirm_req.connect(self._on_confirm)
        self._agent.position_alert.connect(self._on_position_alert)
        self._agent.start()
        self._session_start = datetime.now()
        self._running = True
        self._dot.setStyleSheet(f"color:{C['green']};font-size:18px;")
        self._status_lbl.setText("助理运行中 — 自主监控")
        self._toggle_btn.setText("⏹ 停止助理")
        self._toggle_btn.setStyleSheet(
            self._toggle_btn.styleSheet()
            .replace("#238636", C["dark_red"]).replace("#2ea043", C["red"]))
        self._save_config()

    def _stop_agent(self):
        if self._agent:
            self._agent.stop()
            self._agent.wait(2000)
            self._agent = None
        self._running = False
        self._dot.setStyleSheet(f"color:#21262d;font-size:18px;")
        self._status_lbl.setText("已停止")
        self._toggle_btn.setText("🚀 启动助理")
        self._toggle_btn.setStyleSheet(
            self._toggle_btn.styleSheet()
            .replace(C["dark_red"], "#238636").replace(C["red"], "#2ea043"))

    def closeEvent(self, event):
        try:
            self._stop_agent()
        except Exception:
            pass
        for dlg in list(getattr(self, "_open_symbol_dialogs", [])):
            try:
                dlg.close()
            except Exception:
                pass
        self._open_symbol_dialogs.clear()
        super().closeEvent(event)

    def _emergency_stop(self):
        self._stop_agent()
        self._feed.push("⛔ 紧急停止 — 所有自动操作已暂停", "WARNING")

    # ════════════════════════════════════════════════════════════
    # Agent 信号槽
    # ════════════════════════════════════════════════════════════

    def _on_cycle_done(self, cycle: AgentCycle):
        self._session_cycles += 1
        self._session_actions += len(cycle.actions)
        self._cycle_lbl.setText(
            f"轮次: {cycle.cycle_id} | 决策: {self._session_actions} | 预警: {self._session_alerts}")

    def _on_metrics(self, m: dict):
        bal = m.get("balance", 0)
        pnl = m.get("pnl_pct", 0)
        if bal > self._peak_balance:
            self._peak_balance = bal
        self._m_bal_lbl.setText(f"{bal:,.2f}")
        self._m_pnl_lbl.setText(f"{pnl:+.2f}%")
        self._m_pnl_lbl.setStyleSheet(
            f"color:{C['green'] if pnl >= 0 else C['red']};font-size:15px;font-weight:bold;")
        self._refresh_positions_async()

    def _on_confirm(self, act: AgentAction):
        reply = QMessageBox.question(
            self, "AI 助理请求确认",
            f"操作类型：{act.action_type}\n"
            f"参数：{json.dumps(act.params, ensure_ascii=False)}\n"
            f"理由：{act.reason}\n风险：{act.risk_level}\n\n是否执行？",
            QMessageBox.Yes | QMessageBox.No,
        )
        if self._agent:
            self._agent.set_confirm_result(reply == QMessageBox.Yes)

    def _on_position_alert(self, alerts: list):
        self._session_alerts += 1
        high = [a for a in alerts if a.get("risk") in ("medium", "high")]
        self._feed.push(
            f"🚨 持仓监控预警 — {len(alerts)} 个仓位，{len(high)} 个需关注",
            "WARNING" if high else "INFO")
        dlg = PositionAlertDialog(
            alerts, self.okx_client, self._build_llm_client(), parent=self)
        dlg.show()
        self._cycle_lbl.setText(
            f"轮次: {self._session_cycles} | 决策: {self._session_actions}"
            f" | 预警: {self._session_alerts}")

    # ════════════════════════════════════════════════════════════
    # 绩效更新
    # ════════════════════════════════════════════════════════════

    def _update_performance(self):
        dur = datetime.now() - self._session_start
        h = int(dur.total_seconds() // 3600)
        m = int((dur.total_seconds() % 3600) // 60)
        self._stat_cycles.setText(str(self._session_cycles))
        self._stat_actions.setText(str(self._session_actions))
        self._stat_alerts.setText(str(self._session_alerts))
        self._stat_duration.setText(f"{h}h{m}m")
        if self._peak_balance > 0:
            self._stat_peak.setText(f"{self._peak_balance:,.2f}")
        try:
            cur_str = self._m_bal_lbl.text().replace(",", "")
            cur = float(cur_str) if cur_str and cur_str != "--" else 0
            dd  = max(0, (self._peak_balance - cur) / self._peak_balance * 100) \
                  if self._peak_balance > 0 else 0
            color = C["red"] if dd > 3 else C["yellow"] if dd > 1 else C["green"]
            self._stat_dd.setText(f"{dd:.2f}%")
            self._stat_dd.setStyleSheet(
                f"color:{color};font-size:11px;font-weight:bold;")
        except Exception:
            pass

    # ════════════════════════════════════════════════════════════
    # LLM 客户端 & 配置持久化
    # ════════════════════════════════════════════════════════════

    def _build_llm_client(self) -> Optional[LLMClient]:
        url   = self._url_edit.text().strip()
        key   = self._key_edit.text().strip()
        model = self._model_combo.currentText().strip()
        if not url or not key:
            return None
        return LLMClient(base_url=url, api_key=key, model=model)

    def _test_llm(self):
        client = self._build_llm_client()
        if not client:
            self._feed.push("请填写 API 地址和 Key", "ERROR"); return
        self._feed.push("测试 LLM 连接...", "INFO")
        def _do():
            ok  = client.test_connection()
            msg = "✅ LLM 连接成功" if ok else f"❌ 连接失败: {client.last_error}"
            lv  = "SUCCESS" if ok else "ERROR"
            QTimer.singleShot(0, lambda: self._feed.push(msg, lv))
        threading.Thread(target=_do, daemon=True).start()

    def _save_config(self):
        from pathlib import Path
        p = Path(__file__).resolve().parent.parent.parent / "data" / "assistant_llm_config.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "url":   self._url_edit.text(),
            "key":   self._key_edit.text(),
            "model": self._model_combo.currentText(),
        }, ensure_ascii=False))

    def _load_config(self):
        from pathlib import Path
        p = Path(__file__).resolve().parent.parent.parent / "data" / "assistant_llm_config.json"
        try:
            if p.exists():
                cfg = json.loads(p.read_text())
                self._url_edit.setText(cfg.get("url", ""))
                self._key_edit.setText(cfg.get("key", ""))
                self._model_combo.setCurrentText(cfg.get("model", "deepseek-v4-pro"))
        except Exception:
            pass

    # ════════════════════════════════════════════════════════════
    # 右键菜单 & 双击
    # ════════════════════════════════════════════════════════════

    def _get_row_symbol_and_item(self, row: int):
        sym_item = self._result_table.item(row, 1)   # col 0 = 扫描时间, col 1 = 交易对
        if not sym_item:
            return None, None
        sym = sym_item.text().strip()
        scan_item = next(
            (r for r in self._last_results
             if r.get("symbol", r.get("instId", "")) == sym),
            {"symbol": sym})
        return sym, scan_item

    def _on_table_context_menu(self, pos):
        row = self._result_table.rowAt(pos.y())
        if row < 0:
            return
        sym, scan_item = self._get_row_symbol_and_item(row)
        if not sym:
            return
        from src.qt_compat import QMenu, QAction
        menu = QMenu(self)
        menu.setStyleSheet(self._menu_style())
        act_analyze = menu.addAction("🧠  AI 深度分析（K 线 + 综合评估）")
        act_analyze.setFont(QFont("", -1, QFont.Bold))
        menu.addSeparator()
        act_buy  = menu.addAction("📈  标记买入观察")
        act_sell = menu.addAction("📉  标记卖出观察")
        menu.addSeparator()
        act_copy = menu.addAction("📋  复制交易对名称")
        action   = menu.exec(self._result_table.viewport().mapToGlobal(pos))
        if action == act_analyze:
            self._open_symbol_dialog(sym, scan_item)
        elif action == act_copy:
            from src.qt_compat import QApplication
            QApplication.clipboard().setText(sym)
            self._feed.push(f"已复制: {sym}", "INFO")
        elif action == act_buy:
            self._feed.push(f"📈 {sym} 已标记买入观察", "TRADE")
        elif action == act_sell:
            self._feed.push(f"📉 {sym} 已标记卖出观察", "TRADE")

    def _on_table_double_click(self, index):
        sym, scan_item = self._get_row_symbol_and_item(index.row())
        if sym:
            self._open_symbol_dialog(sym, scan_item)

    def _pos_row_info(self, table: QTableWidget, row: int):
        """从表格某行提取 (sym, side, pnl_pct, upl) 四元组。"""
        def _txt(col):
            it = table.item(row, col)
            return it.text().strip() if it else ""
        return _txt(0), _txt(1), _txt(3), _txt(4)

    def _on_ai_pos_context_menu(self, pos):
        """AI 管控区右键菜单 — 含 AI 操作 + 移至手动区。"""
        row = self._ai_pos_table.rowAt(pos.y())
        if row < 0:
            return
        sym, side, pnl_pct, upl = self._pos_row_info(self._ai_pos_table, row)
        if not sym:
            return
        from src.qt_compat import QMenu
        menu = QMenu(self)
        menu.setStyleSheet(self._menu_style())
        act_analyze = menu.addAction("🧠  AI 深度分析（K线 + 综合评估）")
        act_analyze.setFont(QFont("", -1, QFont.Bold))
        act_close_ai = menu.addAction("📉  AI 建议平仓分析")
        act_risk     = menu.addAction("🛡️  AI 单仓风险评估")
        menu.addSeparator()
        act_to_manual = menu.addAction("👤  移至手动操作区（撤销 AI 授权）")
        act_to_manual.setFont(QFont("", -1))
        menu.addSeparator()
        act_copy = menu.addAction("📋  复制交易对名称")
        action = menu.exec(self._ai_pos_table.viewport().mapToGlobal(pos))
        scan_item = {
            "symbol": sym, "category": "持仓分析",
            "side": side, "direction": side, "opportunity_score": 0,
            "priority_reason": f"AI管控仓位 — 盈亏 {pnl_pct}，浮盈 {upl} USDT",
        }
        if action == act_analyze:
            self._open_symbol_dialog(sym, scan_item)
        elif action == act_close_ai:
            scan_item["category"] = "AI平仓分析"
            scan_item["priority_reason"] = f"AI评估平仓时机 — 盈亏 {pnl_pct}，浮盈 {upl} USDT"
            self._open_symbol_dialog(sym, scan_item)
        elif action == act_risk:
            self._feed.push(f"🛡️ {sym} 风险评估: 盈亏={pnl_pct}  浮盈={upl}U", "RISK")
        elif action == act_to_manual:
            self._move_to_manual(sym)
        elif action == act_copy:
            from src.qt_compat import QApplication
            QApplication.clipboard().setText(sym)

    def _on_manual_pos_context_menu(self, pos):
        """手动操作区右键菜单 — 只读分析 + 移至 AI 区。"""
        row = self._manual_pos_table.rowAt(pos.y())
        if row < 0:
            return
        sym, side, pnl_pct, upl = self._pos_row_info(self._manual_pos_table, row)
        if not sym:
            return
        from src.qt_compat import QMenu
        menu = QMenu(self)
        menu.setStyleSheet(self._menu_style())
        act_kline = menu.addAction("📊  K线图分析（只读）")
        act_kline.setFont(QFont("", -1, QFont.Bold))
        menu.addSeparator()
        act_to_ai = menu.addAction("🤖  移至 AI 管控区（授权 AI 托管）")
        menu.addSeparator()
        act_copy  = menu.addAction("📋  复制交易对名称")
        # AI 禁止提示（置灰）
        menu.addSeparator()
        no_ai = menu.addAction("⛔  AI 助理禁止自动操作此区域")
        no_ai.setEnabled(False)
        action = menu.exec(self._manual_pos_table.viewport().mapToGlobal(pos))
        if action == act_kline:
            scan_item = {
                "symbol": sym, "category": "手动持仓分析（只读）",
                "side": side, "direction": side, "opportunity_score": 0,
                "priority_reason": f"手动仓位 — 盈亏 {pnl_pct}，浮盈 {upl} USDT",
            }
            self._open_symbol_dialog(sym, scan_item)
        elif action == act_to_ai:
            self._move_to_ai(sym)
        elif action == act_copy:
            from src.qt_compat import QApplication
            QApplication.clipboard().setText(sym)

    def _on_pos_double_click_table(self, index, table: QTableWidget):
        """统一双击处理：打开 K 线分析弹窗（仅查看，不触发操作）。"""
        row = index.row()
        sym, side, pnl, upl = self._pos_row_info(table, row)
        if not sym:
            return
        zone = "AI管控" if table is self._ai_pos_table else "手动"
        scan_item = {
            "symbol": sym, "category": f"持仓分析（{zone}）",
            "side": side,
            "priority_reason": f"持仓综合分析 — 盈亏 {pnl}，浮盈 {upl} USDT",
        }
        self._open_symbol_dialog(sym, scan_item)

    # ── 持仓区域管理 ──────────────────────────────────────────────
    def _move_to_ai(self, inst_id: str):
        """将持仓移入 AI 管控区。"""
        self._ai_managed_ids.add(inst_id)
        self._feed.push(
            f"🤖 {inst_id} 已授权 AI 管控（AI 助理可自动操作）", "INFO")
        self._save_managed_ids()
        # 立即刷新分区显示
        if self._pos_rows:
            self._on_pos_data(self._pos_rows)

    def _move_to_manual(self, inst_id: str):
        """将持仓移至手动操作区（撤销 AI 授权）。"""
        self._ai_managed_ids.discard(inst_id)
        self._feed.push(
            f"👤 {inst_id} 已移至手动区（AI 助理禁止自动操作）", "WARNING")
        self._save_managed_ids()
        if self._pos_rows:
            self._on_pos_data(self._pos_rows)

    def _save_managed_ids(self):
        """持久化 AI 管控仓位列表。"""
        from pathlib import Path
        p = Path(__file__).resolve().parent.parent.parent / "data" / "ai_managed_positions.json"
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(list(self._ai_managed_ids), ensure_ascii=False))
        except Exception:
            pass

    def _load_managed_ids(self):
        """加载持久化的 AI 管控仓位列表。"""
        from pathlib import Path
        p = Path(__file__).resolve().parent.parent.parent / "data" / "ai_managed_positions.json"
        try:
            if p.exists():
                data = json.loads(p.read_text())
                if isinstance(data, list):
                    self._ai_managed_ids = set(data)
        except Exception:
            pass

    def _manual_kline_analysis(self):
        """手动区选中行 → 打开 K 线分析弹窗。"""
        row = self._manual_pos_table.currentRow()
        if row < 0:
            self._feed.push("请先在手动区选中一个持仓", "WARNING"); return
        sym, side, pnl, upl = self._pos_row_info(self._manual_pos_table, row)
        if not sym:
            return
        scan_item = {
            "symbol": sym, "category": "手动持仓分析",
            "side": side, "direction": side, "opportunity_score": 0,
            "priority_reason": f"手动仓位 — 盈亏 {pnl}，浮盈 {upl} USDT",
        }
        self._open_symbol_dialog(sym, scan_item)

    # 保留旧名兼容 _on_pos_double_click（外部如有引用）
    def _on_pos_double_click(self, index):
        self._on_pos_double_click_table(index, self._ai_pos_table)

    def _open_symbol_dialog(self, sym: str, scan_item: dict):
        from src.ui.symbol_analysis_dialog import SymbolAnalysisDialog
        dlg = SymbolAnalysisDialog(
            symbol=sym, scan_item=scan_item,
            okx_client=self.okx_client,
            llm_client=self._build_llm_client(),
            parent=self,
        )
        try:
            dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass
        self._open_symbol_dialogs.append(dlg)
        dlg.destroyed.connect(lambda *_: self._discard_symbol_dialog(dlg))
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _discard_symbol_dialog(self, dlg):
        try:
            self._open_symbol_dialogs = [item for item in self._open_symbol_dialogs if item is not dlg]
        except Exception:
            self._open_symbol_dialogs = []

    def _menu_style(self) -> str:
        return (f"QMenu {{background:{C['panel']}; color:{C['text']}; "
                f"border:1px solid {C['border']}; font-size:12px;}}"
                f"QMenu::item:selected {{background:#1f6feb;}}"
                f"QMenu::separator {{background:{C['border']}; height:1px; margin:3px 0;}}")

    # ════════════════════════════════════════════════════════════════
    # 信号追踪 & 策略绩效
    # ════════════════════════════════════════════════════════════════

    def _record_signals_for_tracking(self, results: list):
        recorded = 0
        for item in results:
            try:
                klines_map = item.get("klines_map", {})
                if klines_map:
                    self._tf_tracker.record_signal_with_trends(item, klines_map)
                self._signal_tracker.record_signal(item)
                recorded += 1
            except Exception:
                pass
        if recorded:
            self._feed.push(f"📍 已记录 {recorded} 条信号用于追踪验证", "INFO")
            QTimer.singleShot(500, self._refresh_signal_tracking_tab)
            QTimer.singleShot(500, self._refresh_strategy_perf_tab)

    def _validate_signals_async(self):
        """后台线程：验证已记录信号的实际走势并刷新 UI"""
        if not self.okx_client:
            return
        def _do():
            try:
                n1 = self._signal_tracker.validate_outstanding(okx_client=self.okx_client)
                n2 = self._tf_tracker.validate_predictions(okx_client=self.okx_client)
                if n1 + n2 > 0:
                    QTimer.singleShot(0, self._refresh_signal_tracking_tab)
                    QTimer.singleShot(0, self._refresh_strategy_perf_tab)
                    QTimer.singleShot(0, lambda: self._feed.push(
                        f"🔄 验证完成：{n1} 条信号价格验证，{n2} 条时间框架验证", "INFO"))
            except Exception as e:
                print(f"[信号验证] 异常: {e}")
        threading.Thread(target=_do, daemon=True).start()

    def _refresh_signal_tracking_tab(self):
        """刷新信号追踪表格"""
        try:
            signals = list(reversed(self._signal_tracker._signals))

            # 更新策略下拉框
            all_strats = sorted(set(s.get("strategy", "") for s in signals if s.get("strategy")))
            current = self._track_strategy_combo.currentText()
            self._track_strategy_combo.blockSignals(True)
            self._track_strategy_combo.clear()
            self._track_strategy_combo.addItem("全部策略")
            for s in all_strats:
                self._track_strategy_combo.addItem(s)
            idx = self._track_strategy_combo.findText(current)
            self._track_strategy_combo.setCurrentIndex(max(idx, 0))
            self._track_strategy_combo.blockSignals(False)

            # 筛选
            filter_strat  = self._track_strategy_combo.currentText()
            filter_status = self._track_filter_combo.currentText()
            if filter_strat != "全部策略":
                signals = [s for s in signals if s.get("strategy") == filter_strat]

            def _overall(sig):
                vals = sig.get("validations", {})
                outcomes = [v.get("outcome") for v in vals.values()
                            if isinstance(v, dict) and "outcome" in v]
                if not outcomes:
                    return "待验证"
                wins   = outcomes.count("win")
                losses = outcomes.count("loss")
                if wins > losses:
                    return "盈利"
                if losses > wins:
                    return "亏损"
                return "中性"

            if filter_status != "全部":
                signals = [s for s in signals if _overall(s) == filter_status]

            # 统计
            total = len(signals)
            validated = [s for s in signals if s.get("validations") and
                         any(isinstance(v, dict) and "outcome" in v
                             for v in s["validations"].values())]
            wins_cnt   = sum(1 for s in validated if _overall(s) == "盈利")
            losses_cnt = sum(1 for s in validated if _overall(s) == "亏损")
            wr = wins_cnt / max(len(validated), 1) * 100
            self._track_total_lbl.setText(f"已记录: {total} 条")
            self._track_stats_lbl.setText(
                f"已验证 {len(validated)} 条 | 胜率 {wr:.0f}% ({wins_cnt}胜/{losses_cnt}负)")

            # 填表
            self._track_table.setRowCount(0)
            OUTCOME_COLORS = {"win": C["green"], "loss": C["red"], "neutral": C["dim"]}
            WIN_MAP = {"win": "盈利", "loss": "亏损", "neutral": "中性", "invalid": "--"}
            OVERALL_COLORS = {"盈利": C["green"], "亏损": C["red"], "中性": C["dim"], "待验证": C["yellow"]}

            def _it(text, fg=None, align=Qt.AlignCenter, bold=False):
                it = QTableWidgetItem(str(text))
                it.setTextAlignment(align)
                if fg:
                    it.setForeground(QColor(fg))
                if bold:
                    f = QFont(); f.setBold(True); it.setFont(f)
                return it

            for sig in signals[:200]:
                row = self._track_table.rowCount()
                self._track_table.insertRow(row)

                dt_str      = sig.get("datetime", "")[:16]
                direction   = sig.get("direction", "?")
                dir_color   = (C["green"] if direction in ("BUY", "LONG")
                               else C["red"] if direction in ("SELL", "SHORT") else C["dim"])
                entry_price = float(sig.get("entry_price", 0) or 0)
                score       = float(sig.get("score", 0) or 0)
                overall     = _overall(sig)
                vals        = sig.get("validations", {})

                def _val_cell(h_key, v=vals):
                    vv = v.get(str(h_key))
                    if not isinstance(vv, dict):
                        return "--", C["dim"]
                    outcome = vv.get("outcome", "")
                    pnl     = vv.get("pnl", 0)
                    text    = f"{WIN_MAP.get(outcome,'?')} {pnl:+.1f}%" if isinstance(pnl, (int, float)) else WIN_MAP.get(outcome, "--")
                    color   = OUTCOME_COLORS.get(outcome, C["dim"])
                    return text, color

                self._track_table.setItem(row, 0, _it(dt_str, C["dim"], Qt.AlignLeft | Qt.AlignVCenter))
                self._track_table.setItem(row, 1, _it(sig.get("symbol", ""), None, Qt.AlignLeft | Qt.AlignVCenter))
                self._track_table.setItem(row, 2, _it(sig.get("strategy", ""), C["dim"], Qt.AlignLeft | Qt.AlignVCenter))
                self._track_table.setItem(row, 3, _it(direction, dir_color, bold=True))
                self._track_table.setItem(row, 4, _it(f"{score:.0f}"))
                self._track_table.setItem(row, 5, _it(f"{entry_price:.4f}" if entry_price else "--", C["dim"]))
                self._track_table.setItem(row, 6, _it(overall, OVERALL_COLORS.get(overall, C["dim"]), bold=True))
                for col, h_key in zip(range(7, 11), (2, 6, 24, 72)):
                    text, color = _val_cell(h_key)
                    self._track_table.setItem(row, col, _it(text, color))
        except Exception as e:
            print(f"[信号追踪] 刷新失败: {e}")

    def _refresh_strategy_perf_tab(self):
        """刷新策略绩效表格"""
        try:
            days_map = {"近7天": 7, "近14天": 14, "近30天": 30}
            days = days_map.get(self._perf_days_combo.currentText(), 30)
            strategies = self._signal_tracker.all_strategies()
            self._perf_table.setRowCount(0)

            def _it(text, fg=None, align=Qt.AlignCenter, bold=False):
                it = QTableWidgetItem(str(text))
                it.setTextAlignment(align)
                if fg:
                    it.setForeground(QColor(fg))
                if bold:
                    f = QFont(); f.setBold(True); it.setFont(f)
                return it

            for strat in sorted(strategies):
                stats = self._signal_tracker.strategy_stats(strat, days=days, min_signals=1)
                total = stats.get("total", 0)
                if total == 0:
                    continue
                validated = stats.get("validated_count", 0)
                wr        = stats.get("win_rate", 0.0)
                avg_win   = stats.get("avg_win_pct", 0.0)
                avg_loss  = stats.get("avg_loss_pct", 0.0)
                pf        = stats.get("profit_factor", 0.0)
                sharpe    = stats.get("sharpe", 0.0)

                wr_color = C["green"] if wr >= 55 else C["yellow"] if wr >= 45 else C["red"]
                pf_color = C["green"] if pf >= 1.5 else C["yellow"] if pf >= 1.0 else C["red"]

                row = self._perf_table.rowCount()
                self._perf_table.insertRow(row)
                self._perf_table.setItem(row, 0, _it(strat, C["text"], Qt.AlignLeft | Qt.AlignVCenter))
                self._perf_table.setItem(row, 1, _it(str(total)))
                self._perf_table.setItem(row, 2, _it(str(validated)))
                self._perf_table.setItem(row, 3, _it(f"{wr:.1f}%", wr_color, bold=True))
                self._perf_table.setItem(row, 4, _it(f"+{avg_win:.2f}%", C["green"]))
                self._perf_table.setItem(row, 5, _it(f"-{avg_loss:.2f}%", C["red"]))
                self._perf_table.setItem(row, 6, _it(f"{pf:.2f}", pf_color, bold=True))
                self._perf_table.setItem(row, 7, _it(f"{sharpe:.2f}",
                                                      C["green"] if sharpe > 0.5 else C["dim"]))
        except Exception as e:
            print(f"[策略绩效] 刷新失败: {e}")

    def _on_perf_strategy_selected(self):
        """点击绩效表某行 → 显示规则建议"""
        try:
            row = self._perf_table.currentRow()
            if row < 0:
                return
            strat_item = self._perf_table.item(row, 0)
            if not strat_item:
                return
            strat = strat_item.text()
            recs  = self._tf_tracker.adjustment_recommendations(strat, days=30)
            if recs:
                self._rec_text.setHtml(
                    "<br>".join(f'<span style="color:{C["text"]}">{r}</span>' for r in recs))
            else:
                stats = self._signal_tracker.strategy_stats(strat)
                total = stats.get("total", 0)
                wr    = stats.get("win_rate", 0.0)
                if total < 5:
                    self._rec_text.setPlainText(
                        f"策略「{strat}」信号量不足（仅 {total} 条），积累更多信号后自动生成建议。")
                elif wr >= 55:
                    self._rec_text.setHtml(
                        f'<span style="color:{C["green"]}">✅ 策略「{strat}」胜率 {wr:.1f}% 表现良好，当前参数无需调整。</span>')
                else:
                    self._rec_text.setHtml(
                        f'<span style="color:{C["yellow"]}">📊 策略「{strat}」胜率 {wr:.1f}%，点击「AI 分析参数」获取智能建议。</span>')
        except Exception as e:
            print(f"[绩效建议] 异常: {e}")

    def _request_ai_param_advice(self):
        """调用 LLM 分析策略绩效并给出参数修改建议"""
        client = self._build_llm_client()
        if not client:
            self._feed.push("请先填写 LLM API 地址和 Key", "ERROR"); return

        row = self._perf_table.currentRow()
        strat_name = ""
        if row >= 0:
            it = self._perf_table.item(row, 0)
            if it:
                strat_name = it.text()

        days_map = {"近7天": 7, "近14天": 14, "近30天": 30}
        days = days_map.get(self._perf_days_combo.currentText(), 30)
        strategies = self._signal_tracker.all_strategies()

        perf_lines = []
        for s in sorted(strategies):
            stats = self._signal_tracker.strategy_stats(s, days=days, min_signals=1)
            if stats.get("total", 0) == 0:
                continue
            tf_stats = self._tf_tracker.accuracy_by_timeframe(s, days=days)
            tf_acc = " / ".join(
                f"{tf}:{tf_stats.get(tf, {}).get('accuracy', 0):.0f}%"
                for tf in ("1D", "4H", "1H")
            )
            perf_lines.append(
                f"• {s}：信号 {stats['total']} 条，胜率 {stats.get('win_rate', 0):.1f}%，"
                f"均盈 +{stats.get('avg_win_pct', 0):.2f}% / 均亏 -{stats.get('avg_loss_pct', 0):.2f}%，"
                f"盈亏比 {stats.get('profit_factor', 0):.2f}，时间框架准确率({tf_acc})"
            )

        score_lines = []
        if strat_name:
            for item in self._signal_tracker.score_vs_winrate(strat_name, days=days):
                score_lines.append(
                    f"  评分 {item['score_range']}：{item['signals']} 信号，胜率 {item['win_rate']}%")

        focus_hint = f"重点分析策略：{strat_name}" if strat_name else "分析所有策略"
        perf_text  = "\n".join(perf_lines) if perf_lines else "暂无足够数据"
        score_text = "\n".join(score_lines) if score_lines else "  暂无评分分布数据"

        prompt = (
            f"你是量化交易策略优化专家。以下是扫描策略近{days}天的验证绩效数据：\n\n"
            f"{perf_text}\n\n"
            + (f"策略「{strat_name}」的评分区间胜率分布：\n{score_text}\n\n" if strat_name else "")
            + f"任务（{focus_hint}）：\n"
            "1. 判断哪些策略/时间框架表现不佳，分析可能原因\n"
            "2. 针对日线趋势、小时线企稳、BB压缩等核心条件，给出具体参数调整建议\n"
            "3. 如果高分信号胜率不高，给出评分权重的优化方向\n"
            "4. 按优先级列出 3-5 条可操作的修改建议，每条注明预期效果\n"
            "请用简洁中文直接给结论和建议，不需要重述数据。"
        )

        self._ai_advice_btn.setEnabled(False)
        self._rec_text.setHtml(f'<span style="color:{C["blue"]}">🧠 AI 正在分析策略绩效数据...</span>')

        def _do():
            try:
                result = client.chat_completion([
                    {"role": "system", "content": "你是量化交易策略优化专家，擅长从验证数据中发现策略弱点并给出参数修改建议。"},
                    {"role": "user",   "content": prompt},
                ], max_tokens=1000, temperature=0.3)
                advice_text = result.get("content", "") if isinstance(result, dict) else str(result)

                pending: dict = {}
                import re
                for line in advice_text.split("\n"):
                    for kw in ("h1_stab_bars", "min_score", "h1_min_rebound", "h1_max_dryup",
                               "trend_breakout_pct", "deep_pullback_pct", "h1_per_bar_atr_ratio"):
                        if kw in line:
                            m = re.search(r'[\d.]+', line)
                            if m:
                                try:
                                    pending[kw] = float(m.group())
                                except ValueError:
                                    pass

                QTimer.singleShot(0, lambda t=advice_text, p=pending: self._on_ai_advice_done(t, p))
            except Exception as e:
                QTimer.singleShot(0, lambda: self._rec_text.setHtml(
                    f'<span style="color:{C["red"]}">❌ AI 分析失败: {e}</span>'))
                QTimer.singleShot(0, lambda: self._ai_advice_btn.setEnabled(True))

        threading.Thread(target=_do, daemon=True).start()

    def _on_ai_advice_done(self, advice_text: str, pending_params: dict):
        self._ai_advice_btn.setEnabled(True)
        self._pending_param_advice = pending_params
        lines = advice_text.strip().split("\n")
        html_parts = []
        for line in lines:
            stripped = line.lstrip()
            if stripped[:2] in ("1.", "2.", "3.", "4.", "5.") or stripped.startswith("•"):
                html_parts.append(f'<span style="color:{C["yellow"]}">{line}</span>')
            elif any(kw in line for kw in ("⚠️", "降低", "减小", "过松", "弱")):
                html_parts.append(f'<span style="color:{C["orange"]}">{line}</span>')
            elif any(kw in line for kw in ("✅", "提高", "增大", "优秀", "良好")):
                html_parts.append(f'<span style="color:{C["green"]}">{line}</span>')
            elif line.strip():
                html_parts.append(f'<span style="color:{C["text"]}">{line}</span>')
        self._rec_text.setHtml("<br>".join(html_parts))
        self._feed.push("🧠 AI 参数建议已生成，请在策略绩效页查看", "ANALYSIS")
        if pending_params and self._chk_mod.isChecked():
            self._apply_advice_btn.setEnabled(True)

    def _apply_param_advice(self):
        """将 AI 建议参数应用到当前扫描器运行配置"""
        if not self._chk_mod.isChecked():
            QMessageBox.warning(self, "权限不足", "请先勾选「允许修改策略参数」权限")
            return
        if not self._pending_param_advice:
            return
        if not self.scanner_page:
            self._feed.push("未连接扫描器页面，无法应用参数", "WARNING")
            return
        applied = []
        try:
            engine = getattr(self.scanner_page, "engine", None)
            if engine and hasattr(engine, "config"):
                for key, val in self._pending_param_advice.items():
                    engine.config[key] = val
                    applied.append(f"{key}={val}")
        except Exception as e:
            self._feed.push(f"应用参数失败: {e}", "ERROR")
            return
        if applied:
            self._feed.push(f"✏️ 已应用 AI 建议参数: {', '.join(applied)}", "SUCCESS")
            QMessageBox.information(self, "参数已更新",
                "以下参数已更新（将在下次扫描时生效）：\n\n" + "\n".join(applied))
            self._apply_advice_btn.setEnabled(False)
            self._pending_param_advice = {}

    # ════════════════════════════════════════════════════════════
    # 外部注入
    # ════════════════════════════════════════════════════════════

    def inject_dependencies(self, okx_client=None, trade_executor=None, scanner_page=None):
        self.okx_client     = okx_client
        self.trade_executor = trade_executor
        self.scanner_page   = scanner_page
        if self._agent:
            self._agent.okx_client     = okx_client
            self._agent.trade_executor = trade_executor
            self._agent.scanner_page   = scanner_page


# ════════════════════════════════════════════════════════════════
# 持仓风险预警弹窗
# ════════════════════════════════════════════════════════════════

class PositionAlertDialog(QDialog):

    _REC_COLOR = {
        "减仓":     C["orange"], "部分减仓": C["orange"],
        "加仓":     C["green"],  "保持持仓": C["blue"],
        "立即止损": C["red"],    "止盈出场": C["yellow"],
        "关注风险": C["yellow"],
    }
    _RISK_COLOR = {"low": C["green"], "medium": C["yellow"], "high": C["red"]}

    def __init__(self, alerts: list, okx_client=None, llm_client=None, parent=None):
        super().__init__(parent)
        self._alerts     = alerts
        self._okx_client = okx_client
        self._llm_client = llm_client
        self._open_symbol_dialogs: List[QDialog] = []
        high = sum(1 for a in alerts if a.get("risk") in ("medium", "high"))
        icon = "🚨" if high else "📡"
        self.setWindowTitle(f"{icon} 持仓监控预警 — {len(alerts)} 个仓位")
        self.setMinimumSize(920, 480)
        self.resize(1020, 540)
        self.setStyleSheet(
            f"QDialog{{background:{C['bg']};color:{C['text']};}}"
            f"QLabel{{color:{C['text']};}}")
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # 标题栏
        hdr = QFrame()
        hdr.setStyleSheet(
            f"QFrame{{background:{C['panel']};border:1px solid {C['border']};"
            f"border-radius:8px;}}")
        hlay = QHBoxLayout(hdr)
        hlay.setContentsMargins(14, 8, 14, 8)
        high_cnt = sum(1 for a in self._alerts if a.get("risk") in ("medium", "high"))
        icon = "🚨" if high_cnt else "📡"
        title = _label(
            f"{icon}  持仓监控预警 · {datetime.now().strftime('%H:%M:%S')}",
            C["text"], 14, bold=True)
        hlay.addWidget(title)
        sub_color = C["red"] if high_cnt else C["dim"]
        sub = _label(f"  共 {len(self._alerts)} 个仓位 · {high_cnt} 个需关注",
                     sub_color, 11)
        hlay.addWidget(sub)
        hlay.addStretch()
        note = _label("双击行 → K线深度分析", C["dim"], 10)
        hlay.addWidget(note)
        root.addWidget(hdr)

        # 表格
        self._table = QTableWidget(0, 8)
        self._table.setHorizontalHeaderLabels([
            "交易对", "方向", "均价", "浮盈亏%", "24h变动", "风险", "AI建议", "理由"
        ])
        hh = self._table.horizontalHeader()
        for i in range(7):
            hh.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(7, QHeaderView.Stretch)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.verticalHeader().setVisible(False)
        self._table.setStyleSheet(
            f"QTableWidget{{background:{C['bg']};color:{C['text']};"
            f"gridline-color:#21262d;font-size:12px;}}"
            f"QHeaderView::section{{background:{C['panel']};color:{C['dim']};"
            f"border:none;padding:5px;font-weight:bold;}}")
        self._table.doubleClicked.connect(self._on_row_dbl)
        root.addWidget(self._table, 1)
        self._fill_table()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = _btn("  关闭  ", "#21262d", C["border"], h=34)
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

    def _fill_table(self):
        self._table.setRowCount(0)
        for a in self._alerts:
            row = self._table.rowCount()
            self._table.insertRow(row)
            upl_ratio = a.get("upl_ratio", 0)
            rec_chg   = a.get("recent_chg", 0)
            risk      = a.get("risk", "low")
            rec       = a.get("recommendation", "保持持仓")
            reason    = a.get("rec_reason", "")
            pnl_c  = C["green"] if upl_ratio >= 0 else C["red"]
            chg_c  = C["green"] if rec_chg >= 0 else C["red"]
            side   = a.get("side", "?")
            side_c = (C["green"] if "long" in side.lower() or "buy" in side.lower()
                      else C["red"] if "short" in side.lower() or "sell" in side.lower()
                      else C["dim"])

            def _it(text, fg=None, align=Qt.AlignCenter, bold=False):
                it = QTableWidgetItem(str(text))
                it.setTextAlignment(align)
                if fg: it.setForeground(QColor(fg))
                if bold:
                    f = QFont(); f.setBold(True); it.setFont(f)
                return it

            self._table.setItem(row, 0, _it(a["inst_id"],
                align=Qt.AlignLeft | Qt.AlignVCenter))
            self._table.setItem(row, 1, _it(side, fg=side_c))
            avg = a.get("avg_px", 0)
            self._table.setItem(row, 2, _it(f"{avg:.4f}" if avg else "--", fg=C["dim"]))
            self._table.setItem(row, 3, _it(f"{upl_ratio:+.2f}%", fg=pnl_c, bold=True))
            self._table.setItem(row, 4, _it(f"{rec_chg:+.2f}%", fg=chg_c))
            self._table.setItem(row, 5, _it(
                risk.upper(), fg=self._RISK_COLOR.get(risk, C["dim"]),
                bold=(risk != "low")))
            self._table.setItem(row, 6, _it(
                rec, fg=self._REC_COLOR.get(rec, C["dim"]), bold=True))
            self._table.setItem(row, 7, _it(
                reason, align=Qt.AlignLeft | Qt.AlignVCenter))

            bg = "#2d1a1a" if risk == "high" else "#1e1e12" if risk == "medium" else None
            if bg:
                for c in range(8):
                    it = self._table.item(row, c)
                    if it:
                        it.setBackground(QColor(bg))

    def _on_row_dbl(self, index):
        row = index.row()
        if row < 0 or row >= len(self._alerts):
            return
        a   = self._alerts[row]
        sym = a["inst_id"]
        scan_item = {
            "symbol":    sym, "category": "持仓分析",
            "side":      a.get("side", "?"), "direction": a.get("side", "?"),
            "opportunity_score": 0,
            "priority_reason": (
                f"持仓监控 — 浮盈亏{a['upl_ratio']:+.1f}% | "
                f"24h{a['recent_chg']:+.1f}% | AI:{a.get('recommendation','')} | "
                f"{a.get('rec_reason','')}"),
        }
        from src.ui.symbol_analysis_dialog import SymbolAnalysisDialog
        dlg = SymbolAnalysisDialog(
            symbol=sym, scan_item=scan_item,
            okx_client=self._okx_client,
            llm_client=self._llm_client,
            parent=self,
        )
        try:
            dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass
        self._open_symbol_dialogs.append(dlg)
        dlg.destroyed.connect(lambda *_: self._discard_symbol_dialog(dlg))
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _discard_symbol_dialog(self, dlg):
        try:
            self._open_symbol_dialogs = [item for item in self._open_symbol_dialogs if item is not dlg]
        except Exception:
            self._open_symbol_dialogs = []
