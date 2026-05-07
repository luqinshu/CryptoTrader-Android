"""
交易对深度分析弹窗
右键点击扫描结果/持仓 → 单图多周期切换 K 线 + 技术指标 + AI 综合分析
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except Exception:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm

from src.qt_compat import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QTextEdit, QFrame, QSplitter, QCheckBox,
    QProgressBar,
    Qt, QTimer, Signal,
    QFont, QColor,
)
from src.ai_agent.llm_client import LLMClient

# ── 颜色常量 ──────────────────────────────────────────────────
DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
BORDER    = "#30363d"
TEXT_MAIN = "#c9d1d9"
TEXT_DIM  = "#8b949e"
GREEN     = "#3fb950"
RED       = "#f85149"
BLUE      = "#58a6ff"
YELLOW    = "#e3b341"
ORANGE    = "#f0883e"
PURPLE    = "#bc8cff"

# ── 周期配置（label 全用 ASCII，避免字体渲染问题） ─────────────
TIMEFRAMES = [
    ("1m",  "1m",   60),
    ("3m",  "3m",   60),
    ("5m",  "5m",   60),
    ("15m", "15m",  80),
    ("1H",  "1H",   80),
    ("4H",  "4H",   80),
    ("1D",  "1D",  100),
]
TF_DISPLAY = {   # 按钮上的显示文字
    "1m":  "1分钟",
    "3m":  "3分钟",
    "5m":  "5分钟",
    "15m": "15分钟",
    "1H":  "1小时",
    "4H":  "4小时",
    "1D":  "日线",
}


# ── 技术指标计算 ──────────────────────────────────────────────

def _ema(prices: List[float], period: int) -> List[float]:
    if not prices:
        return []
    k   = 2.0 / (period + 1)
    out = [prices[0]]
    for p in prices[1:]:
        out.append(p * k + out[-1] * (1 - k))
    return out


def _sma(prices: List[float], period: int) -> List[Optional[float]]:
    out = []
    for i in range(len(prices)):
        if i + 1 < period:
            out.append(None)
        else:
            out.append(sum(prices[i - period + 1:i + 1]) / period)
    return out


def _bollinger(prices: List[float], period: int = 20, mult: float = 2.0):
    upper, mid, lower = [], [], []
    for i in range(len(prices)):
        window = prices[max(0, i - period + 1):i + 1]
        m = sum(window) / len(window)
        std = (sum((p - m) ** 2 for p in window) / len(window)) ** 0.5
        mid.append(m)
        upper.append(m + mult * std)
        lower.append(m - mult * std)
    return upper, mid, lower


def _macd(prices: List[float], fast=12, slow=26, signal=9):
    if len(prices) < slow:
        n = len(prices)
        return [0.0] * n, [0.0] * n, [0.0] * n
    e_fast   = _ema(prices, fast)
    e_slow   = _ema(prices, slow)
    macd_l   = [f - s for f, s in zip(e_fast, e_slow)]
    signal_l = _ema(macd_l, signal)
    hist     = [m - s for m, s in zip(macd_l, signal_l)]
    return macd_l, signal_l, hist


def _parse_klines(raw_data: list) -> List[dict]:
    candles = []
    for row in raw_data:
        try:
            candles.append({
                "ts":    int(row[0]) / 1000,
                "open":  float(row[1]),
                "high":  float(row[2]),
                "low":   float(row[3]),
                "close": float(row[4]),
                "vol":   float(row[5]),
            })
        except (IndexError, ValueError, TypeError):
            continue
    return sorted(candles, key=lambda x: x["ts"])


# ── K 线画布（单图，支持切换） ────────────────────────────────

class KLineCanvas(FigureCanvas):

    def __init__(self, parent=None):
        self.fig = Figure(facecolor=DARK_BG)
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.patch.set_facecolor(DARK_BG)
        # 指标开关状态（由外部设置）
        self.show_ema  = True
        self.show_bb   = True
        self.show_macd = True
        self.show_vol  = True
        self._current_tf = ""
        # 滚轮缩放状态
        self._drawn_candles: List[dict] = []
        self._ax_price = None
        self._ax_macd  = None
        self._ax_vol   = None
        self._cid_scroll = None

    def _ax_style(self, ax):
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=TEXT_DIM, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.grid(axis="y", color=BORDER, linewidth=0.4, linestyle="--", alpha=0.5)

    def plot(self, candles: List[dict], tf: str = ""):
        if not candles:
            return
        # 断开旧的滚轮事件
        if self._cid_scroll is not None:
            try:
                self.fig.canvas.mpl_disconnect(self._cid_scroll)
            except Exception:
                pass
            self._cid_scroll = None

        self._current_tf = tf
        self._drawn_candles = candles
        self.fig.clear()
        self._ax_price = None
        self._ax_macd  = None
        self._ax_vol   = None

        # 布局：price(5) / macd(2) / vol(1.5)  ——  根据开关动态计算
        n_rows   = 1 + (1 if self.show_macd else 0) + (1 if self.show_vol else 0)
        ratios   = [5]
        if self.show_macd:
            ratios.append(2)
        if self.show_vol:
            ratios.append(1.5)

        gs = self.fig.add_gridspec(
            n_rows, 1,
            height_ratios=ratios,
            hspace=0.04,
            left=0.02, right=0.88,
            top=0.94,  bottom=0.06,
        )

        row_idx = 0
        ax_price = self.fig.add_subplot(gs[row_idx])
        self._ax_price = ax_price
        row_idx += 1

        ax_macd = None
        ax_vol  = None
        if self.show_macd:
            ax_macd = self.fig.add_subplot(gs[row_idx], sharex=ax_price)
            self._ax_macd = ax_macd
            row_idx += 1
        if self.show_vol:
            ax_vol = self.fig.add_subplot(gs[row_idx], sharex=ax_price)
            self._ax_vol = ax_vol

        closes = [c["close"] for c in candles]
        opens  = [c["open"]  for c in candles]
        highs  = [c["high"]  for c in candles]
        lows   = [c["low"]   for c in candles]
        vols   = [c["vol"]   for c in candles]
        xs     = list(range(len(candles)))

        # ── 蜡烛图 ──────────────────────────────────────────
        for ax in [ax_price, ax_macd, ax_vol]:
            if ax:
                self._ax_style(ax)

        width = 0.6
        for i, c in enumerate(candles):
            up    = c["close"] >= c["open"]
            color = GREEN if up else RED
            ax_price.plot([i, i], [c["low"], c["high"]], color=color, linewidth=0.8, zorder=2)
            body_h = max(abs(c["close"] - c["open"]),
                         (c["high"] - c["low"]) * 0.002)
            body_y = min(c["close"], c["open"])
            rect = Rectangle((i - width / 2, body_y), width, body_h,
                              facecolor=color, edgecolor=color, linewidth=0, zorder=3)
            ax_price.add_patch(rect)

        # ── EMA ─────────────────────────────────────────────
        if self.show_ema and len(closes) >= 5:
            for period, color, lw in [(5, "#00ccff", 1.0),
                                       (20, ORANGE,   1.0),
                                       (60, PURPLE,   1.0)]:
                if len(closes) >= period:
                    ema_v = _ema(closes, period)
                    ax_price.plot(xs, ema_v, color=color, linewidth=lw,
                                  label=f"EMA{period}", zorder=4)

        # ── 布林带 ──────────────────────────────────────────
        if self.show_bb and len(closes) >= 20:
            bb_up, bb_mid, bb_lo = _bollinger(closes, 20, 2.0)
            ax_price.plot(xs, bb_up,  color=BLUE,  linewidth=0.8, linestyle="--",
                          label="BB Up", alpha=0.8, zorder=4)
            ax_price.plot(xs, bb_mid, color=YELLOW, linewidth=0.8, linestyle="--",
                          label="BB Mid", alpha=0.8, zorder=4)
            ax_price.plot(xs, bb_lo,  color=BLUE,  linewidth=0.8, linestyle="--",
                          label="BB Lo", alpha=0.8, zorder=4)
            ax_price.fill_between(xs, bb_up, bb_lo,
                                  color=BLUE, alpha=0.06, zorder=1)

        # 当前价格标注
        last_px = closes[-1]
        color_last = GREEN if closes[-1] >= closes[-2] else RED if len(closes) >= 2 else TEXT_DIM
        ax_price.axhline(last_px, color=color_last, linewidth=0.6,
                         linestyle="--", alpha=0.7, zorder=1)
        ax_price.text(len(candles) + 0.3, last_px,
                      f"{last_px:.4f}", color=color_last, fontsize=8,
                      va="center", ha="left")

        # 图例
        ax_price.legend(loc="upper left", fontsize=7, framealpha=0.1,
                        labelcolor=TEXT_DIM, facecolor=PANEL_BG,
                        edgecolor=BORDER, ncol=3)

        # 周期 + 最新价标题
        pct = ((closes[-1] / closes[0]) - 1) * 100 if closes[0] else 0
        sign = "+" if pct >= 0 else ""
        pct_color = GREEN if pct >= 0 else RED
        ax_price.set_title(
            f"{self._current_tf}   {last_px:.4f}   {sign}{pct:.2f}%",
            color=TEXT_MAIN, fontsize=10, pad=6, loc="left",
        )

        ax_price.set_xlim(-0.5, len(candles) - 0.5)
        price_lo = min(lows) * 0.998
        price_hi = max(highs) * 1.002
        ax_price.set_ylim(price_lo, price_hi)
        plt.setp(ax_price.get_xticklabels(), visible=False)

        # ── MACD ────────────────────────────────────────────
        if ax_macd is not None:
            macd_l, sig_l, hist_l = _macd(closes)
            for i, h in enumerate(hist_l):
                color = GREEN if h >= 0 else RED
                ax_macd.bar(i, h, color=color, alpha=0.7, width=width)
            ax_macd.plot(xs, macd_l, color=BLUE,   linewidth=0.9, label="MACD")
            ax_macd.plot(xs, sig_l,  color=ORANGE,  linewidth=0.9, label="Signal")
            ax_macd.axhline(0, color=BORDER, linewidth=0.5)
            ax_macd.legend(loc="upper left", fontsize=7, framealpha=0.1,
                           labelcolor=TEXT_DIM, facecolor=PANEL_BG, edgecolor=BORDER)
            ax_macd.set_ylabel("MACD", color=TEXT_DIM, fontsize=8,
                               rotation=0, labelpad=28)
            plt.setp(ax_macd.get_xticklabels(), visible=False if ax_vol else True)

        # ── 成交量 ──────────────────────────────────────────
        if ax_vol is not None:
            avg_vol = sum(vols) / len(vols) if vols else 1
            for i, c in enumerate(candles):
                up    = c["close"] >= c["open"]
                color = GREEN if up else RED
                alpha = 0.85 if vols[i] > avg_vol else 0.5
                ax_vol.bar(i, vols[i], color=color, alpha=alpha, width=width)
            ax_vol.set_ylabel("VOL", color=TEXT_DIM, fontsize=8,
                              rotation=0, labelpad=28)
            ax_vol.yaxis.set_major_formatter(
                mticker.FuncFormatter(
                    lambda x, _: (f"{x/1e6:.1f}M" if x >= 1e6
                                  else f"{x/1e3:.0f}K")))

        # X 轴时间标签（显示在最下方的子图）
        bottom_ax = ax_vol or ax_macd or ax_price
        n = len(candles)
        tick_step = max(1, n // 8)
        tick_pos  = list(range(0, n, tick_step))
        tick_lbl  = []
        for p in tick_pos:
            ts = candles[p]["ts"]
            dt = datetime.fromtimestamp(ts)
            if tf in ("1D",):
                tick_lbl.append(dt.strftime("%m/%d"))
            elif tf in ("4H", "1H"):
                tick_lbl.append(dt.strftime("%m/%d\n%H:%M"))
            else:
                tick_lbl.append(dt.strftime("%H:%M"))
        bottom_ax.set_xticks(tick_pos)
        bottom_ax.set_xticklabels(tick_lbl, fontsize=7, color=TEXT_DIM)

        self.fig.canvas.draw_idle()
        # 连接滚轮缩放
        self._cid_scroll = self.fig.canvas.mpl_connect(
            "scroll_event", self._on_scroll)

    def _on_scroll(self, event):
        """滚轮缩放：以鼠标位置为中心缩放 X 轴，Y 轴自动跟随可见蜡烛范围"""
        if self._ax_price is None or not self._drawn_candles:
            return
        valid_axes = [ax for ax in [self._ax_price, self._ax_macd, self._ax_vol]
                      if ax is not None]
        if event.inaxes not in valid_axes:
            return

        n = len(self._drawn_candles)
        x_min, x_max = self._ax_price.get_xlim()
        factor = 0.75 if event.button == "up" else 1.35
        x_cur  = event.xdata if event.xdata is not None else (x_min + x_max) / 2

        new_min = x_cur - (x_cur - x_min) * factor
        new_max = x_cur + (x_max - x_cur) * factor

        # 边界限制
        new_min = max(-0.5, new_min)
        new_max = min(n - 0.5, new_max)
        if new_max - new_min < 4:   # 最少显示 4 根蜡烛
            return

        self._ax_price.set_xlim(new_min, new_max)

        # Y 轴自动缩放到可见蜡烛
        i0 = max(0, int(new_min))
        i1 = min(n - 1, int(new_max) + 1)
        vis = self._drawn_candles[i0 : i1 + 1]
        if vis:
            lo  = min(c["low"]  for c in vis)
            hi  = max(c["high"] for c in vis)
            pad = (hi - lo) * 0.05 or hi * 0.001
            self._ax_price.set_ylim(lo - pad, hi + pad)

        self.fig.canvas.draw_idle()

    def show_placeholder(self, msg: str = "Loading..."):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(DARK_BG)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.tick_params(colors=TEXT_DIM)
        ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                ha="center", va="center", color=TEXT_DIM, fontsize=13)
        self.fig.canvas.draw_idle()


# ── 弹窗主体 ─────────────────────────────────────────────────

class SymbolAnalysisDialog(QDialog):

    _sig_kline_done  = Signal(str, list)
    _sig_ai_done     = Signal(str)
    _sig_kline_error = Signal(str, str)
    _sig_ticker_done = Signal(dict)

    def __init__(self, symbol: str, scan_item: dict,
                 okx_client=None, llm_client: LLMClient = None,
                 parent=None):
        super().__init__(parent)
        self.symbol     = symbol
        self.scan_item  = scan_item
        self.okx_client = okx_client
        self.llm_client = llm_client
        self._candles: Dict[str, List[dict]] = {}
        self._current_tf = "15m"
        self._closing = False

        self.setWindowTitle(f"Deep Analysis  ·  {symbol}")
        self.setMinimumSize(1280, 740)
        self.resize(1380, 820)
        self.setStyleSheet(f"""
            QDialog   {{background:{DARK_BG}; color:{TEXT_MAIN};}}
            QLabel    {{color:{TEXT_MAIN};}}
            QCheckBox {{color:{TEXT_DIM}; font-size:11px;}}
            QCheckBox::indicator {{width:13px; height:13px;}}
        """)

        self._sig_kline_done.connect(self._on_kline_done)
        self._sig_ai_done.connect(self._on_ai_done)
        self._sig_kline_error.connect(self._on_kline_error)
        self._sig_ticker_done.connect(self._on_ticker_done)

        self._build_ui()
        self._start_loading()

    # ── UI ───────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        root.addWidget(self._build_info_bar())

        body = QSplitter(Qt.Horizontal)
        body.addWidget(self._build_chart_panel())
        body.addWidget(self._build_analysis_panel())
        body.setStretchFactor(0, 3)
        body.setStretchFactor(1, 2)
        root.addWidget(body, 1)

        root.addWidget(self._build_footer())

    def _build_info_bar(self) -> QFrame:
        f = QFrame()
        f.setFixedHeight(52)
        f.setStyleSheet(f"QFrame{{background:{PANEL_BG};border:1px solid {BORDER};"
                        f"border-radius:8px;}}")
        lay = QHBoxLayout(f)
        lay.setContentsMargins(14, 6, 14, 6)

        sym_lbl = QLabel(self.symbol)
        sym_lbl.setStyleSheet(f"color:{TEXT_MAIN};font-size:17px;font-weight:bold;")
        lay.addWidget(sym_lbl)

        category  = self.scan_item.get("category", "--")
        direction = str(self.scan_item.get("side",
                        self.scan_item.get("direction", "--"))).upper()
        score     = float(self.scan_item.get("opportunity_score",
                          self.scan_item.get("score", 0)) or 0)
        meta_s    = f"  {category}  |  {direction}"
        if score:
            meta_s += f"  |  score {score:.1f}"
        meta_lbl = QLabel(meta_s)
        meta_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;")
        lay.addWidget(meta_lbl)

        lay.addStretch()

        self._price_lbl  = QLabel("--")
        self._price_lbl.setStyleSheet(
            f"color:{BLUE};font-size:15px;font-weight:bold;margin-right:6px;")
        self._change_lbl = QLabel("")
        self._change_lbl.setStyleSheet("font-size:13px;font-weight:bold;")
        lay.addWidget(self._price_lbl)
        lay.addWidget(self._change_lbl)
        return f

    # ── 图表面板 ─────────────────────────────────────────────

    def _build_chart_panel(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 4, 0)
        lay.setSpacing(6)

        # 第一行：周期选择按钮
        tf_row = QHBoxLayout()
        tf_row.setSpacing(4)
        tf_lbl = QLabel("周期:")
        tf_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;")
        tf_row.addWidget(tf_lbl)

        self._tf_btns: Dict[str, QPushButton] = {}
        for tf, _, _ in TIMEFRAMES:
            btn = QPushButton(TF_DISPLAY[tf])
            btn.setFixedSize(58, 28)
            btn.setCheckable(True)
            btn.setChecked(tf == self._current_tf)
            btn.setStyleSheet(self._tf_btn_style(tf == self._current_tf))
            btn.clicked.connect(lambda checked, t=tf: self._switch_tf(t))
            self._tf_btns[tf] = btn
            tf_row.addWidget(btn)

        tf_row.addSpacing(16)

        # 第二行右侧：指标开关
        ind_lbl = QLabel("指标:")
        ind_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;")
        tf_row.addWidget(ind_lbl)

        self._chk_ema  = self._ind_chk("EMA", True)
        self._chk_bb   = self._ind_chk("BB",  True)
        self._chk_macd = self._ind_chk("MACD",True)
        self._chk_vol  = self._ind_chk("VOL", True)
        for chk in [self._chk_ema, self._chk_bb, self._chk_macd, self._chk_vol]:
            tf_row.addWidget(chk)

        tf_row.addStretch()
        lay.addLayout(tf_row)

        # 画布
        self._canvas = KLineCanvas()
        self._canvas.show_placeholder("Loading K-line data...")
        lay.addWidget(self._canvas, 1)
        return w

    def _tf_btn_style(self, active: bool) -> str:
        bg = BLUE if active else "#21262d"
        return (f"QPushButton{{background:{bg};color:{'white' if active else TEXT_DIM};"
                f"border:1px solid {BORDER};border-radius:5px;"
                f"font-size:11px;font-weight:{'bold' if active else 'normal'};}}"
                f"QPushButton:hover{{background:{'#388bfd' if active else '#30363d'};}}")

    def _ind_chk(self, label: str, checked: bool) -> QCheckBox:
        chk = QCheckBox(label)
        chk.setChecked(checked)
        chk.setStyleSheet(f"QCheckBox{{color:{TEXT_DIM};font-size:11px;margin-right:4px;}}"
                          f"QCheckBox::indicator{{width:13px;height:13px;}}")
        chk.toggled.connect(self._refresh_chart)
        return chk

    # ── 分析面板 ─────────────────────────────────────────────

    def _build_analysis_panel(self) -> QWidget:
        w   = QWidget()
        w.setMinimumWidth(360)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 0, 0, 0)
        lay.setSpacing(6)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setFixedHeight(3)
        self._progress.setStyleSheet(
            f"QProgressBar{{background:{PANEL_BG};border:none;border-radius:2px;}}"
            f"QProgressBar::chunk{{background:{BLUE};border-radius:2px;}}")
        lay.addWidget(self._progress)

        # 信号摘要
        sig = QFrame()
        sig.setStyleSheet(f"QFrame{{background:{PANEL_BG};border:1px solid {BORDER};"
                          f"border-radius:6px;padding:4px;}}")
        sg  = QGridLayout(sig)
        sg.setSpacing(4)
        reason = self.scan_item.get("priority_reason",
                 self.scan_item.get("reason", "--"))[:120]
        fields = [
            ("机会类型", self.scan_item.get("category", "--")),
            ("建议方向", str(self.scan_item.get("side",
                            self.scan_item.get("direction", "--"))).upper()),
            ("评分",    f"{float(self.scan_item.get('opportunity_score',
                         self.scan_item.get('score', 0)) or 0):.1f}"),
            ("信号理由", reason),
        ]
        ls = f"color:{TEXT_DIM};font-size:11px;"
        vs = f"color:{TEXT_MAIN};font-size:11px;font-weight:bold;"
        for r, (k, v) in enumerate(fields):
            kl = QLabel(k + ":")
            kl.setStyleSheet(ls)
            vl = QLabel(str(v))
            vl.setStyleSheet(vs)
            vl.setWordWrap(True)
            sg.addWidget(kl, r, 0, Qt.AlignTop)
            sg.addWidget(vl, r, 1, Qt.AlignTop)
        lay.addWidget(sig)

        ai_title = QLabel("🤖  AI 综合分析")
        ai_title.setStyleSheet(
            f"color:{TEXT_DIM};font-size:12px;font-weight:bold;")
        lay.addWidget(ai_title)

        self._ai_text = QTextEdit()
        self._ai_text.setReadOnly(True)
        self._ai_text.setStyleSheet(
            f"QTextEdit{{background:{PANEL_BG};color:{TEXT_MAIN};"
            f"border:1px solid {BORDER};border-radius:6px;"
            f"font-size:12px;line-height:1.7;padding:8px;}}")
        self._ai_text.setHtml(
            f'<div style="color:{TEXT_DIM};text-align:center;padding:40px;">'
            f'🧠 正在获取数据并分析中...</div>')
        lay.addWidget(self._ai_text, 1)

        self._reanalyze_btn = QPushButton("🔄 重新 AI 分析")
        self._reanalyze_btn.setEnabled(False)
        self._reanalyze_btn.setMinimumHeight(34)
        self._reanalyze_btn.setStyleSheet(
            f"QPushButton{{background:{BLUE};color:white;border:none;border-radius:6px;"
            f"font-weight:bold;font-size:12px;}}"
            f"QPushButton:hover{{background:#388bfd;}}"
            f"QPushButton:disabled{{background:#21262d;color:#484f58;}}")
        self._reanalyze_btn.clicked.connect(self._run_ai_analysis)
        lay.addWidget(self._reanalyze_btn)
        return w

    def _build_footer(self) -> QFrame:
        f   = QFrame()
        lay = QHBoxLayout(f)
        lay.setContentsMargins(0, 0, 0, 0)
        self._status_lbl = QLabel("正在加载数据...")
        self._status_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;")
        lay.addWidget(self._status_lbl)
        lay.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.setFixedSize(90, 32)
        close_btn.setStyleSheet(
            f"QPushButton{{background:#21262d;color:{TEXT_MAIN};border:1px solid {BORDER};"
            f"border-radius:6px;font-size:12px;}}"
            f"QPushButton:hover{{background:{BORDER};}}")
        close_btn.clicked.connect(self.close)
        lay.addWidget(close_btn)
        return f

    # ── 数据加载 ─────────────────────────────────────────────

    def _start_loading(self):
        if not self.okx_client:
            self._canvas.show_placeholder("OKX not connected")
            self._ai_text.setHtml(
                f'<div style="color:{YELLOW};">未连接 OKX，无法获取 K 线数据。</div>')
            self._progress.setRange(0, 1)
            return
        threading.Thread(target=self._fetch_ticker, daemon=True).start()
        for tf, _, limit in TIMEFRAMES:
            threading.Thread(target=self._fetch_kline,
                             args=(tf, limit), daemon=True).start()

    def _fetch_ticker(self):
        try:
            res  = self.okx_client.get_ticker(self.symbol)
            data = res["data"][0] if res.get("code") == "0" and res.get("data") else {}
            self._sig_ticker_done.emit(data)
        except Exception:
            self._sig_ticker_done.emit({})

    def _fetch_kline(self, tf: str, limit: int):
        try:
            res = self.okx_client.get_kline(self.symbol, bar=tf, limit=limit)
            if res.get("code") == "0" and res.get("data"):
                self._sig_kline_done.emit(tf, _parse_klines(res["data"]))
            else:
                self._sig_kline_error.emit(tf, res.get("msg", "API error"))
        except Exception as e:
            self._sig_kline_error.emit(tf, str(e)[:40])

    # ── 信号回调 ─────────────────────────────────────────────

    def _on_ticker_done(self, data: dict):
        if self._closing:
            return
        last = data.get("last", data.get("lastPr", "--"))
        base = data.get("sodUtc8", data.get("open24h", ""))
        try:
            pct   = (float(last) - float(base)) / float(base) * 100
            color = GREEN if pct >= 0 else RED
            sign  = "+" if pct >= 0 else ""
            self._price_lbl.setText(f"{float(last):.4f}")
            self._change_lbl.setText(f"{sign}{pct:.2f}%")
            self._change_lbl.setStyleSheet(
                f"color:{color};font-size:13px;font-weight:bold;")
        except Exception:
            self._price_lbl.setText(str(last))

    def _on_kline_done(self, tf: str, candles: List[dict]):
        if self._closing:
            return
        self._candles[tf] = candles
        if tf == self._current_tf:
            self._draw_current()
        self._check_all_loaded()

    def _on_kline_error(self, tf: str, msg: str):
        if self._closing:
            return
        self._candles.setdefault(tf, [])
        if tf == self._current_tf:
            self._canvas.show_placeholder(f"{tf}: {msg}")
        self._check_all_loaded()

    def _check_all_loaded(self):
        done  = len(self._candles)
        total = len(TIMEFRAMES)
        self._status_lbl.setText(f"K线加载 {done}/{total}")
        if done >= total:
            self._progress.setRange(0, 1)
            self._progress.setValue(0)
            self._status_lbl.setText("K线加载完成，AI分析中...")
            self._reanalyze_btn.setEnabled(True)
            if self.llm_client:
                QTimer.singleShot(100, self._run_ai_analysis)
            else:
                self._ai_text.setHtml(
                    f'<div style="color:{YELLOW};">未配置LLM，无法自动分析。</div>')

    # ── 图表切换 ─────────────────────────────────────────────

    def _switch_tf(self, tf: str):
        self._current_tf = tf
        for t, btn in self._tf_btns.items():
            active = (t == tf)
            btn.setChecked(active)
            btn.setStyleSheet(self._tf_btn_style(active))
        self._draw_current()

    def _refresh_chart(self):
        """指标开关变化时重绘"""
        self._draw_current()

    def _draw_current(self):
        candles = self._candles.get(self._current_tf, [])
        if not candles:
            self._canvas.show_placeholder(f"Loading {self._current_tf}...")
            return
        self._canvas.show_ema  = self._chk_ema.isChecked()
        self._canvas.show_bb   = self._chk_bb.isChecked()
        self._canvas.show_macd = self._chk_macd.isChecked()
        self._canvas.show_vol  = self._chk_vol.isChecked()
        self._canvas.plot(candles, tf=self._current_tf)

    # ── AI 分析 ──────────────────────────────────────────────

    def _run_ai_analysis(self):
        if not self.llm_client:
            return
        self._reanalyze_btn.setEnabled(False)
        self._progress.setRange(0, 0)
        self._ai_text.setHtml(
            f'<div style="color:{TEXT_DIM};padding:10px;">🧠 AI分析中，请稍候...</div>')
        threading.Thread(target=self._do_ai_analysis, daemon=True).start()

    def _do_ai_analysis(self):
        category  = self.scan_item.get("category", "--")
        direction = str(self.scan_item.get("side",
                        self.scan_item.get("direction", "--"))).upper()
        score     = float(self.scan_item.get("opportunity_score",
                          self.scan_item.get("score", 0)) or 0)
        reason    = self.scan_item.get("priority_reason",
                    self.scan_item.get("reason", "--"))
        is_position = category in ("持仓分析", "平仓时机分析")

        # 构建多周期摘要
        summaries = []
        for tf, label, _ in TIMEFRAMES:
            candles = self._candles.get(tf, [])
            if not candles:
                summaries.append(f"[{tf}] no data")
                continue
            closes = [c["close"] for c in candles]
            highs  = [c["high"]  for c in candles]
            lows   = [c["low"]   for c in candles]
            vols   = [c["vol"]   for c in candles]
            last   = closes[-1]
            chg    = (last / closes[0] - 1) * 100 if closes[0] else 0
            ema5   = _ema(closes, 5)[-1]
            ema20  = _ema(closes, 20)[-1]
            trend  = "up" if ema5 > ema20 else "down" if ema5 < ema20 else "flat"
            bb_up, bb_mid, bb_lo = _bollinger(closes, 20)
            bb_pos = "upper" if last > bb_up[-1] else \
                     "lower" if last < bb_lo[-1] else "mid"
            macd_l, sig_l, hist_l = _macd(closes)
            macd_cross = ("golden" if hist_l[-1] > 0 and hist_l[-2] <= 0
                          else "dead" if hist_l[-1] < 0 and hist_l[-2] >= 0
                          else "hold")
            avg_vol = sum(vols) / len(vols) if vols else 1
            vol_r   = vols[-1] / avg_vol
            high20  = max(highs[-20:]) if len(highs) >= 20 else max(highs)
            low20   = min(lows[-20:])  if len(lows)  >= 20 else min(lows)
            summaries.append(
                f"[{tf}] close={last:.4f} chg={chg:+.2f}% trend={trend} "
                f"EMA5={ema5:.4f} EMA20={ema20:.4f} BB={bb_pos} "
                f"MACD={macd_cross} volRatio={vol_r:.2f}x "
                f"high20={high20:.4f} low20={low20:.4f} n={len(candles)}"
            )
        kline_text = "\n".join(summaries)

        if is_position:
            system = """You are a professional crypto position management advisor.
The user currently holds this position. Analyze from a position management perspective.

Cover these points in your response (use HTML format, Chinese language):
1. 📊 Technical structure across timeframes, current price location
2. 📈 Momentum: EMA alignment, trend strength, volume confirmation
3. ⚡ Multi-timeframe resonance (bullish/bearish/divergence)
4. 🎯 Key levels: support/resistance relative to current price
5. 💡 Position management recommendation (one of):
   继续持有 / 加仓 / 减仓 / 止盈 / 止损 / 立即平仓
   Include: suggested take-profit price, stop-loss price, position sizing advice
6. ⚠️ Key risks

HTML colors: 持有/加仓=color:#3fb950, 减仓/止盈=color:#e3b341, 止损/平仓=color:#f85149"""
            user = (f"Symbol: {self.symbol}\n"
                    f"Position side: {direction}\n"
                    f"P&L info: {reason}\n\n"
                    f"Multi-timeframe K-line summary:\n{kline_text}\n\n"
                    f"Provide comprehensive position management analysis in Chinese HTML format.")
        else:
            system = """You are a top crypto technical analyst.
Analyze the given multi-timeframe K-line data and provide a comprehensive trading recommendation.

Cover these points (use HTML format, Chinese language):
1. 📊 Technical patterns across timeframes, support/resistance structure
2. 📈 Momentum analysis: EMA alignment, trend strength, volume-price relationship
3. ⚡ Multi-timeframe resonance (bullish/bearish/divergence)
4. 🎯 Key price levels: major support, resistance, stop-loss zones
5. 💡 Trading recommendation:
   - Direction: 加仓/买入/做多/观望/减仓/卖出/做空/止损
   - Entry zone, take-profit target, stop-loss level
   - Position sizing: 重仓30%+ / 标准仓10-20% / 轻仓5% / 不操作
6. ⚠️ Main risks

HTML colors: 买入/看多=color:#3fb950, 卖出/看空=color:#f85149, neutral=color:#e3b341"""
            user = (f"Symbol: {self.symbol}\n"
                    f"Strategy signal: {category} | {direction} | score {score:.1f}\n"
                    f"Signal reason: {reason}\n\n"
                    f"Multi-timeframe K-line summary:\n{kline_text}\n\n"
                    f"Provide comprehensive technical analysis in Chinese HTML format.")

        raw = self.llm_client.chat(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            timeout=120,
        )
        if raw:
            html = self._fmt(raw)
        else:
            html = f'<div style="color:{RED};">分析失败: {self.llm_client.last_error}</div>'
        self._sig_ai_done.emit(html)

    def _fmt(self, text: str) -> str:
        wrap = (f'<div style="color:{TEXT_MAIN};font-size:12px;'
                f'line-height:1.8;padding:4px;">')
        if "<b>" in text or "<br>" in text or "<div" in text:
            return wrap + text + "</div>"
        lines = []
        for line in text.split("\n"):
            l = line.strip()
            if not l:
                lines.append("<br>")
                continue
            if l.startswith("###"):
                l = f'<b style="color:{BLUE};font-size:13px;">{l[3:].strip()}</b>'
            elif l.startswith("##"):
                l = f'<b style="color:{BLUE};font-size:14px;">{l[2:].strip()}</b>'
            elif l.startswith("#"):
                l = f'<b style="color:{TEXT_MAIN};font-size:15px;">{l[1:].strip()}</b>'
            elif l.startswith(("- ", "* ")):
                l = f"&nbsp;&nbsp;• {l[2:]}"
            for kw, c in [("加仓", GREEN), ("买入", GREEN), ("做多", GREEN),
                           ("卖出", RED),  ("做空", RED),   ("止损", RED),
                           ("观望", YELLOW), ("减仓", ORANGE),
                           ("继续持有", GREEN), ("立即平仓", RED)]:
                l = l.replace(kw, f'<b style="color:{c};">{kw}</b>')
            lines.append(l)
        return wrap + "<br>".join(lines) + "</div>"

    def _on_ai_done(self, html: str):
        if self._closing:
            return
        self._ai_text.setHtml(html)
        self._progress.setRange(0, 1)
        self._status_lbl.setText(f"✅ 分析完成  {datetime.now().strftime('%H:%M:%S')}")
        self._reanalyze_btn.setEnabled(True)

    def closeEvent(self, event):
        self._closing = True
        try:
            if hasattr(self, "_canvas") and getattr(self._canvas, "fig", None) is not None:
                plt.close(self._canvas.fig)
        except Exception:
            pass
        super().closeEvent(event)
