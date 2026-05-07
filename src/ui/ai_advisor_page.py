"""
AI 策略顾问界面：大模型 API 配置 + 对话 + 策略分析。
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


from src.ai_agent.llm_client import LLMClient
from src.ai_agent.strategy_advisor import StrategyAdvisor
from src.ai_agent.code_parser import CodeDiffParser, CodeChange
from src.ai_agent.code_applier import SafeCodeApplier
from src.ai_agent.auto_cycle import AutoCycleController
from src.ai_agent.system_context import SystemContext
from src.ai_agent.multi_agent_trader import MultiAgentTrader
from src.ai_agent.kline_trader import KLineTrader, KLineAnalyzer
from src.ai_agent.kline_monitor import KLineMonitor
from src.ai_agent.self_evolver import SelfEvolver
from src.ai_agent.risk_guard import RiskGuard, RiskAlert
from src.ai_agent.vulnerability_scanner import VulnerabilityScanner, SEV_CRITICAL, SEV_HIGH
from src.ui.risk_alert_dialog import AlertHistoryWidget, get_alert_overlay, show_alert
from src.qt_compat import QApplication, QCheckBox, QColor, QComboBox, QDoubleSpinBox, QFormLayout, QFrame, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QSlider, QSpinBox, QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit, QTimer, QVBoxLayout, QWidget, Qt, Signal

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "ai_config.json"
CONVERSATION_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "ai_conversation.json"


class AIAdvisorPage(QWidget):
    """AI 策略顾问页面"""

    # 跨线程信号：后台线程操作结果返回主线程
    _signal_connection_tested = Signal(bool, str)
    _signal_chat_reply = Signal(str, str)
    _signal_analysis_done = Signal(object)
    _signal_stream_token = Signal(str)
    _signal_kl_done = Signal(object)
    _signal_risk_alert = Signal(object)
    _signal_risk_metrics = Signal(dict)
    _signal_vuln_done = Signal(object)
    _signal_vs_fix_done = Signal(object)   # {"patch": str, "vuln": obj, "fixed_code": str}

    def __init__(self, tracker=None, mutator=None, timeframe_tracker=None,
                 optimizer=None, parent=None):
        super().__init__(parent)
        self.tracker = tracker
        self.mutator = mutator
        self.timeframe_tracker = timeframe_tracker
        self.optimizer = optimizer
        self.llm_client: Optional[LLMClient] = None
        self.advisor: Optional[StrategyAdvisor] = None
        self._busy = False
        self.conversation: List[Dict] = []
        self._busy_since = 0.0  # 记录忙碌开始时间
        self._pending_changes: List[CodeChange] = []
        self._pending_file: str = ""
        self._pending_file_name: str = ""
        self.code_applier: Optional[SafeCodeApplier] = None
        self.auto_cycle: Optional[AutoCycleController] = None
        self.multi_agent: Optional[MultiAgentTrader] = None
        self.kline_trader: Optional[KLineTrader] = None
        self.kline_monitor: Optional[KLineMonitor] = None
        self._pending_strategy: str = ""
        self._pending_stats: Dict = {}
        self.system_ctx = SystemContext()
        self._load_config()
        self._load_conversation()
        self.init_ui()
        self._restore_chat_display()

        # 连接跨线程信号
        self._signal_connection_tested.connect(self._on_connection_tested)
        self._signal_chat_reply.connect(self._on_chat_reply)
        self._signal_analysis_done.connect(self._on_analysis_done)
        self._signal_stream_token.connect(self._on_stream_token)
        self._signal_kl_done.connect(self._on_kl_done)
        self._signal_risk_alert.connect(self._on_risk_alert)
        self._signal_risk_metrics.connect(self._on_risk_metrics)
        self._signal_vuln_done.connect(self._on_vuln_scan_done)
        self._signal_vs_fix_done.connect(self._on_vs_fix_done)

        # 启动后自动测试已有 API 连接
        if self._config.get("api_key", "").strip():
            QTimer.singleShot(1500, self._auto_test_connection)

    # ─── Config Persistence ────────────────────────────────────────

    def _load_config(self):
        self._config = {
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "",
            "model": "deepseek-chat",
            "temperature": 0.3,
            "max_tokens": 4096,
        }
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if CONFIG_PATH.exists():
            try:
                saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                self._config.update(saved)
            except Exception:
                pass

    def _save_config(self):
        try:
            CONFIG_PATH.write_text(
                json.dumps(self._config, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    # ─── 对话持久化 ──────────────────────────────────────────────────

    def _load_conversation(self):
        """从本地文件加载历史对话"""
        if not CONVERSATION_PATH.exists():
            return
        try:
            data = json.loads(CONVERSATION_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self.conversation = data
        except Exception:
            pass

    def _save_conversation(self):
        """将对话保存到本地文件"""
        try:
            CONVERSATION_PATH.parent.mkdir(parents=True, exist_ok=True)
            CONVERSATION_PATH.write_text(
                json.dumps(self.conversation[-200:], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _restore_chat_display(self):
        """在聊天显示区恢复历史对话内容"""
        for msg in self.conversation:
            role = msg.get("role", "")
            content = msg.get("content", "")
            safe = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            if role == "user":
                self.chat_display.append(
                    f'<p style="color:#8B5CF6;"><b>👤 你:</b> {safe}</p>'
                )
            elif role == "assistant":
                self.chat_display.append(
                    f'<div style="background:#1a1035; padding:8px; border-radius:6px; margin:4px 0;">'
                    f'<p style="color:#A78BFA;"><b>🤖 AI:</b></p>'
                    f'<p style="color:#ccc;">{safe}</p></div>'
                )

    # ─── UI ─────────────────────────────────────────────────────────

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("🤖 AI 策略顾问 · 大模型驱动的策略优化")
        title.setStyleSheet("color: #8B5CF6; font-size: 16px; font-weight: bold; padding: 4px;")
        layout.addWidget(title)

        desc = QLabel(
            "连接大模型 API，将扫描信号验证数据 + 策略代码发送给 AI，"
            "让 AI 分析缺陷并给出具体的参数调整和代码修改方案。"
        )
        desc.setStyleSheet("color: #aaaaaa; font-size: 11px; padding-bottom: 6px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        self.main_tabs = QTabWidget()
        self.main_tabs.setStyleSheet("""
            QTabWidget::pane { background-color: #1a1a2e; border: 1px solid #333; border-radius: 4px; }
            QTabBar::tab { background-color: #252540; color: #ccc; padding: 6px 16px;
                           border: 1px solid #333; border-bottom: none; border-radius: 4px 4px 0 0; }
            QTabBar::tab:selected { background-color: #1a1a2e; color: #8B5CF6; font-weight: bold; }
        """)

        self._build_config_tab()
        self._build_chat_tab()
        self._build_analysis_tab()
        self._build_auto_tab()
        self._build_risk_guard_tab()
        self._build_vuln_scan_tab()
        self._build_sysmon_tab()
        self._build_multi_agent_tab()
        self._build_kline_tab()
        self._build_kl_monitor_tab()
        self.main_tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self.main_tabs, 1)

        status_bar = QHBoxLayout()
        status_bar.setContentsMargins(0, 4, 0, 0)
        self.status_label = QLabel("就绪 · 请先配置 API")
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")
        status_bar.addWidget(self.status_label)
        status_bar.addStretch()
        layout.addLayout(status_bar)

    def _wrap_in_scroll(self, inner: QWidget) -> QScrollArea:
        """将内容包在始终显示滚动条的 QScrollArea 中"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollBar:vertical { width: 10px; background: #1a1a2e; }"
            "QScrollBar::handle:vertical { background: #555; border-radius: 4px; min-height: 30px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
        )
        return scroll

    # ─── Config Tab ─────────────────────────────────────────────────

    def _build_config_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 顶部连接状态
        top_bar = QHBoxLayout()
        self.connect_btn = QPushButton("🔗 测试连接")
        self.connect_btn.clicked.connect(self._test_connection)
        self.connect_btn.setStyleSheet(
            "QPushButton { background-color: #8B5CF6; color: white; font-weight: bold; "
            "padding: 5px 14px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #A78BFA; }"
        )
        top_bar.addWidget(self.connect_btn)
        top_bar.addStretch()
        self.config_status = QLabel("未测试")
        self.config_status.setStyleSheet("color: #888; font-size: 11px;")
        top_bar.addWidget(self.config_status)
        layout.addLayout(top_bar)

        group = QGroupBox("🔑 API 配置")
        group.setStyleSheet(
            "QGroupBox { border: 1px solid #555; border-radius: 6px; margin-top: 8px; "
            "padding: 12px; font-weight: bold; color: #8B5CF6; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
        )
        form = QFormLayout()
        form.setSpacing(10)

        self.url_edit = QLineEdit()
        self.url_edit.setText(self._config.get("base_url", ""))
        self.url_edit.setPlaceholderText("https://api.deepseek.com/v1")
        self.url_edit.setStyleSheet("QLineEdit { background:#0d1117; color:#ccc; border:1px solid #444; padding:6px; border-radius:4px; }")
        self.url_edit.textChanged.connect(lambda: self._on_config_changed())
        form.addRow("API 地址:", self.url_edit)

        # 快捷预设按钮
        preset_bar = QHBoxLayout()
        for label, url, model in [
            ("DeepSeek", "https://api.deepseek.com/v1", "deepseek-chat"),
            ("OpenCode Go", "https://opencode.ai/zen/go/v1", "deepseek-v4-pro"),
            ("OpenAI", "https://api.openai.com/v1", "gpt-4o"),
            ("Ollama", "http://localhost:11434/v1", "llama3"),
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(26)
            btn.setStyleSheet(
                "QPushButton { background:#252540; color:#ccc; border:1px solid #444; "
                "border-radius:4px; font-size:11px; padding:2px 10px; }"
                "QPushButton:hover { background:#333; color:#8B5CF6; }"
            )
            btn.clicked.connect(lambda _, u=url, m=model: self._apply_preset(u, m))
            preset_bar.addWidget(btn)
        preset_bar.addStretch()
        form.addRow("快捷:", preset_bar)

        self.key_edit = QLineEdit()
        self.key_edit.setText(self._config.get("api_key", ""))
        self.key_edit.setEchoMode(QLineEdit.Password)
        self.key_edit.setPlaceholderText("sk-xxxxxxxxxxxxxxxx")
        self.key_edit.setStyleSheet("QLineEdit { background:#0d1117; color:#ccc; border:1px solid #444; padding:6px; border-radius:4px; }")
        self.key_edit.textChanged.connect(lambda: self._on_config_changed())
        form.addRow("API Key:", self.key_edit)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems([
            "deepseek-chat", "deepseek-reasoner", "deepseek-v3-0324",
            "gpt-4o", "gpt-4o-mini", "gpt-4-turbo",
            "claude-3-5-sonnet-20241022", "claude-3-opus-20240229",
            "qwen-max", "qwen-plus",
            "gemini-2.0-flash", "gemini-2.0-pro",
            "llama-3.3-70b", "mixtral-8x7b",
        ])
        self.model_combo.setCurrentText(self._config.get("model", "deepseek-chat"))
        self.model_combo.setStyleSheet("QComboBox { background:#0d1117; color:#ccc; border:1px solid #444; padding:5px; border-radius:4px; }")
        self.model_combo.currentTextChanged.connect(lambda: self._on_config_changed())
        form.addRow("模型:", self.model_combo)

        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(self._config.get("temperature", 0.3))
        self.temp_spin.setStyleSheet("QDoubleSpinBox { background:#0d1117; color:#ccc; border:1px solid #444; padding:5px; }")
        self.temp_spin.valueChanged.connect(lambda: self._on_config_changed())
        form.addRow("Temperature:", self.temp_spin)

        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(512, 128000)
        self.max_tokens_spin.setSingleStep(512)
        self.max_tokens_spin.setValue(self._config.get("max_tokens", 4096))
        self.max_tokens_spin.setStyleSheet("QSpinBox { background:#0d1117; color:#ccc; border:1px solid #444; padding:5px; }")
        self.max_tokens_spin.valueChanged.connect(lambda: self._on_config_changed())
        form.addRow("Max Tokens:", self.max_tokens_spin)

        group.setLayout(form)
        layout.addWidget(group)

        info = QLabel(
            "💡 支持所有 OpenAI 兼容 API（DeepSeek / OpenAI / Claude 代理 / 本地模型 / Ollama）。\n"
            "切换模型后点击「🔗 测试连接」确认可用。API Key 仅存储本地，不会上传。"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #777; font-size: 10px; padding: 8px;")
        layout.addWidget(info)

        layout.addStretch()
        self.main_tabs.addTab(self._wrap_in_scroll(tab), "🔑 API配置")

    # ─── Chat Tab ───────────────────────────────────────────────────

    def _build_chat_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet(
            "QTextEdit { background-color: #0a0a15; color: #ccc; border: 1px solid #333; "
            "border-radius: 4px; font-size: 14px; padding: 8px; }"
        )
        layout.addWidget(self.chat_display, 1)

        input_bar = QHBoxLayout()
        self.chat_input = QTextEdit()
        self.chat_input.setMaximumHeight(100)
        self.chat_input.setPlaceholderText("输入你的问题或需求...")
        self.chat_input.setStyleSheet(
            "QTextEdit { background:#0d1117; color:#ccc; border:1px solid #444; "
            "border-radius:4px; padding:8px; font-size: 13px; }"
        )
        input_bar.addWidget(self.chat_input, 1)

        send_btn = QPushButton("发送 📤")
        send_btn.clicked.connect(self._send_chat)
        send_btn.setStyleSheet(
            "QPushButton { background-color: #8B5CF6; color: white; font-weight: bold; "
            "padding: 10px 24px; border-radius: 4px; font-size: 13px; }"
            "QPushButton:hover { background-color: #A78BFA; }"
        )
        input_bar.addWidget(send_btn)
        layout.addLayout(input_bar)

        self.main_tabs.addTab(self._wrap_in_scroll(tab), "💬 对话")

    # ─── Analysis Tab ───────────────────────────────────────────────

    def _build_analysis_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("<b style='color:#8B5CF6;'>策略:</b>"))

        self.strategy_combo = QComboBox()
        self.strategy_combo.setStyleSheet(
            "QComboBox { background:#0d1117; color:#ccc; border:1px solid #444; padding:4px 8px; border-radius:4px; }"
        )
        bar.addWidget(self.strategy_combo, 1)

        self.analyze_btn = QPushButton("🔍 AI分析策略")
        self.analyze_btn.clicked.connect(self._analyze_strategy)
        self.analyze_btn.setStyleSheet(
            "QPushButton { background-color: #8B5CF6; color: white; font-weight: bold; "
            "padding: 6px 18px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #A78BFA; }"
        )
        bar.addWidget(self.analyze_btn)
        layout.addLayout(bar)

        self.review_btn = QPushButton("📝 AI代码审查（结合优化器参数）")
        self.review_btn.clicked.connect(self._review_code)
        self.review_btn.setStyleSheet(
            "QPushButton { background-color: #059669; color: white; font-weight: bold; "
            "padding: 6px 18px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #34D399; }"
        )
        layout.addWidget(self.review_btn)

        self.analysis_output = QTextEdit()
        self.analysis_output.setReadOnly(True)
        self.analysis_output.setStyleSheet(
            "QTextEdit { background-color: #0a0a15; color: #ccc; border: 1px solid #333; "
            "border-radius: 4px; font-family: 'Menlo', monospace; font-size: 13px; padding: 8px; }"
        )
        layout.addWidget(self.analysis_output, 2)

        # --- 代码修改预览 + 应用操作 ---
        diff_label = QLabel("📝 AI 建议的代码修改预览")
        diff_label.setStyleSheet("color: #ffd700; font-size: 14px; font-weight: bold; padding: 6px 0 2px;")
        layout.addWidget(diff_label)

        self.diff_preview = QTextEdit()
        self.diff_preview.setReadOnly(True)
        self.diff_preview.setMaximumHeight(220)
        self.diff_preview.setPlaceholderText("AI 分析完成后，代码修改建议将显示在这里…")
        self.diff_preview.setStyleSheet(
            "QTextEdit { background-color: #0f0f1a; color: #ccc; border: 1px solid #444; "
            "border-radius: 4px; font-family: 'Menlo', monospace; font-size: 12px; padding: 6px; }"
        )
        layout.addWidget(self.diff_preview)

        apply_bar = QHBoxLayout()

        self.auto_apply_check = QCheckBox("🤖 批准后自动应用（跳过预览确认）")
        self.auto_apply_check.setStyleSheet("QCheckBox { color: #ccc; font-size: 11px; }")
        apply_bar.addWidget(self.auto_apply_check)

        apply_bar.addStretch()

        self.apply_btn = QPushButton("✅ 应用到策略文件")
        self.apply_btn.clicked.connect(self._apply_suggested_changes)
        self.apply_btn.setEnabled(False)
        self.apply_btn.setStyleSheet(
            "QPushButton { background-color: #059669; color: white; font-weight: bold; "
            "padding: 6px 18px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #34D399; }"
            "QPushButton:disabled { background-color: #444; color: #888; }"
        )
        apply_bar.addWidget(self.apply_btn)

        self.rollback_btn = QPushButton("⏪ 回退上次修改")
        self.rollback_btn.clicked.connect(self._rollback_last)
        self.rollback_btn.setEnabled(False)
        self.rollback_btn.setStyleSheet(
            "QPushButton { background-color: #cc3333; color: white; font-weight: bold; "
            "padding: 6px 14px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #dd4444; }"
            "QPushButton:disabled { background-color: #444; color: #888; }"
        )
        apply_bar.addWidget(self.rollback_btn)

        layout.addLayout(apply_bar)

        self.analysis_progress = QProgressBar()
        self.analysis_progress.setRange(0, 0)
        self.analysis_progress.setVisible(False)
        layout.addWidget(self.analysis_progress)

        self.main_tabs.addTab(self._wrap_in_scroll(tab), "🔬 策略分析")

    def _build_auto_tab(self):
        """自主进化控制面板"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 控制栏
        ctrl_group = QGroupBox("⚙️ 自主进化控制")
        ctrl_group.setStyleSheet(
            "QGroupBox { border: 1px solid #555; border-radius: 6px; margin-top: 8px; "
            "padding: 12px; font-weight: bold; color: #00cc66; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
        )
        ctrl_layout = QHBoxLayout(ctrl_group)

        self.auto_cycle_toggle = QPushButton("🤖 自主进化: OFF")
        self.auto_cycle_toggle.setCheckable(True)
        self.auto_cycle_toggle.clicked.connect(self._toggle_auto_cycle)
        self.auto_cycle_toggle.setStyleSheet(
            "QPushButton { background-color: #444; color: #ccc; font-size: 13px; font-weight: bold; "
            "padding: 8px 20px; border-radius: 6px; border: 2px solid #666; }"
            "QPushButton:checked { background-color: #00aa66; color: white; border-color: #00ff88; }"
        )
        ctrl_layout.addWidget(self.auto_cycle_toggle)

        conf_layout = QFormLayout()
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(50, 95)
        self.confidence_slider.setValue(70)
        self.confidence_slider.setTickInterval(5)
        self.confidence_slider.valueChanged.connect(
            lambda v: self.conf_label.setText(f"最低置信度: {v}%")
        )
        self.conf_label = QLabel("最低置信度: 70%")
        self.conf_label.setStyleSheet("color: #ccc; font-size: 11px;")
        conf_layout.addRow(self.conf_label, self.confidence_slider)
        ctrl_layout.addLayout(conf_layout)

        ctrl_layout.addStretch()
        layout.addWidget(ctrl_group)

        # 自主进化引擎控制栏
        evolver_group = QGroupBox("🧬 AI 自主进化引擎（元学习）")
        evolver_group.setStyleSheet(
            "QGroupBox { border: 1px solid #555; border-radius: 6px; margin-top: 6px; "
            "padding: 10px; font-weight: bold; color: #c084fc; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
        )
        evolver_h = QHBoxLayout(evolver_group)

        self.evolver_toggle = QPushButton("🧬 自进化引擎: OFF")
        self.evolver_toggle.setCheckable(True)
        self.evolver_toggle.clicked.connect(self._toggle_self_evolver)
        self.evolver_toggle.setStyleSheet(
            "QPushButton { background-color: #3b2060; color: #c084fc; font-size: 12px; font-weight: bold; "
            "padding: 6px 16px; border-radius: 6px; border: 2px solid #7c3aed; }"
            "QPushButton:checked { background-color: #7c3aed; color: white; border-color: #c084fc; }"
        )
        evolver_h.addWidget(self.evolver_toggle)

        self.reflect_btn = QPushButton("🔍 立即元反思")
        self.reflect_btn.clicked.connect(self._manual_reflect)
        self.reflect_btn.setStyleSheet(
            "QPushButton { background-color: #1e1040; color: #a78bfa; padding: 5px 12px; "
            "border-radius: 4px; border: 1px solid #7c3aed; font-size: 11px; }"
            "QPushButton:hover { background-color: #2d1a6e; }"
        )
        evolver_h.addWidget(self.reflect_btn)

        self.evolver_status = QLabel("引擎未启动")
        self.evolver_status.setStyleSheet("color: #888; font-size: 10px; padding-left: 8px;")
        evolver_h.addWidget(self.evolver_status, 1)
        layout.addWidget(evolver_group)

        # 元洞察展示
        self.insight_display = QTextEdit()
        self.insight_display.setReadOnly(True)
        self.insight_display.setMaximumHeight(60)
        self.insight_display.setPlaceholderText("AI 元学习洞察将显示在这里…")
        self.insight_display.setStyleSheet(
            "QTextEdit { background-color: #120830; color: #c084fc; border: 1px solid #7c3aed; "
            "border-radius: 4px; font-size: 10px; padding: 4px; font-style: italic; }"
        )
        layout.addWidget(self.insight_display)

        # 健康面板
        health_group = QGroupBox("💓 策略健康监控")
        health_group.setStyleSheet(
            "QGroupBox { border: 1px solid #444; border-radius: 6px; margin-top: 6px; "
            "padding: 10px; font-weight: bold; color: #ffd700; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; }"
        )
        health_layout = QVBoxLayout(health_group)

        self.health_table = QTableWidget()
        self.health_table.setColumnCount(5)
        self.health_table.setHorizontalHeaderLabels([
            "策略", "7日胜率", "信号数", "待验证", "健康度"
        ])
        self.health_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.health_table.setMaximumHeight(150)
        self.health_table.setStyleSheet(
            "QTableWidget { background-color: #0d1117; color: #ccc; gridline-color: #333; font-size: 11px; }"
        )
        health_layout.addWidget(self.health_table)
        layout.addWidget(health_group)

        # 变更历史
        history_group = QGroupBox("📜 自主变更历史")
        history_group.setStyleSheet(
            "QGroupBox { border: 1px solid #444; border-radius: 6px; margin-top: 6px; "
            "padding: 10px; font-weight: bold; color: #88ccff; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; }"
        )
        hist_layout = QVBoxLayout(history_group)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels([
            "时间", "策略", "变更", "结果", "变化%"
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.setStyleSheet(
            "QTableWidget { background-color: #0d1117; color: #ccc; gridline-color: #333; font-size: 10px; }"
        )
        hist_layout.addWidget(self.history_table)
        layout.addWidget(history_group)

        # 周期日志
        self.cycle_log = QTextEdit()
        self.cycle_log.setReadOnly(True)
        self.cycle_log.setMaximumHeight(100)
        self.cycle_log.setStyleSheet(
            "QTextEdit { background-color: #0a0a15; color: #88ff88; border: 1px solid #333; "
            "border-radius: 4px; font-family: 'Menlo', monospace; font-size: 10px; padding: 4px; }"
        )
        layout.addWidget(self.cycle_log)

        # 刷新定时器
        self._auto_refresh_timer = QTimer(self)
        self._auto_refresh_timer.timeout.connect(self._refresh_auto_tab)
        self._auto_refresh_timer.start(15000)

        self.main_tabs.addTab(self._wrap_in_scroll(tab), "🤖 自主进化")

    # ─── Auto Cycle Control ─────────────────────────────────────────

    def _toggle_auto_cycle(self):
        """切换自主进化循环"""
        if self.auto_cycle_toggle.isChecked():
            self.auto_cycle_toggle.setText("🤖 自主进化: ON")
            self.auto_cycle_toggle.setChecked(True)
            self._start_auto_cycle()
        else:
            self.auto_cycle_toggle.setText("🤖 自主进化: OFF")
            self.auto_cycle_toggle.setChecked(False)
            self._stop_auto_cycle()

    def _start_auto_cycle(self):
        if self.auto_cycle and self.auto_cycle.isRunning():
            return
        if not self.advisor:
            self._get_or_create_client()
            if not self.llm_client:
                QMessageBox.warning(self, "提示", "请先配置 API")
                self.auto_cycle_toggle.setChecked(False)
                self.auto_cycle_toggle.setText("🤖 自主进化: OFF")
                return
            self.advisor = StrategyAdvisor(
                self.llm_client, self.tracker, self.mutator,
                self.timeframe_tracker, self.optimizer,
            )
        if not self.code_applier:
            self.code_applier = SafeCodeApplier()

        conf = self.confidence_slider.value()
        self.auto_cycle = AutoCycleController(
            tracker=self.tracker,
            advisor=self.advisor,
            code_applier=self.code_applier,
            parser=CodeDiffParser,
            cycle_interval_min=30,
            min_confidence=conf,
            verify_wait_signals=10,
            max_drawdown_after_apply=5.0,
            auto_apply_enabled=True,
        )
        self.auto_cycle.log_signal.connect(self._on_cycle_log)
        self.auto_cycle.status_signal.connect(lambda s: self._cycle_log(f"状态: {s}"))
        self.auto_cycle.analysis_requested.connect(self._on_auto_analysis_requested)
        self.auto_cycle.change_applied.connect(self._on_auto_change_applied)
        self.auto_cycle.change_rolled_back.connect(self._on_auto_rollback)
        self.auto_cycle.start()
        self._cycle_log("🤖 自主进化循环已启动（每30分钟检查一次）")

    def _stop_auto_cycle(self):
        if self.auto_cycle:
            self.auto_cycle.stop()
            self.auto_cycle.wait(5000)
            self.auto_cycle = None
        self._cycle_log("⏹ 自主进化循环已停止")

    def _toggle_self_evolver(self):
        if self.evolver_toggle.isChecked():
            self._start_self_evolver()
        else:
            self._stop_self_evolver()

    def _start_self_evolver(self):
        evolver = getattr(self, 'self_evolver', None)
        if evolver and evolver.is_running():
            return
        if not self.advisor:
            self._get_or_create_client()
            if not self.llm_client:
                QMessageBox.warning(self, "提示", "请先配置 API Key")
                self.evolver_toggle.setChecked(False)
                self.evolver_toggle.setText("🧬 自进化引擎: OFF")
                return
            self.advisor = StrategyAdvisor(
                self.llm_client, self.tracker, self.mutator,
                self.timeframe_tracker, self.optimizer,
            )
        if not self.code_applier:
            self.code_applier = SafeCodeApplier()

        self.self_evolver = SelfEvolver(
            advisor=self.advisor,
            tracker=self.tracker,
            code_applier=self.code_applier,
            cycle_interval_min=60,
            verify_wait_signals=10,
            max_drawdown_allowed=5.0,
            min_signals_to_analyze=5,
            reflect_every_n_cycles=3,
            log_callback=lambda msg, lvl: self._on_evolver_log(msg, lvl),
        )
        self.self_evolver.start()
        self.evolver_toggle.setText("🧬 自进化引擎: ON")
        self.evolver_status.setText("🟢 运行中（每60分钟一轮）")
        self._cycle_log("🧬 AI自主进化引擎已启动（元学习模式）")

    def _stop_self_evolver(self):
        evolver = getattr(self, 'self_evolver', None)
        if evolver:
            evolver.stop()
            self.self_evolver = None
        self.evolver_toggle.setText("🧬 自进化引擎: OFF")
        self.evolver_status.setText("引擎未启动")
        self._cycle_log("🧬 AI自主进化引擎已停止")

    def _on_evolver_log(self, msg: str, level: str):
        from PySide6.QtCore import QMetaObject, Qt, Q_ARG
        # 线程安全：通过 QTimer 切回主线程
        colors = {"INFO": "#c084fc", "SUCCESS": "#00ff88", "WARNING": "#ffaa00", "ERROR": "#ff6666"}
        color = colors.get(level, "#ccc")
        QTimer.singleShot(0, lambda: self._cycle_log(f'<span style="color:{color}">{msg}</span>'))

        # 更新元洞察展示
        if "元洞察" in msg or "meta_insight" in msg.lower():
            QTimer.singleShot(0, lambda: self._update_insight_display())

    def _update_insight_display(self):
        evolver = getattr(self, 'self_evolver', None)
        if not evolver:
            return
        status = evolver.get_status()
        insight = status.get("latest_insight", "")
        if insight:
            self.insight_display.setPlainText(f"💡 {insight}")
        rates = status.get("success_rate_by_mode", {})
        rate_str = " | ".join(f"{k}:{v}%" for k, v in rates.items()) if rates else "暂无"
        self.evolver_status.setText(
            f"🟢 周期#{status.get('cycle', 0)} | 成功率: {rate_str}"
        )

    def _manual_reflect(self):
        evolver = getattr(self, 'self_evolver', None)
        if not evolver:
            QMessageBox.information(self, "提示", "请先启动自进化引擎")
            return
        self._cycle_log("🔍 手动触发元反思...")

        def _do():
            insight = evolver.manual_reflect()
            QTimer.singleShot(0, lambda: self._on_manual_reflect_done(insight))

        threading.Thread(target=_do, daemon=True).start()

    def _on_manual_reflect_done(self, insight: str):
        if insight and insight != "（历史数据不足，无法反思）":
            self.insight_display.setPlainText(f"💡 {insight}")
            self._cycle_log(f"✅ 元反思完成: {insight[:80]}...")
        else:
            self._cycle_log("⚠️ " + (insight or "元反思失败"))

    def _on_cycle_log(self, msg: str, level: str):
        colors = {"INFO": "#88ccff", "SUCCESS": "#00ff88", "WARNING": "#ffaa00", "ERROR": "#ff6666"}
        self._cycle_log(f'<span style="color:{colors.get(level, "#ccc")}">{msg}</span>')

    def _cycle_log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.cycle_log.append(f'<span style="color:#555">[{ts}]</span> {msg}')
        doc = self.cycle_log.document()
        if doc.blockCount() > 150:
            cursor = self.cycle_log.textCursor()
            cursor.movePosition(cursor.Start)
            for _ in range(30):
                cursor.movePosition(cursor.EndOfBlock, cursor.KeepAnchor)
                cursor.removeSelectedText()
                cursor.deleteChar()

    def _on_auto_analysis_requested(self, strategy_name: str):
        """自主循环请求分析某个策略"""
        if self._busy:
            return
        self._pending_strategy = strategy_name
        self.analysis_progress.setVisible(True)

        # 构建 prompt 分析
        advisor = self.advisor
        client = self.llm_client
        stats = self.tracker.strategy_stats(strategy_name, days=7)
        raw_code = advisor._load_strategy_code(strategy_name) or "# 未找到"
        code = advisor._truncate_code(raw_code, max_lines=120)
        code = code.replace("{", "{{").replace("}", "}}")
        signals = self.tracker.strategy_recent_signals(strategy_name, limit=5)
        signals_brief = [{
            "time": s.get("datetime", "")[-16:],
            "symbol": s.get("symbol", ""), "dir": s.get("direction", ""),
            "score": round(float(s.get("score", 0) or 0), 1),
        } for s in signals]
        from src.ai_agent.prompt_templates import ANALYSIS_PROMPT, SYSTEM_PROMPT
        prompt = ANALYSIS_PROMPT.format(
            strategy_name=strategy_name, days=7,
            total_signals=stats.get("total", 0),
            win_rate=stats.get("win_rate", 0),
            profit_factor=stats.get("profit_factor", 0),
            net_pnl=stats.get("net_pnl", 0),
            signal_samples=json.dumps(signals_brief, indent=2, ensure_ascii=False),
            timeframe_accuracy=json.dumps(
                self.timeframe_tracker.accuracy_by_timeframe(strategy_name, 7) if self.timeframe_tracker else {},
                indent=2, ensure_ascii=False,
            ),
            param_history=json.dumps(advisor._format_param_history(strategy_name), indent=2, ensure_ascii=False),
            strategy_code=code,
        )
        # 只注入策略信号统计数据，不发送账户/持仓信息（避免触发 LLM 安全护栏）
        signals = self._get_signal_summary_text()
        if signals:
            system_note = f"\n\n## 策略运行统计\n{signals}\n"
            prompt += system_note

        def _do():
            try:
                reply = client.chat(
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": prompt}],
                    timeout=180,
                )
                if reply:
                    fname = advisor._find_strategy_file(strategy_name)
                    changes = CodeDiffParser.parse_analysis(reply, target_file=fname or "")
                    result = {"success": True, "changes": changes, "strategy_name": strategy_name}
                else:
                    result = {"success": False, "error": client.last_error or "无响应"}
            except Exception as e:
                result = {"success": False, "error": str(e)}
            # 在主线程处理
            QTimer.singleShot(0, lambda: self._on_auto_analysis_result(result))

        threading.Thread(target=_do, daemon=True).start()

    def _on_auto_analysis_result(self, result: Dict):
        self.analysis_progress.setVisible(False)
        if not result.get("success"):
            self._cycle_log(f"❌ 自动分析失败: {result.get('error', '')}")
            return
        changes = result.get("changes", [])
        strategy_name = result.get("strategy_name", "")
        if not changes:
            self._cycle_log(f"📊 {strategy_name} AI 未找到可应用的修改")
            return

        # 过滤低置信度变更
        conf = self.confidence_slider.value() / 100.0
        approved = [c for c in changes if c.confidence >= conf]
        if not approved:
            self._cycle_log(f"⏸ {strategy_name} 所有变更置信度低于 {conf*100:.0f}%，跳过")
            return

        # 应用变更
        fname = self.advisor._find_strategy_file(strategy_name)
        if not fname:
            return
        success, msg, applied = self.code_applier.apply_changes(fname, approved)
        if success:
            stats = self.tracker.strategy_stats(strategy_name, days=7)
            self.auto_cycle.on_changes_applied(strategy_name, applied, stats)
            if self.mutator:
                self.mutator._load_mutation_points()
            self._cycle_log(f"✅ {strategy_name}: {len(applied)} 项修改已应用，等待验证...")
            self._pending_changes = approved
            self._pending_file = fname
            self._pending_file_name = strategy_name
            self._show_diff_preview(approved)
        else:
            self._cycle_log(f"❌ 应用失败: {msg}")

    def _on_auto_change_applied(self, strategy_name: str, changes: list):
        pass

    def _on_auto_rollback(self, strategy_name: str, reason: str):
        self._cycle_log(f"⏪ {strategy_name} 已回退: {reason}")

    def _refresh_auto_tab(self):
        """定时刷新健康面板和历史"""
        if not hasattr(self, 'health_table'):
            return

        # 健康面板：优先用 auto_cycle，其次用 tracker 直接查
        if self.auto_cycle and self.auto_cycle.isRunning():
            report = self.auto_cycle.get_health_report()
        elif self.tracker:
            report = [
                {
                    "strategy": name,
                    "win_rate_7d": self.tracker.strategy_stats(name, days=7).get("win_rate", 0),
                    "signals_7d": self.tracker.strategy_stats(name, days=7).get("total", 0),
                    "pending_verify": False,
                    "status": "healthy",
                }
                for name in (self.tracker.all_strategies() or [])
            ]
        else:
            report = []

        for r in report:
            r["status"] = "healthy" if r.get("win_rate_7d", 0) >= 50 else "critical"

        self.health_table.setRowCount(len(report))
        for i, r in enumerate(report):
            self.health_table.setItem(i, 0, QTableWidgetItem(r.get("strategy", "")))
            wr_item = QTableWidgetItem(f"{r.get('win_rate_7d', 0):.1f}%")
            wr_color = "#00ff88" if r.get("win_rate_7d", 0) >= 50 else "#ff6666"
            wr_item.setForeground(QColor(wr_color))
            self.health_table.setItem(i, 1, wr_item)
            self.health_table.setItem(i, 2, QTableWidgetItem(str(r.get("signals_7d", 0))))
            pending_flag = r.get("pending_verify")
            if not pending_flag:
                ev = getattr(self, 'self_evolver', None)
                if ev and ev.get_status().get("pending_verifications"):
                    pending_flag = r.get("strategy") in ev.get_status()["pending_verifications"]
            self.health_table.setItem(i, 3, QTableWidgetItem("⏳" if pending_flag else ""))
            status = "✅" if r["status"] == "healthy" else "⚠️"
            self.health_table.setItem(i, 4, QTableWidgetItem(status))

        # 历史：合并 auto_cycle 和 self_evolver 两个来源
        history = []
        if self.auto_cycle:
            history.extend(self.auto_cycle.get_change_history())
        if getattr(self, 'self_evolver', None):
            history.extend(self.self_evolver.get_history(20))
        history.sort(key=lambda x: x.get("time", ""), reverse=True)

        self.history_table.setRowCount(len(history))
        for i, h in enumerate(history[:50]):
            self.history_table.setItem(i, 0, QTableWidgetItem(h.get("time", "")[-16:]))
            self.history_table.setItem(i, 1, QTableWidgetItem(h.get("strategy", "")[:12]))
            self.history_table.setItem(i, 2, QTableWidgetItem(h.get("description", "")[:50]))
            outcome = h.get("outcome", "")
            item_out = QTableWidgetItem(outcome)
            item_out.setForeground(QColor("#00ff88" if "improved" in outcome else ("#ff6666" if "roll" in outcome else "#ffaa00")))
            self.history_table.setItem(i, 3, item_out)
            delta = h.get("delta", 0)
            d_item = QTableWidgetItem(f"{delta:+.1f}%")
            d_item.setForeground(QColor("#00ff88" if delta > 0 else "#ff6666"))
            self.history_table.setItem(i, 4, d_item)

        # 更新自进化引擎状态
        evolver = getattr(self, 'self_evolver', None)
        if evolver and evolver.is_running():
            self._update_insight_display()

    # ─── Multi-Agent Trading Tab ─────────────────────────────────────

    def _build_multi_agent_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        title = QLabel("🤖🤝🤖 多智能体交易决策系统")
        title.setStyleSheet("color: #ff6b6b; font-size: 14px; font-weight: bold; padding: 4px;")
        layout.addWidget(title)

        desc = QLabel("市场分析师 + 风控分析师 + 执行代理协同决策")
        desc.setStyleSheet("color: #aaa; font-size: 11px; padding-bottom: 4px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        ctrl = QHBoxLayout()
        self.ma_toggle = QPushButton("🧠 多智能体: OFF")
        self.ma_toggle.setCheckable(True)
        self.ma_toggle.clicked.connect(self._toggle_multi_agent)
        self.ma_toggle.setMaximumHeight(40)
        self.ma_toggle.setStyleSheet(
            "QPushButton { background-color: #444; color: #ccc; font-size: 12px; font-weight: bold; "
            "padding: 6px 16px; border-radius: 4px; border: 1px solid #666; }"
            "QPushButton:checked { background-color: #cc5500; color: white; }"
        )
        ctrl.addWidget(self.ma_toggle)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.ma_log = QTextEdit()
        self.ma_log.setReadOnly(True)
        self.ma_log.setMaximumHeight(120)
        self.ma_log.setStyleSheet(
            "QTextEdit { background-color: #0a0a15; color: #ccc; border: 1px solid #333; "
            "border-radius: 4px; font-family: 'Menlo', monospace; font-size: 12px; padding: 6px; }"
        )
        layout.addWidget(self.ma_log)

        self.ma_history_table = QTableWidget()
        self.ma_history_table.setColumnCount(5)
        self.ma_history_table.setHorizontalHeaderLabels(["时间", "币种", "决策", "仓位", "理由"])
        self.ma_history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ma_history_table.setMaximumHeight(100)
        self.ma_history_table.setStyleSheet(
            "QTableWidget { background-color: #0d1117; color: #ccc; gridline-color: #333; font-size: 10px; }"
        )
        layout.addWidget(self.ma_history_table)

        layout.addWidget(self.ma_history_table)

        # 对话输入
        chat_bar = QHBoxLayout()
        self.ma_chat_input = QLineEdit()
        self.ma_chat_input.setPlaceholderText("输入指令：分析 BTC | 推荐交易 | 检查风控 | 查看持仓...")
        self.ma_chat_input.setStyleSheet(
            "QLineEdit { background:#0d1117; color:#ccc; border:1px solid #444; "
            "padding:8px; border-radius:4px; font-size:12px; }"
        )
        self.ma_chat_input.returnPressed.connect(self._send_ma_chat)
        chat_bar.addWidget(self.ma_chat_input, 1)

        send_btn = QPushButton("发送")
        send_btn.clicked.connect(self._send_ma_chat)
        send_btn.setStyleSheet(
            "QPushButton { background-color:#cc5500; color:white; font-weight:bold; "
            "padding:8px 16px; border-radius:4px; }"
            "QPushButton:hover { background-color:#ee7700; }"
        )
        chat_bar.addWidget(send_btn)
        layout.addLayout(chat_bar)

        self.main_tabs.addTab(self._wrap_in_scroll(tab), "🧠 多智能体")

    def _toggle_multi_agent(self):
        if self.ma_toggle.isChecked():
            self.ma_toggle.setText("🧠 多智能体: ON")
            self._start_multi_agent()
        else:
            self.ma_toggle.setText("🧠 多智能体: OFF")
            self._stop_multi_agent()

    def _start_multi_agent(self):
        if self.multi_agent and self.multi_agent.isRunning():
            return
        self.multi_agent = MultiAgentTrader(
            llm_client=self.llm_client,
            trade_executor=self.system_ctx.trade_executor,
            scanner=self.system_ctx.scanner,
            tracker=self.system_ctx.tracker,
            timeframe_tracker=self.system_ctx.timeframe_tracker,
            optimizer=self.system_ctx.optimizer,
        )
        self.multi_agent.log_signal.connect(self._on_ma_log)
        self.multi_agent.decision_signal.connect(self._on_ma_decision)
        self.multi_agent.start()
        self._ma_log("🧠 多智能体交易系统已启动")

    def _stop_multi_agent(self):
        if self.multi_agent:
            self.multi_agent.stop()
            self.multi_agent.wait(5000)
            self.multi_agent = None
        self._ma_log("⏹ 多智能体已停止")

    def _on_ma_log(self, msg: str, level: str):
        colors = {"INFO": "#88ccff", "SUCCESS": "#00ff88", "WARNING": "#ffaa00", "ERROR": "#ff6666"}
        ts = datetime.now().strftime("%H:%M:%S")
        self.ma_log.append(f'<span style="color:#555">[{ts}]</span> <span style="color:{colors.get(level,"#ccc")}">{msg}</span>')

    def _ma_log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.ma_log.append(f'<span style="color:#555">[{ts}]</span> {msg}')

    def _send_ma_chat(self):
        """发送指令给多智能体系统"""
        text = self.ma_chat_input.text().strip()
        if not text:
            return
        self.ma_chat_input.clear()
        self._ma_log(f'<span style="color:#ff6b6b;"><b>👤 你:</b> {text}</span>')

        # 有 LLM 则用 AI 回复，否则规则处理
        if self.llm_client:
            self._ma_chat_with_llm(text)
        else:
            self._ma_chat_rules(text)

    def _ma_chat_with_llm(self, text: str):
        """用 LLM 处理用户指令"""
        # 收集系统上下文
        ctx_parts = []
        if self.system_ctx.trade_executor:
            try:
                bal = self.system_ctx.trade_executor.get_usdt_balance()
                ctx_parts.append(f"余额: ${bal:.0f}")
            except Exception:
                pass
        if self.system_ctx.scanner:
            try:
                sigs = getattr(self.system_ctx.scanner, 'last_scan_signals', [])
                ctx_parts.append(f"扫描信号: {len(sigs)}条")
            except Exception:
                pass
        if self.system_ctx.tracker:
            try:
                strategies = self.system_ctx.tracker.all_strategies()
                s_summary = []
                for s in strategies[:3]:
                    st = self.system_ctx.tracker.strategy_stats(s, days=7)
                    s_summary.append(f"{s}({st.get('win_rate',0):.0f}%)")
                ctx_parts.append(f"策略: {', '.join(s_summary)}")
            except Exception:
                pass

        context = "; ".join(ctx_parts) if ctx_parts else "暂无系统数据"

        prompt = f"""你是加密货币多智能体交易系统的指挥中心。用户向你下达指令。

系统状态: {context}
用户指令: {text}

请回复（1-3句话，直接实质内容，不要问候语）：
- 如果是分析请求：给出你的判断
- 如果是操作指令：确认你能做什么、不能做什么
- 如果是查询：直接回答"""

        def _do():
            try:
                reply = self.llm_client.chat([
                    {"role": "system", "content": "你是交易系统AI。回复简洁直接，1-3句。"},
                    {"role": "user", "content": prompt},
                ], timeout=60)
            except Exception:
                reply = None
            QTimer.singleShot(0, lambda r=reply: self._on_ma_chat_reply(r))

        threading.Thread(target=_do, daemon=True).start()

    def _on_ma_chat_reply(self, reply: Optional[str]):
        if reply:
            self._ma_log(f'<span style="color:#ffaa00;"><b>🤖 AI:</b> {reply}</span>')
        else:
            self._ma_log('<span style="color:#ff6666;">🤖 AI: 请求失败</span>')

    def _ma_chat_rules(self, text: str):
        """无 LLM 时的规则回答"""
        text_l = text.lower()
        if "风控" in text or "风险" in text:
            bal = 0.0
            if self.system_ctx.trade_executor:
                try:
                    bal = self.system_ctx.trade_executor.get_usdt_balance()
                except Exception:
                    pass
            self._ma_log(f"🤖 系统: 当前余额 ${bal:.0f}，风控正常")
        elif "持仓" in text:
            self._ma_log("🤖 系统: 请查看主界面持仓面板")
        elif "扫描" in text or "信号" in text:
            self._ma_log("🤖 系统: 请切换到「交易对扫描」标签查看实时结果")
        elif "分析" in text:
            symbol = text.split()[-1].upper() if " " in text else "BTC"
            self._ma_log(f"🤖 系统: 请配置 LLM API 以启用 {symbol} 智能分析")
        else:
            self._ma_log("🤖 系统: 收到。需要配置 LLM API 才能进行智能对话。你可以：\n"
                         "  • 输入「风控」查看账户安全\n"
                         "  • 输入「分析 BTC」分析指定币种\n"
                           "  • 输入「推荐交易」获取交易建议")

    # ─── K线智能体 Tab ─────────────────────────────────────────────

    def _build_kline_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        title = QLabel("📊 K线 AI 交易智能体")
        title.setStyleSheet("color: #00ccff; font-size: 14px; font-weight: bold; padding: 4px;")
        layout.addWidget(title)

        desc = QLabel("读懂K线趋势/MACD/RSI/布林带 → 技术评分 → 自动交易")
        desc.setStyleSheet("color: #aaa; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("交易对:"))
        self.kl_symbol = QComboBox()
        self.kl_symbol.setEditable(True)
        self.kl_symbol.addItems(["BTC-USDT","ETH-USDT","SOL-USDT","DOGE-USDT","BNB-USDT","XRP-USDT"])
        self.kl_symbol.setStyleSheet("QComboBox{background:#0d1117;color:#ccc;border:1px solid #444;padding:4px;}")
        ctrl.addWidget(self.kl_symbol, 1)
        ctrl.addWidget(QLabel("周期:"))
        self.kl_bar = QComboBox()
        self.kl_bar.addItems(["1m","3m","5m","15m","1H","4H"])
        self.kl_bar.setCurrentText("5m")
        self.kl_bar.setStyleSheet("QComboBox{background:#0d1117;color:#ccc;border:1px solid #444;padding:4px;}")
        ctrl.addWidget(self.kl_bar)
        layout.addLayout(ctrl)

        btn_bar = QHBoxLayout()
        self.kl_analyze_btn = QPushButton("🔍 分析一次")
        self.kl_analyze_btn.clicked.connect(self._kl_analyze_once)
        self.kl_analyze_btn.setStyleSheet("QPushButton{background:#007acc;color:white;font-weight:bold;padding:6px 16px;border-radius:4px;}QPushButton:hover{background:#0099ee;}")
        btn_bar.addWidget(self.kl_analyze_btn)
        self.kl_auto_toggle = QPushButton("🤖 自动交易: OFF")
        self.kl_auto_toggle.setCheckable(True)
        self.kl_auto_toggle.clicked.connect(self._toggle_kline_trader)
        self.kl_auto_toggle.setStyleSheet("QPushButton{background:#444;color:#ccc;font-weight:bold;padding:6px 16px;border-radius:4px;border:1px solid #666;}QPushButton:checked{background:#00aa66;color:white;}")
        btn_bar.addWidget(self.kl_auto_toggle)
        btn_bar.addStretch()
        layout.addLayout(btn_bar)

        self.kl_status = QLabel("就绪")
        self.kl_status.setStyleSheet("color:#888;font-size:11px;padding:4px;")
        layout.addWidget(self.kl_status)

        self.kl_log = QTextEdit()
        self.kl_log.setReadOnly(True)
        self.kl_log.setMaximumHeight(160)
        self.kl_log.setStyleSheet("QTextEdit{background:#0a0a15;color:#ccc;border:1px solid #333;border-radius:4px;font-family:Menlo,monospace;font-size:12px;padding:6px;}")
        layout.addWidget(self.kl_log)

        self.main_tabs.addTab(self._wrap_in_scroll(tab), "📊 K线智能体")

    def _kl_analyze_once(self):
        okx = self._find_okx_client()
        if not okx:
            QMessageBox.warning(self, "提示", "请先在主界面确认 OKX 连接成功")
            return
        symbol = self.kl_symbol.currentText().strip()
        bar = self.kl_bar.currentText()
        okx = self.system_ctx.trade_executor.okx_client
        self.kl_status.setText(f"⏳ 分析 {symbol} {bar}...")
        self._kl_log(f"🔍 拉取 {symbol} {bar} K线...")

        def _do():
            try:
                resp = okx.get_history_kline(symbol, bar=bar, limit=80)
                if resp.get("code") != "0" or not resp.get("data"):
                    self._signal_kl_done.emit({"error": "K线获取失败"})
                    return
                klines = [list(row) for row in reversed(resp["data"])]
                result = KLineAnalyzer.analyze(klines)
                self._signal_kl_done.emit(result)
            except Exception as e:
                self._signal_kl_done.emit({"error": str(e)})
        threading.Thread(target=_do, daemon=True).start()

    def _on_kl_done(self, result: Dict):
        if result.get("error"):
            self._kl_error(result["error"])
        else:
            self._kl_display(self.kl_symbol.currentText(), self.kl_bar.currentText(), result)

    def _kl_error(self, msg: str):
        self.kl_status.setText(f"❌ {msg}")
        self._kl_log(f"❌ {msg}")

    def _kl_display(self, symbol, bar, result):
        if "error" in result:
            self.kl_status.setText(f"❌ {result['error']}")
            self._kl_log(f"❌ {result['error']}")
            return
        signal = result["signal"]
        sig_color = "#00ff88" if signal == "BUY" else ("#ff6666" if signal == "SELL" else "#ffaa00")
        self.kl_status.setText(f"📊 {symbol} {bar} | {signal} | 评分{result['score']:.0f}")
        html = f"""<div style="font-family:Menlo;font-size:12px;line-height:1.6;">
<p><b style="color:#00ccff;">{symbol} {bar}</b> <b style="color:{sig_color};">信号:{signal} 评分:{result['score']:.0f}</b></p>
<hr style="border-color:#333;">
<p>趋势:{result['trend']} | EMA12:{result['ema12']} EMA26:{result['ema26']} EMA50:{result['ema50']}</p>
<p>MACD:{result['macd']}({result['macd_signal']}) | RSI:{result['rsi']}({result['rsi_state']})</p>
<p>布林:上{result['bb_upper']}中{result['bb_mid']}下{result['bb_lower']} | 位置:{result['bb_position']}%宽{result['bb_width']}%</p>
<p>量:{result['volume_ratio']}x({result['volume_signal']}) | ATR:{result['atr_pct']}%</p>
<p>涨跌:1h{result['change_1h']}% 6h{result['change_6h']}% 24h{result['change_24h']}%</p>
</div>"""
        self.kl_log.setHtml(html)

    def _toggle_kline_trader(self):
        if self.kl_auto_toggle.isChecked():
            self.kl_auto_toggle.setText("🤖 自动交易: ON")
            self._start_kline_trader()
        else:
            self.kl_auto_toggle.setText("🤖 自动交易: OFF")
            self._stop_kline_trader()

    def _start_kline_trader(self):
        if self.kline_trader and self.kline_trader.isRunning():
            return
        okx = self._find_okx_client()
        if not okx:
            QMessageBox.warning(self, "提示", "请先在主界面确认 OKX 连接成功")
            self.kl_auto_toggle.setChecked(False)
            return
        self.kline_trader = KLineTrader(
            okx_client=self.system_ctx.trade_executor.okx_client,
            llm_client=self.llm_client,
            trade_executor=self.system_ctx.trade_executor,
            symbol=self.kl_symbol.currentText().strip(),
            bar=self.kl_bar.currentText(),
            interval_sec=300,
        )
        self.kline_trader.log_signal.connect(self._on_kl_log)
        self.kline_trader.start()
        self._kl_log(f"🤖 {self.kl_symbol.currentText()} 自动交易已启动")

    def _stop_kline_trader(self):
        if self.kline_trader:
            self.kline_trader.stop()
            self.kline_trader.wait(5000)
            self.kline_trader = None
        self._kl_log("⏹ 自动交易已停止")

    def _on_kl_log(self, msg: str, level: str):
        colors = {"INFO":"#88ccff","SUCCESS":"#00ff88","WARNING":"#ffaa00","ERROR":"#ff6666"}
        ts = datetime.now().strftime("%H:%M:%S")
        self.kl_log.append(f'<span style="color:#555">[{ts}]</span> <span style="color:{colors.get(level,"#ccc")}">{msg}</span>')

    def _kl_log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.kl_log.append(f'<span style="color:#555">[{ts}]</span> {msg}')

    # ─── K线实时监控 Tab ───────────────────────────────────────────

    def _build_kl_monitor_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        title = QLabel("📡 3分钟K线实时监控")
        title.setStyleSheet("color: #ff6b6b; font-size: 14px; font-weight: bold; padding: 4px;")
        layout.addWidget(title)

        desc = QLabel("监控池交易对 → 3m K线 → 剧烈变动弹窗 + 企稳买入信号")
        desc.setStyleSheet("color: #aaa; font-size: 11px; padding-bottom: 4px;")
        layout.addWidget(desc)

        ctrl = QHBoxLayout()
        self.mon_toggle = QPushButton("📡 实时监控: OFF")
        self.mon_toggle.setCheckable(True)
        self.mon_toggle.clicked.connect(self._toggle_kline_monitor)
        self.mon_toggle.setStyleSheet(
            "QPushButton{background:#444;color:#ccc;font-weight:bold;padding:6px 16px;border-radius:4px;border:1px solid #666;}"
            "QPushButton:checked{background:#cc3300;color:white;}"
        )
        ctrl.addWidget(self.mon_toggle)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.mon_log = QTextEdit()
        self.mon_log.setReadOnly(True)
        self.mon_log.setStyleSheet(
            "QTextEdit{background:#0a0a15;color:#ccc;border:1px solid #333;"
            "border-radius:4px;font-family:Menlo,monospace;font-size:12px;padding:6px;}"
        )
        layout.addWidget(self.mon_log, 1)

        self.main_tabs.addTab(self._wrap_in_scroll(tab), "📡 实时监控")

    def _toggle_kline_monitor(self):
        if self.mon_toggle.isChecked():
            self.mon_toggle.setText("📡 实时监控: ON")
            self._start_kline_monitor()
        else:
            self.mon_toggle.setText("📡 实时监控: OFF")
            self._stop_kline_monitor()

    def _start_kline_monitor(self):
        if self.kline_monitor and self.kline_monitor.isRunning():
            return
        okx = self._find_okx_client()
        if not okx:
            QMessageBox.warning(self, "提示", "请先在主界面确认 OKX 连接成功")
            self.mon_toggle.setChecked(False)
            return
        self.kline_monitor = KLineMonitor(
            okx_client=okx,
            monitor_pool=self.system_ctx.monitor_pool,
        )
        self.kline_monitor.alert_signal.connect(self._on_mon_alert)
        self.kline_monitor.buy_signal.connect(self._on_mon_buy)
        self.kline_monitor.start()
        self._mon_log("📡 3m K线实时监控已启动")

    def _find_okx_client(self):
        """多路径查找 OKX 客户端"""
        # 1. system_ctx
        executor = self.system_ctx.trade_executor
        okx = getattr(executor, 'okx_client', None) if executor else None
        if okx:
            return okx
        # 2. 主窗口
        main_win = self.window()
        if main_win:
            okx = getattr(main_win, 'okx_client', None)
            if okx:
                return okx
            exec_ = getattr(main_win, 'trade_executor', None)
            if exec_:
                okx = getattr(exec_, 'okx_client', None)
                if okx:
                    return okx
        # 3. 全局单例: 从 main 函数创建的 OKXClient
        try:
            from src.api.okx_client import OKXClient
            from src.utils.env_config import get_okx_config
            cfg = get_okx_config()
            if cfg:
                return OKXClient(
                    api_key=cfg.get("api_key", ""),
                    secret_key=cfg.get("secret_key", ""),
                    passphrase=cfg.get("passphrase", ""),
                    is_demo=cfg.get("demo", True),
                )
        except Exception:
            pass
        return None

    def _stop_kline_monitor(self):
        if self.kline_monitor:
            self.kline_monitor.stop()
            self.kline_monitor.wait(3000)
            self.kline_monitor = None
        self._mon_log("⏹ 监控已停止")

    def _on_mon_alert(self, symbol: str, alert_type: str, message: str):
        ts = datetime.now().strftime("%H:%M:%S")
        if alert_type == "violent":
            color = "#ff4444"
            self._mon_log(f'<span style="color:#555">[{ts}]</span> <span style="color:{color};">'
                          f'⚠️ {symbol} 剧烈变动: {message}</span>')
        elif alert_type == "diverge":
            color = "#ffaa00"
            self._mon_log(f'<span style="color:#555">[{ts}]</span> <span style="color:{color};">'
                          f'⚡ {symbol} 变盘预警: {message}</span>')
        elif alert_type == "buy_ready":
            color = "#00ff88"
            self._mon_log(f'<span style="color:#555">[{ts}]</span> <span style="color:{color};">'
                          f'🎯 {symbol} 企稳买入: {message}</span>')

        # 弹窗提醒
        popup = QMessageBox()
        popup.setWindowTitle(f"{symbol} 告警")
        popup.setIcon(QMessageBox.Warning if alert_type != "buy_ready" else QMessageBox.Information)
        popup.setText(message)
        popup.exec()

    def _on_mon_buy(self, symbol: str, price: float, reason: str):
        self._mon_log(f'<span style="color:#00ff88;">💡 买入机会: {symbol} @ {price} - {reason}</span>')

    def _mon_log(self, msg: str):
        self.mon_log.append(msg)

    # ─── Config ──────────────────────────────────────────────────────

    def _on_config_changed(self):
        self._config["base_url"] = self.url_edit.text().strip()
        self._config["api_key"] = self.key_edit.text().strip()
        self._config["model"] = self.model_combo.currentText().strip()
        self._config["temperature"] = self.temp_spin.value()
        self._config["max_tokens"] = self.max_tokens_spin.value()
        self._save_config()

    def _apply_preset(self, url: str, model: str):
        """一键填入预设 API 配置"""
        self.url_edit.setText(url)
        self.model_combo.setCurrentText(model)
        self._on_config_changed()

    def _get_or_create_client(self) -> bool:
        """获取或创建 LLM 客户端，返回是否成功"""
        if not self._config.get("api_key"):
            QMessageBox.warning(self, "提示", "请先填入 API Key")
            return False
        self._on_config_changed()
        self.llm_client = LLMClient(
            base_url=self._config["base_url"],
            api_key=self._config["api_key"],
            model=self._config["model"],
            temperature=self._config["temperature"],
            max_tokens=self._config["max_tokens"],
        )
        return True

    def _auto_test_connection(self):
        """启动时自动测试已有 API 连接（静默，不弹窗）"""
        if not self._config.get("api_key", "").strip():
            return
        self._on_config_changed()
        client = LLMClient(
            base_url=self._config["base_url"],
            api_key=self._config["api_key"],
            model=self._config["model"],
            temperature=self._config["temperature"],
            max_tokens=self._config["max_tokens"],
        )

        def _do():
            try:
                ok = client.test_connection()
                err = client.last_error
            except Exception as e:
                ok = False
                err = str(e)
            self._signal_connection_tested.emit(ok, err)

        threading.Thread(target=_do, daemon=True).start()

    # ─── Busy 状态管理（带超时保护）────────────────────────────────

    def _set_busy(self, status: str):
        import time
        self._busy = True
        self._busy_since = time.time()
        self.analyze_btn.setEnabled(False)
        self.review_btn.setEnabled(False)
        self.connect_btn.setEnabled(False)
        self.status_label.setText(status)

    def _clear_busy(self, status: str = ""):
        self._busy = False
        self._busy_since = 0.0
        self.analyze_btn.setEnabled(True)
        self.review_btn.setEnabled(True)
        self.connect_btn.setEnabled(True)
        self.analysis_progress.setVisible(False)
        if status:
            self.status_label.setText(status)

    def _is_busy_ok(self) -> bool:
        """返回 False 表示忙碌中，带 120s 超时自动重置"""
        if not self._busy:
            return True
        import time
        if time.time() - self._busy_since > 120:
            self._clear_busy("⚠️ 上次操作超时，请重试")
            return True
        return False

    # ─── API 连接 ───────────────────────────────────────────────────

    def _test_connection(self):
        if not self._is_busy_ok():
            QMessageBox.warning(self, "提示", "上一个操作仍在进行中…")
            return
        if not self._get_or_create_client():
            return
        if not self.llm_client:
            return
        self._set_busy("⏳ 测试连接中...")

        client = self.llm_client

        def _do():
            try:
                # 使用详细诊断模式
                detail = client.verbose_test()
                ok = "✅" in detail
                err = detail if not ok else ""
            except Exception as e:
                ok = False
                err = f"异常: {e}"
            self._signal_connection_tested.emit(ok, err)

        threading.Thread(target=_do, daemon=True).start()

    def _on_connection_tested(self, ok: bool, error: str):
        base = self._config.get("base_url", "")
        model = self._config.get("model", "")
        url = base.rstrip("/") + "/chat/completions"

        if ok:
            self.config_status.setText("✅ 已连接")
            self.config_status.setStyleSheet("color: #00ff88; font-size: 11px;")
            self.llm_client = LLMClient(
                base_url=self._config["base_url"],
                api_key=self._config["api_key"],
                model=self._config["model"],
                temperature=self._config["temperature"],
                max_tokens=self._config["max_tokens"],
            )
            self.connect_btn.setStyleSheet(
                "QPushButton { background-color: #059669; color: white; font-weight: bold; "
                "padding: 5px 14px; border-radius: 4px; }"
                "QPushButton:hover { background-color: #34D399; }"
            )
            if self.tracker and self.tracker._signals:
                self.advisor = StrategyAdvisor(
                    self.llm_client, self.tracker, self.mutator,
                    self.timeframe_tracker, self.optimizer,
                )
        else:
            # 自动测试失败只更新状态栏和按钮，不弹窗
            if self._busy:
                detail = (
                    f"连接失败，诊断信息：\n\n{error}\n\n"
                    f"---\n请求地址: {url}\n模型: {model}\n\n"
                    f"常见原因:\n"
                    f"1. API Key 未填入或格式错误\n"
                    f"2. 未订阅服务 → https://opencode.ai/auth\n"
                    f"3. 模型名不匹配\n4. 网络不通或需代理"
                )
                self._clear_busy()
                QMessageBox.critical(self, "连接失败", detail[:800])
            self.config_status.setText("❌ 失败")
            self.config_status.setStyleSheet("color: #ff6666; font-size: 11px;")
            self.connect_btn.setStyleSheet(
                "QPushButton { background-color: #cc3333; color: white; font-weight: bold; "
                "padding: 5px 14px; border-radius: 4px; }"
                "QPushButton:hover { background-color: #dd4444; }"
            )

    # ─── Chat ───────────────────────────────────────────────────────

    def _send_chat(self):
        text = self.chat_input.toPlainText().strip()
        if not text:
            return
        if not self._is_busy_ok():
            QMessageBox.warning(self, "提示", "上一个操作仍在进行中…")
            return
        if not self.llm_client:
            ok = self._get_or_create_client()
            if not ok or not self.llm_client:
                return

        self.chat_input.clear()
        self.chat_display.append(
            f'<p style="color:#8B5CF6;"><b>👤 你:</b> {text}</p>'
        )

        messages = [{"role": "system", "content": "你是一个专业的量化交易助手，精通加密货币市场分析、Python策略开发、特征工程。回答要精确、专业。答案要基于数据逻辑而不是猜测。"}]
        messages.extend(self.conversation[-10:])
        messages.append({"role": "user", "content": text})

        self._set_busy("⏳ AI 思考中...")

        client = self.llm_client

        def _do():
            try:
                reply = client.chat(messages)
            except Exception:
                reply = None
            self._signal_chat_reply.emit(text, reply or "")

        threading.Thread(target=_do, daemon=True).start()

    def _on_chat_reply(self, question: str, reply: str):
        if reply:
            self._clear_busy("✅ 回复完成")
            self.conversation.append({"role": "user", "content": question})
            self.conversation.append({"role": "assistant", "content": reply})
            self._save_conversation()
            display = reply.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            self.chat_display.append(
                f'<div style="background:#1a1035; padding:8px; border-radius:6px; margin:4px 0;">'
                f'<p style="color:#A78BFA;"><b>🤖 AI:</b></p>'
                f'<p style="color:#ccc;">{display}</p></div>'
            )
        else:
            self._clear_busy(f"❌ {self.llm_client.last_error}")

    # ─── Analysis ───────────────────────────────────────────────────

    def _analyze_strategy(self):
        if not self._is_busy_ok():
            QMessageBox.warning(self, "提示", "上一个操作仍在进行中…")
            return
        strategy = self.strategy_combo.currentText()
        if not strategy:
            QMessageBox.warning(self, "提示", "请先选择策略（切换到其他页面再切回来刷新列表）")
            return
        if not self.llm_client:
            ok = self._get_or_create_client()
            if not ok or not self.llm_client:
                QMessageBox.warning(self, "提示", "请先在「API配置」页填入密钥并测试连接")
                return
        if not self.advisor:
            self.advisor = StrategyAdvisor(
                self.llm_client, self.tracker, self.mutator,
                self.timeframe_tracker, self.optimizer,
            )

        self._set_busy(f"⏳ AI 分析 {strategy} 中...")
        self.analysis_progress.setVisible(True)

        # 先显示占位
        self.analysis_output.setHtml(
            '<p style="color:#8B5CF6;">⏳ AI 分析中，请稍候…</p>'
        )

        advisor = self.advisor
        client = self.llm_client

        # 构建 prompt（复用 advisor 的逻辑）
        stats = self.tracker.strategy_stats(strategy, days=30)
        raw_code = advisor._load_strategy_code(strategy) or "# 未找到"
        code = advisor._truncate_code(raw_code, max_lines=120)
        code = code.replace("{", "{{").replace("}", "}}")
        signals = self.tracker.strategy_recent_signals(strategy, limit=5)
        signals_brief = [{
            "time": s.get("datetime", "")[-16:],
            "symbol": s.get("symbol", ""),
            "dir": s.get("direction", ""),
            "score": round(float(s.get("score", 0) or 0), 1),
            "validations": {
                k[:2] + "h": v for k, v in s.get("validations", {}).items()
                if isinstance(v, dict)
            } if s.get("validations") else {},
        } for s in signals]
        from src.ai_agent.prompt_templates import ANALYSIS_PROMPT, SYSTEM_PROMPT

        prompt = ANALYSIS_PROMPT.format(
            strategy_name=strategy, days=30,
            total_signals=stats.get("total", 0),
            win_rate=stats.get("win_rate", 0),
            profit_factor=stats.get("profit_factor", 0),
            net_pnl=stats.get("net_pnl", 0),
            signal_samples=json.dumps(signals_brief, indent=2, ensure_ascii=False, default=str),
            timeframe_accuracy=json.dumps(
                self.timeframe_tracker.accuracy_by_timeframe(strategy, 30),
                indent=2, ensure_ascii=False,
            ),
            param_history=json.dumps(advisor._format_param_history(strategy), indent=2, ensure_ascii=False),
            strategy_code=code,
        )

        def _do():
            try:
                # 流式请求
                full = []
                token_stream = self._signal_stream_token

                def on_token(t):
                    full.append(t)
                    # 批量发送，减少信号频率
                    if len(full) % 5 == 0 or "\n" in t:
                        pass
                    token_stream.emit(t)

                reply = client.chat(
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": prompt}],
                    stream=True,
                    on_token=on_token,
                    timeout=180,
                )
                if reply:
                    result = {
                        "strategy_name": strategy,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "stats": stats,
                        "advice": reply,
                    }
                else:
                    result = {"error": client.last_error or "LLM 请求失败"}
            except Exception as e:
                import traceback
                result = {"error": f"分析异常: {e}\n{traceback.format_exc()[-300:]}"}
            self._signal_analysis_done.emit(result)

        threading.Thread(target=_do, daemon=True).start()

    def _on_stream_token(self, token: str):
        """流式输出：每收到 token 追加到分析窗口"""
        cursor = self.analysis_output.textCursor()
        cursor.movePosition(cursor.End)
        safe = token.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        safe = safe.replace("\n", "<br>")
        cursor.insertHtml(
            f'<span style="color:#ccc; font-family:Menlo,monospace; font-size:11px;">{safe}</span>'
        )
        self.analysis_output.ensureCursorVisible()

    def _on_analysis_done(self, result: Optional[Dict]):
        self._clear_busy()

        if result and result.get("error"):
            self.analysis_output.setHtml(
                f'<p style="color:#ff6666;">❌ {result["error"]}</p>'
            )
            self.status_label.setText("❌ 分析失败")
            self.diff_preview.setHtml("")
            self.apply_btn.setEnabled(False)
            return

        if not result or not result.get("advice"):
            self.analysis_output.setHtml(
                '<p style="color:#ff6666;">❌ LLM 无响应</p>'
            )
            self.status_label.setText("❌ 无响应")
            return

        advice = result.get("advice", "")
        strategy_name = result.get("strategy_name", "")

        # 在流式内容后添加分隔
        self.analysis_output.append(
            f'\n<hr style="border-color:#333;"><p style="color:#059669;">✅ AI 分析完成</p>'
        )

        # 解析代码变更
        if self.advisor and self.mutator:
            fname = self.advisor._find_strategy_file(strategy_name)
            if fname:
                changes = CodeDiffParser.parse_analysis(advice, target_file=fname)
                self._pending_changes = changes
                self._pending_file = fname
                self._pending_file_name = strategy_name

                if changes:
                    self._show_diff_preview(changes)
                    if self.auto_apply_check.isChecked():
                        self._apply_suggested_changes()
                else:
                    self.diff_preview.setHtml(
                        '<p style="color:#ffaa00;">⚠️ 分析完成，但未提取到可用的代码修改建议。'
                        '请查看上方分析结果手动调整。</p>'
                    )
                    self.apply_btn.setEnabled(False)
            else:
                self.diff_preview.setHtml(
                    f'<p style="color:#ffaa00;">⚠️ 未找到策略文件 {strategy_name}</p>'
                )
                self.apply_btn.setEnabled(False)

        self.status_label.setText(f"✅ {strategy_name} 分析完成")

    def _review_code(self):
        """直接用优化器最优参数修改代码，不走 LLM"""
        if not self._is_busy_ok():
            QMessageBox.warning(self, "提示", "上一个操作仍在进行中…")
            return
        strategy = self.strategy_combo.currentText()
        if not strategy:
            QMessageBox.warning(self, "提示", "请先选择策略")
            return
        if not self.optimizer:
            self.analysis_output.setHtml('<p style="color:#ffaa00;">⚠️ 优化器未连接</p>')
            return
        if not self.advisor:
            if not self.llm_client:
                self._get_or_create_client()
            if not self.advisor:
                self.advisor = StrategyAdvisor(
                    self.llm_client, self.tracker, self.mutator,
                    self.timeframe_tracker, self.optimizer,
                )

        fname = self.advisor._find_strategy_file(strategy)
        if not fname:
            self.analysis_output.setHtml(f'<p style="color:#ff6666;">❌ 未找到策略文件</p>')
            self.status_label.setText("❌ 未找到文件")
            return

        code = self.advisor._load_strategy_code(strategy)
        if not code:
            self.analysis_output.setHtml(f'<p style="color:#ff6666;">❌ 无法读取策略文件</p>')
            return

        opt_params = self.optimizer.get_best_params(strategy, use_exploration=False)
        current_params = self.optimizer.get_optimized_params(strategy)

        # 构建代码变更：每个优化参数生成一个 CodeChange
        changes: List[CodeChange] = []
        for key, target_val in opt_params.items():
            cur_val = ""
            if isinstance(current_params.get(key), dict):
                cur_val = str(current_params[key].get("value", ""))
            else:
                cur_val = str(current_params.get(key, ""))

            target_str = str(round(float(target_val), 4))
            cur_str = str(round(float(cur_val), 4)) if cur_val and cur_val != "None" else ""

            # 从源码中提取这个参数的实际值
            for pattern in [
                rf'{re.escape(key)}\s*=\s*([\d.]+)',
                rf'["\']?{re.escape(key)}["\']?\s*:\s*([\d.]+)',
            ]:
                m = re.search(pattern, code)
                if m:
                    actual_cur = m.group(1)
                    changes.append(CodeChange(
                        description=f"{key}: {actual_cur} → {target_str}",
                        file_path=fname,
                        old_code=f"{key} = {actual_cur}",
                        new_code=f"{key} = {target_str}",
                        change_type="param",
                        reason="优化器 Thompson Sampling 后验分析：该参数值在历史数据中胜率/盈亏比最高",
                    ))
                    break
            else:
                # 没找到精确匹配，用代码中的关键词搜索
                self.status_label.setText(f"⚠️ 参数 {key} 未在源码中找到")

        if not changes:
            self.analysis_output.setHtml(
                '<p style="color:#ffaa00;">⚠️ 优化器参数与源码一致，无需修改。</p>'
            )
            self.status_label.setText("✅ 已是最优")
            return

        self._pending_changes = changes
        self._pending_file = fname
        self._pending_file_name = strategy
        self._show_diff_preview(changes)

        self.analysis_output.setHtml(
            f"<div style='background:#111; padding:12px; border-radius:6px;'>"
            f"<h3 style='color:#059669;'>📝 优化器参数 → 代码变更 ({len(changes)} 项)</h3>"
            f"<p style='color:#ccc;'>{strategy} 的优化器最优参数已解析为代码修改。"
            f"请查看下方 diff 预览，确认后点击「✅ 应用到策略文件」。</p>"
            f"</div>"
        )
        self.status_label.setText(f"✅ {len(changes)} 项优化建议已就绪")

    # ─── Code Apply / Rollback ───────────────────────────────────────

    def _show_diff_preview(self, changes: List[CodeChange]):
        """在 diff 预览区展示代码修改建议（含理由/预期提升）"""
        html_lines = ['<div style="font-family:Menlo,monospace; font-size:10px; line-height:1.4;">']
        for c in changes:
            color = "#ffd700" if c.change_type == "param" else "#88ccff"
            # 修改项标题
            html_lines.append(
                f'<p style="color:{color}; margin:6px 0 2px; font-size:11px;"><b>{c.description}</b></p>'
            )
            # 理由
            if c.reason:
                html_lines.append(
                    f'<p style="color:#a0e0ff; margin:1px 0; padding-left:12px;">📊 理由: {c.reason[:200]}</p>'
                )
            # 预期提升
            if c.expected_improvement:
                html_lines.append(
                    f'<p style="color:#80ffaa; margin:1px 0; padding-left:12px;">📈 预期: {c.expected_improvement[:200]}</p>'
                )
            # 代码 diff
            html_lines.append(
                f'<p style="color:#ff6666; margin:1px 0; padding-left:12px;">- {c.old_code[:120]}</p>'
            )
            html_lines.append(
                f'<p style="color:#00ff88; margin:1px 0; padding-left:12px;">+ {c.new_code[:120]}</p>'
            )
            html_lines.append('<hr style="border-color:#2a2a3a; margin:3px 0;">')
        html_lines.append('</div>')
        self.diff_preview.setHtml("".join(html_lines))
        self.apply_btn.setEnabled(True)
        self.rollback_btn.setEnabled(True)

    def _apply_suggested_changes(self):
        """将 AI 建议的代码修改写入策略文件"""
        if not self._pending_changes:
            QMessageBox.warning(self, "提示", "没有待应用的修改建议。请先执行 AI 分析。")
            return
        if not self._pending_file:
            QMessageBox.warning(self, "提示", "未确定目标策略文件。")
            return

        if not self.code_applier:
            self.code_applier = SafeCodeApplier()

        reply = QMessageBox.question(
            self, "确认应用修改",
            f"将 {len(self._pending_changes)} 项代码修改写入\n{self._pending_file}\n\n"
            f"修改摘要:\n" + "\n".join(
                f"  • {c.description}" for c in self._pending_changes[:10]
            ) + "\n\n系统已自动备份原文件，确认继续？",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        success, msg, applied = self.code_applier.apply_changes(
            self._pending_file, self._pending_changes
        )

        if success:
            self.status_label.setText(f"✅ {msg}")
            self.apply_btn.setStyleSheet(
                "QPushButton { background-color: #059669; color: white; font-weight: bold; "
                "padding: 6px 18px; border-radius: 4px; }"
            )
            QMessageBox.information(self, "应用成功", msg)
            # 刷新 mutator 的变异点缓存
            if self.mutator:
                self.mutator._load_mutation_points()
        else:
            self.status_label.setText(f"❌ 应用失败")
            QMessageBox.critical(self, "应用失败", msg)

    def _rollback_last(self):
        """回退到最近的 AI 备份"""
        if not self._pending_file:
            QMessageBox.warning(self, "提示", "没有可回退的策略文件。")
            return

        if not self.code_applier:
            self.code_applier = SafeCodeApplier()

        reply = QMessageBox.question(
            self, "确认回退",
            f"回退 {self._pending_file} 到最近的 AI 备份？\n当前修改将被撤销。",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        success, msg = self.code_applier.rollback(self._pending_file)
        if success:
            self.status_label.setText(f"✅ {msg}")
            if self.mutator:
                self.mutator._load_mutation_points()
            QMessageBox.information(self, "回退成功", msg)
        else:
            QMessageBox.critical(self, "回退失败", msg)

    def _on_tab_changed(self, index: int):
        """切换标签页时刷新策略列表"""
        tab_title = self.main_tabs.tabText(index)
        if "漏洞扫描" in tab_title and self.tracker:
            existing = {self.vs_strategy_combo.itemText(i)
                        for i in range(self.vs_strategy_combo.count())}
            for s in sorted(self.tracker.all_strategies() or []):
                if s not in existing:
                    self.vs_strategy_combo.addItem(s)

    # ─── Risk Guard Tab ─────────────────────────────────────────────

    def _build_risk_guard_tab(self):
        """🛡️ 风控守卫标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 控制行
        ctrl = QHBoxLayout()
        self.rg_toggle = QPushButton("🛡️ 风控守卫: OFF")
        self.rg_toggle.setCheckable(True)
        self.rg_toggle.clicked.connect(self._toggle_risk_guard)
        self.rg_toggle.setStyleSheet(
            "QPushButton { background-color: #3a1a00; color: #ff9900; font-size: 13px; font-weight: bold; "
            "padding: 8px 20px; border-radius: 6px; border: 2px solid #995500; }"
            "QPushButton:checked { background-color: #994400; color: #ffcc00; border-color: #ffaa00; }"
        )
        ctrl.addWidget(self.rg_toggle)

        self.rg_run_once = QPushButton("⚡ 立即检查")
        self.rg_run_once.clicked.connect(self._risk_run_once)
        self.rg_run_once.setStyleSheet(
            "QPushButton { background-color: #1a1a2e; color: #ffa500; padding: 6px 14px; "
            "border-radius: 4px; border: 1px solid #885500; font-size: 11px; }"
            "QPushButton:hover { background-color: #2a2a3e; }"
        )
        ctrl.addWidget(self.rg_run_once)

        self.rg_status = QLabel("守卫未启动")
        self.rg_status.setStyleSheet("color: #888; font-size: 11px; padding-left: 10px;")
        ctrl.addWidget(self.rg_status, 1)
        layout.addLayout(ctrl)

        # 实时指标面板
        metrics_group = QGroupBox("📊 实时指标")
        metrics_group.setStyleSheet(
            "QGroupBox { border: 1px solid #664400; border-radius: 6px; margin-top: 6px; "
            "padding: 10px; font-weight: bold; color: #ffaa00; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
        )
        mg_layout = QHBoxLayout(metrics_group)

        self.rg_metric_labels: Dict[str, QLabel] = {}
        for key, display in [
            ("total_positions", "持仓数"),
            ("total_pnl_pct", "总盈亏%"),
            ("max_leverage", "最高杠杆"),
            ("balance", "余额USDT"),
            ("api_latency_ms", "API延迟ms"),
            ("memory_mb", "内存MB"),
        ]:
            vbox = QVBoxLayout()
            name_lbl = QLabel(display)
            name_lbl.setStyleSheet("color: #888; font-size: 10px;")
            name_lbl.setAlignment(Qt.AlignCenter)
            val_lbl = QLabel("—")
            val_lbl.setStyleSheet("color: #ffcc00; font-size: 14px; font-weight: bold;")
            val_lbl.setAlignment(Qt.AlignCenter)
            vbox.addWidget(name_lbl)
            vbox.addWidget(val_lbl)
            mg_layout.addLayout(vbox)
            self.rg_metric_labels[key] = val_lbl

        layout.addWidget(metrics_group)

        # 阈值配置（快捷）
        thresh_group = QGroupBox("⚙️ 风险阈值配置")
        thresh_group.setStyleSheet(
            "QGroupBox { border: 1px solid #444; border-radius: 6px; margin-top: 6px; "
            "padding: 8px; font-weight: bold; color: #aaa; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
        )
        from PySide6.QtWidgets import QFormLayout as _FL
        thresh_form = _FL(thresh_group)
        thresh_form.setSpacing(6)

        self.rg_max_loss = QDoubleSpinBox()
        self.rg_max_loss.setRange(-50, 0)
        self.rg_max_loss.setValue(-8.0)
        self.rg_max_loss.setSuffix(" %")
        self.rg_max_loss.setStyleSheet("QDoubleSpinBox { background:#0d1117; color:#ccc; border:1px solid #444; padding:4px; }")
        thresh_form.addRow("单仓最大亏损:", self.rg_max_loss)

        self.rg_max_leverage = QDoubleSpinBox()
        self.rg_max_leverage.setRange(1, 50)
        self.rg_max_leverage.setValue(15.0)
        self.rg_max_leverage.setSuffix(" x")
        self.rg_max_leverage.setStyleSheet("QDoubleSpinBox { background:#0d1117; color:#ccc; border:1px solid #444; padding:4px; }")
        thresh_form.addRow("杠杆警告阈值:", self.rg_max_leverage)

        self.rg_min_winrate = QDoubleSpinBox()
        self.rg_min_winrate.setRange(0, 100)
        self.rg_min_winrate.setValue(40.0)
        self.rg_min_winrate.setSuffix(" %")
        self.rg_min_winrate.setStyleSheet("QDoubleSpinBox { background:#0d1117; color:#ccc; border:1px solid #444; padding:4px; }")
        thresh_form.addRow("策略胜率警告线:", self.rg_min_winrate)

        save_thresh_btn = QPushButton("💾 应用配置")
        save_thresh_btn.clicked.connect(self._apply_risk_config)
        save_thresh_btn.setStyleSheet(
            "QPushButton { background:#664400; color:#ffcc00; padding:4px 12px; border-radius:4px; font-size:11px; }"
            "QPushButton:hover { background:#885500; }"
        )
        thresh_form.addRow("", save_thresh_btn)
        layout.addWidget(thresh_group)

        # 预警历史
        hist_group = QGroupBox("🚨 预警历史记录")
        hist_group.setStyleSheet(
            "QGroupBox { border: 1px solid #333; border-radius: 6px; margin-top: 6px; "
            "padding: 8px; font-weight: bold; color: #ff6666; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
        )
        hist_v = QVBoxLayout(hist_group)
        self.alert_history_widget = AlertHistoryWidget()
        hist_v.addWidget(self.alert_history_widget)
        layout.addWidget(hist_group, 1)

        self.main_tabs.addTab(self._wrap_in_scroll(tab), "🛡️ 风控守卫")

    def _toggle_risk_guard(self):
        if self.rg_toggle.isChecked():
            self._start_risk_guard()
        else:
            self._stop_risk_guard()

    def _start_risk_guard(self):
        if self.risk_guard and self.risk_guard.isRunning():
            return

        # 获取 exchange client
        exchange = getattr(self, 'exchange', None) or getattr(self, '_exchange', None)
        if not exchange:
            # 尝试从其他页面组件拿
            parent = self.parent()
            while parent:
                for attr in ('exchange', '_exchange', 'exchange_client'):
                    if hasattr(parent, attr):
                        exchange = getattr(parent, attr)
                        break
                if exchange:
                    break
                parent = parent.parent() if hasattr(parent, 'parent') else None

        from src.ai_agent.risk_guard import DEFAULT_CONFIG
        cfg = dict(DEFAULT_CONFIG)
        cfg["max_single_loss_pct"] = self.rg_max_loss.value()
        cfg["max_leverage"] = self.rg_max_leverage.value()
        cfg["min_win_rate_warn"] = self.rg_min_winrate.value()

        self.risk_guard = RiskGuard(
            okx_client=exchange,
            tracker=self.tracker,
            config=cfg,
        )
        self.risk_guard.alert_signal.connect(lambda a: self._signal_risk_alert.emit(a))
        self.risk_guard.metrics_signal.connect(lambda m: self._signal_risk_metrics.emit(m))
        self.risk_guard.start()

        self.rg_toggle.setText("🛡️ 风控守卫: ON")
        self.rg_status.setText("🟢 运行中（每30秒检查）")
        self.rg_status.setStyleSheet("color: #00ff88; font-size: 11px; padding-left: 10px;")

    def _stop_risk_guard(self):
        if self.risk_guard:
            self.risk_guard.stop()
            self.risk_guard.wait(1000)
            self.risk_guard = None
        self.rg_toggle.setText("🛡️ 风控守卫: OFF")
        self.rg_status.setText("守卫已停止")
        self.rg_status.setStyleSheet("color: #888; font-size: 11px; padding-left: 10px;")

    def _risk_run_once(self):
        """手动触发一次风控检查"""
        if not self.risk_guard:
            self._start_risk_guard()
            if not self.risk_guard:
                return
        def _do():
            alerts = self.risk_guard.run_once()
            for a in alerts:
                self._signal_risk_alert.emit(a)
        threading.Thread(target=_do, daemon=True).start()

    def _apply_risk_config(self):
        if not self.risk_guard:
            return
        self.risk_guard.update_config({
            "max_single_loss_pct": self.rg_max_loss.value(),
            "max_leverage": self.rg_max_leverage.value(),
            "min_win_rate_warn": self.rg_min_winrate.value(),
        })
        self.rg_status.setText("✅ 配置已更新")

    def _on_risk_alert(self, alert: RiskAlert):
        """收到风控预警：更新历史面板 + 弹出浮窗"""
        self.alert_history_widget.add_alert(alert)
        # 弹出非阻塞预警窗
        try:
            overlay = get_alert_overlay()
            overlay.push_alert(alert)
        except Exception:
            pass

    def _on_risk_metrics(self, metrics: dict):
        """更新实时指标展示"""
        fmt_map = {
            "total_positions": lambda v: str(int(v)),
            "total_pnl_pct": lambda v: f"{v:+.2f}%",
            "max_leverage": lambda v: f"{v:.1f}x",
            "balance": lambda v: f"{v:,.0f}",
            "api_latency_ms": lambda v: f"{v:.0f}",
            "memory_mb": lambda v: f"{v:.0f}",
        }
        for key, lbl in self.rg_metric_labels.items():
            val = metrics.get(key)
            if val is None:
                continue
            try:
                text = fmt_map[key](val)
            except Exception:
                text = str(val)
            lbl.setText(text)

            # 颜色：根据危险程度
            if key == "total_pnl_pct":
                lbl.setStyleSheet(f"color: {'#00ff88' if val >= 0 else '#ff4444'}; font-size:14px; font-weight:bold;")
            elif key == "max_leverage" and val > 15:
                lbl.setStyleSheet("color: #ff4444; font-size:14px; font-weight:bold;")
            elif key == "api_latency_ms" and val > 2000:
                lbl.setStyleSheet("color: #ffaa00; font-size:14px; font-weight:bold;")
            else:
                lbl.setStyleSheet("color: #ffcc00; font-size:14px; font-weight:bold;")

    # ─── Vulnerability Scan Tab ─────────────────────────────────────

    def _build_vuln_scan_tab(self):
        """🔬 漏洞扫描标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 控制行
        ctrl = QHBoxLayout()

        self.vs_strategy_combo = QComboBox()
        self.vs_strategy_combo.setEditable(False)
        self.vs_strategy_combo.addItem("全部策略文件")
        if self.tracker:
            for s in sorted(self.tracker.all_strategies() or []):
                self.vs_strategy_combo.addItem(s)
        self.vs_strategy_combo.setStyleSheet(
            "QComboBox { background:#0d1117; color:#ccc; border:1px solid #444; padding:4px 8px; border-radius:4px; }"
        )
        ctrl.addWidget(QLabel("目标:"))
        ctrl.addWidget(self.vs_strategy_combo, 1)

        self.vs_llm_check = QCheckBox("启用 LLM 深度审查")
        self.vs_llm_check.setStyleSheet("QCheckBox { color: #ccc; font-size: 11px; }")
        ctrl.addWidget(self.vs_llm_check)

        self.vs_scan_btn = QPushButton("🔬 开始扫描")
        self.vs_scan_btn.clicked.connect(self._start_vuln_scan)
        self.vs_scan_btn.setStyleSheet(
            "QPushButton { background-color: #1a3a20; color: #00ff88; font-weight: bold; "
            "padding: 6px 16px; border-radius: 4px; border: 1px solid #00aa44; }"
            "QPushButton:hover { background-color: #224422; }"
        )
        ctrl.addWidget(self.vs_scan_btn)
        layout.addLayout(ctrl)

        # 扫描状态
        self.vs_status = QLabel("就绪，选择目标后点击「开始扫描」")
        self.vs_status.setStyleSheet("color: #888; font-size: 11px; padding: 4px;")
        layout.addWidget(self.vs_status)

        self.vs_progress = QProgressBar()
        self.vs_progress.setRange(0, 0)
        self.vs_progress.setVisible(False)
        self.vs_progress.setStyleSheet(
            "QProgressBar { background:#1a1a2e; border:1px solid #333; border-radius:3px; height:6px; }"
            "QProgressBar::chunk { background:#00aa44; border-radius:3px; }"
        )
        layout.addWidget(self.vs_progress)

        # 统计摘要行
        summary_h = QHBoxLayout()
        self.vs_summary_labels: Dict[str, QLabel] = {}
        for key, label, color in [
            ("CRITICAL", "严重", "#ff4444"),
            ("HIGH", "高危", "#ff8800"),
            ("MEDIUM", "中危", "#ffcc00"),
            ("LOW", "低危", "#aaaaaa"),
        ]:
            vbox = QVBoxLayout()
            cnt_lbl = QLabel("0")
            cnt_lbl.setStyleSheet(f"color: {color}; font-size: 22px; font-weight: bold;")
            cnt_lbl.setAlignment(Qt.AlignCenter)
            name_lbl = QLabel(label)
            name_lbl.setStyleSheet(f"color: {color}; font-size: 10px;")
            name_lbl.setAlignment(Qt.AlignCenter)
            vbox.addWidget(cnt_lbl)
            vbox.addWidget(name_lbl)
            summary_h.addLayout(vbox)
            self.vs_summary_labels[key] = cnt_lbl
        layout.addLayout(summary_h)

        # 漏洞列表
        self.vs_table = QTableWidget()
        self.vs_table.setColumnCount(5)
        self.vs_table.setHorizontalHeaderLabels(["严重度", "类型", "标题", "文件:行", "建议修复"])
        self.vs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.vs_table.setStyleSheet(
            "QTableWidget { background-color: #0d1117; color: #ccc; gridline-color: #222; font-size: 11px; }"
            "QHeaderView::section { background-color: #161b22; color: #ccc; padding: 4px; border: none; }"
        )
        self.vs_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.vs_table.itemSelectionChanged.connect(self._on_vs_selection_changed)
        layout.addWidget(self.vs_table, 1)

        # ── AI 修复操作栏 ──────────────────────────────────────────────
        fix_bar = QHBoxLayout()

        self.vs_fix_btn = QPushButton("🔧 AI修复选中漏洞")
        self.vs_fix_btn.setEnabled(False)
        self.vs_fix_btn.clicked.connect(self._vs_fix_selected)
        self.vs_fix_btn.setStyleSheet(
            "QPushButton { background-color: #1a2540; color: #88aaff; font-weight: bold; "
            "padding: 6px 16px; border-radius: 4px; border: 1px solid #3355aa; }"
            "QPushButton:hover { background-color: #223366; }"
            "QPushButton:disabled { background-color: #1a1a2e; color: #555; border-color: #333; }"
        )
        fix_bar.addWidget(self.vs_fix_btn)

        self.vs_fix_all_btn = QPushButton("⚡ AI修复全部高危+")
        self.vs_fix_all_btn.setEnabled(False)
        self.vs_fix_all_btn.clicked.connect(self._vs_fix_all_high)
        self.vs_fix_all_btn.setStyleSheet(
            "QPushButton { background-color: #2a1a00; color: #ffaa44; font-weight: bold; "
            "padding: 6px 16px; border-radius: 4px; border: 1px solid #885500; }"
            "QPushButton:hover { background-color: #3a2200; }"
            "QPushButton:disabled { background-color: #1a1a2e; color: #555; border-color: #333; }"
        )
        fix_bar.addWidget(self.vs_fix_all_btn)

        fix_bar.addStretch()

        self.vs_apply_btn = QPushButton("✅ 应用修复到文件")
        self.vs_apply_btn.setEnabled(False)
        self.vs_apply_btn.clicked.connect(self._vs_apply_fix)
        self.vs_apply_btn.setStyleSheet(
            "QPushButton { background-color: #059669; color: white; font-weight: bold; "
            "padding: 6px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #34D399; }"
            "QPushButton:disabled { background-color: #444; color: #888; }"
        )
        fix_bar.addWidget(self.vs_apply_btn)

        layout.addLayout(fix_bar)

        # ── 修复进度条 ─────────────────────────────────────────────────
        self.vs_fix_progress = QProgressBar()
        self.vs_fix_progress.setRange(0, 0)
        self.vs_fix_progress.setVisible(False)
        self.vs_fix_progress.setStyleSheet(
            "QProgressBar { background:#1a1a2e; border:1px solid #333; border-radius:3px; height:4px; }"
            "QProgressBar::chunk { background:#3355aa; border-radius:3px; }"
        )
        layout.addWidget(self.vs_fix_progress)

        # ── 修复 diff 预览 ─────────────────────────────────────────────
        self.vs_fix_preview = QTextEdit()
        self.vs_fix_preview.setReadOnly(True)
        self.vs_fix_preview.setMaximumHeight(130)
        self.vs_fix_preview.setPlaceholderText("AI生成的修复补丁将显示在这里，确认后点击「应用修复到文件」…")
        self.vs_fix_preview.setStyleSheet(
            "QTextEdit { background:#080d14; color:#ccc; border:1px solid #2a3a5a; "
            "border-radius:4px; font-family:Menlo,monospace; font-size:10px; padding:6px; }"
        )
        layout.addWidget(self.vs_fix_preview)

        # LLM 洞察
        self.vs_llm_output = QTextEdit()
        self.vs_llm_output.setReadOnly(True)
        self.vs_llm_output.setMaximumHeight(70)
        self.vs_llm_output.setPlaceholderText("LLM 深度审查洞察将显示在这里…")
        self.vs_llm_output.setStyleSheet(
            "QTextEdit { background:#0a100a; color:#88ff88; border:1px solid #334; "
            "border-radius:4px; font-size:10px; padding:4px; font-style:italic; }"
        )
        layout.addWidget(self.vs_llm_output)

        self.main_tabs.addTab(self._wrap_in_scroll(tab), "🔬 漏洞扫描")

    def _start_vuln_scan(self):
        """启动漏洞扫描（后台线程）"""
        if self.mutator:
            strategies_dir = str(getattr(self.mutator, '_strategies_dir', ''))
        else:
            strategies_dir = str(Path(__file__).resolve().parent.parent.parent / "strategies")

        if not Path(strategies_dir).exists():
            self.vs_status.setText(f"⚠️ 策略目录不存在: {strategies_dir}")
            return
        if not self.vuln_scanner:
            self.vuln_scanner = VulnerabilityScanner(
                strategies_dir=strategies_dir,
                llm_client=self.llm_client,
                log_callback=lambda msg: QTimer.singleShot(0, lambda m=msg: self.vs_status.setText(m)),
            )

        # 策略列表懒刷新：如有新策略则追加（"全部策略文件" 始终在 index 0）
        if self.tracker:
            existing = {self.vs_strategy_combo.itemText(i)
                        for i in range(self.vs_strategy_combo.count())}
            for s in sorted(self.tracker.all_strategies() or []):
                if s not in existing:
                    self.vs_strategy_combo.addItem(s)

        target = self.vs_strategy_combo.currentText()
        use_llm = self.vs_llm_check.isChecked() and bool(self.llm_client)

        self.vs_scan_btn.setEnabled(False)
        self.vs_progress.setVisible(True)
        self.vs_status.setText("⏳ 扫描中...")

        scanner = self.vuln_scanner

        def _do():
            if target == "全部策略文件":
                report = scanner.scan_all(use_llm=use_llm)
            else:
                report = scanner.scan_strategy(target, use_llm=use_llm)
            self._signal_vuln_done.emit(report)

        threading.Thread(target=_do, daemon=True).start()

    def _on_vuln_scan_done(self, report):
        """扫描完成，渲染结果"""
        self.vs_scan_btn.setEnabled(True)
        self.vs_progress.setVisible(False)

        if report.error:
            self.vs_status.setText(f"❌ {report.error}")
            return

        self.vs_status.setText(
            f"✅ 扫描完成（{report.duration_sec}s）— {report.summary}"
        )

        # 更新计数
        from src.ai_agent.vulnerability_scanner import SEV_LOW, SEV_MEDIUM
        counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for v in report.vulnerabilities:
            if v.severity in counts:
                counts[v.severity] += 1
        for key, lbl in self.vs_summary_labels.items():
            lbl.setText(str(counts.get(key, 0)))

        # 存储供修复引用
        self._vs_vulns = report.vulnerabilities
        self._vs_pending_fix = None
        self.vs_apply_btn.setEnabled(False)
        self.vs_fix_preview.clear()
        has_high = any(v.severity in ("CRITICAL", "HIGH") for v in self._vs_vulns)
        self.vs_fix_all_btn.setEnabled(has_high and bool(self.llm_client))

        # 填表
        vulns = report.vulnerabilities
        self.vs_table.setRowCount(len(vulns))
        sev_colors = {"CRITICAL": "#ff4444", "HIGH": "#ff8800", "MEDIUM": "#ffcc00", "LOW": "#aaaaaa"}
        for i, v in enumerate(vulns):
            sev_item = QTableWidgetItem(v.severity)
            sev_item.setForeground(QColor(sev_colors.get(v.severity, "#ccc")))
            self.vs_table.setItem(i, 0, sev_item)
            self.vs_table.setItem(i, 1, QTableWidgetItem(v.category))
            self.vs_table.setItem(i, 2, QTableWidgetItem(v.title))
            from pathlib import Path as _P
            loc = f"{_P(v.file_path).name}:{v.line_no}" if v.file_path else ""
            self.vs_table.setItem(i, 3, QTableWidgetItem(loc))
            self.vs_table.setItem(i, 4, QTableWidgetItem(v.suggested_fix[:80] if v.suggested_fix else ""))

        # LLM 洞察
        if report.llm_insights:
            import json as _j
            try:
                data = _j.loads(report.llm_insights[report.llm_insights.find("{"):report.llm_insights.rfind("}")+1])
                summary = data.get("summary", "")
                actions = data.get("priority_actions", [])
                score = data.get("overall_score", "?")
                self.vs_llm_output.setHtml(
                    f'<p style="color:#88ff88;">📊 评分: {score}/100 — {summary}</p>'
                    f'<p style="color:#aaffaa;">⚡ 优先修复: {" / ".join(str(a) for a in actions[:3])}</p>'
                )
            except Exception:
                self.vs_llm_output.setPlainText(report.llm_insights[:500])

        # 发出严重漏洞预警弹窗
        critical_count = counts.get("CRITICAL", 0)
        if critical_count > 0:
            from src.ai_agent.risk_guard import RiskAlert as _RA
            alert = _RA(
                level="CRITICAL",
                category="SYSTEM",
                title=f"发现 {critical_count} 个严重代码漏洞",
                message="策略代码存在严重安全/逻辑问题，可能导致意外损失",
                detail="请切换到「漏洞扫描」标签查看详情并修复",
                suggested_action="立即检查并修复严重漏洞后再运行策略",
            )
            try:
                get_alert_overlay().push_alert(alert)
            except Exception:
                pass

    # ── 漏洞修复 ──────────────────────────────────────────────────────

    def _on_vs_selection_changed(self):
        rows = self.vs_table.selectionModel().selectedRows()
        has_sel = bool(rows) and bool(self.llm_client)
        self.vs_fix_btn.setEnabled(has_sel)

    def _vs_fix_selected(self):
        """AI修复选中的单条漏洞"""
        rows = self.vs_table.selectionModel().selectedRows()
        if not rows:
            return
        idx = rows[0].row()
        if idx >= len(self._vs_vulns):
            return
        vuln = self._vs_vulns[idx]
        self._vs_do_fix([vuln], f"修复: {vuln.title}")

    def _vs_fix_all_high(self):
        """AI修复全部高危及以上漏洞"""
        high_vulns = [v for v in self._vs_vulns if v.severity in ("CRITICAL", "HIGH")]
        if not high_vulns:
            return
        self._vs_do_fix(high_vulns, f"批量修复 {len(high_vulns)} 个高危/严重漏洞")

    def _vs_do_fix(self, vulns, label: str):
        """调用 LLM 生成修复补丁（后台线程）"""
        if not self.llm_client:
            QMessageBox.warning(self, "提示", "请先配置并连接 API")
            return

        self.vs_fix_btn.setEnabled(False)
        self.vs_fix_all_btn.setEnabled(False)
        self.vs_apply_btn.setEnabled(False)
        self.vs_fix_progress.setVisible(True)
        self.vs_fix_preview.setHtml(
            f'<p style="color:#88aaff;">⏳ AI 正在生成修复方案：{label}…</p>'
        )

        client = self.llm_client

        def _do():
            results = []
            for vuln in vulns:
                patch = self._call_llm_fix(client, vuln)
                if patch:
                    results.append({"vuln": vuln, "patch": patch})
            self._signal_vs_fix_done.emit(results)

        threading.Thread(target=_do, daemon=True).start()

    def _call_llm_fix(self, client, vuln) -> Optional[Dict]:
        """向 LLM 请求具体修复补丁，返回 {old_code, new_code, explanation}"""
        # 读取文件上下文
        code_context = ""
        if vuln.file_path:
            try:
                lines = Path(vuln.file_path).read_text(encoding="utf-8").splitlines()
                lo = max(0, vuln.line_no - 10)
                hi = min(len(lines), vuln.line_no + 10)
                code_context = "\n".join(
                    f"{i+1:4d}│ {lines[i]}" for i in range(lo, hi)
                )
            except Exception:
                pass

        prompt = f"""你是代码安全修复专家。请对以下漏洞生成精确的修复补丁。

漏洞信息：
- 严重度: {vuln.severity}
- 类型: {vuln.category}
- 标题: {vuln.title}
- 描述: {vuln.description}
- 文件: {vuln.file_path}
- 行号: {vuln.line_no}
- 修复建议: {vuln.suggested_fix}

代码上下文（行号 {max(1, vuln.line_no-9)} 起）：
```python
{code_context or vuln.code_snippet or "（无上下文）"}
```

要求：
1. 只输出需要替换的最小代码片段，不要重写整个文件
2. old_code 必须是文件中实际存在的字符串（原样复制，保留缩进）
3. new_code 是替换后的安全代码
4. 如果无需修改代码（如仅建议性问题），old_code 和 new_code 相同

只输出 JSON，格式：
{{
  "old_code": "原始代码片段（精确匹配）",
  "new_code": "修复后的代码片段",
  "explanation": "一句话说明修复逻辑"
}}"""

        try:
            reply = client.chat([
                {"role": "system", "content": "你是代码安全修复专家。只输出 JSON，不要解释。"},
                {"role": "user", "content": prompt},
            ], timeout=60)
        except Exception:
            return None

        if not reply:
            return None

        try:
            j_start = reply.find("```json") + 7
            j_end = reply.rfind("```")
            raw = reply[j_start:j_end] if j_start > 6 and j_end > j_start else reply
            j_start2 = raw.find("{")
            j_end2 = raw.rfind("}") + 1
            data = json.loads(raw[j_start2:j_end2])
            return {
                "vuln": vuln,
                "old_code": data.get("old_code", ""),
                "new_code": data.get("new_code", ""),
                "explanation": data.get("explanation", ""),
            }
        except Exception:
            return None

    def _on_vs_fix_done(self, results: list):
        """LLM 修复完成，展示 diff 预览"""
        self.vs_fix_progress.setVisible(False)
        self.vs_fix_btn.setEnabled(bool(self._vs_vulns) and bool(self.llm_client))
        has_high = any(v.severity in ("CRITICAL", "HIGH") for v in self._vs_vulns)
        self.vs_fix_all_btn.setEnabled(has_high and bool(self.llm_client))

        if not results:
            self.vs_fix_preview.setHtml(
                '<p style="color:#ff6666;">❌ AI 未能生成有效修复补丁，请检查漏洞详情后手动修复</p>'
            )
            return

        # 过滤掉 old_code == new_code 的无效补丁
        valid = [r for r in results if r and r.get("old_code") and r["old_code"] != r.get("new_code")]
        if not valid:
            self.vs_fix_preview.setHtml(
                '<p style="color:#ffaa00;">⚠️ AI 认为此漏洞无需代码修改（建议性问题），请参考漏洞描述手动处理</p>'
            )
            return

        self._vs_pending_fix = valid

        # 渲染 diff 预览
        html = ['<div style="font-family:Menlo,monospace; font-size:10px; line-height:1.5;">']
        for r in valid:
            vuln = r["vuln"]
            sev_color = {"CRITICAL": "#ff4444", "HIGH": "#ff8800", "MEDIUM": "#ffcc00"}.get(vuln.severity, "#aaa")
            html.append(
                f'<p style="color:{sev_color}; font-size:11px; margin:6px 0 2px;">'
                f'<b>[{vuln.severity}] {vuln.title}</b>'
                f'<span style="color:#666; font-size:10px;"> — {vuln.file_path.split("/")[-1] if vuln.file_path else ""}:{vuln.line_no}</span>'
                f'</p>'
            )
            if r.get("explanation"):
                html.append(f'<p style="color:#88aaff; margin:1px 0 3px; padding-left:8px;">💡 {r["explanation"]}</p>')
            for line in r["old_code"].splitlines():
                safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                html.append(f'<p style="color:#ff6666; margin:0; padding-left:8px; background:#1a0808;">- {safe}</p>')
            for line in r["new_code"].splitlines():
                safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                html.append(f'<p style="color:#00ff88; margin:0; padding-left:8px; background:#081a08;">+ {safe}</p>')
            html.append('<hr style="border-color:#1a2a3a; margin:4px 0;">')
        html.append('</div>')

        self.vs_fix_preview.setHtml("".join(html))
        self.vs_apply_btn.setEnabled(True)

    def _vs_apply_fix(self):
        """将 AI 生成的补丁写入文件"""
        if not self._vs_pending_fix:
            return

        # 按文件分组
        from collections import defaultdict
        by_file: Dict[str, list] = defaultdict(list)
        for r in self._vs_pending_fix:
            fp = getattr(r["vuln"], "file_path", "")
            if fp:
                by_file[fp].append(r)

        if not by_file:
            QMessageBox.warning(self, "提示", "没有关联文件的修复补丁")
            return

        summary = "\n".join(
            f"• {Path(fp).name}: {len(patches)} 处" for fp, patches in by_file.items()
        )
        reply = QMessageBox.question(
            self, "确认应用 AI 修复",
            f"将以下修复写入文件（原文件已自动备份）：\n\n{summary}\n\n确认继续？",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        if not self.code_applier:
            self.code_applier = SafeCodeApplier()

        success_count = 0
        fail_msgs = []
        for fp, patches in by_file.items():
            try:
                source = Path(fp).read_text(encoding="utf-8")
                # 先备份
                backup = Path(fp).with_suffix(".py.vuln_bak")
                backup.write_text(source, encoding="utf-8")
                # 逐条替换
                modified = source
                for r in patches:
                    old = r.get("old_code", "")
                    new = r.get("new_code", "")
                    if old and old in modified:
                        modified = modified.replace(old, new, 1)
                        success_count += 1
                    else:
                        fail_msgs.append(f"{Path(fp).name}: 未找到匹配代码片段 — {r['vuln'].title[:40]}")
                Path(fp).write_text(modified, encoding="utf-8")
            except Exception as e:
                fail_msgs.append(f"{Path(fp).name}: {e}")

        msg = f"✅ 成功修复 {success_count} 处漏洞"
        if fail_msgs:
            msg += f"\n⚠️ {len(fail_msgs)} 处未匹配：\n" + "\n".join(fail_msgs[:5])

        self.vs_status.setText(msg.split("\n")[0])
        self.vs_apply_btn.setEnabled(False)
        self._vs_pending_fix = None

        if fail_msgs:
            QMessageBox.warning(self, "部分修复失败", msg)
        else:
            QMessageBox.information(self, "修复完成", msg)

        # 刷新 mutator 缓存
        if self.mutator:
            self.mutator._load_mutation_points()

    # ─── System Monitor Tab ─────────────────────────────────────────

    def _build_sysmon_tab(self):
        """📊 系统监控标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 实时系统指标
        sys_group = QGroupBox("🖥️ 系统资源")
        sys_group.setStyleSheet(
            "QGroupBox { border: 1px solid #444; border-radius: 6px; margin-top: 6px; "
            "padding: 10px; font-weight: bold; color: #88ccff; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
        )
        sys_grid = QHBoxLayout(sys_group)

        self.sysmon_labels: Dict[str, QLabel] = {}
        for key, display in [
            ("cpu_pct", "CPU %"),
            ("mem_pct", "内存 %"),
            ("mem_mb", "内存 MB"),
            ("thread_count", "线程数"),
            ("fd_count", "句柄数"),
        ]:
            vbox = QVBoxLayout()
            n = QLabel(display)
            n.setStyleSheet("color: #666; font-size: 10px;")
            n.setAlignment(Qt.AlignCenter)
            v = QLabel("—")
            v.setStyleSheet("color: #88ccff; font-size: 16px; font-weight: bold;")
            v.setAlignment(Qt.AlignCenter)
            vbox.addWidget(n)
            vbox.addWidget(v)
            sys_grid.addLayout(vbox)
            self.sysmon_labels[key] = v

        layout.addWidget(sys_group)

        # 交易引擎状态
        engine_group = QGroupBox("⚙️ 交易引擎状态")
        engine_group.setStyleSheet(
            "QGroupBox { border: 1px solid #444; border-radius: 6px; margin-top: 6px; "
            "padding: 10px; font-weight: bold; color: #88ccff; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
        )
        engine_form = QFormLayout(engine_group)
        engine_form.setSpacing(8)

        self.sysmon_auto_trading = QLabel("—")
        self.sysmon_auto_trading.setStyleSheet("color: #ccc; font-size: 12px;")
        engine_form.addRow("自动交易:", self.sysmon_auto_trading)

        self.sysmon_manual_orders = QLabel("—")
        self.sysmon_manual_orders.setStyleSheet("color: #ccc; font-size: 12px;")
        engine_form.addRow("手动挂单:", self.sysmon_manual_orders)

        self.sysmon_scan_status = QLabel("—")
        self.sysmon_scan_status.setStyleSheet("color: #ccc; font-size: 12px;")
        engine_form.addRow("扫描状态:", self.sysmon_scan_status)

        self.sysmon_evolver_status = QLabel("—")
        self.sysmon_evolver_status.setStyleSheet("color: #ccc; font-size: 12px;")
        engine_form.addRow("自进化引擎:", self.sysmon_evolver_status)

        self.sysmon_risk_guard = QLabel("—")
        self.sysmon_risk_guard.setStyleSheet("color: #ccc; font-size: 12px;")
        engine_form.addRow("风控守卫:", self.sysmon_risk_guard)

        layout.addWidget(engine_group)

        # 事件日志
        log_group = QGroupBox("📋 系统事件日志")
        log_group.setStyleSheet(
            "QGroupBox { border: 1px solid #333; border-radius: 6px; margin-top: 6px; "
            "padding: 8px; font-weight: bold; color: #aaa; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
        )
        log_v = QVBoxLayout(log_group)
        self.sysmon_log = QTextEdit()
        self.sysmon_log.setReadOnly(True)
        self.sysmon_log.setStyleSheet(
            "QTextEdit { background:#060c06; color:#88ff88; border:1px solid #1a3a1a; "
            "border-radius:4px; font-family:Menlo,monospace; font-size:10px; padding:4px; }"
        )
        log_v.addWidget(self.sysmon_log)
        layout.addWidget(log_group, 1)

        # 定时刷新
        self._sysmon_timer = QTimer(self)
        self._sysmon_timer.timeout.connect(self._refresh_sysmon)
        self._sysmon_timer.start(5000)

        self.main_tabs.addTab(self._wrap_in_scroll(tab), "📊 系统监控")

    def _refresh_sysmon(self):
        """每5秒刷新系统监控数据"""
        try:
            import psutil, os
            proc = psutil.Process(os.getpid())
            self.sysmon_labels["cpu_pct"].setText(f"{proc.cpu_percent(interval=None):.1f}")
            mi = proc.memory_info()
            mb = mi.rss / 1024 / 1024
            self.sysmon_labels["mem_mb"].setText(f"{mb:.0f}")
            self.sysmon_labels["mem_pct"].setText(f"{proc.memory_percent():.1f}")
            self.sysmon_labels["thread_count"].setText(str(proc.num_threads()))
            try:
                self.sysmon_labels["fd_count"].setText(str(proc.num_fds()))
            except Exception:
                self.sysmon_labels["fd_count"].setText("N/A")
        except ImportError:
            for lbl in self.sysmon_labels.values():
                lbl.setText("N/A")

        # 引擎状态
        rg = getattr(self, 'risk_guard', None)
        rg_on = rg is not None and rg.isRunning()
        self.sysmon_risk_guard.setText("🟢 运行中" if rg_on else "⚫ 停止")
        self.sysmon_risk_guard.setStyleSheet(
            f"color: {'#00ff88' if rg_on else '#888'}; font-size: 12px;"
        )

        ev = getattr(self, 'self_evolver', None)
        ev_on = ev is not None and ev.is_running()
        self.sysmon_evolver_status.setText("🟢 运行中" if ev_on else "⚫ 停止")
        self.sysmon_evolver_status.setStyleSheet(
            f"color: {'#c084fc' if ev_on else '#888'}; font-size: 12px;"
        )

        ac_on = self.auto_cycle is not None and self.auto_cycle.isRunning()
        self.sysmon_auto_trading.setText("🟢 运行中" if ac_on else "⚫ 停止")
        self.sysmon_auto_trading.setStyleSheet(
            f"color: {'#00cc66' if ac_on else '#888'}; font-size: 12px;"
        )

        self.sysmon_scan_status.setText("就绪")
        self.sysmon_manual_orders.setText("—")

    def sysmon_log_event(self, msg: str):
        """向系统监控日志追加一条事件"""
        ts = datetime.now().strftime("%H:%M:%S")
        self.sysmon_log.append(
            f'<span style="color:#555">[{ts}]</span> '
            f'<span style="color:#88ff88">{msg}</span>'
        )

    # ─── Public Integration Methods ─────────────────────────────────

    def refresh_strategies(self):
        """更新策略下拉列表"""
        if not self.tracker:
            return
        current = self.strategy_combo.currentText()
        self.strategy_combo.clear()
        strategies = sorted(self.tracker.all_strategies())
        self.strategy_combo.addItems(strategies)
        if current in strategies:
            self.strategy_combo.setCurrentText(current)

    def wire_system_context(self, **kwargs):
        """注入系统各模块引用：在 main_window 完成所有页面创建后调用"""
        for key, val in kwargs.items():
            setattr(self.system_ctx, key, val)
        self.tracker = getattr(self.system_ctx, 'tracker', self.tracker)
        self.mutator = getattr(self.system_ctx, 'mutator', self.mutator)
        self.timeframe_tracker = getattr(self.system_ctx, 'timeframe_tracker', self.timeframe_tracker)
        self.optimizer = getattr(self.system_ctx, 'optimizer', self.optimizer)
        self.refresh_strategies()

    def _get_signal_summary_text(self) -> str:
        """获取纯策略信号统计（不含账户数据）"""
        if not self.tracker:
            return ""
        strategies = self.tracker.all_strategies()
        lines = []
        for name in sorted(strategies):
            stats = self.tracker.strategy_stats(name, days=7)
            total = stats.get("total", 0)
            if total < 3:
                continue
            lines.append(
                f"{name}: {total}条信号, 正确率{stats.get('win_rate',0):.1f}%, "
                f"盈亏比{stats.get('profit_factor',0):.2f}, 净值{stats.get('net_pnl',0):.2f}%"
            )
        return "\n".join(lines) if lines else ""
