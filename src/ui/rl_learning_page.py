"""
强化学习进化界面：展示策略绩效、参数演化与自动调优。
"""

from __future__ import annotations

import atexit
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


from src.rl_optimizer.tracker import SignalTracker
from src.rl_optimizer.optimizer import ParameterOptimizer
from src.rl_optimizer.mutator import StrategyMutator
from src.rl_optimizer.timeframe_tracker import MultiTimeframeTracker
from src.rl_optimizer.auto_trainer import RLAutoTrainer
from src.qt_compat import QColor, QFrame, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QProgressBar, QPushButton, QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit, QTimer, QVBoxLayout, QWidget, Qt, Signal


class RLLearningPage(QWidget):
    """机器自主学习进化页面"""

    def __init__(self, okx_client=None, parent=None):
        super().__init__(parent)
        self.okx_client = okx_client
        self.tracker = SignalTracker()
        self.optimizer = ParameterOptimizer()
        self.mutator = StrategyMutator()
        self.timeframe_tracker = MultiTimeframeTracker()
        self._strategies_tab: Optional[QWidget] = None
        self._param_timer = QTimer(self)
        self._param_timer.timeout.connect(self._auto_validate)
        self._para_tables: Dict[str, QTableWidget] = {}
        self._validating = False

        # 自主训练器
        self.auto_trainer: Optional[RLAutoTrainer] = None
        self._auto_mode = False

        # 防抖刷新：合并连续刷新请求，只执行最后一次
        self._refresh_debounce = QTimer(self)
        self._refresh_debounce.setSingleShot(True)
        self._refresh_debounce.timeout.connect(self._do_refresh_all)
        self._refresh_pending = False

        self.init_ui()
        self._param_timer.start(600000)

        # 注册 atexit 处理器：防止强制退出时 QThread 未清理导致 SIGABRT
        atexit.register(self._safe_stop_trainer)

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("🧠 自主强化学习进化系统")
        title.setStyleSheet("color: #00ccff; font-size: 16px; font-weight: bold; padding: 4px;")
        layout.addWidget(title)

        desc = QLabel(
            "系统自动记录每次扫描信号 → 追踪实际走势验证对错 → "
            "根据胜率/盈亏比自动调优策略参数 → 持续迭代进化"
        )
        desc.setStyleSheet("color: #aaaaaa; font-size: 11px; padding-bottom: 8px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        action_bar = QHBoxLayout()
        self.refresh_btn = QPushButton("🔄 验证待审信号")
        self.refresh_btn.clicked.connect(self._run_validation)
        self.refresh_btn.setStyleSheet(
            "QPushButton { background-color: #007acc; color: white; font-weight: bold; "
            "padding: 6px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #0099ee; }"
        )
        action_bar.addWidget(self.refresh_btn)

        self.update_btn = QPushButton("📈 更新参数评分")
        self.update_btn.clicked.connect(self._update_optimizer)
        self.update_btn.setStyleSheet(
            "QPushButton { background-color: #28a745; color: white; font-weight: bold; "
            "padding: 6px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #34d058; }"
        )
        action_bar.addWidget(self.update_btn)

        self.deploy_btn = QPushButton("🚀 发布最优参数")
        self.deploy_btn.clicked.connect(self._deploy_best_params)
        self.deploy_btn.setStyleSheet(
            "QPushButton { background-color: #cc6600; color: white; font-weight: bold; "
            "padding: 6px 14px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #ee8800; }"
        )
        action_bar.addWidget(self.deploy_btn)

        self.auto_mode_btn = QPushButton("🤖 自主进化: OFF")
        self.auto_mode_btn.setCheckable(True)
        self.auto_mode_btn.clicked.connect(self._toggle_auto_mode)
        self.auto_mode_btn.setStyleSheet(
            "QPushButton { background-color: #444; color: #ccc; font-weight: bold; "
            "padding: 6px 16px; border-radius: 4px; border: 2px solid #666; }"
            "QPushButton:checked { background-color: #00aa66; color: white; border-color: #00ff88; }"
            "QPushButton:hover { background-color: #555; }"
        )
        action_bar.addWidget(self.auto_mode_btn)

        action_bar.addStretch()

        date_label = QLabel("时段:")
        date_label.setStyleSheet("color: #888; font-size: 11px;")
        action_bar.addWidget(date_label)

        from PySide6.QtWidgets import QComboBox
        self.date_range_combo = QComboBox()
        self.date_range_combo.addItems(["7天", "14天", "30天", "90天", "全部"])
        self.date_range_combo.setCurrentText("30天")
        self.date_range_combo.currentTextChanged.connect(self._refresh_all)
        self.date_range_combo.setStyleSheet(
            "QComboBox { background:#222; color:#ccc; border:1px solid #444; padding:2px 6px; border-radius:3px; }"
        )
        action_bar.addWidget(self.date_range_combo)

        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")
        action_bar.addWidget(self.status_label)
        layout.addLayout(action_bar)

        # 自主训练日志面板（默认隐藏，开启自主模式后显示）
        self.auto_log_group = QGroupBox("🤖 自主进化日志")
        self.auto_log_group.setVisible(False)
        auto_log_layout = QVBoxLayout(self.auto_log_group)
        self.auto_log_text = QTextEdit()
        self.auto_log_text.setReadOnly(True)
        self.auto_log_text.setMaximumHeight(120)
        self.auto_log_text.setStyleSheet(
            "QTextEdit { background-color: #0a0a15; color: #88ff88; border: 1px solid #333; "
            "border-radius: 4px; font-family: 'Menlo', monospace; font-size: 10px; padding: 4px; }"
        )
        auto_log_layout.addWidget(self.auto_log_text)
        layout.addWidget(self.auto_log_group)

        self.main_tabs = QTabWidget()
        self.main_tabs.setStyleSheet("""
            QTabWidget::pane { background-color: #1a1a2e; border: 1px solid #333; border-radius: 4px; }
            QTabBar::tab { background-color: #252540; color: #ccc; padding: 6px 16px;
                           border: 1px solid #333; border-bottom: none; border-radius: 4px 4px 0 0; }
            QTabBar::tab:selected { background-color: #1a1a2e; color: #00ccff; font-weight: bold; }
        """)
        layout.addWidget(self.main_tabs, 1)

        self._build_strategy_tab()
        self._build_overview_tab()
        self._build_evolution_tab()
        self._build_mutation_tab()
        self._build_analysis_tab()
        self._build_ab_test_tab()
        self._build_timeframe_tab()
        # 首次刷新异步执行，不阻塞 UI
        QTimer.singleShot(100, self._request_refresh)

    def _build_overview_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.overview_table = QTableWidget()
        self.overview_table.setColumnCount(11)
        self.overview_table.setHorizontalHeaderLabels([
            "策略名称", "总信号", "已验证", "胜率%", "平均盈利%",
            "平均亏损%", "盈亏比", "净收益%", "夏普", "进化代数", "参数重要性"
        ])
        self.overview_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.overview_table.setAlternatingRowColors(True)
        self.overview_table.setStyleSheet(
            "QTableWidget { background-color: #0d1117; color: #ccc; gridline-color: #333; }"
            "QTableWidget::item { padding: 4px; }"
        )
        layout.addWidget(self.overview_table)
        self.main_tabs.addTab(tab, "📊 概览")

    def _build_strategy_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.strategy_detail_text = QTextEdit()
        self.strategy_detail_text.setReadOnly(True)
        self.strategy_detail_text.setStyleSheet(
            "QTextEdit { background-color: #0d1117; color: #ccc; border: 1px solid #333; "
            "border-radius: 4px; font-family: 'Menlo', monospace; font-size: 11px; padding: 8px; }"
        )
        layout.addWidget(self.strategy_detail_text)
        self.main_tabs.addTab(tab, "📋 策略明细")

    def _build_evolution_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.evolution_table = QTableWidget()
        self.evolution_table.setColumnCount(6)
        self.evolution_table.setHorizontalHeaderLabels([
            "时间", "策略", "胜率%", "盈亏比", "信号数", "净收益%"
        ])
        self.evolution_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.evolution_table.setAlternatingRowColors(True)
        self.evolution_table.setStyleSheet(
            "QTableWidget { background-color: #0d1117; color: #ccc; gridline-color: #333; }"
            "QTableWidget::item { padding: 4px; }"
        )
        layout.addWidget(self.evolution_table)
        self.main_tabs.addTab(tab, "📈 进化历史")

    def _build_mutation_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        bar = QHBoxLayout()
        self.mutate_btn = QPushButton("🧬 生成变异")
        self.mutate_btn.clicked.connect(self._do_mutation)
        self.mutate_btn.setStyleSheet(
            "QPushButton { background-color: #aa6f00; color: white; font-weight: bold; "
            "padding: 6px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #cc8800; }"
        )
        bar.addWidget(self.mutate_btn)

        self.rollback_btn = QPushButton("⏪ 回退选中版本")
        self.rollback_btn.clicked.connect(self._do_rollback)
        self.rollback_btn.setEnabled(False)
        self.rollback_btn.setStyleSheet(
            "QPushButton { background-color: #cc3333; color: white; font-weight: bold; "
            "padding: 6px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #dd4444; }"
        )
        bar.addWidget(self.rollback_btn)
        bar.addStretch()

        self.mutation_status = QLabel("就绪")
        self.mutation_status.setStyleSheet("color: #888; font-size: 11px;")
        bar.addWidget(self.mutation_status)
        layout.addLayout(bar)

        self.mutation_table = QTableWidget()
        self.mutation_table.setColumnCount(5)
        self.mutation_table.setHorizontalHeaderLabels([
            "版本", "时间", "源文件", "变异数", "变异描述"
        ])
        self.mutation_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.mutation_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.mutation_table.setAlternatingRowColors(True)
        self.mutation_table.setStyleSheet(
            "QTableWidget { background-color: #0d1117; color: #ccc; gridline-color: #333; }"
            "QTableWidget::item { padding: 4px; }"
        )
        self.mutation_table.itemSelectionChanged.connect(
            lambda: self.rollback_btn.setEnabled(
                bool(self.mutation_table.currentItem())
            )
        )
        layout.addWidget(self.mutation_table)
        self.main_tabs.addTab(tab, "🧬 代码变异")

    def _do_mutation(self):
        from src.rl_optimizer.mutator import StrategyMutator
        import json, random

        log_path = Path(self.mutator._evolutions_dir) / "_mutation_log.json"

        strategies = [f for f in sorted(
            Path(self.mutator._strategies_dir).glob("*.py")
        ) if not f.name.startswith("_")]

        if not strategies:
            self.mutation_status.setText("❌ 未找到策略文件")
            return

        fpath = random.choice(strategies)
        result = self.mutator.mutate_strategy(fpath.name, num_mutations=random.randint(1, 3))

        if result and result.get("success"):
            descs = [m["description"] for m in result.get("mutations", [])]
            self.mutation_status.setText(
                f"✅ {result['version_name']} | 变异 {len(descs)} 处"
            )
            self._refresh_mutations()
        else:
            err = result.get("error", "未知错误") if result else "变异失败"
            self.mutation_status.setText(f"❌ {err}")

    def _do_rollback(self):
        row = self.mutation_table.currentRow()
        if row < 0:
            return
        version = self.mutation_table.item(row, 0)
        if not version:
            return
        vname = version.text()
        ok = self.mutator.reset_to_version(vname)
        if ok:
            self.mutation_status.setText(f"✅ 已回退到 {vname}")
            self._refresh_mutations()
        else:
            self.mutation_status.setText(f"❌ 回退失败")

    def _refresh_mutations(self):
        log_path = Path(self.mutator._evolutions_dir) / "_mutation_log.json"
        if not log_path.exists():
            self.mutation_table.setRowCount(0)
            return
        import json
        try:
            log = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            self.mutation_table.setRowCount(0)
            return
        self.mutation_table.setRowCount(len(log))
        for i, entry in enumerate(reversed(log)):
            self.mutation_table.setItem(i, 0, QTableWidgetItem(entry.get("version_name", "")))
            self.mutation_table.setItem(i, 1, QTableWidgetItem(entry.get("timestamp", "")))
            self.mutation_table.setItem(i, 2, QTableWidgetItem(entry.get("source_file", "")))
            muts = entry.get("mutations", [])
            self.mutation_table.setItem(i, 3, QTableWidgetItem(str(len(muts))))
            descs = ", ".join(m.get("description", "") for m in muts)
            self.mutation_table.setItem(i, 4, QTableWidgetItem(descs[:80]))
        if log:
            self.mutation_table.resizeColumnToContents(0)

    def _refresh_all(self):
        self._refresh_overview()
        self._refresh_detail()
        self._refresh_evolution()
        self._refresh_mutations()

    def _refresh_overview(self):
        from PySide6.QtWidgets import QApplication
        strategies = self.tracker.all_strategies()
        self.overview_table.setRowCount(len(strategies))
        for i, name in enumerate(sorted(strategies)):
            if i % 2 == 0:
                QApplication.processEvents()
            stats = self.tracker.strategy_stats(name, self._get_days_filter())
            gen = self.optimizer.strategy_generation(name)
            importance = self.optimizer.param_importance(name)

            items = [
                name,
                str(stats.get("total", 0)),
                str(stats.get("validated_count", 0)),
                f"{stats.get('win_rate', 0):.1f}%",
                f"{stats.get('avg_win_pct', 0):.2f}%",
                f"{stats.get('avg_loss_pct', 0):.2f}%",
                f"{stats.get('profit_factor', 0):.2f}",
                f"{stats.get('net_pnl', 0):.2f}%",
                f"{stats.get('sharpe', 0):.2f}",
                str(gen),
                ", ".join(f"{k}:{v.get('value',0):.2f}" for k, v in sorted(importance.items(), key=lambda x: -x[1]["importance"])[:3]),
            ]
            for j, text in enumerate(items):
                item = QTableWidgetItem(text)
                if j == 3:
                    wr = stats.get("win_rate", 0)
                    item.setForeground(QColor("#00ff88") if wr >= 50 else QColor("#ff6666"))
                self.overview_table.setItem(i, j, item)

    def _refresh_detail(self):
        from PySide6.QtWidgets import QApplication
        strategies = self.tracker.all_strategies()
        html = "<h3 style='color:#00aaff;'>各策略参数进化详情</h3>"
        for i, name in enumerate(sorted(strategies)):
            if i % 2 == 0:
                QApplication.processEvents()
            stats = self.tracker.strategy_stats(name, self._get_days_filter())
            params = self.optimizer.get_optimized_params(name)
            gen = self.optimizer.strategy_generation(name)
            recent = self.tracker.strategy_recent_signals(name, 5)

            html += f"<div style='margin:10px 0; padding:8px; background:#111; border-radius:4px;'>"
            html += f"<h4 style='color:#ffd700; margin:0 0 4px;'>{name} (第 {gen} 代)</h4>"
            html += f"<p style='color:#ccc; font-size:11px; margin:2px 0;'>"
            html += f"总信号: {stats.get('total',0)} | 胜率: {stats.get('win_rate',0):.1f}% | "
            html += f"盈亏比: {stats.get('profit_factor',0):.2f} | 净收益: {stats.get('net_pnl',0):.2f}%</p>"

            if params:
                html += "<p style='color:#888; font-size:10px; margin:2px 0;'>优化参数: "
                param_strs = []
                for k, v in params.items():
                    if isinstance(v, dict):
                        param_strs.append(f"{k}={v.get('value', '?')}")
                    else:
                        param_strs.append(f"{k}={v}")
                html += " | ".join(param_strs)
                html += "</p>"

            if recent:
                html += "<table style='font-size:10px; color:#aaa; width:100%; border-collapse:collapse;'>"
                html += "<tr style='color:#888;'><td>时间</td><td>标的</td><td>方向</td><td>评分</td><td>盈亏%</td></tr>"
                for r in recent:
                    pnl = float(r.get("pnl_pct", 0) or 0)
                    pnl_color = "#00ff88" if pnl > 0 else ("#ff6666" if pnl < 0 else "#888")
                    html += f"<tr><td>{r.get('datetime','')[-8:]}</td>"
                    html += f"<td>{r.get('symbol','')}</td>"
                    html += f"<td>{r.get('direction','')}</td>"
                    html += f"<td>{float(r.get('score',0)):.1f}</td>"
                    html += f"<td style='color:{pnl_color};'>{pnl:+.1f}%</td></tr>"
                html += "</table>"
            html += "</div>"
        self.strategy_detail_text.setHtml(html)

    def _refresh_evolution(self):
        history = self.optimizer.evolution_summary()
        self.evolution_table.setRowCount(len(history))
        for i, h in enumerate(history):
            self.evolution_table.setItem(i, 0, QTableWidgetItem(h.get("time", "")))
            self.evolution_table.setItem(i, 1, QTableWidgetItem(h.get("strategy", "")))
            item_wr = QTableWidgetItem(f"{h.get('win_rate', 0):.1f}%")
            item_wr.setForeground(QColor("#00ff88") if h.get("win_rate", 0) >= 50 else QColor("#ff6666"))
            self.evolution_table.setItem(i, 2, item_wr)
            self.evolution_table.setItem(i, 3, QTableWidgetItem(f"{h.get('profit_factor', 0):.2f}"))
            self.evolution_table.setItem(i, 4, QTableWidgetItem(str(h.get("total_signals", 0))))
            item_pnl = QTableWidgetItem(f"{h.get('net_pnl', 0):+.2f}%")
            item_pnl.setForeground(QColor("#00ff88") if h.get("net_pnl", 0) >= 0 else QColor("#ff6666"))
            self.evolution_table.setItem(i, 5, item_pnl)

    def _run_validation(self):
        if not self.okx_client:
            QMessageBox.warning(self, "提示", "未连接 OKX，无法获取当前价格验证信号")
            return
        if self._validating:
            return
        self._validating = True
        self.refresh_btn.setEnabled(False)
        self.status_label.setText("⏳ 正在验证信号...")

        def _do():
            try:
                updated_sig = self.tracker.validate_outstanding(self.okx_client)
                updated_tf = self.timeframe_tracker.validate_predictions(self.okx_client)
                QTimer.singleShot(0, lambda: self._on_validation_done(updated_sig + updated_tf))
            except Exception as e:
                QTimer.singleShot(0, lambda: self._on_validation_error(str(e)))

        threading.Thread(target=_do, daemon=True).start()

    def _on_validation_done(self, count: int):
        self._validating = False
        self.refresh_btn.setEnabled(True)
        self.status_label.setText(f"✅ 已验证 {count} 条信号")
        self._update_optimizer()

    def _on_validation_error(self, msg: str):
        self._validating = False
        self.refresh_btn.setEnabled(True)
        self.status_label.setText(f"❌ 验证失败: {msg}")

    def _update_optimizer(self):
        strategies = self.tracker.all_strategies()
        updated = 0
        for name in strategies:
            stats = self.tracker.strategy_stats(name, self._get_days_filter())
            samples = self.tracker.strategy_learning_samples(name, self._get_days_filter())
            if stats.get("total", 0) >= 3 and len(samples) >= 3:
                self.optimizer.update_from_samples(name, samples, stats)
                updated += 1
                fname = self._find_strategy_file(name)
                version_stats = self.tracker.strategy_version_stats(name, self._get_days_filter())
                history = self.mutator.get_evolution_history(fname)
                for entry in history:
                    version_name = entry.get("version_name")
                    if not version_name:
                        continue
                    checksum = self.mutator.version_checksum(version_name)
                    metrics = version_stats.get(checksum)
                    if metrics and metrics.get("signal_count", 0) >= 2:
                        self.mutator.update_fitness(version_name, float(metrics.get("win_rate", 0)))

                current_checksum = self.mutator.file_checksum(fname)
                current_metrics = version_stats.get(current_checksum)
                current_fitness = float(current_metrics.get("win_rate", stats.get("win_rate", 0)) if current_metrics else stats.get("win_rate", 0))
                self.mutator.apply_best_if_better(fname, current_fitness)

        self.status_label.setText(f"✅ 已更新 {updated} 个策略+反馈变异适应度")
        self._request_refresh()

    def _find_strategy_file(self, strategy_name: str) -> str:
        """根据策略显示名查找文件名，回退到第一个匹配项"""
        from pathlib import Path
        for f in Path(self.mutator._strategies_dir).glob("*.py"):
            try:
                content = f.read_text(encoding="utf-8")
                if f'name = "{strategy_name}"' in content or f'STRATEGY_NAME = "{strategy_name}"' in content:
                    return f.name
            except Exception:
                continue
        # 回退：取所有策略文件的第一个匹配关键词
        keywords = strategy_name.replace(" ", "").replace("AI", "").replace("截面", "").replace("因子", "")
        for f in Path(self.mutator._strategies_dir).glob("*.py"):
            if keywords and keywords.lower() in f.name.lower():
                return f.name
        return list(Path(self.mutator._strategies_dir).glob("*.py"))[0].name if list(Path(self.mutator._strategies_dir).glob("*.py")) else ""

    def _get_days_filter(self) -> int:
        t = self.date_range_combo.currentText() if hasattr(self, 'date_range_combo') else "30天"
        return {"7天": 7, "14天": 14, "30天": 30, "90天": 90}.get(t, 9999)

    def _build_analysis_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        hsplit = QSplitter(Qt.Orientation.Horizontal)

        # 左: 评分区间 vs 胜率
        left_frame = QFrame()
        left_frame.setStyleSheet("QFrame { background: #111; border-radius: 4px; padding: 8px; }")
        ll = QVBoxLayout(left_frame)
        ll.addWidget(QLabel("<b style='color:#00aaff;'>评分区间 → 胜率分析</b>"))
        self.score_winrate_table = QTableWidget()
        self.score_winrate_table.setColumnCount(4)
        self.score_winrate_table.setHorizontalHeaderLabels(["评分区间", "信号数", "胜率%", "建议"])
        self.score_winrate_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.score_winrate_table.setStyleSheet(
            "QTableWidget { background-color: #0d1117; color: #ccc; gridline-color: #333; }"
            "QTableWidget::item { padding: 3px; }"
        )
        ll.addWidget(self.score_winrate_table)
        hsplit.addWidget(left_frame)

        # 右: 参数重要性排名
        right_frame = QFrame()
        right_frame.setStyleSheet("QFrame { background: #111; border-radius: 4px; padding: 8px; }")
        rl = QVBoxLayout(right_frame)
        rl.addWidget(QLabel("<b style='color:#00aaff;'>参数重要性排名</b>"))

        self.importance_table = QTableWidget()
        self.importance_table.setColumnCount(5)
        self.importance_table.setHorizontalHeaderLabels(["策略", "参数名", "当前值", "胜率%", "重要性"])
        self.importance_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.importance_table.setStyleSheet(
            "QTableWidget { background-color: #0d1117; color: #ccc; gridline-color: #333; }"
            "QTableWidget::item { padding: 3px; }"
        )
        rl.addWidget(self.importance_table)
        hsplit.addWidget(right_frame)

        hsplit.setSizes([400, 400])
        layout.addWidget(hsplit)

        self.main_tabs.addTab(tab, "📉 深度分析")

    def _build_ab_test_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        hbar = QHBoxLayout()
        hbar.addWidget(QLabel("<b style='color:#00aaff;'>A/B 策略对比</b>"))
        hbar.addStretch()

        from PySide6.QtWidgets import QComboBox
        self.ab_strat_a = QComboBox()
        self.ab_strat_a.setStyleSheet("QComboBox { background:#222; color:#ccc; border:1px solid #444; padding:3px 8px; }")
        self.ab_strat_b = QComboBox()
        self.ab_strat_b.setStyleSheet("QComboBox { background:#222; color:#ccc; border:1px solid #444; padding:3px 8px; }")
        self.ab_strat_a.currentTextChanged.connect(lambda: self._refresh_ab_test())
        self.ab_strat_b.currentTextChanged.connect(lambda: self._refresh_ab_test())

        hbar.addWidget(QLabel("策略A:"))
        hbar.addWidget(self.ab_strat_a)
        hbar.addWidget(QLabel("  vs  策略B:"))
        hbar.addWidget(self.ab_strat_b)
        layout.addLayout(hbar)

        self.ab_compare_label = QLabel("选择两个策略进行对比")
        self.ab_compare_label.setStyleSheet("color: #ccc; font-size: 13px; padding: 10px;")
        self.ab_compare_label.setWordWrap(True)
        layout.addWidget(self.ab_compare_label)

        self.ab_detail_table = QTableWidget()
        self.ab_detail_table.setColumnCount(4)
        self.ab_detail_table.setHorizontalHeaderLabels(["指标", "策略A", "策略B", "优胜"])
        self.ab_detail_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ab_detail_table.setStyleSheet(
            "QTableWidget { background-color: #0d1117; color: #ccc; gridline-color: #333; }"
            "QTableWidget::item { padding: 4px; }"
        )
        layout.addWidget(self.ab_detail_table)

        self.main_tabs.addTab(tab, "⚖️ A/B对比")

    def _refresh_analysis(self):
        strategies = self.tracker.all_strategies()
        days = self._get_days_filter()
        effective_days = 9999 if days >= 9999 else days

        # Score vs Winrate
        all_data = []
        for name in sorted(strategies):
            data = self.tracker.score_vs_winrate(name, effective_days)
            for d in data:
                d["strategy"] = name
                all_data.append(d)
        self.score_winrate_table.setRowCount(len(all_data))
        best_range = ""
        best_wr = 0
        for i, d in enumerate(sorted(all_data, key=lambda x: x["score_range"])):
            self.score_winrate_table.setItem(i, 0, QTableWidgetItem(
                f"[{d['strategy'][:8]}] {d['score_range']}"
            ))
            self.score_winrate_table.setItem(i, 1, QTableWidgetItem(str(d["signals"])))
            item = QTableWidgetItem(f"{d['win_rate']:.1f}%")
            item.setForeground(QColor("#00ff88") if d["win_rate"] >= 55 else QColor("#ff6666"))
            self.score_winrate_table.setItem(i, 2, item)
            suggestion = ""
            if d["win_rate"] >= 60 and d["signals"] > 3:
                suggestion = "✅ 最优区间"
                if d["win_rate"] > best_wr:
                    best_wr = d["win_rate"]
                    best_range = d["score_range"]
            elif d["win_rate"] < 45 and d["signals"] > 3:
                suggestion = "⚠️ 避免此区间"
            self.score_winrate_table.setItem(i, 3, QTableWidgetItem(suggestion))
            if d["win_rate"] > best_wr:
                best_wr = d["win_rate"]
                best_range = f"[{d['strategy'][:8]}] {d['score_range']}"

        # 建议最优阈值
        if best_range:
            self.score_winrate_table.setRowCount(len(all_data) + 1)
            item = QTableWidgetItem(f"💡 最优评分阈值: {best_range} (胜率 {best_wr:.1f}%)")
            item.setForeground(QColor("#ffd700"))
            self.score_winrate_table.setItem(len(all_data), 0, item)
            self.score_winrate_table.setSpan(len(all_data), 0, 1, 4)

        # Parameter Importance
        all_params = []
        for name in sorted(strategies):
            importance = self.optimizer.param_importance(name)
            for key, val in importance.items():
                if val.get("trials", 0) > 0:
                    all_params.append({
                        "strategy": name,
                        "param": key,
                        "value": val.get("value", 0),
                        "win_rate": val.get("win_rate", 50),
                        "importance": val.get("importance", 0),
                    })
        self.importance_table.setRowCount(len(all_params))
        for i, p in enumerate(sorted(all_params, key=lambda x: -x["importance"])):
            self.importance_table.setItem(i, 0, QTableWidgetItem(p["strategy"][:12]))
            self.importance_table.setItem(i, 1, QTableWidgetItem(p["param"]))
            self.importance_table.setItem(i, 2, QTableWidgetItem(f"{p['value']:.3f}"))
            item_wr = QTableWidgetItem(f"{p['win_rate']:.1f}%")
            item_wr.setForeground(QColor("#00ff88") if p["win_rate"] >= 55 else QColor("#ff6666"))
            self.importance_table.setItem(i, 3, item_wr)
            importance_str = "●" * int(p["importance"]) + "◌" * max(0, 5 - int(p["importance"]))
            self.importance_table.setItem(i, 4, QTableWidgetItem(importance_str))

    def _refresh_ab_test(self):
        a = self.ab_strat_a.currentText()
        b = self.ab_strat_b.currentText()
        if not a or not b:
            return
        days = self._get_days_filter()
        stats_a = self.tracker.strategy_stats(a, days)
        stats_b = self.tracker.strategy_stats(b, days)

        metrics = [
            ("总信号数", "total", False),
            ("胜率 %", "win_rate", True),
            ("平均盈利 %", "avg_win_pct", True),
            ("平均亏损 %", "avg_loss_pct", False),
            ("盈亏比", "profit_factor", True),
            ("净收益 %", "net_pnl", True),
            ("夏普比率", "sharpe", True),
        ]
        self.ab_detail_table.setRowCount(len(metrics))
        for i, (label, key, higher_better) in enumerate(metrics):
            va = stats_a.get(key, 0)
            vb = stats_b.get(key, 0)
            winner = ""
            if va and vb:
                if higher_better:
                    winner = "A 胜" if va > vb else ("B 胜" if vb > va else "平")
                else:
                    winner = "A 胜" if va < vb else ("B 胜" if vb < va else "平")
            self.ab_detail_table.setItem(i, 0, QTableWidgetItem(label))
            self.ab_detail_table.setItem(i, 1, QTableWidgetItem(f"{va}" if isinstance(va, (int, float)) else str(va)))
            self.ab_detail_table.setItem(i, 2, QTableWidgetItem(f"{vb}" if isinstance(vb, (int, float)) else str(vb)))
            w_item = QTableWidgetItem(winner)
            w_item.setForeground(QColor("#00ff88" if "A" in winner else ("#ffd700" if "平" in winner else "#ff6666")))
            self.ab_detail_table.setItem(i, 3, w_item)

        # Populate combos
        strategies = self.tracker.all_strategies()
        if self.ab_strat_a.count() == 0:
            for s in strategies:
                self.ab_strat_a.addItem(s)
                self.ab_strat_b.addItem(s)

        # Summary
        a_win = stats_a.get("win_rate", 0)
        b_win = stats_b.get("win_rate", 0)
        if a_win > 55 and b_win > 0 and a_win > b_win + 3:
            self.ab_compare_label.setText(
                f"<span style='color:#00ff88;font-weight:bold;'>✅ 策略A ({a}) 胜率 {a_win:.1f}% > 策略B {b_win:.1f}%，建议采纳A</span>"
            )
        elif b_win > 55 and b_win > a_win + 3:
            self.ab_compare_label.setText(
                f"<span style='color:#00ff88;font-weight:bold;'>✅ 策略B ({b}) 胜率 {b_win:.1f}% > 策略A {a_win:.1f}%，建议采纳B</span>"
            )
        else:
            self.ab_compare_label.setText(
                f"<span style='color:#ffd700;'>⚖️ 两策略表现接近 (A: {a_win:.1f}% vs B: {b_win:.1f}%)，需要更多数据</span>"
            )

    def _build_timeframe_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.tf_accuracy_table = QTableWidget()
        self.tf_accuracy_table.setColumnCount(6)
        self.tf_accuracy_table.setHorizontalHeaderLabels([
            "策略", "1D准确率", "4H准确率", "1H准确率", "3m准确率", "调整建议"
        ])
        self.tf_accuracy_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tf_accuracy_table.setStyleSheet(
            "QTableWidget { background-color: #0d1117; color: #ccc; gridline-color: #333; }"
            "QTableWidget::item { padding: 4px; }"
        )
        layout.addWidget(self.tf_accuracy_table)

        self.tf_recommendations = QLabel()
        self.tf_recommendations.setStyleSheet("color: #ccc; padding: 10px; font-size: 12px;")
        self.tf_recommendations.setWordWrap(True)
        layout.addWidget(self.tf_recommendations)

        self.main_tabs.addTab(tab, "⏱ 周期准确率")

    def _refresh_timeframe(self):
        days = self._get_days_filter()
        effective = 9999 if days >= 9999 else days
        acc = self.timeframe_tracker.accuracy_by_strategy(effective)

        self.tf_accuracy_table.setRowCount(len(acc))
        row = 0
        for name in sorted(acc.keys()):
            tf_data = acc[name]
            self.tf_accuracy_table.setItem(row, 0, QTableWidgetItem(str(name)[:20]))
            for col, tf in enumerate(["1D", "4H", "1H", "3m"], 1):
                d = tf_data.get(tf, {})
                acc_val = d.get("accuracy", 0)
                total = d.get("total", 0)
                text = f"{acc_val:.1f}% ({total})" if total > 0 else "--"
                item = QTableWidgetItem(text)
                if acc_val >= 60:
                    item.setForeground(QColor("#00ff88"))
                elif acc_val >= 45:
                    item.setForeground(QColor("#ffd700"))
                elif total > 0:
                    item.setForeground(QColor("#ff6666"))
                self.tf_accuracy_table.setItem(row, col, item)
            recs = self.timeframe_tracker.adjustment_recommendations(name, effective)
            self.tf_accuracy_table.setItem(row, 5, QTableWidgetItem("; ".join(recs)[:80]))
            row += 1

        summary = []
        for name in sorted(acc.keys()):
            recs = self.timeframe_tracker.adjustment_recommendations(name, effective)
            if recs:
                summary.append(f"<b>{name}:</b> {' | '.join(recs)}")
        self.tf_recommendations.setText("<br>".join(summary) if summary else "暂无数据")

    def record_signal_with_trends(self, signal: Dict, klines_map: Dict):
        self.record_signal(signal)
        if klines_map:
            self.timeframe_tracker.record_signal_with_trends(signal, klines_map)

    def _deploy_best_params(self):
        from PySide6.QtWidgets import QMessageBox
        strategies = self.tracker.all_strategies()
        deployed = []
        for name in strategies:
            stats = self.tracker.strategy_stats(name, self._get_days_filter())
            params = self.optimizer.get_best_params(name, use_exploration=False)
            if not params or stats.get("total", 0) < 3:
                continue
            fname = self._find_strategy_file(name)
            if not fname:
                continue
            # Apply best mutation if available
            applied = self.mutator.apply_best_if_better(fname, stats.get("win_rate", 0), min_improvement=2.0)
            if applied:
                deployed.append(f"{name} → 已应用最优变异")
            elif params:
                deployed.append(f"{name} → 当前即最优 ({stats.get('win_rate',0):.1f}%)")
        msg = "\n".join(deployed) if deployed else "无策略需要更新（需至少3个验证信号）"
        self.status_label.setText(f"✅ 已发布优化参数")
        QMessageBox.information(self, "发布结果", msg)

    def _request_refresh(self):
        """防抖刷新：300ms 内多次请求只执行最后一次"""
        if self._refresh_pending:
            return
        self._refresh_pending = True
        QTimer.singleShot(300, self._do_refresh_all)

    def _do_refresh_all(self):
        """实际刷新，但分步执行避免单次阻塞主线程"""
        self._refresh_pending = False
        self._refresh_overview()
        self._refresh_detail()
        self._refresh_evolution()
        self._refresh_mutations()
        self._refresh_analysis()
        self._refresh_timeframe()
        if hasattr(self, 'ab_strat_a'):
            self._refresh_ab_test()

    def _auto_validate(self):
        if self.okx_client and not self._validating:
            self._run_validation()

    def record_signal(self, signal: Dict):
        """从扫描页面接收信号记录"""
        self.tracker.record_signal(signal)

    def set_client(self, okx_client):
        self.okx_client = okx_client
        if self.auto_trainer:
            self.auto_trainer.okx_client = okx_client

    def _toggle_auto_mode(self):
        """切换自主进化模式"""
        self._auto_mode = self.auto_mode_btn.isChecked()
        if self._auto_mode:
            self.auto_mode_btn.setText("🤖 自主进化: ON")
            self.auto_mode_btn.setChecked(True)
            self.auto_log_group.setVisible(True)
            self._param_timer.stop()
            self._start_auto_trainer()
        else:
            self.auto_mode_btn.setText("🤖 自主进化: OFF")
            self.auto_mode_btn.setChecked(False)
            self._stop_auto_trainer()
            self._param_timer.start(600000)

    def _start_auto_trainer(self):
        """启动自主训练器"""
        if self.auto_trainer and self.auto_trainer.isRunning():
            return
        self.auto_trainer = RLAutoTrainer(
            tracker=self.tracker,
            optimizer=self.optimizer,
            mutator=self.mutator,
            timeframe_tracker=self.timeframe_tracker,
            okx_client=self.okx_client,
        )
        self.auto_trainer.log_signal.connect(self._on_auto_log)
        self.auto_trainer.status_signal.connect(self._on_auto_status)
        self.auto_trainer.validation_done.connect(self._on_auto_validation_done)
        self.auto_trainer.optimization_done.connect(self._on_auto_optimization_done)
        self.auto_trainer.deployment_done.connect(self._on_auto_deployment_done)
        self.auto_trainer.start()
        self._append_auto_log("🚀 自主训练器已启动", "#00ff88")

    def _stop_auto_trainer(self):
        """停止自主训练器"""
        if self.auto_trainer:
            self._safe_stop_trainer()
            self.auto_trainer = None
        self.auto_log_group.setVisible(False)
        self._append_auto_log("⏹ 自主训练器已停止", "#ffaa00")

    def _safe_stop_trainer(self):
        """安全停止训练器（CTRL+C 安全）"""
        trainer = self.auto_trainer
        if not trainer:
            return
        try:
            trainer.stop()
            trainer.quit()
            if not trainer.wait(5000):
                trainer.terminate()
                trainer.wait(2000)
        except Exception:
            pass

    def _on_auto_log(self, message: str, level: str):
        """处理自主训练器日志"""
        colors = {
            "INFO": "#88ccff",
            "SUCCESS": "#00ff88",
            "WARNING": "#ffaa00",
            "ERROR": "#ff6666",
        }
        self._append_auto_log(message, colors.get(level, "#cccccc"))

    def _append_auto_log(self, message: str, color: str = "#cccccc"):
        """追加日志到自主训练日志面板"""
        ts = datetime.now().strftime("%H:%M:%S")
        self.auto_log_text.append(f'<span style="color:#555">[{ts}]</span> <span style="color:{color}">{message}</span>')
        # 保持日志不超过 200 行
        doc = self.auto_log_text.document()
        if doc.blockCount() > 200:
            cursor = self.auto_log_text.textCursor()
            cursor.movePosition(cursor.Start)
            for _ in range(50):
                cursor.movePosition(cursor.EndOfBlock, cursor.KeepAnchor)
                cursor.removeSelectedText()
                cursor.deleteChar()

    def _on_auto_status(self, status: str):
        """处理自主训练器状态更新"""
        self.status_label.setText(status)

    def _on_auto_validation_done(self, count: int):
        self._request_refresh()

    def _on_auto_optimization_done(self, count: int):
        self._request_refresh()

    def _on_auto_deployment_done(self, strategy_name: str):
        self._request_refresh()
        QMessageBox.information(
            self, "策略已自动部署",
            f"策略 {strategy_name} 已自动部署优化版本。\n"
            f"系统将持续监控表现，如胜率下降将自动回退。"
        )

    def closeEvent(self, event):
        """安全关闭，停止自主训练器"""
        if self.auto_trainer and self.auto_trainer.isRunning():
            self.auto_trainer.stop()
            self.auto_trainer.wait(3000)
        if hasattr(self, '_param_timer'):
            self._param_timer.stop()
        super().closeEvent(event)
