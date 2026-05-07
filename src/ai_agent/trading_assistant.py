"""
AI 智能交易助理 - 自主决策代理
具有查看持仓、监控交易、分析扫描结果、执行操作的完整权限。
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.qt_compat import QThread, Signal
from src.trading.position_registry import position_registry


# ──────────────────────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────────────────────

@dataclass
class Permission:
    can_trade:          bool = False   # 允许买入/做空
    can_close:          bool = True    # 允许平仓
    can_scan:           bool = True    # 允许触发扫描
    can_modify_strategy:bool = False   # 允许修改策略参数
    can_stop_strategy:  bool = False   # 允许停止策略
    max_single_usdt:    float = 100.0  # 单笔最大 USDT
    max_daily_loss_pct: float = 5.0    # 日内最大亏损 %（触发停止）
    require_confirm:    bool = True    # 大单需要用户确认

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Permission":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class AgentAction:
    action_type: str          # buy / close / scan / alert / adjust / hold
    params:      dict         # action-specific params
    reason:      str          # LLM 推理说明
    risk_level:  str = "low"  # low / medium / high
    result:      str = ""     # 执行结果（事后填充）
    success:     bool = False
    timestamp:   str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


@dataclass
class AgentCycle:
    cycle_id:   int
    timestamp:  str
    analysis:   str
    actions:    List[AgentAction]
    next_in:    int = 60
    learning:   str = ""


# ──────────────────────────────────────────────────────────────
# 主 Agent 线程
# ──────────────────────────────────────────────────────────────

class TradingAssistant(QThread):
    """
    自主 AI 交易助理。
    每隔 interval 秒：采集全局状态 → LLM 决策 → 执行 → 记录。
    """

    # UI 信号
    cycle_done      = Signal(object)   # AgentCycle
    action_log      = Signal(str, str) # (message, level)
    metrics_upd     = Signal(dict)     # 仪表盘数据
    confirm_req     = Signal(object)   # AgentAction 需要用户确认
    position_alert  = Signal(list)     # 持仓风险预警：list[dict]

    def __init__(
        self,
        llm_client=None,
        okx_client=None,
        trade_executor=None,
        scanner_page=None,
        permission: Permission = None,
        interval: int = 60,
        parent=None,
    ):
        super().__init__(parent)
        self.llm_client     = llm_client
        self.okx_client     = okx_client
        self.trade_executor = trade_executor
        self.scanner_page   = scanner_page  # 可访问扫描结果
        self.permission     = permission or Permission()
        self.interval       = interval

        self._stop_flag  = threading.Event()
        self._cycle_id   = 0
        self._history: List[AgentCycle] = []
        self._confirm_result: Optional[bool] = None
        self._confirm_event  = threading.Event()
        self._last_pos_check: float = 0.0          # 上次持仓监控时间戳
        self._pos_monitor_interval: int = 180      # 3 分钟快速监控间隔

        self._data_path = Path(__file__).resolve().parent.parent.parent / "data" / "assistant_memory.json"
        self._data_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_history()

    # ── 生命周期 ──────────────────────────────────────────────

    def run(self):
        self.action_log.emit("🤖 AI 交易助理已启动，开始自主监控...", "INFO")
        while not self._stop_flag.is_set():
            try:
                self._run_cycle()
            except Exception as e:
                self.action_log.emit(f"⚠️ Agent 循环异常: {e}", "ERROR")
                self.action_log.emit(traceback.format_exc(), "DEBUG")
            # 等待下一轮（可中途唤醒）
            self._stop_flag.wait(self.interval)

    def stop(self):
        self._stop_flag.set()
        self.action_log.emit("🛑 AI 助理已停止", "INFO")

    # ── 核心循环 ──────────────────────────────────────────────

    def _run_cycle(self):
        self._cycle_id += 1
        self.action_log.emit(f"── 第 {self._cycle_id} 轮分析开始 ──", "INFO")

        # 1. 采集状态
        state = self._gather_state()
        self._emit_metrics(state)

        # 2. LLM 决策
        if not self.llm_client:
            self.action_log.emit("未配置 LLM，跳过 AI 决策", "WARNING")
            return

        decision = self._make_decision(state)
        if not decision:
            return

        analysis = decision.get("analysis", "")
        actions_raw = decision.get("actions", [])
        next_in = int(decision.get("next_check_in", self.interval))
        learning = decision.get("learning", "")

        self.action_log.emit(f"📊 分析：{analysis}", "ANALYSIS")

        # 3. 执行动作
        executed: List[AgentAction] = []
        for raw in actions_raw:
            act = AgentAction(
                action_type=raw.get("type", "hold"),
                params=raw,
                reason=raw.get("reason", ""),
                risk_level=raw.get("risk_level", "low"),
            )
            self._execute_action(act)
            executed.append(act)

        # 4. 记录本轮
        cycle = AgentCycle(
            cycle_id=self._cycle_id,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            analysis=analysis,
            actions=executed,
            next_in=next_in,
            learning=learning,
        )
        self._history.append(cycle)
        if len(self._history) > 100:
            self._history = self._history[-100:]
        self._save_history()

        self.cycle_done.emit(cycle)
        self.interval = max(30, min(600, next_in))
        self.action_log.emit(f"✅ 本轮完成，下次检查 {self.interval}s 后", "INFO")

        # 扫描信号为空时，每 3 分钟快速监控持仓
        if not state["scan_results"] and state["positions"]:
            now = time.time()
            if now - self._last_pos_check >= self._pos_monitor_interval:
                self._last_pos_check = now
                self._quick_position_monitor(state)

    # ── 状态采集 ──────────────────────────────────────────────

    def _gather_state(self) -> dict:
        state: dict = {
            "timestamp": datetime.now().isoformat(),
            "balance_usdt": 0.0,
            "positions": [],
            "scan_results": [],
            "daily_pnl_pct": 0.0,
            "history_summary": self._history_summary(),
        }

        # 账户余额
        if self.trade_executor:
            try:
                state["balance_usdt"] = self.trade_executor.get_usdt_balance()
            except Exception:
                pass

            # 持仓（PositionInfo 字段：entry_price / unrealized_pnl / pnl_percent）
            try:
                positions = self.trade_executor.get_positions()
                for inst_id, pos in positions.items():
                    side_obj  = getattr(pos, "side", None)
                    side_str  = side_obj.name if hasattr(side_obj, "name") else str(side_obj)
                    upl       = float(getattr(pos, "unrealized_pnl", 0) or 0)
                    upl_ratio = float(getattr(pos, "pnl_percent",    0) or 0)  # 已是 %
                    state["positions"].append({
                        "inst_id":   inst_id,
                        "side":      side_str,
                        "size":      float(getattr(pos, "size",        0) or 0),
                        "avg_px":    float(getattr(pos, "entry_price", 0) or 0),
                        "upl":       upl,
                        "upl_ratio": upl_ratio / 100,  # 统一存为小数，显示时 ×100
                    })
                    state["daily_pnl_pct"] += upl_ratio
            except Exception:
                pass

        # 扫描结果（取最近 top 20）
        if self.scanner_page:
            try:
                results = getattr(self.scanner_page, "_scan_results", [])
                for r in results[:20]:
                    state["scan_results"].append({
                        "symbol": getattr(r, "symbol", str(r)),
                        "score":  getattr(r, "score", 0),
                        "signal": getattr(r, "signal", ""),
                        "reason": getattr(r, "reason", ""),
                    })
            except Exception:
                pass

        return state

    def _emit_metrics(self, state: dict):
        self.metrics_upd.emit({
            "balance": state["balance_usdt"],
            "positions": len(state["positions"]),
            "pnl_pct": state["daily_pnl_pct"],
            "scan_count": len(state["scan_results"]),
            "cycle": self._cycle_id,
        })

    # ── LLM 决策 ──────────────────────────────────────────────

    def _make_decision(self, state: dict) -> Optional[dict]:
        positions_str = json.dumps(state["positions"], ensure_ascii=False, indent=2)
        scan_str = json.dumps(state["scan_results"][:10], ensure_ascii=False, indent=2)
        perm = self.permission

        system_prompt = f"""你是一个专业量化交易 AI 助理，负责自主管理加密货币投资组合。

【当前权限】
- 买入/做空: {'✅允许' if perm.can_trade else '❌禁止'}
- 平仓: {'✅允许' if perm.can_close else '❌禁止'}
- 触发扫描: {'✅允许' if perm.can_scan else '❌禁止'}
- 修改策略: {'✅允许' if perm.can_modify_strategy else '❌禁止'}
- 单笔上限: {perm.max_single_usdt} USDT
- 日亏损止损线: {perm.max_daily_loss_pct}%

【当前账户状态】
余额: {state['balance_usdt']:.2f} USDT
持仓数: {len(state['positions'])}
日内盈亏: {state['daily_pnl_pct']:.2f}%

【当前持仓】
{positions_str}

【最新扫描信号（Top 10）】
{scan_str}

【历史决策摘要】
{state['history_summary']}

【任务】
基于以上信息，制定本轮操作计划。严格遵守权限限制。
输出必须为合法 JSON，格式如下：
{{
  "analysis": "市场分析（1-3句）",
  "risk_level": "low|medium|high",
  "actions": [
    {{
      "type": "buy|close|scan|alert|hold|adjust",
      "inst_id": "BTC-USDT-SWAP",
      "usdt_amount": 50,
      "side": "buy|sell",
      "reason": "操作理由",
      "risk_level": "low|medium|high"
    }}
  ],
  "next_check_in": 60,
  "learning": "本轮学到的规律（可选）"
}}

action type 说明：
- buy: 买入/做空，需要 inst_id、usdt_amount、side
- close: 平仓，需要 inst_id
- scan: 触发新一轮扫描，无需额外参数
- alert: 发出风险预警，需要 message 字段
- hold: 无操作，继续观察
- adjust: 调整风控参数，需要 field 和 value
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "请分析当前市场状态并输出操作决策 JSON。"},
        ]

        self.action_log.emit("🧠 正在调用 LLM 决策...", "INFO")
        raw = self.llm_client.chat(messages, timeout=60)
        if not raw:
            self.action_log.emit(f"LLM 无响应: {self.llm_client.last_error}", "ERROR")
            return None

        # 提取 JSON
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except json.JSONDecodeError as e:
            self.action_log.emit(f"JSON 解析失败: {e}\n原始: {raw[:200]}", "ERROR")
        return None

    # ── 动作执行 ──────────────────────────────────────────────

    def _execute_action(self, act: AgentAction):
        perm = self.permission
        t = act.action_type

        if t == "hold":
            self.action_log.emit(f"⏸ 保持观察：{act.reason}", "INFO")
            act.success = True
            return

        if t == "alert":
            msg = act.params.get("message", act.reason)
            self.action_log.emit(f"🚨 预警：{msg}", "WARNING")
            act.success = True
            act.result = "预警已发出"
            return

        if t == "scan":
            if not perm.can_scan:
                self.action_log.emit("扫描权限已关闭，跳过", "WARNING")
                act.result = "权限拒绝"
                return
            self.action_log.emit(f"🔍 触发市场扫描：{act.reason}", "INFO")
            if self.scanner_page:
                try:
                    from src.qt_compat import QTimer
                    QTimer.singleShot(0, lambda: self._trigger_scan())
                    act.success = True
                    act.result = "扫描已触发"
                except Exception as e:
                    act.result = f"触发失败: {e}"
            return

        if t == "buy":
            if not perm.can_trade:
                self.action_log.emit("交易权限已关闭，跳过买入", "WARNING")
                act.result = "权限拒绝"
                return
            usdt = float(act.params.get("usdt_amount", 0))
            inst_id = act.params.get("inst_id", "")
            if usdt > perm.max_single_usdt:
                self.action_log.emit(
                    f"⛔ {inst_id} 下单 {usdt} USDT 超过单笔上限 {perm.max_single_usdt}", "WARNING")
                act.result = "超出单笔上限"
                return
            # ── 注册器检查：该标的是否已被其他系统管理 ────────────────────────
            if not position_registry.try_lock(inst_id, 'TradingAssistant'):
                _owner = position_registry.get_owner(inst_id)
                self.action_log.emit(
                    f"⛔ {inst_id} 已由 {_owner} 持有，TradingAssistant 买入被拒绝",
                    "WARNING",
                )
                act.result = f"注册器拒绝（{_owner} 持有中）"
                return
            if perm.require_confirm and act.risk_level != "low":
                confirmed = self._request_confirm(act)
                if not confirmed:
                    position_registry.release(inst_id, 'TradingAssistant')
                    act.result = "用户取消"
                    return
            self.action_log.emit(f"📈 执行买入 {inst_id} {usdt} USDT：{act.reason}", "TRADE")
            if self.trade_executor:
                try:
                    # 全局资金池检查：防止 AI agent 与自动交易同时超额
                    from src.trading.risk_manager import RiskGuard
                    total_exposure = sum(
                        float(getattr(p, 'notional', 0) or 0)
                        for p in (self.trade_executor.get_positions().values())
                    )
                    max_total = float(self.trade_executor.get_usdt_balance()) * 3.0
                    if total_exposure + usdt > max_total:
                        position_registry.release(inst_id, 'TradingAssistant')
                        act.result = f"全局敞口超限({total_exposure:.0f}+{usdt}>{max_total:.0f})"
                        self.action_log.emit(f"⛔ {act.result}", "WARNING")
                        return
                    ratio = usdt / max(self.trade_executor.get_usdt_balance(), 1)
                    result = self.trade_executor.execute_buy(inst_id, position_ratio=ratio)
                    act.success = result.success
                    act.result = f"成交={getattr(result, 'filled_size', '?')}" if result.success else result.message
                    if not result.success:
                        position_registry.release(inst_id, 'TradingAssistant')
                    self.action_log.emit(
                        f"{'✅' if result.success else '❌'} 买入结果: {act.result}",
                        "SUCCESS" if result.success else "ERROR")
                except Exception as e:
                    position_registry.release(inst_id, 'TradingAssistant')
                    act.result = str(e)
                    self.action_log.emit(f"❌ 买入异常: {e}", "ERROR")
            return

        if t == "close":
            if not perm.can_close:
                self.action_log.emit("平仓权限已关闭，跳过", "WARNING")
                act.result = "权限拒绝"
                return
            inst_id = act.params.get("inst_id", "")
            # 平仓前检查注册器：不允许平掉其他系统持有的仓位
            _owner = position_registry.get_owner(inst_id)
            if _owner and _owner != 'TradingAssistant':
                act.result = f"平仓拒绝：{inst_id} 由 {_owner} 持有"
                self.action_log.emit(f"⛔ {act.result}", "WARNING")
                return
            self.action_log.emit(f"📉 执行平仓 {inst_id}：{act.reason}", "TRADE")
            if self.trade_executor:
                try:
                    result = self.trade_executor.execute_stop_loss(inst_id)
                    act.success = result.success
                    act.result = "平仓成功" if result.success else result.message
                    if result.success:
                        position_registry.release(inst_id, 'TradingAssistant')
                    self.action_log.emit(
                        f"{'✅' if result.success else '❌'} 平仓结果: {act.result}",
                        "SUCCESS" if result.success else "ERROR")
                except Exception as e:
                    act.result = str(e)
                    self.action_log.emit(f"❌ 平仓异常: {e}", "ERROR")
            return

        if t == "adjust":
            f_name = act.params.get("field", "")
            f_val  = act.params.get("value")
            self.action_log.emit(f"⚙️ 调整参数 {f_name}={f_val}：{act.reason}", "INFO")
            if hasattr(self.permission, f_name):
                setattr(self.permission, f_name, f_val)
                act.success = True
                act.result = f"已设置 {f_name}={f_val}"
            return

    # ── 持仓快速监控（扫描空窗期每 3 分钟执行） ──────────────────

    def _quick_position_monitor(self, state: dict):
        positions = state["positions"]
        self.action_log.emit(
            f"📡 持仓快速监控 — {len(positions)} 个仓位", "INFO")

        alerts = []
        for pos in positions:
            inst_id   = pos["inst_id"]
            upl_ratio = float(pos.get("upl_ratio", 0) or 0) * 100  # state 存小数，显示转 %
            upl       = float(pos.get("upl", 0) or 0)
            side      = pos.get("side", "?")
            avg_px    = float(pos.get("avg_px", 0) or 0)

            # 获取实时 ticker 24h 涨跌
            recent_chg = 0.0
            last_px    = 0.0
            if self.okx_client:
                try:
                    res = self.okx_client.get_ticker(inst_id)
                    if res.get("code") == "0" and res.get("data"):
                        d       = res["data"][0]
                        last_px = float(d.get("last", 0) or 0)
                        open8   = float(d.get("sodUtc8", d.get("open24h", last_px)) or last_px)
                        if open8:
                            recent_chg = (last_px - open8) / open8 * 100
                except Exception:
                    pass

            # 判断风险等级
            flags = []
            risk  = "low"
            RISK_LEVELS = ["low", "medium", "high"]

            if upl_ratio <= -5:
                risk = "high"; flags.append(f"浮亏{upl_ratio:.1f}%")
            elif upl_ratio <= -2:
                risk = "medium"; flags.append(f"浮亏{upl_ratio:.1f}%")

            if abs(recent_chg) >= 5:
                risk = RISK_LEVELS[max(RISK_LEVELS.index(risk), 2)]
                flags.append(f"24h变动{recent_chg:+.1f}%")
            elif abs(recent_chg) >= 3:
                risk = RISK_LEVELS[max(RISK_LEVELS.index(risk), 1)]
                flags.append(f"24h变动{recent_chg:+.1f}%")

            if upl_ratio >= 8:
                flags.append(f"浮盈{upl_ratio:.1f}%，可考虑止盈")
                if risk == "low":
                    risk = "medium"

            alerts.append({
                "inst_id":   inst_id,
                "side":      side,
                "upl_ratio": upl_ratio,
                "upl":       upl,
                "last_px":   last_px,
                "avg_px":    avg_px,
                "recent_chg": recent_chg,
                "risk":      risk,
                "flags":     flags,
                "recommendation": "",
                "rec_reason":     "",
                "urgency":        risk,
            })

        if not alerts:
            return

        # LLM 给出操作建议
        if self.llm_client:
            alerts = self._llm_position_advice(alerts)

        # 记录到日志
        for a in alerts:
            color_lv = "WARNING" if a["risk"] != "low" else "INFO"
            flag_str = "、".join(a["flags"]) if a["flags"] else "正常"
            self.action_log.emit(
                f"📊 {a['inst_id']} {a['side']} | 浮盈亏{a['upl_ratio']:+.1f}% | "
                f"{flag_str} → {a['recommendation']}", color_lv)

        self.position_alert.emit(alerts)

    def _llm_position_advice(self, alerts: list) -> list:
        """调用 LLM 为每个持仓给出建议"""
        pos_text = "\n".join(
            "- {inst_id} {side} | 浮盈亏{upl:.1f}% | 24h{chg:+.1f}% | 风险:{risk} | {flags}".format(
                inst_id=a["inst_id"], side=a["side"],
                upl=a["upl_ratio"], chg=a["recent_chg"],
                risk=a["risk"],
                flags="、".join(a["flags"]) if a["flags"] else "正常")
            for a in alerts
        )
        system = (
            "你是专业持仓风控顾问。根据持仓状态给出简洁操作建议。\n"
            "每个持仓必须给出以下之一：减仓 / 部分减仓 / 加仓 / 保持持仓 / 立即止损 / 止盈出场\n"
            "输出严格合法 JSON，格式：\n"
            '{"positions":{"BTC-USDT-SWAP":{"recommendation":"保持持仓","reason":"20字内理由","urgency":"low|medium|high"}}}'
        )
        raw = self.llm_client.chat(
            [{"role": "system", "content": system},
             {"role": "user",   "content": f"持仓状态:\n{pos_text}\n请给出建议。"}],
            timeout=30,
        )
        if raw:
            try:
                s = raw.find("{"); e = raw.rfind("}") + 1
                data = json.loads(raw[s:e])
                pos_data = data.get("positions", {})
                for a in alerts:
                    iid = a["inst_id"]
                    if iid in pos_data:
                        a["recommendation"] = pos_data[iid].get("recommendation", "保持持仓")
                        a["rec_reason"]     = pos_data[iid].get("reason", "")
                        a["urgency"]        = pos_data[iid].get("urgency", a["risk"])
            except Exception:
                pass
        # 兜底
        for a in alerts:
            if not a["recommendation"]:
                a["recommendation"] = "保持持仓" if a["risk"] == "low" else "关注风险"
        return alerts

    def _trigger_scan(self):
        try:
            if self.scanner_page and hasattr(self.scanner_page, "start_scan"):
                self.scanner_page.start_scan()
        except Exception as e:
            self.action_log.emit(f"扫描触发失败: {e}", "ERROR")

    def _request_confirm(self, act: AgentAction) -> bool:
        self._confirm_result = None
        self._confirm_event.clear()
        self.confirm_req.emit(act)
        self._confirm_event.wait(timeout=30)
        return bool(self._confirm_result)

    def set_confirm_result(self, result: bool):
        self._confirm_result = result
        self._confirm_event.set()

    # ── 历史记录 ──────────────────────────────────────────────

    def _history_summary(self) -> str:
        if not self._history:
            return "无历史记录"
        lines = []
        for c in self._history[-5:]:
            n_success = sum(1 for a in c.actions if a.success)
            lines.append(f"[{c.timestamp}] {c.analysis[:60]}... 执行{len(c.actions)}个动作，{n_success}成功")
        return "\n".join(lines)

    def _save_history(self):
        try:
            data = []
            for c in self._history[-50:]:
                data.append({
                    "cycle_id": c.cycle_id,
                    "timestamp": c.timestamp,
                    "analysis": c.analysis,
                    "actions": [asdict(a) for a in c.actions],
                    "next_in": c.next_in,
                    "learning": c.learning,
                })
            self._data_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception:
            pass

    def _load_history(self):
        try:
            if self._data_path.exists():
                data = json.loads(self._data_path.read_text())
                for d in data[-20:]:
                    actions = [AgentAction(**a) for a in d.get("actions", [])]
                    self._history.append(AgentCycle(
                        cycle_id=d["cycle_id"],
                        timestamp=d["timestamp"],
                        analysis=d["analysis"],
                        actions=actions,
                        next_in=d.get("next_in", 60),
                        learning=d.get("learning", ""),
                    ))
        except Exception:
            pass

    def get_history(self) -> List[AgentCycle]:
        return list(self._history)
