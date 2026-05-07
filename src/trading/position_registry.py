"""
全局持仓注册器（Singleton）

防止 ScanDrivenAutoTrader / StrategyRunner / TradingAssistant / MultiAgentTrader
对同一交易对重复建仓或方向冲突。

规则：
  • 开仓前先调用 try_lock(inst_id, system) → False 则中止
  • 平仓/退出后调用 release(inst_id, system)（仅释放自身持有的锁）
  • 强制接管用 force_takeover(inst_id, new_system) → 通知原 owner 清状态
  • 任何系统可查询 get_owner(inst_id) 判断谁在管理该标的
"""
from __future__ import annotations

import threading
from typing import Callable, Dict, List, Optional

# callback(inst_id: str, old_owner: str, new_owner: str)
TakeoverCallback = Callable[[str, str, str], None]


class _PositionRegistry:
    """持仓注册器核心（线程安全）。"""

    def __init__(self) -> None:
        self._mu: threading.Lock = threading.Lock()
        self._owners: Dict[str, str] = {}            # inst_id → system_name
        self._takeover_cbs: List[TakeoverCallback] = []  # 强制接管通知回调

    # ── 回调注册 ────────────────────────────────────────────────────────────

    def register_takeover_callback(self, cb: TakeoverCallback) -> None:
        """
        注册强制接管通知回调。

        cb 签名为 (inst_id: str, old_owner: str, new_owner: str)。
        当 force_takeover 从 *old_owner* 抢走 *inst_id* 时触发，
        *old_owner* 应在此回调中清空自己的内部状态机。
        """
        with self._mu:
            self._takeover_cbs.append(cb)

    def unregister_takeover_callback(self, cb: TakeoverCallback) -> None:
        """移除先前注册的接管通知回调。"""
        with self._mu:
            try:
                self._takeover_cbs.remove(cb)
            except ValueError:
                pass

    # ── 基本操作 ────────────────────────────────────────────────────────────

    def try_lock(self, inst_id: str, system: str) -> bool:
        """
        尝试为 *system* 锁定 *inst_id*。

        Returns:
            True  — 成功锁定（新锁 或 自身续锁）
            False — 已被 **其他** 系统占用
        """
        with self._mu:
            owner = self._owners.get(inst_id, "")
            if owner and owner != system:
                return False
            self._owners[inst_id] = system
            return True

    def release(self, inst_id: str, system: str) -> None:
        """
        释放 *inst_id* 的锁（仅当持有者与 *system* 一致时才释放）。
        跨系统强制释放请使用 force_takeover()。
        """
        with self._mu:
            if self._owners.get(inst_id, "") != system:
                return
            self._owners.pop(inst_id, None)

    def force_takeover(self, inst_id: str, new_system: str) -> str:
        """
        强制接管 *inst_id* 的锁，不论当前持有者是谁。

        与 release('') 的区别：显式语义 + 通知原 owner。
        返回旧 owner 名称（空字符串表示原本无人持有）。

        调用方应先平仓/清理持仓，再调用 force_takeover 接管注册器，
        这样旧 owner 收到回调时外部持仓已不存在，只需清理内部状态机。
        """
        with self._mu:
            old_owner = self._owners.get(inst_id, "")
            if old_owner == new_system:
                return old_owner
            self._owners[inst_id] = new_system
            cbs = list(self._takeover_cbs)  # 锁内复制，锁外回调

        # 通知旧 owner 清理内部状态（锁外执行，避免死锁）
        if old_owner and old_owner != new_system:
            for cb in cbs:
                try:
                    cb(inst_id, old_owner, new_system)
                except Exception:
                    pass
        return old_owner

    def is_locked(self, inst_id: str) -> bool:
        with self._mu:
            return inst_id in self._owners

    def get_owner(self, inst_id: str) -> str:
        with self._mu:
            return self._owners.get(inst_id, "")

    def snapshot(self) -> Dict[str, str]:
        """返回当前注册表快照（调试用）。"""
        with self._mu:
            return dict(self._owners)

    def release_by_system(self, system: str) -> int:
        """释放指定系统持有的所有锁，返回释放数量（系统停止时批量清理）。"""
        with self._mu:
            keys = [k for k, v in self._owners.items() if v == system]
            for k in keys:
                del self._owners[k]
            return len(keys)


# ── 模块级单例 ──────────────────────────────────────────────────────────────
position_registry: _PositionRegistry = _PositionRegistry()
