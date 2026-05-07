"""
策略顾问：将扫描结果、验证数据、策略代码喂给大模型，获取优化建议。
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.ai_agent.llm_client import LLMClient
from src.ai_agent.prompt_templates import (
    SYSTEM_PROMPT,
    ANALYSIS_PROMPT,
    CODE_REVIEW_PROMPT,
    MUTATION_PROMPT,
    SELF_REFLECTION_PROMPT,
    SCANNING_EVOLUTION_PROMPT,
    STRATEGY_HEALTH_PROMPT,
)


class StrategyAdvisor:
    """
    量化策略 AI 顾问。
    连接大模型，基于扫描+验证数据生成代码优化建议。

    用法:
        advisor = StrategyAdvisor(llm_client, tracker, mutator, timeframe_tracker)
        result = advisor.analyze_strategy("AI订单流动量")
    """

    def __init__(self, llm_client: LLMClient, tracker, mutator, timeframe_tracker, optimizer=None):
        self.llm = llm_client
        self.tracker = tracker
        self.mutator = mutator
        self.timeframe_tracker = timeframe_tracker
        self.optimizer = optimizer
        self.conversation_history: List[Dict[str, Any]] = []

    def analyze_strategy(self, strategy_name: str, days: int = 30) -> Optional[Dict[str, Any]]:
        """完整分析一个策略并返回优化建议"""
        stats = self.tracker.strategy_stats(strategy_name, days=days)
        if stats.get("total", 0) < 3:
            return {"error": "数据不足，至少需要 3 条信号"}

        # 截断策略代码：只保留参数/配置部分（前60行 + 含 '=' 'param' 'config' 的关键行）
        raw_code = self._load_strategy_code(strategy_name) or "# 未找到策略代码"
        code = self._truncate_code(raw_code, max_lines=120)
        code = code.replace("{", "{{").replace("}", "}}")

        # 精简信号数据
        signals = self.tracker.strategy_recent_signals(strategy_name, limit=10)
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

        tf_accuracy = self.timeframe_tracker.accuracy_by_timeframe(strategy_name, days)
        param_history = self._format_param_history(strategy_name)

        prompt = ANALYSIS_PROMPT.format(
            strategy_name=strategy_name,
            days=days,
            total_signals=stats.get("total", 0),
            win_rate=stats.get("win_rate", 0),
            profit_factor=stats.get("profit_factor", 0),
            net_pnl=stats.get("net_pnl", 0),
            signal_samples=json.dumps(signals_brief, indent=2, ensure_ascii=False, default=str),
            timeframe_accuracy=json.dumps(tf_accuracy, indent=2, ensure_ascii=False),
            param_history=json.dumps(param_history, indent=2, ensure_ascii=False),
            strategy_code=code,
        )

        reply = self.llm.chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ], timeout=180)

        if not reply:
            return {"error": self.llm.last_error or "LLM 请求失败"}

        result = {
            "strategy_name": strategy_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stats": stats,
            "advice": reply,
            "raw_message": reply,
        }
        self.conversation_history.append(result)
        return result

    @staticmethod
    def _truncate_code(code: str, max_lines: int = 120) -> str:
        """截断策略代码：保留首部 + 含关键模式的行"""
        lines = code.split("\n")
        if len(lines) <= max_lines:
            return code
        head = lines[:60]
        tail = []
        for line in lines[60:]:
            if re.search(r'\b(min_score|max_|weight|threshold|penalty|bonus|learning_rate|max_depth|lambda)\b', line, re.IGNORECASE):
                tail.append(line)
            elif re.search(r'^\s*(self\.)?\w+\s*=\s*[\d.]+', line):
                tail.append(line)
        combined = head + tail[-60:]
        return "\n".join(combined[:max_lines])

    def review_code(
        self, strategy_name: str, strategy_code: str, optimized_params: Dict[str, Any]
    ) -> Optional[str]:
        """让 AI 审查代码并提出具体修改"""
        stats = self.tracker.strategy_stats(strategy_name, days=30)
        safe_code = (strategy_code or "").replace("{", "{{").replace("}", "}}")
        prompt = CODE_REVIEW_PROMPT.format(
            strategy_code=safe_code,
            optimized_params=json.dumps(optimized_params, indent=2, ensure_ascii=False),
            win_rate=stats.get("win_rate", 0),
            total_signals=stats.get("total", 0),
        )
        return self.llm.chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])

    def suggest_mutations(
        self, strategy_name: str, current_params: Dict[str, float],
        lo: float = 0, hi: float = 100
    ) -> Optional[List[Dict]]:
        """让 AI 建议参数变异值"""
        param_effects = self._format_param_effects(strategy_name)
        prompt = MUTATION_PROMPT.format(
            current_params=json.dumps(current_params, indent=2, ensure_ascii=False),
            param_effects=json.dumps(param_effects, indent=2, ensure_ascii=False),
            lo=lo,
            hi=hi,
        )
        reply = self.llm.chat([
            {"role": "system", "content": "你是量化交易参数优化专家。只输出 JSON，不要解释。"},
            {"role": "user", "content": prompt},
        ])
        if not reply:
            return None
        try:
            json_start = reply.find("```json") + 7
            json_end = reply.rfind("```")
            if json_start > 6 and json_end > json_start:
                return json.loads(reply[json_start:json_end])
            json_start = reply.find("[")
            json_end = reply.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(reply[json_start:json_end])
        except (json.JSONDecodeError, ValueError):
            return None
        return None

    def _load_strategy_code(self, strategy_name: str) -> Optional[str]:
        fname = self._find_strategy_file(strategy_name)
        if not fname:
            return None
        fpath = Path(self.mutator._strategies_dir) / fname
        try:
            return fpath.read_text(encoding="utf-8")
        except Exception:
            return None

    def _find_strategy_file(self, strategy_name: str) -> str:
        strategies_dir = Path(self.mutator._strategies_dir)
        # 精确匹配：查找含 name = "xxx" 或 STRATEGY_NAME = "xxx" 的文件
        for f in strategies_dir.glob("*.py"):
            if f.name.startswith("_"):
                continue
            try:
                content = f.read_text(encoding="utf-8")
                if f'name = "{strategy_name}"' in content or f'STRATEGY_NAME = "{strategy_name}"' in content:
                    return f.name
            except Exception:
                continue
        # 关键词模糊匹配：策略名去掉通用词后与文件名比对
        stop_words = {"AI", "截面", "因子", "扫描", "策略", "组合", "引擎"}
        keywords = strategy_name
        for w in stop_words:
            keywords = keywords.replace(w, "")
        keywords = keywords.replace(" ", "").strip()
        if keywords:
            for f in strategies_dir.glob("*.py"):
                if f.name.startswith("_"):
                    continue
                if keywords.lower() in f.name.lower():
                    return f.name
        # 无匹配时返回空字符串，不猜测（避免操作错误文件）
        return ""

    def _format_param_history(self, strategy_name: str) -> List[Dict]:
        if not self.optimizer:
            return []
        return [
            h for h in self.optimizer.evolution_summary()
            if h.get("strategy") == strategy_name
        ][-10:]

    def _format_param_effects(self, strategy_name: str) -> List[Dict]:
        if not self.optimizer:
            return []
        importance = self.optimizer.param_importance(strategy_name)
        return [
            {"param": k, "value": v.get("value", 0), "win_rate": v.get("win_rate", 0),
             "trials": v.get("trials", 0), "importance": v.get("importance", 0)}
            for k, v in sorted(importance.items(), key=lambda x: -x[1]["importance"])[:10]
        ]

    # ─── 自我反思：元学习 ───────────────────────────────────────────

    def self_reflect(self, change_history: List[Dict], days: int = 7) -> Optional[Dict]:
        """
        让 AI 分析过去优化建议的成效，提炼规律，指导下一轮优化。
        返回结构化的元学习洞察，供 SelfEvolver 使用。
        """
        if not change_history:
            return None

        # 收集各策略当前绩效
        current_perf = {}
        if self.tracker:
            for name in (self.tracker.all_strategies() or []):
                s = self.tracker.strategy_stats(name, days=days)
                current_perf[name] = {
                    "win_rate": s.get("win_rate", 0),
                    "total": s.get("total", 0),
                    "profit_factor": s.get("profit_factor", 0),
                }

        prompt = SELF_REFLECTION_PROMPT.format(
            n_records=len(change_history),
            history_json=json.dumps(change_history[-30:], indent=2, ensure_ascii=False, default=str),
            current_perf_json=json.dumps(current_perf, indent=2, ensure_ascii=False),
        )

        reply = self.llm.chat([
            {"role": "system", "content": "你是一个能自我学习的量化策略优化 AI。只输出 JSON，不要任何解释。"},
            {"role": "user", "content": prompt},
        ], timeout=120)

        if not reply:
            return None

        try:
            j_start = reply.find("```json") + 7
            j_end = reply.rfind("```")
            if j_start > 6 and j_end > j_start:
                return json.loads(reply[j_start:j_end])
            j_start = reply.find("{")
            j_end = reply.rfind("}") + 1
            if j_start >= 0 and j_end > j_start:
                return json.loads(reply[j_start:j_end])
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def evolve_scanning_params(
        self,
        strategy_name: str,
        days: int = 14,
        meta_insight: str = "",
    ) -> Optional[Dict]:
        """
        深度进化扫描策略：基于成功/失败信号特征 + 元学习洞察，生成新的扫描逻辑优化。
        相比 analyze_strategy，这里会分析信号质量分布，而不只是统计数据。
        """
        stats = self.tracker.strategy_stats(strategy_name, days=days)
        if stats.get("total", 0) < 5:
            return {"error": "数据不足"}

        # 按结果拆分信号样本
        all_signals = self.tracker.strategy_recent_signals(strategy_name, limit=50)
        success_signals = [s for s in all_signals if s.get("validated") and s.get("net_pnl", 0) > 0][:10]
        failure_signals = [s for s in all_signals if s.get("validated") and s.get("net_pnl", 0) <= 0][:10]

        def _brief(signals):
            return [{
                "symbol": s.get("symbol", ""),
                "dir": s.get("direction", ""),
                "score": round(float(s.get("score", 0) or 0), 1),
                "net_pnl": round(float(s.get("net_pnl", 0) or 0), 2),
                "params_snapshot": {
                    k: v for k, v in (s.get("param_snapshot") or {}).items()
                }
            } for s in signals]

        # 获取参数空间信息（如果 optimizer 有的话）
        param_space = {}
        if self.optimizer:
            param_space = getattr(self.optimizer, "_param_space", {}).get(strategy_name, {})

        evolution_history = self._format_param_history(strategy_name)
        raw_code = self._load_strategy_code(strategy_name) or "# 未找到策略代码"
        code = self._truncate_code(raw_code, max_lines=80)
        code = code.replace("{", "{{").replace("}", "}}")

        prompt = SCANNING_EVOLUTION_PROMPT.format(
            strategy_name=strategy_name,
            total_signals=stats.get("total", 0),
            win_rate=stats.get("win_rate", 0),
            profit_factor=stats.get("profit_factor", 0),
            net_pnl=stats.get("net_pnl", 0),
            sharpe=round(stats.get("sharpe", 0), 2),
            failure_patterns=json.dumps(_brief(failure_signals), indent=2, ensure_ascii=False, default=str),
            success_patterns=json.dumps(_brief(success_signals), indent=2, ensure_ascii=False, default=str),
            param_space=json.dumps(param_space, indent=2, ensure_ascii=False),
            evolution_history=json.dumps(evolution_history, indent=2, ensure_ascii=False, default=str),
            meta_insight=meta_insight or "（暂无元学习洞察）",
        )

        reply = self.llm.chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ], timeout=180)

        if not reply:
            return {"error": self.llm.last_error or "LLM 无响应"}

        result = {
            "strategy_name": strategy_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stats": stats,
            "advice": reply,
            "mode": "evolution",
            "meta_insight": meta_insight,
        }
        self.conversation_history.append(result)
        return result

    def quick_health_check(self, strategy_names: List[str], days: int = 7) -> Optional[Dict]:
        """快速诊断多个策略的健康状态，返回优先级排序的修复建议。"""
        snapshot = {}
        sig_dist = {}
        for name in strategy_names:
            s = self.tracker.strategy_stats(name, days=days)
            snapshot[name] = {
                "win_rate": s.get("win_rate", 0),
                "total": s.get("total", 0),
                "profit_factor": s.get("profit_factor", 0),
            }
            recent = self.tracker.strategy_recent_signals(name, limit=5)
            sig_dist[name] = [
                {"dir": sig.get("direction", ""), "score": round(float(sig.get("score", 0) or 0), 1)}
                for sig in recent
            ]

        prompt = STRATEGY_HEALTH_PROMPT.format(
            health_snapshot=json.dumps(snapshot, indent=2, ensure_ascii=False),
            signal_distribution=json.dumps(sig_dist, indent=2, ensure_ascii=False),
        )

        reply = self.llm.chat([
            {"role": "system", "content": "你是量化策略诊断专家。只输出 JSON，不要解释。"},
            {"role": "user", "content": prompt},
        ], timeout=60)

        if not reply:
            return None
        try:
            j_start = reply.find("```json") + 7
            j_end = reply.rfind("```")
            if j_start > 6 and j_end > j_start:
                return json.loads(reply[j_start:j_end])
            j_start = reply.find("{")
            j_end = reply.rfind("}") + 1
            if j_start >= 0 and j_end > j_start:
                return json.loads(reply[j_start:j_end])
        except (json.JSONDecodeError, ValueError):
            pass
        return None
