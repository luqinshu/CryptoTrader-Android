"""
RL自主训练器：后台持续运行，自动验证信号→优化参数→变异代码→安全部署。
v2: 修复 update_from_stats 空实现问题，用真实数据驱动优化循环。
"""

from __future__ import annotations

import gc
import hashlib
import random
import re
import shutil
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from src.qt_compat import QThread, Signal


class RLAutoTrainer(QThread):
    """
    自主强化学习训练器。

    真实学习循环（每 10 分钟）：
    1. 验证信号 → 存入 tracker
    2. tracker 有足够验证数据 → optimizer.update_from_samples()
       （参数值 → 实际胜率 → 贝叶斯后验评分）
    3. optimizer.get_best_params() → Thompson Sampling 选出最优参数
    4. 将最优参数写入策略代码（定向变异）
    5. 评估安全门槛 → 部署

    安全门控：
    - 最少验证样本：10
    - 最低胜率：50%
    - 参数变异必须由 optimizer 数据驱动
    - 部署冷却期：24 小时
    """

    log_signal = Signal(str, str)
    status_signal = Signal(str)
    validation_done = Signal(int)
    optimization_done = Signal(int)
    deployment_done = Signal(str)

    MIN_SAMPLES = 10
    MIN_WIN_RATE = 50.0
    DEPLOY_COOLDOWN_HOURS = 24
    CYCLE_INTERVAL_SEC = 600

    def __init__(self, tracker, optimizer, mutator, timeframe_tracker, okx_client=None):
        super().__init__()
        self.tracker = tracker
        self.optimizer = optimizer
        self.mutator = mutator
        self.timeframe_tracker = timeframe_tracker
        self.okx_client = okx_client
        self._stop_flag = False
        self._running = False
        self._last_deploy_time: Dict[str, float] = {}

    def run(self):
        self._running = True
        self.log_signal.emit("🤖 自主训练器已启动", "INFO")
        self.status_signal.emit("🤖 运行中")

        cycle = 0
        while not self._stop_flag:
            cycle += 1
            try:
                self._run_cycle(cycle)
            except Exception as e:
                self.log_signal.emit(f"⚠️ 训练循环异常: {e}", "ERROR")
                import traceback
                self.log_signal.emit(traceback.format_exc()[-200:], "ERROR")
            gc.collect()
            # 每 10 分钟一个周期，但每 500ms 检查一次停止标志
            for _ in range(1200):
                if self._stop_flag:
                    break
                self.msleep(500)

        self._running = False
        self.status_signal.emit("⏹ 已停止")
        self.log_signal.emit("🤖 自主训练器已停止", "INFO")

    def stop(self):
        self._stop_flag = True

    def _run_cycle(self, cycle: int):
        self.log_signal.emit(f"--- 训练周期 #{cycle} ---", "INFO")

        # === 第 1 步：验证信号 ===
        val_count = self._validate_signals()
        # === 第 2 步：用真实数据更新优化器 ===
        opt_count = self._update_optimizer_from_data()
        # === 第 3 步：定向变异 + 安全部署 ===
        dep_count = self._optimize_and_deploy()

        if val_count == 0 and opt_count == 0 and dep_count == 0:
            self.log_signal.emit("⏳ 本周期无操作（等待更多信号数据）", "INFO")

    def _validate_signals(self) -> int:
        """验证待审信号，返回验证数"""
        if not self.okx_client:
            self.log_signal.emit("⚠️ 未连接 OKX，跳过验证", "WARNING")
            return 0

        validated = self.tracker.validate_outstanding(self.okx_client)
        tf_validated = self.timeframe_tracker.validate_predictions(self.okx_client)
        total = validated + tf_validated
        if total > 0:
            self.log_signal.emit(f"✅ 验证 {total} 条信号 ({validated} 常规 + {tf_validated} 多周期)", "SUCCESS")
            self.validation_done.emit(total)
        else:
            self.log_signal.emit("📊 无待验证信号", "INFO")
        return total

    def _update_optimizer_from_data(self) -> int:
        """
        用真实验证数据更新参数优化器。
        - tracker 中有已验证的信号
        - optimizer.update_from_samples() 使用具体参数值和对应胜率
          更新每个参数值的贝叶斯后验分布
        - 这是真正的学习步骤
        """
        strategies = self.tracker.all_strategies()
        updated = 0
        for name in strategies:
            stats = self.tracker.strategy_stats(name, days=30)
            samples = self.tracker.strategy_learning_samples(name, days=30)
            if stats.get("total", 0) >= self.MIN_SAMPLES and len(samples) >= self.MIN_SAMPLES:
                self.optimizer.update_from_samples(name, samples, stats)
                updated += 1
                wr = stats.get("win_rate", 0)
                self.log_signal.emit(
                    f"📈 {name}: 更新优化器 ({len(samples)} 样本, 胜率 {wr:.1f}%)", "SUCCESS"
                )
        if updated > 0:
            self.optimization_done.emit(updated)
        return updated

    def _optimize_and_deploy(self) -> int:
        """
        用 optimizer 数据驱动定向变异 + 安全部署。
        不再使用随机变异，而是：
        1. 用 get_best_params() 获取 Thompson 采样最优参数
        2. 将最优参数写入策略文件（定向变异）
        3. 用 optimizer 的后验胜率作为适应度
        """
        strategies = self.tracker.all_strategies()
        deployed = 0
        for name in strategies:
            fname = self._find_strategy_file(name)
            if not fname:
                continue

            fpath = Path(self.mutator._strategies_dir) / fname
            if not fpath.exists():
                continue

            stats = self.tracker.strategy_stats(name, days=30)
            if stats.get("total", 0) < self.MIN_SAMPLES:
                continue

            wr = stats.get("win_rate", 0)
            if wr < self.MIN_WIN_RATE:
                continue

            # 冷却期检查
            last = self._last_deploy_time.get(name, 0)
            if time.time() - last < self.DEPLOY_COOLDOWN_HOURS * 3600:
                continue

            # 用 Thompson Sampling 获取最优参数
            best_params = self.optimizer.get_best_params(name, use_exploration=False)
            if not best_params:
                continue

            # 获取参数空间，检验参数是否有优化潜力
            param_space = getattr(self.optimizer, "_param_space", {}).get(name, {})
            has_improvement = False
            current_params = self.optimizer.get_optimized_params(name)
            for key, val in best_params.items():
                cur_val = current_params.get(key, {}).get("value") if isinstance(current_params.get(key), dict) else current_params.get(key)
                if cur_val is not None and abs(float(val) - float(cur_val)) > 0.01:
                    has_improvement = True
                    break

            if not has_improvement:
                continue

            # 定向变异：将策略代码中的参数值替换为优化后的值
            content = fpath.read_text(encoding="utf-8")
            original = content
            mutations = []
            for key, new_val in best_params.items():
                lo, hi = param_space.get(key, (0, 999))
                new_val_clamped = round(max(lo, min(hi, float(new_val))), 4)
                # 尝试多种匹配模式
                patterns = [
                    f'"{key}"\\s*:\\s*([\\d.]+)',     # "key": 72.0
                    f"'{key}'\\s*:\\s*([\\d.]+)",       # 'key': 72.0
                    rf'\b{key}\s*=\s*([\d.]+)',          # key = 72.0
                ]
                replaced = False
                for pat in patterns:
                    new_content, count = re.subn(
                        pat,
                        lambda m, k=key, v=new_val_clamped: m.group(0).replace(m.group(1), str(v)),
                        content,
                        count=1
                    )
                    if count > 0 and new_content != content:
                        old_match = re.search(pat, content)
                        old_val = old_match.group(1) if old_match else "?"
                        mutations.append({
                            "description": f"{key}: {old_val} → {new_val_clamped} (优化)",
                            "old_value": float(old_val) if old_match else 0,
                            "new_value": new_val_clamped,
                            "type": "optimizer_driven",
                        })
                        content = new_content
                        replaced = True
                        break
                if not replaced:
                    self.log_signal.emit(f"  ⚠️ 策略中未找到参数 '{key}'，跳过", "INFO")

            if not mutations:
                self.log_signal.emit(f"  ⚖️ {name} 参数已最优，无需变异", "INFO")
                continue

            # 语法检查
            try:
                compile(content, f"<{fname}>", "exec")
            except SyntaxError as e:
                self.log_signal.emit(f"  ❌ {name} 变异语法错误: {e}", "ERROR")
                continue

            # 保存变异版本
            checksum = hashlib.md5(content.encode()).hexdigest()[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = fname.replace(".py", "")
            version_name = f"{base}_v{timestamp}_{checksum}.py"
            version_path = Path(self.mutator._evolutions_dir) / version_name
            version_path.write_text(content, encoding="utf-8")

            # 从 optimizer 估计适应度（基于 posterior win rate）
            param_tracker = getattr(self.optimizer, "_param_tracker", {}).get(name, {})
            fitness_scores = []
            for key, new_val in best_params.items():
                tracker = param_tracker.get(key, {})
                for bucket, stat in tracker.get("value_stats", {}).items():
                    if abs(float(stat.get("value", 0)) - float(new_val)) < 0.001:
                        s = float(stat.get("success", 0))
                        f = float(stat.get("failure", 0))
                        n = float(stat.get("neutral", 0))
                        posterior = (s + 1.0) / (s + f + n + 2.0) * 100.0
                        fitness_scores.append(posterior)
                        break

            estimated_fitness = np.mean(fitness_scores) if fitness_scores else max(wr, 50.0)

            # 记录变异
            mutation_entry = {
                "success": True,
                "version_name": version_name,
                "source_file": fname,
                "timestamp": timestamp,
                "checksum": checksum,
                "mutations": mutations,
                "version_path": str(version_path),
                "fitness": round(estimated_fitness, 2),
                "optimized_params": best_params,
            }
            self.mutator._log_mutation(mutation_entry)
            self.mutator.update_fitness(version_name, estimated_fitness)

            # 安全门控：需要显著改进（胜率提升≥4% 且超过 54%）并达到最小样本量
            min_trials = max(int(best_stat.get("trials", 0) or 0), 0)
            if estimated_fitness > max(wr + 4.0, 54.0) and min_trials >= 5:
                # 备份当前文件
                bak = Path(self.mutator._backups_dir) / f"before_deploy_{timestamp}_{fname}"
                shutil.copy2(str(fpath), str(bak))
                # 部署
                fpath.write_text(content, encoding="utf-8")
                self._last_deploy_time[name] = time.time()
                deployed += 1
                self.log_signal.emit(
                    f"🚀 {name} 部署优化参数 (适应度 {estimated_fitness:.1f}%)", "SUCCESS"
                )
                self.deployment_done.emit(name)
                # 刷新突变点缓存
                self.mutator._load_mutation_points()
            else:
                self.log_signal.emit(
                    f"  ⏸ {name} 优化潜力不足 (适应度 {estimated_fitness:.1f}% vs 当前 {wr:.1f}%)", "INFO"
                )

        return deployed

    def _find_strategy_file(self, strategy_name: str) -> str:
        strategies_dir = Path(self.mutator._strategies_dir)
        for f in strategies_dir.glob("*.py"):
            if f.name.startswith("_"):
                continue
            try:
                content = f.read_text(encoding="utf-8")
                if f'name = "{strategy_name}"' in content or f'STRATEGY_NAME = "{strategy_name}"' in content:
                    return f.name
            except Exception:
                continue
        keywords = strategy_name.replace(" ", "").replace("AI", "").replace("截面", "").replace("因子", "")
        for f in strategies_dir.glob("*.py"):
            if keywords and keywords.lower() in f.name.lower():
                return f.name
        files = [f for f in strategies_dir.glob("*.py") if not f.name.startswith("_")]
        return files[0].name if files else ""
