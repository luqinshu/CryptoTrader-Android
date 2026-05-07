"""
策略基因变异引擎 v2：去重匹配 + 原文件备份 + 适应度评分追踪。
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class StrategyMutator:
    """v2 基因变异器：发现变异点 → 安全变异 → 备份 → 评分追踪"""

    def __init__(self, strategies_dir: Optional[str] = None):
        self._strategies_dir = Path(strategies_dir) if strategies_dir else Path(__file__).resolve().parent.parent.parent / "strategies"
        self._evolutions_dir = self._strategies_dir / "_evolutions"
        self._backups_dir = self._strategies_dir / "_backups"
        self._evolutions_dir.mkdir(parents=True, exist_ok=True)
        self._backups_dir.mkdir(parents=True, exist_ok=True)

        # mutation_points: {filename: [(description, old_val, range, pattern, position, type)]}
        self._mutation_points: Dict[str, List[Dict]] = {}
        self._load_mutation_points()

    def _load_mutation_points(self):
        for fpath in sorted(self._strategies_dir.glob("*.py")):
            if fpath.name.startswith("_"):
                continue
            try:
                content = fpath.read_text(encoding="utf-8")
                self._discover_mutations(fpath.name, content)
            except Exception:
                continue

    def _discover_mutations(self, fname: str, content: str):
        points = []

        # 1. CONFIG_SCHEMA 中的 "default": 数值
        #    匹配: "default": 72.0  或  'default': 5_000_000
        for m in re.finditer(r'"(default)"\s*:\s*([\d._]+)', content):
            val_str = m.group(2).replace("_", "")
            try:
                val = float(val_str)
            except ValueError:
                continue
            lo = max(val * 0.4, val / 2.0)
            hi = val * 2.5
            points.append({
                "description": f"参数默认值 {val}",
                "old_value": val,
                "range": (lo, hi),
                "pattern": m.group(0),
                "position": m.start(),
                "type": "config_default",
            })

        # 2. _BASE_WEIGHTS 字典值（权重类，0.01~0.30）
        #   仅匹配在 _BASE_WEIGHTS 或 WEIGHTS 附近的值
        for m in re.finditer(r'"(weight|reversal|momentum|trend|low_vol|liquidity|volume|funding|oi|chain|llm|event|developer|early|ema|rsi|macd|donchian)"\s*:\s*([\d.]+)', content, re.IGNORECASE):
            try:
                val = float(m.group(2))
            except ValueError:
                continue
            if val < 0.01 or val > 0.50:
                continue
            lo = max(val * 0.4, 0.005)
            hi = min(val * 2.5, 0.80)
            points.append({
                "description": f"权重 {m.group(1)}={val}",
                "old_value": val,
                "range": (lo, hi),
                "pattern": m.group(0),
                "position": m.start(),
                "type": "weight",
            })

        # 3. 缩放系数 (@ 0.65~0.85 范围)
        for m in re.finditer(r'"(momentum|trend|reversal|low_vol|liquidity|volume_impulse|funding_contra|oi|on_chain|network|llm_sentiment|event|developer|early_trend|ema|rsi|macd|donchian|volume_price|ai_trend|accumulation|quality|fresh)"\s*:\s*([\d.]+)', content, re.IGNORECASE):
            try:
                val = float(m.group(2))
            except ValueError:
                continue
            if 0.40 <= val <= 1.20:
                lo = max(val * 0.55, 0.25)
                hi = min(val * 1.6, 2.0)
                points.append({
                    "description": f"缩放 {m.group(1)}={val}",
                    "old_value": val,
                    "range": (lo, hi),
                    "pattern": m.group(0),
                    "position": m.start(),
                    "type": "scale",
                })

        # 去重：同位置只保留第一个匹配（最有意义的）
        seen_positions = set()
        deduped = []
        for p in sorted(points, key=lambda x: x["position"]):
            if p["position"] not in seen_positions:
                deduped.append(p)
                seen_positions.add(p["position"])

        self._mutation_points[fname] = deduped

    def mutation_points_for(self, filename: str) -> List[Dict]:
        return self._mutation_points.get(filename, [])

    def mutate_strategy(self, filename: str, num_mutations: int = 1) -> Optional[Dict[str, Any]]:
        fpath = self._strategies_dir / filename
        if not fpath.exists():
            return None

        content = fpath.read_text(encoding="utf-8")
        points = self._mutation_points.get(filename, [])
        if not points:
            return None

        selected = random.sample(points, min(num_mutations, len(points)))
        mutated_content = content
        mutations_applied = []

        for pt in sorted(selected, key=lambda x: -x["position"]):
            lo, hi = pt["range"]
            # 修复：lo = min(val*0.4, val/2.0) → 取更小值
            lo_calc = min(val * 0.4, val / 2.0)
            delta = (hi - lo) * random.uniform(-0.15, 0.15)
            new_val = round(max(lo, min(hi, pt["old_value"] + delta)), 4)
            old_pattern = pt["pattern"]
            # 精确替换 old_val 所在的数值
            new_pattern = old_pattern.replace(str(pt["old_value"]), str(new_val), 1)

            pos = pt["position"]
            mutated_content = mutated_content[:pos] + mutated_content[pos:].replace(old_pattern, new_pattern, 1)
            mutations_applied.append({
                "description": pt["description"],
                "old_value": pt["old_value"],
                "new_value": new_val,
                "type": pt["type"],
            })

        try:
            compile(mutated_content, f"<{filename}>", "exec")
        except SyntaxError as e:
            return {"success": False, "error": f"语法错误: {e}"}

        # 备份原文件（仅首次）
        backup_path = self._backups_dir / filename
        if not backup_path.exists():
            shutil.copy2(str(fpath), str(backup_path))

        checksum = hashlib.md5(mutated_content.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = filename.replace(".py", "")
        version_name = f"{base}_v{timestamp}_{checksum}.py"
        version_path = self._evolutions_dir / version_name
        version_path.write_text(mutated_content, encoding="utf-8")

        result = {
            "success": True,
            "version_name": version_name,
            "source_file": filename,
            "timestamp": timestamp,
            "checksum": checksum,
            "mutations": mutations_applied,
            "version_path": str(version_path),
            "fitness": 0.0,
        }

        self._log_mutation(result)
        return result

    def _log_mutation(self, result: Dict):
        log_path = self._evolutions_dir / "_mutation_log.json"
        try:
            log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
        except Exception:
            log = []
        log.append(result)
        log_path.write_text(json.dumps(log[-300:], indent=2, ensure_ascii=False), encoding="utf-8")

    def update_fitness(self, version_name: str, fitness: float):
        """更新某个变异版本的适应度评分（由优化器反馈）"""
        log_path = self._evolutions_dir / "_mutation_log.json"
        try:
            log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
        except Exception:
            return
        for entry in log:
            if entry.get("version_name") == version_name:
                entry["fitness"] = round(fitness, 2)
                entry["fitness_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_path.write_text(json.dumps(log[-300:], indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def version_checksum(version_name: str) -> str:
        stem = Path(version_name).stem
        parts = stem.rsplit("_", 1)
        return parts[-1] if len(parts) == 2 else ""

    def file_checksum(self, filename: str) -> str:
        fpath = self._strategies_dir / filename
        if not fpath.exists():
            return ""
        try:
            return hashlib.md5(fpath.read_text(encoding="utf-8").encode("utf-8")).hexdigest()[:8]
        except Exception:
            return ""

    def get_best_mutation(self, filename: str) -> Optional[Dict]:
        """返回胜率最高的变异版本"""
        log_path = self._evolutions_dir / "_mutation_log.json"
        if not log_path.exists():
            return None
        try:
            log = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        candidates = [
            e for e in log
            if e.get("source_file") == filename and e.get("success") and e.get("fitness", 0) > 0
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e["fitness"])

    def get_evolution_history(self, filename: str) -> List[Dict]:
        log_path = self._evolutions_dir / "_mutation_log.json"
        if not log_path.exists():
            return []
        try:
            log = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return [e for e in log if e.get("source_file") == filename]

    def all_evolution_entries(self) -> List[Dict]:
        log_path = self._evolutions_dir / "_mutation_log.json"
        if not log_path.exists():
            return []
        try:
            return json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def reset_to_version(self, version_name: str) -> bool:
        version_path = self._evolutions_dir / version_name
        if not version_path.exists():
            return False
        base = version_name.split("_v")[0]
        target_name = base + ".py"
        target_path = self._strategies_dir / target_name

        # 备份当前版本
        if target_path.exists():
            bak = self._backups_dir / f"before_reset_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{target_name}"
            shutil.copy2(str(target_path), str(bak))

        shutil.copy2(str(version_path), str(target_path))
        return True

    def apply_best_if_better(self, filename: str, current_fitness: float, min_improvement: float = 3.0) -> bool:
        """如果最佳变异比当前表现好，自动应用"""
        best = self.get_best_mutation(filename)
        if not best:
            return False
        if best.get("fitness", 0) > current_fitness + min_improvement:
            return self.reset_to_version(best["version_name"])
        return False
