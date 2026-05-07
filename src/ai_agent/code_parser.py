"""
AI 代码建议解析器：从 LLM 返回的分析文本中提取可执行的代码修改。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CodeChange:
    """表示一个具体的代码修改"""

    description: str           # 修改描述
    file_path: str             # 目标文件路径
    old_code: str              # 原代码片段
    new_code: str              # 新代码片段
    line_number: int = 0       # 行号（如果能找到）
    change_type: str = "param" # param / logic / weight / threshold
    confidence: float = 1.0    # AI 建议的置信度
    applied: bool = False      # 是否已应用
    reason: str = ""           # AI 给出的理由（数据分析依据）
    expected_improvement: str = ""  # 预期提升（胜率/回撤变化）


class CodeDiffParser:
    """解析 AI 返回的建议，提取代码变更列表"""

    @staticmethod
    def parse_analysis(text: str, target_file: str = "") -> List[CodeChange]:
        """
        从 AI 分析文本中提取所有代码修改建议。
        支持格式:
          ```
          # 原代码
          old
          # 改为
          new
          
          修改理由: ...
          预期提升: ...
          ```
        """
        changes: List[CodeChange] = []

        # === 格式 1: 代码块中的 "原代码 / 改为" 模式 ===
        # 将文本切分成"段"，每段以 "# 原代码" 开头
        blocks = re.split(r'#\s*(?:原代码|修改前|before)[\s:：]*', text)
        for i, block in enumerate(blocks[1:], 1):
            parts = re.split(r'#\s*(?:改为|修改后|after)[\s:：]*', block, maxsplit=1)
            if len(parts) < 2:
                continue
            old_code = parts[0].strip()
            remaining = parts[1]

            # 提取 new_code（到下一个 # 原代码 或 修改理由 之前）
            new_end = re.split(
                r'(?:修改理由|预期提升|风险提示|\n#\s*(?:原代码|修改前|before))[\s:：]*',
                remaining,
                maxsplit=1,
            )
            new_code = new_end[0].strip()
            rest = new_end[1] if len(new_end) > 1 else ""

            if not old_code or not new_code or old_code == new_code:
                continue

            # 提取理由和预期提升
            reason = ""
            improvement = ""
            reason_m = re.search(r'修改理由[\s:：]*[:：]?\s*(.+?)(?=\n(?:预期提升|风险提示|$)|\Z)',
                                rest, re.DOTALL)
            if reason_m:
                reason = reason_m.group(1).strip()
            imp_m = re.search(r'预期提升[\s:：]*[:：]?\s*(.+?)(?=\n(?:修改理由|风险提示|$)|\Z)',
                              rest, re.DOTALL)
            if imp_m:
                improvement = imp_m.group(1).strip()

            desc = CodeDiffParser._describe_change(old_code, new_code)
            changes.append(CodeChange(
                description=desc,
                file_path=target_file,
                old_code=old_code.strip(),
                new_code=new_code.strip(),
                change_type="param" if "=" in old_code and "=" in new_code else "logic",
                reason=reason,
                expected_improvement=improvement,
            ))

        # === 格式 2: 参数建议格式 "参数名: 当前值 → 建议值" ===
        param_pattern = re.findall(
            r'[-*]\s*`?(\w+)`?\s*[:=]\s*([\d.]+)\s*[→>]\s*([\d.]+)',
            text, re.MULTILINE
        )
        for param_name, old_val, new_val in param_pattern:
            if any(c.description and param_name in c.description and c.reason for c in changes):
                continue
            reason = ""
            improvement = ""
            idx = text.find(f"{param_name}: {old_val} → {new_val}")
            if idx < 0:
                idx = text.find(f"{param_name} : {old_val} → {new_val}")
            if idx > 0:
                after = text[idx + 60:]
                reason_m = re.search(r'(?:理由|原因|依据)[:：]\s*(.+?)(?=\n|$)', after)
                if reason_m:
                    reason = reason_m.group(1).strip()[:120]
                imp_m = re.search(r'(?:预期|提升|改善)[:：]\s*(.+?)(?=\n|$)', after)
                if imp_m:
                    improvement = imp_m.group(1).strip()[:120]
            changes.append(CodeChange(
                description=f"参数 {param_name}: {old_val} → {new_val}",
                file_path=target_file,
                old_code=f"{param_name} = {old_val}",
                new_code=f"{param_name} = {new_val}",
                change_type="param",
                reason=reason,
                expected_improvement=improvement,
            ))

        # === 格式 3: regex 参数赋值 ===
        param_kv = re.findall(
            r'(?:建议|修改|调整)\s*[:：]\s*(\w+)\s*[:=]\s*([\d.]+)\s*[→>]\s*([\d.]+)',
            text
        )
        for param_name, old_val, new_val in param_kv:
            if not any(c.description and param_name in c.description for c in changes):
                changes.append(CodeChange(
                    description=f"{param_name}: {old_val} → {new_val}",
                    file_path=target_file,
                    old_code=f"{param_name} = {old_val}",
                    new_code=f"{param_name} = {new_val}",
                    change_type="param",
                ))

        return changes

    @staticmethod
    def _parse_block(block: str, target_file: str) -> List[CodeChange]:
        """解析单个文本块中的 '原代码 / 改为' 模式"""
        changes = []
        # 模式: # 原代码\n...\n# 改为\n...
        parts = re.split(r'#\s*(?:原代码|修改前|before)[\s:：]*', block, maxsplit=1)
        if len(parts) < 2:
            return changes

        remaining = parts[1]
        segments = re.split(r'#\s*(?:改为|修改后|after)[\s:：]*', remaining)

        for i in range(0, len(segments) - 1, 1):
            old = segments[i].strip()
            new = segments[i + 1].strip() if i + 1 < len(segments) else ""
            if not old or not new:
                continue
            if old == new:
                continue

            desc = CodeDiffParser._describe_change(old, new)
            changes.append(CodeChange(
                description=desc,
                file_path=target_file,
                old_code=old,
                new_code=new,
                change_type="param" if "=" in old and "=" in new else "logic",
            ))

        return changes

    @staticmethod
    def _describe_change(old: str, new: str) -> str:
        """生成变更的简短描述"""
        old_first = old.split("\n")[0].strip()
        new_first = new.split("\n")[0].strip()
        if "=" in old_first and "=" in new_first:
            old_key, old_val = old_first.split("=", 1)
            _, new_val = new_first.split("=", 1)
            return f"{old_key.strip()}: {old_val.strip()} → {new_val.strip()}"
        return f"代码变更 ({len(old)}→{len(new)} 字符)"

    @staticmethod
    def to_markdown_diff(changes: List[CodeChange]) -> str:
        """将变更列表转换为 Markdown diff 格式展示"""
        lines = ["## 📝 AI 建议的代码修改\n"]
        for i, c in enumerate(changes, 1):
            lines.append(f"### 修改 {i}: {c.description}")
            if c.reason:
                lines.append(f"**理由**: {c.reason}")
            lines.append(f"**类型**: {c.change_type}")
            lines.append(f"**文件**: {c.file_path}")
            lines.append("")
            lines.append("```diff")
            for old_line in c.old_code.split("\n"):
                lines.append(f"- {old_line}")
            for new_line in c.new_code.split("\n"):
                lines.append(f"+ {new_line}")
            lines.append("```")
            lines.append("")
        return "\n".join(lines)
