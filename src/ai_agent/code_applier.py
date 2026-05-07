"""
安全代码应用器：备份→语法检查→应用→回退。
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.ai_agent.code_parser import CodeChange


class SafeCodeApplier:
    """
    安全应用代码修改：
    1. 备份原文件
    2. 语法检查新代码
    3. 应用修改
    4. 如失败，自动回退
    """

    def __init__(self, strategies_dir: Optional[str] = None):
        self._strategies_dir = Path(strategies_dir) if strategies_dir else Path(__file__).resolve().parent.parent.parent / "strategies"
        self._backups_dir = self._strategies_dir / "_ai_backups"
        self._backups_dir.mkdir(parents=True, exist_ok=True)

    def apply_changes(
        self,
        file_path: str,
        changes: List[CodeChange],
        dry_run: bool = False,
    ) -> Tuple[bool, str, List[str]]:
        """
        应用一组代码修改。
        Returns: (success, message, applied_description_list)
        """
        fp = self._strategies_dir / file_path
        if not fp.exists():
            return False, f"文件不存在: {fp}", []

        try:
            original = fp.read_text(encoding="utf-8")
        except Exception as e:
            return False, f"无法读取文件: {e}", []

        if dry_run:
            applied = self._simulate(original, changes)
            return True, f"干跑: {len(applied)} 项匹配，{len(changes) - len(applied)} 项未匹配", applied

        # 备份
        backup_path = self._backups_dir / f"{file_path}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
        try:
            shutil.copy2(str(fp), str(backup_path))
        except Exception as e:
            return False, f"备份失败: {e}", []

        # 逐项应用
        modified = original
        applied_descs = []
        for change in changes:
            new_content = self._apply_one(modified, change)
            if new_content == modified:
                continue  # 未匹配，跳过
            modified = new_content
            applied_descs.append(change.description)

        if not applied_descs:
            return False, "所有修改均未匹配到源代码，请检查建议是否针对当前文件", []

        # 语法检查
        try:
            compile(modified, f"<{file_path}>", "exec")
        except SyntaxError as e:
            # 回退
            shutil.copy2(str(backup_path), str(fp))
            return False, f"修改后代码语法错误: {e}\n\n已自动回退到备份: {backup_path.name}", []

        # 写入
        try:
            fp.write_text(modified, encoding="utf-8")
        except Exception as e:
            shutil.copy2(str(backup_path), str(fp))
            return False, f"写入文件失败: {e}\n\n已自动回退", []

        return True, f"成功应用 {len(applied_descs)} 项修改\n备份: {backup_path.name}", applied_descs

    def _apply_one(self, content: str, change: CodeChange) -> str:
        """尝试将一项修改应用到代码中"""
        old = change.old_code.strip()

        if old in content:
            return content.replace(old, change.new_code.strip(), 1)

        # 变量/参数赋值模式: key = old_value → key = new_value
        if "=" in old:
            parts = old.split("=", 1)
            key = parts[0].strip()
            old_val = parts[1].strip()
            new_val = change.new_code.split("=", 1)[1].strip() if "=" in change.new_code else change.new_code.strip()

            # 匹配多种格式
            for pattern in [
                rf'\b{re.escape(key)}\s*=\s*{re.escape(old_val)}\b',
                rf'["\']?{re.escape(key)}["\']?\s*[:=]\s*{re.escape(old_val)}',
            ]:
                match = re.search(pattern, content)
                if match:
                    new_str = match.group(0).replace(old_val, new_val, 1)
                    return content[:match.start()] + new_str + content[match.end():]

        return content  # 未匹配

    def _simulate(self, content: str, changes: List[CodeChange]) -> List[str]:
        """模拟应用，返回能匹配的修改描述"""
        descs = []
        for change in changes:
            if self._apply_one(content, change) != content:
                descs.append(change.description)
        return descs

    def rollback(self, file_path: str) -> Tuple[bool, str]:
        """回退到最近的备份"""
        fp = self._strategies_dir / file_path
        backups = sorted(
            self._backups_dir.glob(f"{file_path}.*.bak"),
            reverse=True,
        )
        if not backups:
            return False, "没有找到备份文件"
        latest = backups[0]
        try:
            shutil.copy2(str(latest), str(fp))
            return True, f"已回退到: {latest.name}"
        except Exception as e:
            return False, f"回退失败: {e}"
