"""
Telegram 消息发送工具 - 轻量级，无需启动完整 Bot
所有 HTTP 请求在后台线程执行，不阻塞 UI
"""

import json
import threading
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional


class TelegramSender:
    """通过 Telegram Bot API 发送消息"""

    def __init__(self, token: str = None, chat_id: str = None):
        """
        Args:
            token: Bot Token (从 telegram_config.txt 读取)
            chat_id: Chat ID (从 src/telegram_config.json 读取)
        """
        self._token = token
        self._chat_id = chat_id

        # 自动加载配置
        if not self._token or not self._chat_id:
            self._auto_load_config()

    def _auto_load_config(self):
        """自动从项目配置文件加载"""
        config_dir = Path(__file__).resolve().parent.parent  # src/

        if not self._token:
            token_path = config_dir.parent / 'telegram_config.txt'
            if token_path.exists():
                self._token = token_path.read_text().strip()

        if not self._chat_id:
            json_path = config_dir / 'telegram_config.json'
            if json_path.exists():
                try:
                    cfg = json.loads(json_path.read_text())
                    self._chat_id = cfg.get('chat_id', '')
                except Exception:
                    pass

    def send(self, text: str, parse_mode: str = 'Markdown') -> bool:
        """异步发送消息到 Telegram (后台线程，不阻塞 UI)

        Args:
            text: 消息文本
            parse_mode: 'Markdown' 或 'HTML'

        Returns:
            始终返回 True (异步发送不等待结果)
        """
        if not self._token or not self._chat_id:
            print("[TelegramSender] 未配置 Token 或 Chat ID，跳过发送")
            return False

        # 在后台线程发送，避免阻塞 UI
        def _do_send():
            with self._send_lock:
                now = time.time()
                if now - self._last_send_time < self._min_interval:
                    time.sleep(self._min_interval - (now - self._last_send_time))
                    now = time.time()
                self._last_send_time = now
            try:
                url = f"https://api.telegram.org/bot{self._token}/sendMessage"
                data = {
                    'chat_id': self._chat_id,
                    'text': text,
                    'parse_mode': parse_mode,
                }
                encoded = urllib.parse.urlencode(data).encode('utf-8')
                req = urllib.request.Request(url, data=encoded, method='POST')
                with urllib.request.urlopen(req, timeout=10) as resp:
                    result = json.loads(resp.read().decode())
                    if not result.get('ok'):
                        print(f"[TelegramSender] 发送失败: {result.get('description', 'unknown')}")
            except Exception as e:
                print(f"[TelegramSender] 发送异常: {e}")

        threading.Thread(target=_do_send, daemon=True).start()
        return True

    def send_signal_alert(self, inst_id: str, signal_type: str, direction: str,
                          score: float, message: str, price: float = 0,
                          timestamp: str = '') -> bool:
        """发送交易信号提醒（格式化消息）"""
        emoji_map = {
            '趋势突破': '🚀',
            '大幅回调': '📉',
            '企稳突破': '🎯',
            '放量异动': '📊',
            '动量背离': '⚡',
        }
        emoji = emoji_map.get(signal_type, '📡')
        dire_emoji = '🟢➡️' if direction == 'BUY' else ('🔴⬇️' if direction == 'SHORT' else '⚪')

        text = (
            f"{emoji} *{signal_type}* {dire_emoji}\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📊 交易对: `{inst_id}`\n"
            f"🎯 方向: *{direction}*\n"
            f"⭐ 评分: *{score:.0f}* 分\n"
        )
        if price > 0:
            text += f"💰 价格: ${price:.4f}\n"
        text += (
            f"📝 {message}\n"
            f"🕐 {timestamp or ''}"
        )
        return self.send(text)

    @property
    def is_configured(self) -> bool:
        return bool(self._token and self._chat_id)


# 全局单例
_sender: Optional[TelegramSender] = None


def get_telegram_sender() -> TelegramSender:
    global _sender
    if _sender is None:
        _sender = TelegramSender()
    return _sender
