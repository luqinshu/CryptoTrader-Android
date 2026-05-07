"""
LLM客户端：支持 OpenAI 兼容 API，可随时切换不同大模型。
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import requests


class LLMClient:
    """
    大模型 API 客户端。
    支持所有 OpenAI 兼容接口（OpenAI / DeepSeek / Claude / 本地模型）。

    用法:
        client = LLMClient(
            base_url="https://api.deepseek.com/v1",
            api_key="sk-xxxx",
            model="deepseek-chat",
        )
        reply = client.chat([
            {"role": "system", "content": "你是一个量化交易专家"},
            {"role": "user", "content": "分析这段策略代码..."},
        ])
    """

    DEFAULT_TIMEOUT = 120

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        model: str = "deepseek-chat",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        self.base_url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.last_error = ""

    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
        timeout: Optional[int] = None,
    ) -> Optional[str]:
        """发送对话请求"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        req_timeout = timeout or self.DEFAULT_TIMEOUT

        try:
            if stream:
                return self._stream_request(headers, payload, on_token, req_timeout)
            resp = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=req_timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            else:
                self.last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                return None
        except requests.exceptions.Timeout:
            self.last_error = f"请求超时({req_timeout}s) - 请尝试减少数据量或增大超时"
            return None
        except requests.exceptions.ConnectionError:
            self.last_error = f"无法连接到 {self.base_url}"
            return None
        except Exception as e:
            self.last_error = str(e)
            return None

    def _stream_request(self, headers, payload, on_token, timeout):
        try:
            resp = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=timeout,
                stream=True,
            )
            if resp.status_code != 200:
                self.last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                return None
            full = []
            for line in resp.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                chunk = line[6:].decode("utf-8", errors="ignore")
                if chunk == "[DONE]":
                    break
                try:
                    data = json.loads(chunk)
                    token = data["choices"][0].get("delta", {}).get("content") or ""
                    if token and on_token:
                        on_token(token)
                    full.append(token)
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
            result = "".join(full)
            return result if result else None
        except requests.exceptions.Timeout:
            self.last_error = f"流式请求超时({timeout}s)"
            return None
        except requests.exceptions.ConnectionError as e:
            self.last_error = f"连接失败: {e}"
            return None
        except Exception as e:
            self.last_error = str(e)
            return None

    def test_connection(self) -> bool:
        """测试 API 连接是否正常（更短超时）"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "回复OK"}],
                "max_tokens": 5,
            }
            resp = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=15,
            )
            if resp.status_code == 200:
                return True
            body = resp.text[:300]
            self.last_error = f"HTTP {resp.status_code}: {body}"
            return False
        except requests.exceptions.Timeout:
            self.last_error = "连接超时(15s)"
            return False
        except requests.exceptions.ConnectionError as e:
            self.last_error = f"连接被拒绝: {e}"
            return False
        except Exception as e:
            self.last_error = f"未知错误: {e}"
            return False

    def verbose_test(self) -> str:
        """详细测试连接，返回完整诊断信息"""
        lines = []
        lines.append(f"目标URL: {self.base_url}")
        lines.append(f"模型: {self.model}")
        lines.append(f"Key前缀: {self.api_key[:8]}..." if len(self.api_key) > 8 else f"Key: (空)")
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "回复OK"}],
                "max_tokens": 5,
            }
            lines.append("发送请求...")
            resp = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=15,
            )
            lines.append(f"状态码: {resp.status_code}")
            lines.append(f"响应头: {dict(resp.headers)}")
            lines.append(f"响应体: {resp.text[:500]}")
            if resp.status_code == 200:
                lines.append("✅ 连接成功")
            elif resp.status_code == 401:
                lines.append("❌ API Key 无效或未订阅 OpenCode Go")
            elif resp.status_code == 404:
                lines.append("❌ 模型名不存在或 API 路径错误")
            elif resp.status_code == 429:
                lines.append("❌ 请求频率超限，稍后重试")
            else:
                lines.append(f"❌ 未预期的状态码")
        except requests.exceptions.Timeout:
            lines.append("❌ 连接超时(15s) - 检查网络/代理")
        except requests.exceptions.ConnectionError as e:
            lines.append(f"❌ 连接失败: {e}")
            lines.append("   检查 URL 是否正确，是否需要代理")
        except Exception as e:
            lines.append(f"❌ 异常: {e}")
        return "\n".join(lines)

    def to_config(self) -> Dict[str, Any]:
        return {
            "base_url": self.base_url.replace("/chat/completions", ""),
            "api_key": self.api_key,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
