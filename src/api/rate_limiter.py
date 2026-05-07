"""
全局 API 限流器（令牌桶算法）

防止多线程并发请求超过 OKX V5 API 限频（公共端点 20次/2s，私有端点 10次/2s）。
所有 OKX API 调用应经过此限流器分发。
"""

import threading
import time
from typing import Dict


class TokenBucketRateLimiter:
    """
    令牌桶限流器。

    参数：
      rate:     每秒允许的请求数（令牌产生速率）
      capacity: 桶容量（最大突发请求数）
    """

    def __init__(self, rate: float = 8.0, capacity: int = 15):
        self._rate = rate
        self._capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 5.0) -> bool:
        """
        获取一个令牌（阻塞直至可用或超时）。

        Returns:
            True  — 成功获取令牌
            False — 超时未获得
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
            # 令牌不足，等待
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            # 计算下一个令牌到达时间
            wait = min(1.0 / self._rate, remaining)
            time.sleep(wait)

    def _refill(self):
        """补充令牌（在锁内调用）。"""
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * self._rate
        if new_tokens > 0:
            self._tokens = min(self._capacity, self._tokens + new_tokens)
            self._last_refill = now

    @property
    def available_tokens(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens


class APIRateLimitManager:
    """
    分类限流管理器。

    针对 OKX V5 不同端点类别实施独立限频：
      - public:  公共端点（行情/K线等）  → 20 次/2秒 = 10/s
      - private: 私有端点（账户/下单等）  → 10 次/2秒 = 5/s
      - trade:   交易端点（下单/撤单等）  → 60 次/2秒 = 30/s（但更保守）
    """

    # 端点前缀分类表
    _TRADE_PATHS = {'/api/v5/trade/'}
    _PUBLIC_PATHS = {'/api/v5/market/', '/api/v5/public/'}

    def __init__(self):
        self._limiters: Dict[str, TokenBucketRateLimiter] = {
            'public':  TokenBucketRateLimiter(rate=8.0, capacity=15),
            'private': TokenBucketRateLimiter(rate=4.0, capacity=8),
            'trade':   TokenBucketRateLimiter(rate=10.0, capacity=20),
        }

    def acquire(self, endpoint: str, timeout: float = 5.0) -> bool:
        """根据端点分类获取令牌。"""
        category = self._classify(endpoint)
        return self._limiters[category].acquire(timeout)

    def _classify(self, endpoint: str) -> str:
        for prefix in self._TRADE_PATHS:
            if endpoint.startswith(prefix):
                return 'trade'
        for prefix in self._PUBLIC_PATHS:
            if endpoint.startswith(prefix):
                return 'public'
        return 'private'


# ── 模块级单例 ──────────────────────────────────────────────────────────────
api_rate_limiter: APIRateLimitManager = APIRateLimitManager()
