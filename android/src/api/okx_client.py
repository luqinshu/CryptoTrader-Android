"""
OKX 交易所 API 客户端
封装 OKX V5 API 接口

错误码约定：
  code="0"   → 成功
  code="-2"  → 客户端限流排队超时（可重试，需退避）
  code="-3"  → 网络请求超时（可重试一次）
  code="-1"  → 其他错误（请求失败/异常，不可盲目重试）
"""
import time
import random
import hashlib
import hmac
import base64
import requests
from requests.exceptions import ConnectionError, ProxyError
from typing import Dict, List, Optional
import urllib3
urllib3.disable_warnings()

from src.api.rate_limiter import api_rate_limiter


class OKXClient:
    """OKX 交易所客户端"""

    def __init__(self, api_key: str, secret_key: str, passphrase: str,
                 testnet: bool = True, proxy_url: str = None):
        """
        初始化 OKX 客户端

        Args:
            api_key: API 密钥
            secret_key: 密钥
            passphrase: 密码短语
            testnet: 是否使用测试网
            proxy_url: 代理地址
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.testnet = testnet
        self.proxy_url = proxy_url

        # API 基础 URL：实盘 / 模拟盘分域名，防止误下单到生产环境
        if testnet:
            self.base_url = "https://demo.okx.com"
        else:
            self.base_url = "https://www.okx.com"

        # 代理配置
        self.proxies = None
        if proxy_url:
            self.proxies = {
                "http": proxy_url,
                "https": proxy_url
            }
        self._proxy_disabled_after_error = False

        # 请求头
        self.headers = {
            "Content-Type": "application/json",
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": "",
            "OK-ACCESS-TIMESTAMP": "",
            "OK-ACCESS-PASSPHRASE": self.passphrase,
        }
        
        # 如果是测试网，增加模拟盘 Header
        if self.testnet:
            self.headers["x-simulated-trading"] = "1"

        # 启动横幅：明确当前环境，防止误操作
        env_label = "模拟盘 TESTNET (demo.okx.com)" if testnet else "⚠️  实盘 PRODUCTION (www.okx.com)  ⚠️"
        print(f"\n{'='*60}\n  OKXClient 初始化\n  {env_label}\n{'='*60}\n")

    def _get_timestamp(self) -> str:
        """获取 ISO 格式时间戳（毫秒精度，OKX V5 要求）"""
        ts = time.time()
        ms = int(ts * 1000) % 1000
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts)) + f".{ms:03d}Z"

    def _sign(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        """生成签名"""
        message = timestamp + method + request_path + body
        mac = hmac.new(
            bytes(self.secret_key, encoding="utf8"),
            bytes(message, encoding="utf8"),
            digestmod=hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode()

    def _request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Dict:
        """发送 HTTP 请求（含限流重试 + 网络重试）。

        错误码：
          "-2" — 限流排队超时（多次重试后仍失败）
          "-3" — 网络请求超时（重试一次后仍失败）
          "-1" — 其他不可重试的错误
        """
        url = self.base_url + endpoint

        # 准备请求体
        body = ""
        if data:
            import json
            body = json.dumps(data)

        # 生成签名 —— GET 请求需将查询参数拼入 requestPath
        sign_path = endpoint
        if method == "GET" and params:
            from urllib.parse import urlencode
            query_string = urlencode(sorted(
                [(k, v) for k, v in (params or {}).items() if v is not None]
            ))
            if query_string:
                sign_path = f"{endpoint}?{query_string}"

        # ── 限流重试循环 ────────────────────────────────────────────────────
        _MAX_RATE_LIMIT_RETRIES = 3   # 最多重试 3 次（含首次）
        for rate_attempt in range(_MAX_RATE_LIMIT_RETRIES):
            # 全局限流：按端点分类排队，防止超过 OKX 限频
            if not api_rate_limiter.acquire(endpoint, timeout=8.0):
                if rate_attempt < _MAX_RATE_LIMIT_RETRIES - 1:
                    backoff = (2 ** rate_attempt) * 0.5 + random.random() * 0.3
                    time.sleep(backoff)
                    continue
                return {"code": "-2",
                        "msg": f"API限流排队超时（重试{_MAX_RATE_LIMIT_RETRIES}次后仍失败）"}
            # 排队完成后生成时间戳和签名
            timestamp = self._get_timestamp()
            sign = self._sign(timestamp, method, sign_path, body)
            headers = self.headers.copy()
            headers["OK-ACCESS-SIGN"] = sign
            headers["OK-ACCESS-TIMESTAMP"] = timestamp
            headers["OK-ACCESS-PASSPHRASE"] = self.passphrase

            try:
                # 发送请求
                if method == "GET":
                    response = self._request_get(url, params=params, headers=headers)
                elif method == "POST":
                    response = self._request_post(url, data=data, headers=headers)
                else:
                    raise ValueError(f"不支持的 HTTP 方法：{method}")

                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout:
                if rate_attempt == 0:
                    time.sleep(0.5)
                    continue   # 回限流重试循环（会重新签名）
                return {"code": "-3", "msg": "网络请求超时（重试后仍失败）"}
            except requests.exceptions.RequestException as e:
                return {"code": "-1", "msg": f"请求失败：{str(e)}"}
            except Exception as e:
                return {"code": "-1", "msg": f"错误：{str(e)}"}

        # 所有限流重试已用完（理论上不会到达这里，但安全兜底）
        return {"code": "-2", "msg": "API限流排队超时"}

    def _active_proxies(self):
        if self._proxy_disabled_after_error:
            return None
        return self.proxies

    def _disable_bad_proxy(self, exc: Exception) -> None:
        if self.proxies and not self._proxy_disabled_after_error:
            self._proxy_disabled_after_error = True
            print(f"[OKXClient] 代理不可用，已自动切换直连: {self.proxy_url} ({exc})")

    def _direct_session(self):
        session = requests.Session()
        session.trust_env = False
        session.verify = False
        return session

    def _request_get(self, url: str, params: Dict = None, headers: Dict = None):
        try:
            proxies = self._active_proxies()
            if proxies:
                return requests.get(url, params=params, headers=headers, proxies=proxies, timeout=10, verify=False)
            with self._direct_session() as session:
                return session.get(url, params=params, headers=headers, timeout=10)
        except (ProxyError, ConnectionError) as exc:
            self._disable_bad_proxy(exc)
            with self._direct_session() as session:
                return session.get(url, params=params, headers=headers, timeout=10)

    def _request_post(self, url: str, data: Dict = None, headers: Dict = None):
        try:
            proxies = self._active_proxies()
            if proxies:
                return requests.post(url, json=data, headers=headers, proxies=proxies, timeout=10, verify=False)
            with self._direct_session() as session:
                return session.post(url, json=data, headers=headers, timeout=10)
        except (ProxyError, ConnectionError) as exc:
            self._disable_bad_proxy(exc)
            with self._direct_session() as session:
                return session.post(url, json=data, headers=headers, timeout=10)

    def get_tickers(self, instType: str = "SPOT") -> Dict:
        """
        获取所有交易对行情

        Args:
            instType: 产品类型 (SPOT, SWAP, FUTURES, OPTION)

        Returns:
            行情数据
        """
        endpoint = "/api/v5/market/tickers"
        params = {"instType": instType}
        return self._request("GET", endpoint, params=params)

    def get_ticker(self, instId: str) -> Dict:
        """
        获取单个交易对行情

        Args:
            instId: 交易对 ID (如 BTC-USDT)

        Returns:
            行情数据
        """
        endpoint = "/api/v5/market/ticker"
        params = {"instId": instId}
        return self._request("GET", endpoint, params=params)

    def get_kline(self, instId: str, bar: str = "1m", limit: int = 100,
                  after: str = None, before: str = None) -> Dict:
        """
        获取 K 线数据 (最新 1440 根)

        Args:
            instId: 交易对 ID
            bar: K 线类型 (1m, 5m, 15m, 30m, 1H, 2H, 4H, 1D, 1W)
            limit: 返回数量 (最多 300)
            after: 获取该时间戳之前（更旧）的数据 (毫秒)
            before: 获取该时间戳之后（更新）的数据 (毫秒)

        Returns:
            K 线数据
        """
        endpoint = "/api/v5/market/candles"
        params = {
            "instId": instId,
            "bar": bar,
            "limit": min(limit, 300)
        }
        if after:
            params["after"] = after
        if before:
            params["before"] = before

        return self._request("GET", endpoint, params=params)

    def get_history_kline(self, instId: str, bar: str = "1m", limit: int = 100,
                          after: str = None, before: str = None) -> Dict:
        """
        获取历史 K 线数据 (1440 根之前的数据)

        Args:
            instId: 交易对 ID
            bar: K 线类型 (1m, 5m, 15m, 30m, 1H, 2H, 4H, 1D, 1W)
            limit: 返回数量 (最多 100)
            after: 获取该时间戳之前（更旧）的数据 (毫秒)
            before: 获取该时间戳之后（更新）的数据 (毫秒)

        Returns:
            K 线数据
        """
        endpoint = "/api/v5/market/history-candles"
        params = {
            "instId": instId,
            "bar": bar,
            "limit": min(limit, 100)
        }
        if after:
            params["after"] = after
        if before:
            params["before"] = before

        return self._request("GET", endpoint, params=params)

    def get_order_book(self, instId: str, limit: int = 10) -> Dict:
        """
        获取订单簿

        Args:
            instId: 交易对 ID
            limit: 深度档位 (5, 10, 20)

        Returns:
            订单簿数据
        """
        endpoint = "/api/v5/market/books"
        params = {
            "instId": instId,
            "sz": limit
        }
        return self._request("GET", endpoint, params=params)

    def get_funding_rate(self, instId: str) -> Dict:
        """获取永续合约资金费率"""
        endpoint = "/api/v5/public/funding-rate"
        params = {"instId": instId}
        return self._request("GET", endpoint, params=params)

    def get_open_interest(self, instId: str, instType: str = "SWAP") -> Dict:
        """获取合约持仓量"""
        endpoint = "/api/v5/public/open-interest"
        params = {"instType": instType, "instId": instId}
        return self._request("GET", endpoint, params=params)

    def get_balance(self) -> Dict:
        """
        获取账户余额

        Returns:
            余额数据
        """
        endpoint = "/api/v5/account/balance"
        return self._request("GET", endpoint)

    def get_positions(self, instId: str = None) -> Dict:
        """
        获取持仓信息

        Args:
            instId: 交易对 ID，为 None 时获取所有持仓

        Returns:
            持仓数据
        """
        endpoint = "/api/v5/account/positions"
        params = {}
        if instId:
            params["instId"] = instId
        return self._request("GET", endpoint, params=params)

    def get_account_config(self) -> Dict:
        """获取账户配置，包括账户模式与持仓模式。"""
        endpoint = "/api/v5/account/config"
        return self._request("GET", endpoint)

    def place_order(self, instId: str, tdMode: str, side: str,
                    ordType: str, sz: str, px: str = None,
                    posSide: str = None, reduceOnly: bool = None) -> Dict:
        """
        下单交易

        Args:
            instId: 交易对 ID
            tdMode: 交易模式 (cash: 现货，cross: 全仓，isolated: 逐仓)
            side: 订单方向 (buy: 买入，sell: 卖出)
            ordType: 订单类型 (market: 市价，limit: 限价)
            sz: 委托数量
            px: 委托价格 (限价单必填)
            posSide: 持仓方向 (net: 单向，long: 多仓，short: 空仓)
            reduceOnly: 是否只减仓

        Returns:
            下单结果
        """
        endpoint = "/api/v5/trade/order"
        data = {
            "instId": instId,
            "tdMode": tdMode,
            "side": side,
            "ordType": ordType,
            "sz": sz
        }
        if px:
            data["px"] = px
        if posSide:
            data["posSide"] = posSide
        if reduceOnly is not None:
            data["reduceOnly"] = bool(reduceOnly)

        return self._request("POST", endpoint, data=data)

    def cancel_order(self, instId: str, ordId: str) -> Dict:
        """
        撤销订单

        Args:
            instId: 交易对 ID
            ordId: 订单 ID

        Returns:
            撤单结果
        """
        endpoint = "/api/v5/trade/cancel-order"
        data = {
            "instId": instId,
            "ordId": ordId
        }
        return self._request("POST", endpoint, data=data)

    def get_order(self, instId: str, ordId: str) -> Dict:
        """
        获取订单详情

        Args:
            instId: 交易对 ID
            ordId: 订单 ID

        Returns:
            订单详情
        """
        endpoint = "/api/v5/trade/order"
        params = {
            "instId": instId,
            "ordId": ordId
        }
        return self._request("GET", endpoint, params=params)

    def get_pending_orders(self, instId: str = None, limit: int = 100) -> Dict:
        """
        获取未成交/部分成交委托

        Args:
            instId: 交易对 ID
            limit: 返回数量

        Returns:
            当前挂单
        """
        endpoint = "/api/v5/trade/orders-pending"
        params = {"limit": limit}
        if instId:
            params["instId"] = instId
        return self._request("GET", endpoint, params=params)

    def get_order_history(self, instId: str = None, limit: int = 100) -> Dict:
        """
        获取历史订单

        Args:
            instId: 交易对 ID
            limit: 返回数量

        Returns:
            历史订单
        """
        endpoint = "/api/v5/trade/orders-history"
        params = {"limit": limit}
        if instId:
            params["instId"] = instId
        return self._request("GET", endpoint, params=params)

    def get_fills(self, instId: str = None, limit: int = 100) -> Dict:
        """
        获取历史成交明细

        Args:
            instId: 交易对 ID
            limit: 返回数量

        Returns:
            历史成交
        """
        endpoint = "/api/v5/trade/fills"
        params = {"limit": limit}
        if instId:
            params["instId"] = instId
        return self._request("GET", endpoint, params=params)

    def test_connection(self) -> bool:
        """
        测试 API 连接

        Returns:
            是否连接成功
        """
        result = self.get_tickers(instType="SWAP")
        return result.get("code") == "0"
