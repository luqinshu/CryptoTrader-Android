"""
OKX 交易所 API 客户端
封装 OKX V5 API 接口
"""

import time
import hashlib
import hmac
import base64
import requests
from typing import Dict, List, Optional


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

        # API 基础 URL - 使用 HTTP 避免代理 SSL 问题
        self.base_url = "http://www.okx.com"

        # 代理配置
        self.proxies = None
        if proxy_url:
            self.proxies = {
                "http": proxy_url,
                "https": proxy_url
            }

        # 请求头
        self.headers = {
            "Content-Type": "application/json",
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": "",
            "OK-ACCESS-TIMESTAMP": "",
            "OK-ACCESS-PASSPHRASE": self.passphrase,
        }

    def _get_timestamp(self) -> str:
        """获取 ISO 格式时间戳"""
        return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())

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
        """发送 HTTP 请求"""
        url = self.base_url + endpoint
        timestamp = self._get_timestamp()

        # 准备请求体
        body = ""
        if data:
            import json
            body = json.dumps(data)

        # 生成签名
        sign = self._sign(timestamp, method, endpoint, body)

        # 设置请求头
        headers = self.headers.copy()
        headers["OK-ACCESS-SIGN"] = sign
        headers["OK-ACCESS-TIMESTAMP"] = timestamp
        headers["OK-ACCESS-PASSPHRASE"] = self.passphrase

        try:
            # 发送请求
            if method == "GET":
                response = requests.get(
                    url,
                    params=params,
                    headers=headers,
                    proxies=self.proxies,
                    timeout=10
                )
            elif method == "POST":
                response = requests.post(
                    url,
                    json=data,
                    headers=headers,
                    proxies=self.proxies,
                    timeout=10
                )
            else:
                raise ValueError(f"不支持的 HTTP 方法：{method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            return {"code": "-1", "msg": "请求超时"}
        except requests.exceptions.RequestException as e:
            return {"code": "-1", "msg": f"请求失败：{str(e)}"}
        except Exception as e:
            return {"code": "-1", "msg": f"错误：{str(e)}"}

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
                  start_time: str = None, end_time: str = None) -> Dict:
        """
        获取 K 线数据

        Args:
            instId: 交易对 ID
            bar: K 线类型 (1m, 5m, 15m, 30m, 1H, 2H, 4H, 1D, 1W)
            limit: 返回数量 (最多 300)
            start_time: 开始时间 (毫秒)
            end_time: 结束时间 (毫秒)

        Returns:
            K 线数据
        """
        endpoint = "/api/v5/market/candles"
        params = {
            "instId": instId,
            "bar": bar,
            "limit": min(limit, 300)
        }
        if start_time:
            params["after"] = start_time
        if end_time:
            params["before"] = end_time

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

    def place_order(self, instId: str, tdMode: str, side: str,
                    ordType: str, sz: str, px: str = None,
                    posSide: str = None) -> Dict:
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

    def test_connection(self) -> bool:
        """
        测试 API 连接

        Returns:
            是否连接成功
        """
        result = self.get_tickers(instType="SPOT")
        return result.get("code") == "0"
