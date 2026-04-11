"""
交易执行器模块 V2.0 (OKX 专业版)
负责在 OKX 交易所执行交易订单，支持原生止盈止损、合约张数适配及风险控制。
"""

import threading
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"

class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"
    NONE = "none"

@dataclass
class OrderResult:
    success: bool
    order_id: str = ""
    message: str = ""
    filled_size: float = 0
    filled_price: float = 0

@dataclass
class PositionInfo:
    inst_id: str
    side: PositionSide
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    pnl_percent: float
    leverage: float = 1


class TradeExecutor:
    """交易执行器 - 深度适配 OKX V5 合约与现货"""

    def __init__(self, okx_client):
        self.okx_client = okx_client
        self.inst_info_cache: Dict[str, Dict] = {} # 交易规则缓存

    def _get_inst_info(self, inst_id: str) -> Dict:
        """获取交易对精度、面值等核心规则"""
        if inst_id in self.inst_info_cache:
            return self.inst_info_cache[inst_id]
        
        # 自动判定产品类型
        inst_type = "SWAP" if "-SWAP" in inst_id else "SPOT"
        res = self.okx_client._request("GET", "/api/v5/public/instruments", params={"instId": inst_id, "instType": inst_type})
        
        if res.get('code') == '0' and res.get('data'):
            info = res['data'][0]
            self.inst_info_cache[inst_id] = info
            return info
        return {}

    def set_leverage(self, inst_id: str, lever: int = 10):
        """设置杠杆 (仅限合约)"""
        if "-SWAP" not in inst_id: return
        return self.okx_client._request("POST", "/api/v5/account/set-leverage", data={
            "instId": inst_id,
            "lever": str(lever),
            "mgnMode": "cross"
        })

    def calculate_lots(self, inst_id: str, usdt_amount: float) -> float:
        """根据 USDT 金额计算正确的下单数量 (现货为币数, 合约为张数)"""
        info = self._get_inst_info(inst_id)
        if not info: return 0
        
        # 获取最新价
        ticker = self.okx_client.get_ticker(inst_id)
        last_price = float(ticker['data'][0]['last'])
        
        if "-SWAP" in inst_id:
            # 合约：张数 = USDT / (价格 * 面值)
            ct_val = float(info.get('ctVal', 1))
            lots = usdt_amount / (last_price * ct_val)
            lot_sz = float(info.get('lotSz', 1))
            # 向下取整到最小张数倍数
            final_sz = int(lots / lot_sz) * lot_sz
        else:
            # 现货：数量 = USDT / 价格
            sz_precision = int(info.get('szPrecision', 8))
            final_sz = round(usdt_amount / last_price, sz_precision)
            
        return final_sz

    def place_smart_order(self, inst_id: str, side: str, pos_side: str, usdt_amount: float, 
                          tp_pct: float = 0.05, sl_pct: float = 0.03) -> OrderResult:
        """
        全自动智能下单：含杠杆设置、张数计算、原生止盈止损。
        tp_pct: 止盈比例 (如 0.05 代表 5%)
        sl_pct: 止损比例 (如 0.03 代表 3%)
        """
        try:
            # 1. 设置杠杆
            self.set_leverage(inst_id, 10)
            
            # 2. 计算下单量
            sz = self.calculate_lots(inst_id, usdt_amount)
            if sz <= 0: return OrderResult(False, message="可用余额不足以完成最小下单单位")
            
            # 3. 计算止盈止损价格
            ticker = self.okx_client.get_ticker(inst_id)
            last_price = float(ticker['data'][0]['last'])
            
            if pos_side == "long":
                tp_price = last_price * (1 + tp_pct)
                sl_price = last_price * (1 - sl_pct)
            else:
                tp_price = last_price * (1 - tp_pct)
                sl_price = last_price * (1 + sl_pct)

            # 4. 构建 OKX V5 订单
            # tpOrderKind/slOrderKind 为 condition 代表策略委托
            data = {
                "instId": inst_id,
                "tdMode": "cross" if "-SWAP" in inst_id else "cash",
                "side": side,
                "posSide": pos_side,
                "ordType": "market",
                "sz": str(sz),
                "tpTriggerPx": f"{tp_price:.8f}".rstrip('0').rstrip('.'),
                "tpOrdPx": "-1", # 市价止盈
                "slTriggerPx": f"{sl_price:.8f}".rstrip('0').rstrip('.'),
                "slOrdPx": "-1"  # 市价止损
            }
            
            res = self.okx_client._request("POST", "/api/v5/trade/order", data=data)
            
            if res.get('code') == '0':
                return OrderResult(True, order_id=res['data'][0]['ordId'], message=f"已成功开启 {pos_side} 仓位，同步挂载服务器端止盈止损。")
            else:
                return OrderResult(False, message=f"下单失败: {res.get('msg')}")

        except Exception as e:
            return OrderResult(False, message=f"执行异常: {str(e)}")

    def get_positions(self) -> Dict[str, PositionInfo]:
        """获取当前活跃持仓 (返回字典格式)"""
        positions = self.get_active_positions()
        return {pos.inst_id: pos for pos in positions}

    def get_active_positions(self) -> List[PositionInfo]:
        """获取当前活跃持仓"""
        res = self.okx_client.get_positions()
        positions = []
        if res.get('code') == '0' and res.get('data'):
            for p in res['data']:
                if float(p['pos']) == 0: continue
                positions.append(PositionInfo(
                    inst_id=p['instId'],
                    side=PositionSide.LONG if p['posSide'] == 'long' else PositionSide.SHORT,
                    size=abs(float(p['pos'])),
                    entry_price=float(p['avgPx']),
                    current_price=float(p['last']),
                    unrealized_pnl=float(p['upl']),
                    pnl_percent=float(p['uplRatio']) * 100,
                    leverage=float(p['lever'])
                ))
        return positions

    def get_usdt_balance(self) -> float:
        """获取 USDT 余额"""
        res = self.okx_client.get_balance()
        if res.get('code') == '0' and res.get('data'):
            details = res['data'][0].get('details', [])
            for d in details:
                if d.get('ccy') == 'USDT':
                    return float(d.get('availBal', 0)) + float(d.get('frozenBal', 0))
            total_equity = res['data'][0].get('totalEq', '0')
            if total_equity:
                return float(total_equity)
        return 0.0

    def get_usdt_balance(self) -> float:
        """获取账户 USDT 余额 (UI 兼容)"""
        res = self.okx_client.get_balance()
        if res.get('code') == '0' and res.get('data'):
            for detail in res['data'][0].get('details', []):
                if detail['ccy'] == 'USDT':
                    return float(detail.get('availEq', 0))
        return 0.0

    def get_positions(self, inst_id: str = None) -> Dict[str, PositionInfo]:
        """获取持仓字典 (UI 兼容)"""
        pos_list = self.get_active_positions()
        pos_dict = {p.inst_id: p for p in pos_list}
        if inst_id:
            return {inst_id: pos_dict[inst_id]} if inst_id in pos_dict else {}
        return pos_dict

    def execute_buy(self, inst_id: str, position_ratio: float = 0.1) -> OrderResult:
        """快速买入/做多 (UI & Runner 兼容)"""
        balance = self.get_usdt_balance()
        return self.place_smart_order(inst_id, "buy", "long", balance * position_ratio)

    def execute_sell(self, inst_id: str, quantity: float) -> OrderResult:
        """平仓做多/卖出 (UI & Runner 兼容)"""
        pos_dict = self.get_positions(inst_id)
        if inst_id in pos_dict:
            pos = pos_dict[inst_id]
            data = {
                "instId": inst_id,
                "tdMode": "cross" if "-SWAP" in inst_id else "cash",
                "side": "sell" if pos.side == PositionSide.LONG else "buy",
                "posSide": "long" if pos.side == PositionSide.LONG else "short",
                "ordType": "market",
                "sz": str(pos.size)
            }
            res = self.okx_client.place_order(**data) # 改用规范方法
            return OrderResult(res.get('code') == '0', message=res.get('msg', ''))
        return OrderResult(False, message="无活跃持仓")

    def execute_short(self, inst_id: str, position_ratio: float = 0.1) -> OrderResult:
        """快速开空 (Runner 兼容)"""
        balance = self.get_usdt_balance()
        return self.place_smart_order(inst_id, "sell", "short", balance * position_ratio)

    def execute_cover(self, inst_id: str, quantity: float) -> OrderResult:
        """平空 (Runner 兼容)"""
        return self.execute_sell(inst_id, quantity)

    def execute_stop_loss(self, inst_id: str) -> OrderResult:
        """手动平仓 (UI 兼容)"""
        return self.execute_sell(inst_id, 0)
