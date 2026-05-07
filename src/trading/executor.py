"""
交易执行器模块 V2.0 (OKX 专业版)
负责在 OKX 交易所执行交易订单，支持原生止盈止损、合约张数适配及风险控制。
"""

import logging
import threading
import time
import math
import uuid
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

_log = logging.getLogger("TradeExecutor")


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
    raw_pos: float = 0.0
    raw_pos_side: str = ""
    mgn_mode: str = ""
    c_time: str = ""  # OKX 持仓创建时间 (毫秒时间戳)
    notional_usd: float = 0.0  # 名义价值


class TradeExecutor:
    """交易执行器 - 深度适配 OKX V5 合约与现货"""

    def __init__(self, okx_client):
        self.okx_client = okx_client
        self._cache_lock = threading.Lock()
        self.inst_info_cache: Dict[str, Dict] = {} # 交易规则缓存
        self._last_balance_diag_key: Optional[str] = None
        self._account_config_cache: Dict[str, object] = {}
        self._account_config_cache_ts: float = 0.0
        self._startup_diag_logged = False

    def _log_balance_diag_once(self, key: str, message: str):
        """同类余额诊断只打印一次，避免刷屏。"""
        if self._last_balance_diag_key == key:
            return
        self._last_balance_diag_key = key
        _log.info(message)

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        """安全转换浮点数"""
        try:
            if value in ("", None):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_leverage(leverage: float) -> int:
        try:
            value = int(leverage or 1)
        except (TypeError, ValueError):
            value = 1
        return max(value, 1)

    def _entry_notional(self, usdt_amount: float, leverage: float) -> float:
        margin = max(self._safe_float(usdt_amount, 0.0), 0.0)
        return margin * self._normalize_leverage(leverage)

    def _required_margin(self, notional: float, leverage: float) -> float:
        lev = self._normalize_leverage(leverage)
        return max(self._safe_float(notional, 0.0), 0.0) / lev

    def estimate_fee_for_notional(self, notional: float, fee_rate: float = 0.0005) -> float:
        return abs(self._safe_float(notional, 0.0)) * fee_rate

    def _log_startup_risk_context_once(self, inst_id: str, leverage: int, mgn_mode: str = "cross"):
        if self._startup_diag_logged:
            return
        self._startup_diag_logged = True
        account_cfg = self._get_account_config(force_refresh=True)
        pos_mode = str(account_cfg.get('posMode') or 'unknown')
        acct_level = str(account_cfg.get('acctLv') or 'unknown')
        lever = self._normalize_leverage(leverage)
        info = self._get_inst_info(inst_id)
        ct_val = str(info.get('ctVal', 'N/A')) if info else 'N/A'
        min_sz = str(info.get('minSz', 'N/A')) if info else 'N/A'
        _log.info(
            f"启动自检 posMode={pos_mode} acctLv={acct_level} "
            f"targetInst={inst_id} targetMgnMode={mgn_mode} "
            f"requestedLeverage={lever}x ctVal={ct_val} minSz={min_sz}"
        )

    def _get_inst_info(self, inst_id: str) -> Dict:
        """获取交易对精度、面值等核心规则（线程安全）"""
        with self._cache_lock:
            if inst_id in self.inst_info_cache:
                return self.inst_info_cache[inst_id]

        # 自动判定产品类型
        inst_type = "SWAP" if "-SWAP" in inst_id else "SPOT"
        res = self.okx_client._request("GET", "/api/v5/public/instruments", params={"instId": inst_id, "instType": inst_type})

        if res.get('code') == '0' and res.get('data'):
            info = res['data'][0]
            with self._cache_lock:
                self.inst_info_cache[inst_id] = info
            return info
        return {}

    def _get_account_config(self, force_refresh: bool = False) -> Dict[str, object]:
        """获取账户配置（线程安全缓存）"""
        now = time.time()
        with self._cache_lock:
            if not force_refresh and self._account_config_cache and (now - self._account_config_cache_ts) < 60:
                return dict(self._account_config_cache)
        try:
            res = self.okx_client.get_account_config()
            rows = res.get('data') or []
            if rows:
                with self._cache_lock:
                    self._account_config_cache = dict(rows[0])
                    self._account_config_cache_ts = now
                return dict(self._account_config_cache)
        except Exception as e:
            _log.warning(f"获取账户配置失败: {e}")
        with self._cache_lock:
            return dict(self._account_config_cache)

    def _get_margin_mode(self, inst_id: str, cfg: Dict = None) -> str:
        """检测指定品种的保证金模式（cross/isolated），默认返回 cross"""
        if not inst_id or "-SWAP" not in inst_id:
            return "cross"
        if cfg is None:
            cfg = self._get_account_config()
        acct_level = str(cfg.get('acctLv', '1'))
        # 查询逐仓模式下该品种的保证金模式
        try:
            res = self.okx_client._request("GET", "/api/v5/account/config", params={})
            rows = res.get('data') or []
            for row in rows:
                if str(row.get('instId', '')).upper() == inst_id.upper():
                    return str(row.get('mgnMode', 'cross')).lower()
        except Exception:
            pass
        return "cross"

    def _extract_order_error_message(self, res: Dict) -> str:
        msg = str(res.get('msg') or '').strip()
        details = res.get('data') or []
        if details and isinstance(details, list):
            item = details[0] if isinstance(details[0], dict) else {}
            detail_msg = str(item.get('sMsg') or item.get('msg') or '').strip()
            detail_code = str(item.get('sCode') or '').strip()
            if detail_msg and detail_code:
                return f"{detail_msg} (sCode={detail_code})"
            if detail_msg:
                return detail_msg
        return msg or "交易所未返回详细错误"

    def _extract_order_error_code(self, res: Dict) -> str:
        details = res.get('data') or []
        if details and isinstance(details, list):
            item = details[0] if isinstance(details[0], dict) else {}
            detail_code = str(item.get('sCode') or '').strip()
            if detail_code:
                return detail_code
        return str(res.get('code') or '').strip()

    def _get_position_mode_params(self, inst_id: str, pos: PositionInfo) -> Dict[str, object]:
        """根据 OKX 当前持仓记录判断平仓单该如何携带 posSide/reduceOnly。"""
        params: Dict[str, object] = {}
        account_cfg = self._get_account_config()
        pos_mode = str(account_cfg.get('posMode') or '').lower()
        if pos_mode == 'net_mode':
            params['reduceOnly'] = True
            params['posSide'] = 'net'
            _log.debug(f" {inst_id} account_posMode=net_mode -> reduceOnly + posSide=net")
            return params
        if pos_mode == 'long_short_mode':
            params['posSide'] = 'long' if pos.side == PositionSide.LONG else 'short'
            _log.debug(f" {inst_id} account_posMode=long_short_mode -> posSide={params['posSide']}")
            return params

        try:
            res = self.okx_client.get_positions(inst_id)
            rows = res.get('data') or []
            active_rows = [
                row for row in rows
                if abs(self._safe_float(row.get('pos'))) > 0
            ]
            if active_rows:
                preview = [
                    {
                        'posSide': str(row.get('posSide') or ''),
                        'pos': row.get('pos'),
                        'mgnMode': row.get('mgnMode'),
                    }
                    for row in active_rows[:3]
                ]
                _log.debug(f" {inst_id} active_rows={preview}")
            if active_rows:
                net_rows = [row for row in active_rows if str(row.get('posSide') or '').lower() == 'net']
                if net_rows:
                    params['reduceOnly'] = True
                    return params

                target_pos_side = 'long' if pos.side == PositionSide.LONG else 'short'
                matched_rows = [
                    row for row in active_rows
                    if str(row.get('posSide') or '').lower() == target_pos_side
                ]
                if matched_rows:
                    params['posSide'] = target_pos_side
                    return params

                # 如果只有一条有效持仓记录且为 long/short，直接跟随该记录，避免拿到空仓占位。
                if len(active_rows) == 1:
                    only_side = str(active_rows[0].get('posSide') or '').lower()
                    if only_side in {'long', 'short'}:
                        params['posSide'] = only_side
                        return params
        except Exception as e:
            _log.debug(f" {inst_id} 获取仓位模式失败: {e}，使用回退逻辑")

        # 回退逻辑：未知模式时，优先按 long/short 模式构造。
        params['posSide'] = 'long' if pos.side == PositionSide.LONG else 'short'
        _log.debug(f" {inst_id} fallback posSide={params['posSide']}")
        return params

    def _build_close_order_context(self, inst_id: str, pos: PositionInfo) -> Dict[str, object]:
        """基于 OKX 原始持仓返回构造最稳妥的平仓/减仓订单上下文。"""
        context = {
            "tdMode": "cross" if "-SWAP" in inst_id else "cash",
            "side": "sell" if pos.side == PositionSide.LONG else "buy",
        }
        raw_pos = self._safe_float(getattr(pos, "raw_pos", 0.0))
        raw_pos_side = str(getattr(pos, "raw_pos_side", "") or "").lower()
        raw_mgn_mode = str(getattr(pos, "mgn_mode", "") or "").lower()
        if raw_mgn_mode in {"cross", "isolated"}:
            context["tdMode"] = raw_mgn_mode

        account_cfg = self._get_account_config()
        pos_mode = str(account_cfg.get("posMode") or "").lower()
        if raw_pos != 0:
            if pos_mode == "net_mode" or raw_pos_side == "net":
                context["reduceOnly"] = True
                context["posSide"] = "net"
                context["side"] = "sell" if raw_pos > 0 else "buy"
                _log.debug(
                    f"{inst_id} raw_pos={raw_pos}, raw_posSide={raw_pos_side or 'net'}, "
                    f"mgnMode={context['tdMode']} -> side={context['side']} reduceOnly+net"
                )
                return context
            if raw_pos_side in {"long", "short"}:
                context["posSide"] = raw_pos_side
                context["side"] = "sell" if raw_pos_side == "long" else "buy"
                _log.debug(
                    f"{inst_id} raw_pos={raw_pos}, raw_posSide={raw_pos_side}, "
                    f"mgnMode={context['tdMode']} -> side={context['side']}"
                )
                return context

        context.update(self._get_position_mode_params(inst_id, pos))
        return context

    def _place_reduce_order_with_net_retry(self, data: Dict, inst_id: str) -> Dict:
        """针对 net_mode 下可能的方向误判，在 51169 时自动反向重试一次。"""
        res = self.okx_client.place_order(**data)
        params = self._get_account_config()
        pos_mode = str(params.get('posMode') or '').lower()
        err_code = self._extract_order_error_code(res)
        if pos_mode == 'net_mode' and err_code == '51169':
            retry_data = dict(data)
            retry_data['side'] = 'buy' if data.get('side') == 'sell' else 'sell'
            _log.debug(f" {inst_id} net_mode 首次方向失败，自动反向重试 side={retry_data['side']}")
            retry_res = self.okx_client.place_order(**retry_data)
            retry_code = self._extract_order_error_code(retry_res)
            if retry_res.get('code') == '0' or retry_code == '0':
                return retry_res
            return retry_res
        return res

    def set_leverage(self, inst_id: str, lever: int = 10) -> bool:
        """设置杠杆（自动检测当前保证金模式）。"""
        if "-SWAP" not in inst_id:
            return True
        MAX_LEVERAGE = 10
        if lever > MAX_LEVERAGE:
            _log.warning(f"请求杠杆 {lever}x 超过安全上限 {MAX_LEVERAGE}x，已自动降至 {MAX_LEVERAGE}x")
            lever = MAX_LEVERAGE
        if lever < 1:
            lever = 1
        try:
            # 检测当前账户保证金模式（cross/isolated）
            cfg = self._get_account_config()
            pos_mode = cfg.get('posMode', 'long_short_mode')
            mgn_mode = self._get_margin_mode(inst_id, cfg)
            if mgn_mode not in ('cross', 'isolated'):
                mgn_mode = 'cross'
            res = self.okx_client._request("POST", "/api/v5/account/set-leverage", data={
                "instId": inst_id,
                "lever": str(lever),
                "mgnMode": mgn_mode
            })
            if res.get('code') != '0':
                _log.warning(f"{inst_id} 设置 {lever}x 失败: {res.get('msg', '未知错误')}")
                return False
            # 杠杆变更后强制刷新账户配置缓存，避免后续下单用旧 posMode
            self._account_config_cache_ts = 0.0
            return True
        except Exception as e:
            _log.warning(f"{inst_id} 设置异常: {e}")
            return False

    def calculate_lots(self, inst_id: str, usdt_amount: float, leverage: float = 1.0) -> float:
        """根据保证金金额计算正确的下单数量 (现货为币数, 合约为张数)。"""
        return self.calculate_lots_with_price(inst_id, usdt_amount, reference_price=None, leverage=leverage)

    def calculate_lots_with_price(self, inst_id: str, usdt_amount: float,
                                  reference_price: Optional[float] = None,
                                  leverage: float = 1.0) -> float:
        """根据给定参考价格计算下单数量"""
        info = self._get_inst_info(inst_id)
        if not info:
            _log.warning(f" {inst_id} 未获取到交易规则，返回 0")
            return 0

        if reference_price is None or reference_price <= 0:
            ticker = self.okx_client.get_ticker(inst_id)
            if ticker.get('code') != '0' or not ticker.get('data'):
                _log.warning(f" {inst_id} 未获取到行情价格，返回 0")
                return 0
            reference_price = float(ticker['data'][0]['last'])
        if reference_price <= 0:
            _log.warning(f" {inst_id} 参考价格无效 {reference_price}，返回 0")
            return 0
        
        margin = max(self._safe_float(usdt_amount, 0.0), 0.0)
        if "-SWAP" in inst_id:
            notional = self._entry_notional(margin, leverage)
            # 合约：张数 = 名义金额 / (价格 * 面值)
            ct_val = float(info.get('ctVal', 1))
            lots = notional / (reference_price * ct_val)
            lot_sz = float(info.get('lotSz', 1))
            # 向下取整到最小张数倍数
            if lot_sz <= 0:
                lot_sz = 1.0
            final_sz = math.floor(lots / lot_sz) * lot_sz
        else:
            # 现货：数量 = 保证金金额 / 价格（现货不使用杠杆）
            min_sz = float(info.get('minSz', 0) or 0)
            lot_sz = float(info.get('lotSz', 0) or 0)
            size = margin / reference_price
            if lot_sz > 0:
                size = math.floor(size / lot_sz) * lot_sz
            if min_sz > 0 and size < min_sz:
                return 0
            final_sz = size
            
        return final_sz

    def estimate_order(self, inst_id: str, usdt_amount: float, price: Optional[float] = None,
                       leverage: float = 1.0) -> Dict:
        """估算订单信息，用于手动交易界面展示"""
        info = self._get_inst_info(inst_id)
        if not info:
            return {"success": False, "message": "未获取到交易规则"}

        market_price = 0.0
        ticker = self.okx_client.get_ticker(inst_id)
        if ticker.get('code') == '0' and ticker.get('data'):
            market_price = float(ticker['data'][0].get('last') or 0)

        reference_price = price if price and price > 0 else market_price
        if reference_price <= 0:
            return {"success": False, "message": "未获取到参考价格"}

        size = self.calculate_lots_with_price(inst_id, usdt_amount, reference_price, leverage=leverage)
        ct_val = float(info.get('ctVal', 1) or 1)
        lot_sz = float(info.get('lotSz', 0) or 0)
        min_sz = float(info.get('minSz', 0) or 0)
        tick_sz = float(info.get('tickSz', 0) or 0)

        estimated_notional = size * reference_price * ct_val if "-SWAP" in inst_id else size * reference_price
        estimated_margin = self._required_margin(estimated_notional, leverage) if "-SWAP" in inst_id else estimated_notional
        estimated_fee = estimated_notional * 0.0005
        return {
            "success": True,
            "market_price": market_price,
            "reference_price": reference_price,
            "size": size,
            "min_size": min_sz,
            "lot_size": lot_sz,
            "tick_size": tick_sz,
            "contract_value": ct_val,
            "estimated_notional": estimated_notional,
            "estimated_margin": estimated_margin,
            "estimated_fee": estimated_fee,
            "is_swap": "-SWAP" in inst_id,
        }

    def get_pending_orders(self, inst_id: Optional[str] = None) -> List[Dict]:
        """获取当前未成交委托"""
        res = self.okx_client.get_pending_orders(instId=inst_id)
        if res.get('code') == '0':
            return res.get('data', [])
        return []

    def get_order_history(self, inst_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """获取历史委托"""
        res = self.okx_client.get_order_history(instId=inst_id, limit=limit)
        if res.get('code') == '0':
            return res.get('data', [])
        return []

    def get_fill_history(self, inst_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """获取历史成交"""
        res = self.okx_client.get_fills(instId=inst_id, limit=limit)
        if res.get('code') == '0':
            return res.get('data', [])
        return []

    def place_smart_order(self, inst_id: str, side: str, pos_side: str, usdt_amount: float,
                          tp_pct: float = 0.05, sl_pct: float = 0.03, leverage: int = 10,
                          order_type: str = "market", price: Optional[float] = None,
                          tgt_type: str = "market") -> OrderResult:
        """
        全自动智能下单：含杠杆设置、张数计算、原生止盈止损。
        order_type: market | limit | stop_market | ioc | fok | post_only
        tgt_type: 止盈止损执行方式 'market'(市价) 或 'limit'(限价)
        """
        OKX_ORDER_TYPES = {
            "market": "market", "limit": "limit",
            "stop_market": "conditional", "ioc": "ioc",
            "fok": "fok", "post_only": "post_only",
        }
        okx_ord_type = OKX_ORDER_TYPES.get(order_type, order_type)
        try:
            # 1. 设置杠杆
            normalized_leverage = self._normalize_leverage(leverage)
            self._log_startup_risk_context_once(inst_id, normalized_leverage, "cross" if "-SWAP" in inst_id else "cash")
            if "-SWAP" in inst_id and not self.set_leverage(inst_id, normalized_leverage):
                return OrderResult(False, message=f"{inst_id} 设置杠杆 {normalized_leverage}x 失败，已取消下单")

            ticker = self.okx_client.get_ticker(inst_id)
            if ticker.get('code') != '0' or not ticker.get('data'):
                return OrderResult(False, message="未获取到最新行情")
            last_price = float(ticker['data'][0]['last'])
            entry_price = price if order_type == "limit" and float(price or 0) > 0 else last_price
            if order_type == "limit" and float(price or 0) <= 0:
                return OrderResult(False, message="限价单价格必须 > 0")

            # 2. 计算下单量
            notional = self._entry_notional(usdt_amount, normalized_leverage) if "-SWAP" in inst_id else max(self._safe_float(usdt_amount, 0.0), 0.0)
            sz = self.calculate_lots_with_price(inst_id, usdt_amount, entry_price, leverage=normalized_leverage)
            if sz <= 0:
                return OrderResult(False, message="可用余额不足以完成最小下单单位")

            # 3. 适配 OKX 持仓模式 —— net_mode(单向) 要求 posSide="net"
            #    long_short_mode(双向) 才使用 "long"/"short"
            account_cfg = self._get_account_config()
            acct_pos_mode = str(account_cfg.get('posMode') or '').lower()
            if acct_pos_mode == 'net_mode':
                effective_pos_side = 'net'
            else:
                effective_pos_side = pos_side  # "long" or "short"

            # 4. 计算止盈止损价格（仍使用原始 pos_side 判断方向）
            if pos_side == "long":
                tp_price = entry_price * (1 + tp_pct)
                sl_price = entry_price * (1 - sl_pct)
                tp_limit_price = tp_price * (1 - 0.001)
            else:
                tp_price = entry_price * (1 - tp_pct)
                sl_price = entry_price * (1 + sl_pct)
                tp_limit_price = tp_price * (1 + 0.001)

            # 5. 构建 OKX V5 订单（含防重客户端订单ID）
            cl_ord_id = f"{inst_id.replace('-','')}_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
            data = {
                "instId": inst_id,
                "tdMode": "cross" if "-SWAP" in inst_id else "cash",
                "side": side,
                "posSide": effective_pos_side,
                "ordType": okx_ord_type,
                "sz": str(sz),
                "clOrdId": cl_ord_id,
                "tpTriggerPx": f"{tp_price:.8f}".rstrip('0').rstrip('.'),
                "tpOrdPx": "-1" if tgt_type == "market" else f"{tp_limit_price:.8f}".rstrip('0').rstrip('.'),
                "slTriggerPx": f"{sl_price:.8f}".rstrip('0').rstrip('.'),
                "slOrdPx": "-1"
            }
            if order_type == "limit":
                data["px"] = f"{entry_price:.8f}".rstrip('0').rstrip('.')

            _log.info(
                f"下单 {inst_id} side={side} posSide={effective_pos_side} sz={sz} "
                f"margin={float(usdt_amount):.4f} notional={notional:.4f} leverage={normalized_leverage}x "
                f"acctMode={acct_pos_mode or 'unknown'} ordType={okx_ord_type}"
            )
            res = self.okx_client._request("POST", "/api/v5/trade/order", data=data)

            if res.get('code') == '0':
                order_id = res['data'][0]['ordId']
                order_kind_text = "限价" if order_type == "limit" else "市价"
                # 轮询获取实际成交均价（市价单通常瞬间成交）
                actual_fill_price = entry_price
                if order_type == "market":
                    polled_price = self._poll_order_fill_price(inst_id, order_id)
                    if polled_price > 0:
                        actual_fill_price = polled_price
                return OrderResult(
                    True,
                    order_id=order_id,
                    message=f"已成功提交{order_kind_text}{effective_pos_side}仓位，并挂载服务器端止盈止损。",
                    filled_size=sz,
                    filled_price=actual_fill_price
                )
            else:
                # 使用详细错误解析（含 sMsg/sCode），避免只显示顶层 "All operations failed"
                detail = self._extract_order_error_message(res)
                _log.error(f" {inst_id} code={res.get('code')} msg={res.get('msg')} detail={detail} "
                      f"data={res.get('data')}")
                return OrderResult(False, message=f"下单失败: {detail}")

        except Exception as e:
            return OrderResult(False, message=f"执行异常: {str(e)}")

    def _poll_order_fill_price(self, inst_id: str, order_id: str,
                                  max_attempts: int = 3, interval: float = 0.5) -> float:
        """
        轮询订单状态，获取实际成交均价。

        OKX 市价单通常瞬间成交，但高波动时可能短暂延迟。
        最多轮询 max_attempts 次，每次间隔 interval 秒。
        返回实际成交均价；若无法获取则返回 0.0（调用方应回退到预估价）。
        """
        for i in range(max_attempts):
            try:
                res = self.okx_client.get_order(inst_id, order_id)
                if res.get('code') == '0' and res.get('data'):
                    order_data = res['data'][0]
                    state = str(order_data.get('state') or '')
                    avg_px = self._safe_float(order_data.get('avgPx'))
                    if state == 'filled' and avg_px > 0:
                        return avg_px
                    if state in ('canceled', 'cancelled'):
                        return 0.0
            except Exception:
                pass
            if i < max_attempts - 1:
                time.sleep(interval)
        return 0.0

    def get_active_positions(self) -> List[PositionInfo]:
        """获取当前活跃持仓"""
        res = self.okx_client.get_positions()
        positions = []
        if res.get('code') == '0' and res.get('data'):
            for p in res['data']:
                raw_pos = self._safe_float(p.get('pos'))
                if raw_pos == 0:
                    continue
                pos_side = str(p.get('posSide') or '').lower()
                if pos_side == 'long':
                    side = PositionSide.LONG
                elif pos_side == 'short':
                    side = PositionSide.SHORT
                else:
                    # OKX 单向持仓模式下常返回 net，需要结合 pos 正负号判断方向。
                    side = PositionSide.SHORT if raw_pos < 0 else PositionSide.LONG
                positions.append(PositionInfo(
                    inst_id=p['instId'],
                    side=side,
                    size=abs(raw_pos),
                    entry_price=self._safe_float(p.get('avgPx')),
                    current_price=self._safe_float(p.get('last') or p.get('markPx')),
                    unrealized_pnl=self._safe_float(p.get('upl')),
                    pnl_percent=self._safe_float(p.get('uplRatio')) * 100,
                    leverage=self._safe_float(p.get('lever'), 1.0),
                    raw_pos=raw_pos,
                    raw_pos_side=pos_side,
                    mgn_mode=str(p.get('mgnMode') or ''),
                    c_time=str(p.get('cTime') or ''),
                    notional_usd=self._safe_float(p.get('notionalUsd'), 0.0),
                ))
        return positions

    def get_usdt_balance(self) -> float:
        """
        统一获取账户 USDT 权益 (针对 OKX V5 模拟盘/实盘深度优化)
        """
        try:
            # 1. 调用余额接口
            res = self.okx_client.get_balance()
            if res.get('code') == '0' and res.get('data'):
                account_data = res['data'][0]

                # 优先获取美金总权益 (totalEq)，这是 OKX 最权威的资产统计
                total_eq = self._safe_float(account_data.get('totalEq'))
                if total_eq > 0:
                    self._last_balance_diag_key = None
                    _log.info(f" 使用 totalEq: {total_eq:.8f}")
                    return total_eq

                iso_eq = self._safe_float(account_data.get('isoEq'))
                if iso_eq > 0:
                    self._last_balance_diag_key = None
                    _log.info(f" 使用 isoEq: {iso_eq:.8f}")
                    return iso_eq

                adj_eq = self._safe_float(account_data.get('adjEq'))
                if adj_eq > 0:
                    self._last_balance_diag_key = None
                    _log.info(f" 使用 adjEq: {adj_eq:.8f}")
                    return adj_eq

                # 如果总权益为0，遍历具体币种细节
                details = account_data.get('details', [])
                if details:
                    _log.info(f" details 数量: {len(details)}")
                detail_snapshots = []
                for d in details:
                    ccy = d.get('ccy', '')
                    snapshot = {
                        'ccy': ccy,
                        'eq': d.get('eq'),
                        'availEq': d.get('availEq'),
                        'availBal': d.get('availBal'),
                        'cashBal': d.get('cashBal'),
                        'eqUsd': d.get('eqUsd'),
                        'frozenBal': d.get('frozenBal'),
                    }
                    detail_snapshots.append(snapshot)

                    if ccy == 'USDT':
                        # 兼容更多可能字段，优先级从更可用的余额到权益
                        candidates = [
                            ('availEq', self._safe_float(d.get('availEq'))),
                            ('availBal', self._safe_float(d.get('availBal'))),
                            ('cashBal', self._safe_float(d.get('cashBal'))),
                            ('eq', self._safe_float(d.get('eq'))),
                            ('eqUsd', self._safe_float(d.get('eqUsd'))),
                        ]
                        for field_name, value in candidates:
                            if value > 0:
                                self._last_balance_diag_key = None
                                _log.info(f" 使用 USDT.{field_name}: {value:.8f}")
                                return value
                        self._log_balance_diag_once(
                            f"usdt_zero:{snapshot}",
                            f"[余额解析] 找到 USDT 明细，但金额均为 0: {snapshot}"
                        )

                if detail_snapshots:
                    preview = detail_snapshots[:5]
                    self._log_balance_diag_once(
                        f"details_preview:{preview}",
                        f"[余额解析] 明细预览(前5条): {preview}"
                    )

                self._log_balance_diag_once(
                    f"account_summary:{account_data.get('totalEq')}:{account_data.get('isoEq')}:{account_data.get('adjEq')}:{len(details)}",
                    "[余额解析] 账户汇总字段: "
                    f"totalEq={account_data.get('totalEq')}, "
                    f"isoEq={account_data.get('isoEq')}, "
                    f"adjEq={account_data.get('adjEq')}, "
                    f"details_count={len(details)}"
                )
            else:
                self._log_balance_diag_once(
                    f"balance_api_error:{res.get('code')}:{res.get('msg')}",
                    "[余额解析] 余额接口返回异常: "
                    f"code={res.get('code')}, msg={res.get('msg')}, "
                    f"has_data={bool(res.get('data'))}"
                )

            # 如果接口返回正常但找不到 USDT，可能是由于没有划转到交易账户
            self._log_balance_diag_once("no_active_usdt", "账户余额解析：未找到活跃的 USDT 资产配置")
            return 0.0
        except Exception as e:
            self._log_balance_diag_once(f"balance_exception:{e}", f"获取余额时发生异常: {e}")
            return 0.0

    def get_positions(self, inst_id: str = None) -> Dict[str, PositionInfo]:
        """获取持仓字典 (UI 兼容)"""
        pos_list = self.get_active_positions()
        pos_dict = {p.inst_id: p for p in pos_list}
        if inst_id:
            return {inst_id: pos_dict[inst_id]} if inst_id in pos_dict else {}
        return pos_dict

    def estimate_position_notional(self, position: PositionInfo) -> float:
        """估算持仓占用名义金额"""
        info = self._get_inst_info(position.inst_id)
        ct_val = float(info.get('ctVal', 1)) if info else 1.0
        if "-SWAP" in position.inst_id:
            return abs(position.size) * position.current_price * ct_val
        return abs(position.size) * position.current_price

    def get_total_position_notional(self) -> float:
        """估算当前总持仓名义金额"""
        return sum(self.estimate_position_notional(pos) for pos in self.get_active_positions())

    def execute_entry(self, inst_id: str, direction: str, usdt_amount: float,
                      leverage: int = 10, tp_pct: float = 0.05, sl_pct: float = 0.03,
                      order_type: str = "market", price: Optional[float] = None) -> OrderResult:
        """按方向统一开仓"""
        normalized = direction.upper()
        if normalized in {"BUY", "LONG"}:
            return self.place_smart_order(
                inst_id, "buy", "long", usdt_amount,
                tp_pct=tp_pct, sl_pct=sl_pct, leverage=leverage,
                order_type=order_type, price=price
            )
        if normalized in {"SELL", "SHORT"}:
            return self.place_smart_order(
                inst_id, "sell", "short", usdt_amount,
                tp_pct=tp_pct, sl_pct=sl_pct, leverage=leverage,
                order_type=order_type, price=price
            )
        return OrderResult(False, message=f"不支持的开仓方向: {direction}")

    def reverse_position(self, inst_id: str, usdt_amount: float,
                         leverage: int = 10, tp_pct: float = 0.05, sl_pct: float = 0.03,
                         order_type: str = "market", price: Optional[float] = None) -> OrderResult:
        """一键反手：先平现有持仓，再按反方向开仓"""
        positions = self.get_positions(inst_id)
        if inst_id not in positions:
            return OrderResult(False, message="当前无持仓，无法反手")

        current_pos = positions[inst_id]
        close_result = self.close_position(inst_id)
        if not close_result.success:
            return OrderResult(False, message=f"反手前平仓失败: {close_result.message}")

        direction = "SHORT" if current_pos.side == PositionSide.LONG else "LONG"
        open_result = self.execute_entry(
            inst_id,
            direction,
            usdt_amount,
            leverage=leverage,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            order_type=order_type,
            price=price
        )
        if open_result.success:
            open_result.message = f"反手成功：已平旧仓并开{direction}"
        return open_result

    def close_position(self, inst_id: str) -> OrderResult:
        """统一平仓接口"""
        return self.execute_stop_loss(inst_id)

    def close_position_partial(self, inst_id: str, ratio: float) -> OrderResult:
        """按比例部分平仓"""
        pos_dict = self.get_positions(inst_id)
        if inst_id not in pos_dict:
            return OrderResult(False, message="无活跃持仓")

        pos = pos_dict[inst_id]
        if ratio <= 0 or ratio > 1:
            return OrderResult(False, message="部分平仓比例必须在 0 到 1 之间")

        close_size = pos.size * ratio
        if "-SWAP" in inst_id:
            info = self._get_inst_info(inst_id)
            lot_sz = float(info.get('lotSz', 1) or 1)
            close_size = math.floor(close_size / lot_sz) * lot_sz if lot_sz > 0 else close_size
        if close_size <= 0:
            return OrderResult(False, message="部分平仓数量不足最小下单单位")

        data = {
            "instId": inst_id,
            "ordType": "market",
            "sz": str(close_size)
        }
        data.update(self._build_close_order_context(inst_id, pos))
        res = self._place_reduce_order_with_net_retry(data, inst_id)
        return OrderResult(
            res.get('code') == '0',
            order_id=(res.get('data') or [{}])[0].get('ordId', ''),
            message=self._extract_order_error_message(res),
            filled_size=close_size if res.get('code') == '0' else 0.0
        )

    def cancel_order(self, inst_id: str, order_id: str) -> OrderResult:
        """撤销指定挂单"""
        res = self.okx_client.cancel_order(inst_id, order_id)
        return OrderResult(
            res.get('code') == '0',
            order_id=order_id,
            message=res.get('msg', '')
        )

    def _cancel_pending_orders(self, inst_id: str) -> None:
        """撤销指定品种的所有挂单（含止盈止损触发单），避免平仓后被孤儿触发单反手开仓。"""
        try:
            pending = self.get_pending_orders(inst_id)
            for order in pending:
                oid = str(order.get('ordId', ''))
                ode_type = str(order.get('ordType', '')).lower()
                if oid and ode_type in ('conditional', 'oco', 'trigger', 'limit', 'market'):
                    self.okx_client.cancel_order(inst_id, oid)
        except Exception:
            pass  # 取消失败不影响主流程

    def execute_buy(self, inst_id: str, position_ratio: float = 0.1) -> OrderResult:
        """快速买入/做多 (UI & Runner 兼容)"""
        balance = self.get_usdt_balance()
        return self.place_smart_order(inst_id, "buy", "long", balance * position_ratio)

    def execute_sell(self, inst_id: str, quantity: float) -> OrderResult:
        """平仓做多/卖出 (UI & Runner 兼容)"""
        return self._execute_close_position(inst_id, quantity)

    def execute_short(self, inst_id: str, position_ratio: float = 0.1) -> OrderResult:
        """快速开空 (Runner 兼容)"""
        balance = self.get_usdt_balance()
        return self.place_smart_order(inst_id, "sell", "short", balance * position_ratio)

    def _execute_close_position(self, inst_id: str, close_size: float = 0) -> OrderResult:
        """通用平仓（平多或平空，_build_close_order_context 自动判断方向）"""
        pos_dict = self.get_positions(inst_id)
        if inst_id not in pos_dict:
            return OrderResult(False, message="无活跃持仓")
        pos = pos_dict[inst_id]
        sz = pos.size if close_size <= 0 else min(close_size, pos.size)
        data = {
            "instId": inst_id,
            "ordType": "market",
            "sz": str(sz)
        }
        data.update(self._build_close_order_context(inst_id, pos))
        res = self._place_reduce_order_with_net_retry(data, inst_id)
        order_id = (res.get('data') or [{}])[0].get('ordId', '')
        filled_price = 0.0
        if res.get('code') == '0' and order_id:
            filled_price = self._poll_order_fill_price(inst_id, order_id)
        return OrderResult(
            res.get('code') == '0',
            order_id=order_id,
            message=self._extract_order_error_message(res),
            filled_size=sz,
            filled_price=filled_price,
        )

    def execute_cover(self, inst_id: str, quantity: float) -> OrderResult:
        """平空 (Runner 兼容)"""
        return self._execute_close_position(inst_id, quantity)

    def execute_stop_loss(self, inst_id: str) -> OrderResult:
        """手动平仓 (UI 兼容) —— 先取消挂单再市价平仓"""
        self._cancel_pending_orders(inst_id)
        return self._execute_close_position(inst_id)
