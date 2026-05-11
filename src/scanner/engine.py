"""
扫描引擎
负责执行扫描策略，获取 OKX 合约交易对数据，并筛选符合条件的交易对
"""

import time
import random
import threading
import os
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.scanner.ranking import enrich_scan_result, sort_scan_results
from src.scanner.strategy_lifecycle import apply_strategy_lifecycle_guard
from src.scanner.market_state_classifier import (
    classify_market_state, STATE_TRENDING, STATE_RANGE, STATE_VOLATILE, STATE_NEUTRAL, VALID_STATES,
)
from src.scanner.cross_sectional_ranker import cross_sectional_rank, generate_long_short_pairs
from src.scanner.signal_lifecycle_tracker import get_tracker, record_signal
from src.scanner.resonance_dashboard import compute_resonance, build_resonance_summary
from src.scanner.volatility_pools import (
    classify_all_symbols, get_pool_params, get_position_pct_hint,
    POOL_LOW, POOL_MEDIUM, POOL_HIGH, POOL_LABELS,
)
from src.data.on_chain_provider import OnChainDataProvider

class ScanEngine:
    """扫描引擎"""

    def __init__(self, okx_client):
        """
        初始化扫描引擎

        Args:
            okx_client: OKX 客户端实例
        """
        self.okx_client = okx_client
        self.strategy = None
        self.is_running = False
        self.last_results = []
        self.last_scan_time = None
        self.total_scanned = 0
        self.total_passed = 0
        self.scan_interval = 60  # 默认 60 秒
        self._scan_thread = None
        self._stop_requested = threading.Event()
        
        # 回调函数
        self.on_scan_start = None
        self.on_scan_progress = None
        self.on_scan_complete = None
        self.on_scan_error = None

    def request_stop(self):
        """请求尽快停止当前扫描。"""
        self.is_running = False
        self._stop_requested.set()

    def _attach_learning_metadata(self, strategy: Any, res: Dict[str, Any], symbol: Any = None) -> Dict[str, Any]:
        if not isinstance(res, dict):
            return res

        strategy_name = getattr(strategy, "_rl_strategy_name", "") or getattr(strategy, "name", type(strategy).__name__)
        res.setdefault("strategy_name", strategy_name)
        res.setdefault("param_snapshot", dict(getattr(strategy, "_rl_param_snapshot", {}) or getattr(strategy, "config", {}) or {}))
        res.setdefault("strategy_code_hash", str(getattr(strategy, "_rl_code_hash", "") or ""))
        strategy_path = str(getattr(strategy, "_rl_strategy_path", "") or "")
        strategy_file = str(getattr(strategy, "_rl_strategy_file", "") or (os.path.basename(strategy_path) if strategy_path else ""))
        res.setdefault("strategy_source_file", strategy_file)
        res.setdefault("strategy_source_path", strategy_path)
        if symbol is not None:
            klines_map = dict(getattr(symbol, "extra_data", {}).get("klines", {}) or {})
            if klines_map:
                res.setdefault("klines_map", klines_map)
        return res

    def get_all_tickers(self) -> List[Dict]:
        """获取所有交易对行情"""
        try:
            print("[扫描引擎] 开始获取 OKX 行情数据...")
            result = self.okx_client.get_tickers(instType="SWAP")
            print(f"[扫描引擎] OKX 返回结果: code={result.get('code') if result else 'None'}")
            
            if result and result.get('code') == '0':
                tickers = result.get('data', [])
                print(f"[扫描引擎] 获取到 {len(tickers)} 个交易对")
                
                # 过滤出 USDT 永续合约
                usdt_swaps = [t for t in tickers if t.get('instId', '').endswith('-USDT-SWAP')]
                print(f"[扫描引擎] 过滤后剩余 {len(usdt_swaps)} 个 USDT 永续合约")
                return usdt_swaps
            else:
                print(f"[扫描引擎] 获取行情失败: {result}")
                return []
        except Exception as e:
            print(f"[扫描引擎] 获取行情异常: {e}")
            import traceback
            traceback.print_exc()
            return []

    def parse_ticker(self, ticker: Dict) -> Any:
        """解析 Ticker 数据为 ScannerSymbol"""
        from src.scanner.base_scanner import ScannerSymbol
        
        # 计算 24h 涨跌幅
        last_price = float(ticker.get('last', 0))
        open_24h = float(ticker.get('open24h', 0))
        price_change_24h = 0.0
        if open_24h > 0:
            price_change_24h = ((last_price - open_24h) / open_24h) * 100
        
        # 修复：优先使用 volCcyQuote，如果为 None 或 0 则使用 vol24h
        vol_ccy = ticker.get('volCcyQuote')
        vol_24h_raw = ticker.get('vol24h', 0)
        volume_24h = float(vol_ccy) if vol_ccy else float(vol_24h_raw or 0)
        
        return ScannerSymbol(
            inst_id=ticker.get('instId'),
            last_price=last_price,
            volume_24h=volume_24h,
            price_change_24h=price_change_24h,
            high_24h=float(ticker.get('high24h', 0)),
            low_24h=float(ticker.get('low24h', 0)),
            open_interest=0.0,
            extra_data={'vol24h': vol_24h_raw, 'open24h': ticker.get('open24h', 0)}
        )

    def get_klines_batch(self, inst_ids: List[str], bars: List[str], limit: int = 50, max_workers: int = 6, progress_callback=None, bar_limits: Dict[str, int] = None) -> Dict[str, Dict[str, List]]:
        """批量获取 K 线数据 - 并发优化版，支持重试机制。
        bar_limits: 每个周期单独的 limit，如 {"1D": 300, "1H": 200}，未指定周期回退到 limit。
        """
        results = {inst_id: {} for inst_id in inst_ids}
        _bar_limits = bar_limits or {}

        print(f"[get_klines_batch] 开始获取 K 线: {len(inst_ids)} 个交易对 × {len(bars)} 个周期 (并发数: {max_workers})")

        # 单个交易对的 K 线获取函数（带重试，限流错误使用指数退避）
        def fetch_klines_for_symbol(inst_id: str, max_retries: int = 2) -> Dict:
            symbol_klines = {}
            for bar in bars:
                if self._stop_requested.is_set():
                    break
                bar_limit = _bar_limits.get(bar, limit)
                success = False
                for attempt in range(max_retries + 1):
                    if self._stop_requested.is_set():
                        break
                    try:
                        res = self.okx_client.get_kline(inst_id, bar=bar, limit=bar_limit)
                        if res and res.get('code') == '0':
                            data = res.get('data', [])
                            if data:
                                symbol_klines[bar] = data
                                success = True
                                break
                            elif attempt < max_retries:
                                time.sleep(0.2)
                                continue
                        elif attempt < max_retries:
                            code = res.get('code', '-1') if res else '-1'
                            if code == '-2':
                                # 限流超时 → 指数退避 + 抖动，让令牌桶恢复
                                backoff = (2 ** attempt) * 0.5 + random.random() * 0.3
                                time.sleep(backoff)
                            else:
                                time.sleep(0.3)
                            continue
                    except Exception as e:
                        if attempt < max_retries:
                            time.sleep(0.3)
                            continue
                if not success and len(inst_ids) <= 5:
                    print(f"[get_klines_batch] ⚠️ {inst_id} {bar} 获取失败（已重试 {max_retries} 次）")
            return {inst_id: symbol_klines}

        # 并发获取 K 线
        success_count = 0
        fail_count = 0
        total_requests = len(inst_ids)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_instid = {executor.submit(fetch_klines_for_symbol, inst_id): inst_id for inst_id in inst_ids}

            for idx, future in enumerate(as_completed(future_to_instid)):
                if self._stop_requested.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    return results
                inst_id = future_to_instid[future]
                try:
                    result = future.result(timeout=30)
                    inst_klines = result[inst_id]

                    if inst_klines:
                        results[inst_id] = inst_klines
                        success_count += 1
                        kline_info = ', '.join([f'{k}:{len(v)}' for k, v in inst_klines.items()])

                        # 节流：每 10 个或最后一个才回调，避免大量信号积压
                        if progress_callback and (idx % 10 == 0 or idx + 1 == total_requests):
                            should_stop = progress_callback(
                                20 + int(10 * (idx + 1) / total_requests),
                                f"📥 [{success_count}/{total_requests}] {inst_id}",
                                inst_id, idx + 1, total_requests, ""
                            )
                            if should_stop:
                                print("[get_klines_batch] 收到停止信号，退出获取 K 线")
                                return results
                    else:
                        fail_count += 1

                except Exception as e:
                    fail_count += 1
                    if len(inst_ids) <= 5:
                        print(f"[get_klines_batch] ❌ {inst_id} 异常: {e}")

        print(f"[get_klines_batch] 完成: 成功 {success_count}/{total_requests}，失败 {fail_count}")
        if progress_callback:
            progress_callback(30, f"✅ K 线获取完成 ({success_count}/{total_requests})", "", success_count, total_requests, "")

        return results

    def enrich_derivative_metrics(self, symbols: List[Any], max_workers: int = 8, max_symbols: int = 120, progress_callback=None):
        """为需要衍生品指标的策略补充资金费率和当前持仓量。"""
        selected = symbols[:max_symbols]
        if not selected:
            return
        symbol_map = {s.inst_id: s for s in selected}

        def fetch_metrics(inst_id: str) -> Dict[str, Any]:
            metrics = {}
            try:
                funding = self.okx_client.get_funding_rate(inst_id)
                if funding and funding.get('code') == '0' and funding.get('data'):
                    item = funding['data'][0]
                    metrics['funding_rate'] = float(item.get('fundingRate') or 0.0)
                    metrics['next_funding_time'] = item.get('nextFundingTime', '')
            except Exception:
                pass
            try:
                oi = self.okx_client.get_open_interest(inst_id)
                if oi and oi.get('code') == '0' and oi.get('data'):
                    item = oi['data'][0]
                    metrics['open_interest'] = float(item.get('oi') or 0.0)
                    metrics['open_interest_ccy'] = float(item.get('oiCcy') or 0.0)
            except Exception:
                pass
            return {'inst_id': inst_id, 'metrics': metrics}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_metrics, symbol.inst_id): symbol.inst_id for symbol in selected}
            for idx, future in enumerate(as_completed(futures), 1):
                try:
                    payload = future.result()
                    symbol = symbol_map.get(payload.get('inst_id'))
                    if symbol and payload.get('metrics'):
                        symbol.extra_data.update(payload['metrics'])
                        if 'open_interest' in payload['metrics']:
                            symbol.open_interest = payload['metrics']['open_interest']
                except Exception:
                    continue
                if progress_callback and idx % 20 == 0:
                    progress_callback(18, f"补充衍生品指标 {idx}/{len(selected)}", "", idx, len(selected), "")

    def enrich_on_chain_metrics(self, symbols: List[Any], progress_callback=None):
        """为策略补充链上指标，写入 symbol.extra_data['on_chain']。"""
        if not symbols:
            return
        provider = OnChainDataProvider()
        if not provider.is_configured():
            print("[扫描引擎] 未配置链上数据源，链上因子保持中性")
            return

        inst_ids = [symbol.inst_id for symbol in symbols if getattr(symbol, 'inst_id', '')]
        try:
            metrics_map = provider.fetch_many(inst_ids)
        except Exception as exc:
            print(f"[扫描引擎] 链上数据源异常，链上因子保持中性: {exc}")
            return

        if not metrics_map:
            print("[扫描引擎] 链上数据源未返回有效指标")
            return

        injected = 0
        for symbol in symbols:
            metrics = metrics_map.get(getattr(symbol, 'inst_id', ''))
            if not metrics:
                continue
            symbol.extra_data['on_chain'] = metrics
            injected += 1
        print(f"[扫描引擎] 已注入链上指标 {injected}/{len(symbols)} 个品种")
        if progress_callback:
            progress_callback(19, f"链上指标补充完成 {injected}/{len(symbols)}", "", injected, len(symbols), "")

    def apply_lifecycle_guard(self, strategy, result: Dict) -> Dict:
        """统一应用扫描策略发挥前提/失效条件守门。"""
        if not isinstance(result, dict):
            return result
        category = result.get('category')
        details = result.get('details')
        if not category and isinstance(details, dict):
            category = details.get('机会类型')
        if not category:
            return result
        config = getattr(strategy, 'config', {}) or {}
        return apply_strategy_lifecycle_guard(result, str(category), config)

    def _record_signals(self, results: List[Dict]) -> None:
        """v4.6: 将扫描结果记录到信号生命周期追踪器"""
        for r in results:
            if not r.get("passed"):
                continue
            symbol = str(r.get("symbol", ""))
            direction = str(r.get("direction", "WAIT")).upper()
            if direction not in ("BUY", "SELL", "LONG", "SHORT"):
                continue
            price = float(r.get("last_price", r.get("entry_price", 0)) or 0)
            score = float(r.get("score", r.get("composite_score", 50)) or 50)
            if symbol and price > 0:
                record_signal(symbol, direction, price, score)

    def run_scan(self, strategy, progress_callback=None, result_callback=None) -> List[Dict]:
        """
        全市场实时扫描
        """
        def emit_progress(val, msg, symbol="", count=0, total=0, remaining=""):
            if progress_callback:
                # print(f"[emit_progress] 调用进度回调: {val}% - {msg}")
                return progress_callback(val, msg, symbol, count, total, remaining)
            return False

        def emit_result(res):
            # ── HIGH-2 FIX：市场状态自适应分数门槛过滤 ──────────────────────
            _ms_min = getattr(strategy, 'config', {}).get('_market_state_min_score')
            if _ms_min is not None:
                _res_score = float(res.get('score', res.get('composite_score', 0)) or 0)
                if _res_score < _ms_min:
                    return
            if result_callback:
                result_callback(res)

        try:
            self.strategy = strategy
            self.is_running = True
            self._stop_requested.clear()
            scan_start_time = time.time()

            # 1. 获取行情 (提前获取以供生命周期和市场状态检测使用)
            if emit_progress(5, "🌐 正在获取 OKX 实时行情...", "", 0, 0, "计算中..."):
                return []
            all_tickers = self.get_all_tickers()
            if not all_tickers:
                raise RuntimeError("未获取到行情数据，请检查网络或 API 设置")
            if emit_progress(8, f"✅ 获取到 {len(all_tickers)} 个交易对", "", len(all_tickers), len(all_tickers), ""):
                return []

            # ── v4.6: 信号生命周期 — 结算上次扫描的待定信号 ──────────────
            tracker = get_tracker()
            # 用当前 ticker 价格结算
            price_map = {}
            for t in all_tickers:
                try:
                    price_map[t["instId"]] = float(t.get("last", 0) or 0)
                except Exception:
                    pass
            resolved = tracker.resolve_all(price_map)
            if resolved:
                print(f"[扫描引擎] 生命周期结算: {sum(resolved.values())} 个信号到期")

            # ── v4.4: 市场状态自适应检测 ──────────────────────────────────
            market_state = STATE_NEUTRAL
            state_conf = 0.0
            state_diag: Dict = {}
            try:
                btc_4h = self.okx_client.get_kline("BTC-USDT-SWAP", bar="4H", limit=120)
                btc_1h = self.okx_client.get_kline("BTC-USDT-SWAP", bar="1H", limit=200)
                btc_1d = self.okx_client.get_kline("BTC-USDT-SWAP", bar="1D", limit=200)
                btc_4h_data = btc_4h.get("data", []) if btc_4h and btc_4h.get("code") == "0" else []
                btc_1h_data = btc_1h.get("data", []) if btc_1h and btc_1h.get("code") == "0" else []
                btc_1d_data = btc_1d.get("data", []) if btc_1d and btc_1d.get("code") == "0" else []
                # 从 tickers 中获取 BTC 24H 涨跌
                btc_ticker = next((t for t in all_tickers if "BTC" in str(t.get("instId","")).upper()), None)
                if btc_ticker:
                    btc_24h = float(btc_ticker.get("last", 0) or 0) / max(float(btc_ticker.get("open24h", 1) or 1), 1e-9) - 1.0
                    btc_24h_pct = btc_24h * 100
                else:
                    btc_24h_pct = 0.0
                if btc_4h_data and btc_1h_data:
                    market_state, state_conf, state_diag = classify_market_state(
                        btc_4h_data, btc_1h_data, btc_1d_data, btc_24h_pct
                    )
                    print(f"[扫描引擎] 市场状态: {market_state} (置信{state_conf:.0%}) "
                          f"| ADX={state_diag.get('metrics',{}).get('adx','?')} "
                          f"| BTC 24H={btc_24h_pct:+.1f}%")
            except Exception as e:
                print(f"[扫描引擎] 市场状态检测异常(使用neutral): {e}")

            # ── HIGH-2 FIX：市场状态注入 + 自适应分数门槛 ──────────────────
            # 1. 无论策略是否实现 _apply_market_state_params，
            #    都将 market_state 写入 strategy.config，供策略内部 get() 读取。
            # 2. 在震荡市（STATE_RANGE）且置信度 ≥ 60% 时，
            #    自动提高引擎层最低分数阈值（+10分），抑制假突破信号。
            if hasattr(strategy, 'config') and isinstance(getattr(strategy, 'config', None), dict):
                strategy.config['market_state']      = market_state
                strategy.config['market_state_conf'] = state_conf

            # 市场状态自适应最低分数（注入为 _market_state_min_score，优先级高于策略自身 min_score）
            _base_min_score = float(getattr(strategy, 'config', {}).get('min_opportunity_score', 60) or 60)
            if market_state == STATE_RANGE and state_conf >= 0.60:
                _effective_min_score = min(_base_min_score + 10.0, 85.0)
                if hasattr(strategy, 'config') and isinstance(getattr(strategy, 'config', None), dict):
                    strategy.config['_market_state_min_score'] = _effective_min_score
                print(f"[扫描引擎] 震荡市自适应：最低分数门槛从 {_base_min_score:.0f} 提升至"
                      f" {_effective_min_score:.0f}（减少假突破信号）")
            elif market_state == STATE_VOLATILE and state_conf >= 0.60:
                _effective_min_score = min(_base_min_score + 5.0, 80.0)
                if hasattr(strategy, 'config') and isinstance(getattr(strategy, 'config', None), dict):
                    strategy.config['_market_state_min_score'] = _effective_min_score
                print(f"[扫描引擎] 高波动市自适应：最低分数门槛从 {_base_min_score:.0f} 提升至"
                      f" {_effective_min_score:.0f}（提高信号质量要求）")
            else:
                if hasattr(strategy, 'config') and isinstance(getattr(strategy, 'config', None), dict):
                    strategy.config.pop('_market_state_min_score', None)

            if hasattr(strategy, "_apply_market_state_params"):
                try:
                    strategy._apply_market_state_params(market_state)
                except Exception:
                    pass

            # 2. 预过滤 (成交量 > 100k)
            filtered_tickers = []
            min_vol = float(getattr(self.strategy, 'config', {}).get('min_volume_24h', 100000) or 100000)
            for t in all_tickers:
                vol = t.get('volCcyQuote') or t.get('vol24h') or 0
                vol = float(vol) if vol else 0
                if vol >= min_vol:
                    filtered_tickers.append(t)

            # ticker 新鲜度检查：超过 30% 数据超过 60s 未更新则打印一次警告
            now_ms = int(time.time() * 1000)
            stale_count = sum(
                1 for t in filtered_tickers
                if int(t.get('ts', 0) or 0) > 0 and now_ms - int(t.get('ts', 0)) > 60_000
            )
            if stale_count > len(filtered_tickers) * 0.3:
                print(f"[扫描引擎警告] {stale_count}/{len(filtered_tickers)} 个 ticker 数据超过 60s 未更新")
            if emit_progress(10, f"✅ 已筛选出 {len(filtered_tickers)} 个活跃品种", "", 0, len(filtered_tickers), "计算中..."):
                return []

            # 3. 解析
            symbols = []
            for t in filtered_tickers:
                try:
                    s = self.parse_ticker(t)
                    # 修正：直接从原始 ticker 't' 中获取 24h 开盘价
                    open_24h = float(t.get('open24h', 0))
                    if open_24h > 0:
                        s.price_change_24h = ((s.last_price - open_24h) / open_24h) * 100
                    symbols.append(s)
                except Exception as e:
                    print(f"解析 Ticker 失败: {e}")
                    continue

            if bool(getattr(strategy, 'requires_derivative_metrics', False)):
                if emit_progress(15, "📊 正在补充资金费率/持仓量指标...", "", 0, len(symbols), ""):
                    return []
                self.enrich_derivative_metrics(symbols, progress_callback=emit_progress)

            if bool(getattr(strategy, 'requires_on_chain_metrics', False)):
                if emit_progress(19, "🔗 正在补充链上指标...", "", 0, len(symbols), ""):
                    return []
                self.enrich_on_chain_metrics(symbols, progress_callback=emit_progress)

            # 4. 获取 K 线 (智能检测策略需要的 K 线周期)
            # 涨跌幅策略不需要 K 线，其他策略需要
            strategy_name = type(strategy).__name__.lower()
            
            if 'gainer' in strategy_name or 'loser' in strategy_name or 'ranking' in strategy_name:
                # 涨跌幅策略不需要 K 线
                needs_klines = False
                bars = []
            elif hasattr(strategy, 'required_bars'):
                # 策略声明了需要的 K 线周期
                needs_klines = True
                bars = strategy.required_bars
            elif hasattr(strategy, 'scan_all_symbols') and callable(getattr(strategy, 'scan_all_symbols')):
                # 有 scan_all_symbols 的策略（如单边大趋势）需要 5m/15m K 线
                needs_klines = True
                bars = ['5m', '15m', '1H', '1D']
            else:
                # 默认策略需要 1D/1H/3m
                needs_klines = True
                bars = ['1D', '1H', '3m']
            
            if needs_klines and symbols and bars:
                # 优化：移除 80 个的限制，获取所有交易对的 K 线
                kline_symbols = symbols
                if emit_progress(20, f"📈 获取 {len(kline_symbols)} 个品种的 K 线数据...", "", 0, len(kline_symbols), ""):
                    return []
                inst_ids = [s.inst_id for s in kline_symbols]

                # 若策略声明了每周期所需根数，则分别使用；否则统一 200
                _bar_limits = getattr(strategy, 'required_bars_limits', None) or {}
                klines_data = self.get_klines_batch(
                    inst_ids, bars, limit=200,
                    bar_limits=_bar_limits,
                    progress_callback=emit_progress,
                )

                kline_map = {s.inst_id: s for s in symbols}
                success_count = 0
                for inst_id, data in klines_data.items():
                    if data:
                        success_count += 1
                        if inst_id in kline_map:
                            kline_map[inst_id].extra_data['klines'] = data

                print(f"[run_scan] K 线获取完成：{success_count}/{len(inst_ids)} 个交易对成功")
                if emit_progress(30, f"✅ K 线数据获取完成 ({success_count}/{len(inst_ids)})", "", success_count, len(inst_ids), ""):
                    return []

                # ── v4.7: 波动率分层 — 按ATR分三档 ─────────────────────────
                if needs_klines:
                    klines_4h_map: Dict[str, List] = {}
                    for sym in kline_symbols:
                        inst_id = str(getattr(sym, "inst_id", ""))
                        km = (getattr(sym, "extra_data", {}) or {}).get("klines", {})
                        klines_4h_map[inst_id] = km.get("4H") or km.get("4h") or []
                    pool_map = classify_all_symbols(kline_symbols, klines_4h_map)
                    # 将池信息注入 symbol
                    for sym in kline_symbols:
                        inst_id = str(getattr(sym, "inst_id", ""))
                        pool, pool_atr = pool_map.get(inst_id, (POOL_MEDIUM, 4.0))
                        sym.extra_data["vol_pool"] = pool
                        sym.extra_data["vol_atr_pct"] = pool_atr
                    # 逐品种传递波动率分类到 extra_data（不再用全局 dominant 覆盖策略）
                    low_n = sum(1 for p, _ in pool_map.values() if p == POOL_LOW)
                    mid_n = sum(1 for p, _ in pool_map.values() if p == POOL_MEDIUM)
                    high_n = sum(1 for p, _ in pool_map.values() if p == POOL_HIGH)
                    print(f"[扫描引擎] 波动率分层: 低波动{low_n} 中波动{mid_n} 高波动{high_n}")

            else:
                print(f"[run_scan] 阶段 4: 跳过 K 线获取（策略：{type(strategy).__name__} 不需要 K 线）")
                if emit_progress(30, "✅ 跳过 K 线获取", "", len(symbols), len(symbols), ""):
                    return []

            # 5. 执行并发分析 - 检查策略是否有批量扫描方法
            final_results = []

            # 如果策略有 scan_all_symbols 方法，使用批量扫描
            if hasattr(strategy, 'scan_all_symbols') and callable(getattr(strategy, 'scan_all_symbols')):
                # 包装 scan_all_symbols，注入停止检查 + 进度回调（不打印逐条日志）
                original_scan_all = strategy.scan_all_symbols

                def scan_with_progress(symbols_data):
                    for idx, s in enumerate(symbols_data):
                        if self._stop_requested.is_set():
                            raise InterruptedError("Stop requested during progress")
                        if idx % 10 == 0:
                            should_stop = emit_progress(
                                40 + int(55 * (idx + 1) / len(symbols_data)),
                                f"🔍 [{idx+1}/{len(symbols_data)}] {s.inst_id}",
                                s.inst_id, idx + 1, len(symbols_data), ""
                            )
                            if should_stop:
                                raise InterruptedError("User requested stop")
                    return original_scan_all(symbols_data)
                
                # 替换策略的 scan_all_symbols 方法
                strategy.scan_all_symbols = scan_with_progress
                
                if emit_progress(40, "🔍 开始批量扫描分析...", "", 0, len(symbols), ""):
                    return []
                
                try:
                    batch_result = strategy.scan_all_symbols(symbols)
                except InterruptedError:
                    print("[run_scan] 批量扫描被用户终止")
                    return []
                finally:
                    # 恢复原始方法
                    strategy.scan_all_symbols = original_scan_all

                print(f"[run_scan] 批量扫描完成: type={batch_result.get('type','unknown')}, 信号数={len(batch_result.get('all_opportunities',[]))}")
                
                # 转换批量结果为引擎格式
                if 'top_gainers' in batch_result:
                    for gainer in batch_result.get('top_gainers', []):
                        gainer['passed'] = True
                        gainer['score'] = 100.0
                        gainer['direction'] = 'BUY'
                        gainer['reason'] = f"涨幅榜 #{gainer.get('rank', '?')}"
                        source_symbol = next((s for s in symbols if s.inst_id == gainer.get('symbol')), None)
                        gainer = self._attach_learning_metadata(strategy, gainer, source_symbol)
                        enrich_scan_result(gainer)
                        final_results.append(gainer)
                        emit_result(gainer)

                if 'top_losers' in batch_result:
                    for loser in batch_result.get('top_losers', []):
                        loser['passed'] = True
                        loser['score'] = 100.0
                        loser['direction'] = 'SELL'
                        loser['reason'] = f"跌幅榜 #{loser.get('rank', '?')}"
                        source_symbol = next((s for s in symbols if s.inst_id == loser.get('symbol')), None)
                        loser = self._attach_learning_metadata(strategy, loser, source_symbol)
                        enrich_scan_result(loser)
                        final_results.append(loser)
                        emit_result(loser)

                # 处理 all_opportunities（如单边大趋势策略）
                if 'all_opportunities' in batch_result:
                    for opp in batch_result.get('all_opportunities', []):
                        opp['passed'] = True
                        opp = self.apply_lifecycle_guard(strategy, opp)
                        if not opp.get('passed', False):
                            continue
                        opp['reason'] = ', '.join(opp.get('signals', []))
                        source_symbol = next((s for s in symbols if s.inst_id == opp.get('symbol')), None)
                        opp = self._attach_learning_metadata(strategy, opp, source_symbol)
                        enrich_scan_result(opp)
                        final_results.append(opp)
                        emit_result(opp)

                final_results = sort_scan_results(final_results)
                # v4.5: 截面排序 — 相对排名替代绝对评分
                final_results = cross_sectional_rank(final_results)
                final_results = compute_resonance(final_results)
                self._record_signals(final_results)
                emit_progress(100, f"✅ 批量扫描完成！截面排序 {len(final_results)} 个信号", "", len(symbols), len(symbols), "0s")
                return final_results
            
            # 否则使用逐个分析模式
            # 过滤出有 K 线数据的交易对
            symbols_with_klines = [s for s in symbols if s.extra_data.get('klines')]
            total_symbols = len(symbols_with_klines)
            
            emit_progress(40, f"🔍 启动多线程实时建模分析（{total_symbols} 个交易对）...", "", 0, total_symbols, "极速处理中")

            def analyze_task(idx, s):
                try:
                    # 每 5 个品种汇报一次进度（不再 print 逐条日志）
                    if idx % 5 == 0:
                        progress = 40 + int(55 * (idx / total_symbols)) if total_symbols > 0 else 50
                        should_stop = emit_progress(progress, f"实时分析: {s.inst_id}", s.inst_id, idx+1, total_symbols, "")
                        if should_stop:
                            return "STOP"

                    res = strategy.scan_symbol(s)
                    res = self.apply_lifecycle_guard(strategy, res)
                    if res and res.get('passed', False):
                        res.update({
                            'symbol': s.inst_id,
                            'last_price': s.last_price,
                            'change_24h': s.price_change_24h,
                            'score': res.get('score', 0)
                        })
                        res = self._attach_learning_metadata(strategy, res, s)
                        enrich_scan_result(res)
                        emit_result(res)   # 立即把结果发回 UI
                        return res
                except Exception as e:
                    # 保留异常 print（便于问题排查，但仅一行）
                    print(f"[扫描引擎] 分析 {s.inst_id} 失败: {e}")
                return None

            with ThreadPoolExecutor(max_workers=min(15, len(symbols_with_klines) or 1)) as executor:
                futures = [executor.submit(analyze_task, i, s) for i, s in enumerate(symbols_with_klines)]
                for future in as_completed(futures):
                    if self._stop_requested.is_set():
                        executor.shutdown(wait=False, cancel_futures=True)
                        return []
                    res = future.result()
                    if res == "STOP":
                        executor.shutdown(wait=False, cancel_futures=True)
                        return []
                    if res: final_results.append(res)

            # 6. 完成 + 截面排序
            final_results = sort_scan_results(final_results)
            final_results = cross_sectional_rank(final_results)
            total_elapsed = time.time() - scan_start_time
            emit_progress(100, f"✅ 扫描完成！截面排序 {len(final_results)} 个信号 (耗时 {total_elapsed:.1f}s)", "", total_symbols, total_symbols, "0s")
            return final_results

        except Exception as e:
            emit_progress(100, f"❌ 扫描失败: {str(e)}")
            raise e
        finally:
            self.is_running = False
