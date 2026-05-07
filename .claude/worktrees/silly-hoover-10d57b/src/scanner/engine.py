"""
扫描引擎
负责执行扫描策略，获取 OKX 合约交易对数据，并筛选符合条件的交易对
"""

import time
import threading
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.scanner.base_scanner import ScannerSymbol, BaseScannerStrategy


class ScanCache:
    """扫描数据缓存 - 减少重复API调用"""
    
    def __init__(self):
        self._symbols_cache: List[str] = []
        self._symbols_expire: float = 0
        self._tickers_cache: Dict[str, Dict] = {}
        self._tickers_expire: float = 0
        self._klines_cache: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        
        self.SYMBOLS_TTL = 300
        self.TICKERS_TTL = 60
        self.KLINES_TTL = 60
    
    def get_symbols(self, fetch_func: Callable) -> List[str]:
        with self._lock:
            now = time.time()
            if now < self._symbols_expire and self._symbols_cache:
                return self._symbols_cache
            symbols = fetch_func()
            self._symbols_cache = symbols
            self._symbols_expire = now + self.SYMBOLS_TTL
            return symbols
    
    def get_tickers(self, fetch_func: Callable) -> List[Dict]:
        with self._lock:
            now = time.time()
            if now < self._tickers_expire and self._tickers_cache:
                return list(self._tickers_cache.values())
            tickers = fetch_func()
            self._tickers_cache = {t['instId']: t for t in tickers}
            self._tickers_expire = now + self.TICKERS_TTL
            return tickers
    
    def get_klines(self, key: str, fetch_func: Callable) -> Any:
        with self._lock:
            now = time.time()
            if key in self._klines_cache and now < self._klines_cache[key]['expire']:
                return self._klines_cache[key]['data']
            data = fetch_func()
            self._klines_cache[key] = {'data': data, 'expire': now + self.KLINES_TTL}
            return data
    
    def invalidate(self):
        with self._lock:
            self._klines_cache.clear()


# 全局缓存实例
_scan_cache = ScanCache()


class ScanEngine:
    """
    扫描引擎
    
    功能：
    - 获取 OKX 合约交易对列表
    - 获取每个交易对的行情数据
    - 执行扫描策略
    - 定时/连续扫描
    """

    def __init__(self, okx_client):
        """
        初始化扫描引擎

        Args:
            okx_client: OKX API 客户端
        """
        self.okx_client = okx_client
        self.strategy: Optional[BaseScannerStrategy] = None
        self.is_running = False
        self.scan_interval = 60  # 扫描间隔（秒）
        self._scan_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # 回调函数
        self.on_scan_start: Optional[Callable] = None
        self.on_scan_progress: Optional[Callable] = None
        self.on_scan_complete: Optional[Callable] = None
        self.on_scan_error: Optional[Callable] = None

        # 扫描结果缓存
        self.last_results: List[Dict] = []
        self.last_scan_time: Optional[datetime] = None
        self.total_scanned: int = 0
        self.total_passed: int = 0

    def set_strategy(self, strategy: BaseScannerStrategy):
        """
        设置扫描策略

        Args:
            strategy: 扫描策略实例
        """
        self.strategy = strategy

    def set_scan_interval(self, interval: int):
        """
        设置扫描间隔

        Args:
            interval: 间隔秒数
        """
        self.scan_interval = max(interval, 10)  # 最小 10 秒

    def get_contract_symbols(self) -> List[str]:
        """获取所有合约交易对列表（带缓存）"""
        return _scan_cache.get_symbols(self._fetch_contract_symbols)
    
    def _fetch_contract_symbols(self) -> List[str]:
        try:
            result = self.okx_client._request(
                "GET",
                "/api/v5/public/instruments",
                params={"instType": "SWAP"}
            )
            if result and result.get('code') == '0':
                symbols = [item.get('instId', '') for item in result.get('data', []) 
                          if item.get('instId', '').endswith('-USDT-SWAP')]
                return symbols
            return []
        except Exception as e:
            print(f"获取合约列表异常: {e}")
            return []

    def get_all_tickers(self) -> List[Dict]:
        """批量获取所有 SWAP 合约行情（带缓存）"""
        return _scan_cache.get_tickers(self._fetch_all_tickers)
    
    def _fetch_all_tickers(self) -> List[Dict]:
        try:
            result = self.okx_client._request(
                "GET",
                "/api/v5/market/tickers",
                params={"instType": "SWAP"}
            )
            if result and result.get('code') == '0':
                return [t for t in result.get('data', []) 
                       if t.get('instId', '').endswith('-USDT-SWAP')]
            return []
        except Exception as e:
            print(f"获取批量行情异常: {e}")
            return []

    def get_ticker_data(self, inst_id: str) -> Optional[Dict]:
        """
        获取单个交易对的行情数据

        Args:
            inst_id: 交易对 ID

        Returns:
            行情数据字典
        """
        try:
            result = self.okx_client.get_ticker(inst_id)
            if result and result.get('code') == '0':
                data = result.get('data', [])
                if data:
                    return data[0]
            return None
        except Exception as e:
            print(f"获取 {inst_id} 行情失败: {e}")
            return None

    def get_klines(self, inst_id: str, bar: str = "1D", limit: int = 100) -> List:
        """获取 K 线数据"""
        cache_key = f"{inst_id}_{bar}_{limit}"
        return _scan_cache.get_klines(cache_key, lambda: self._fetch_klines(inst_id, bar, limit))
    
    def _fetch_klines(self, inst_id: str, bar: str, limit: int) -> List:
        try:
            result = self.okx_client.get_kline(inst_id, bar=bar, limit=limit)
            if result and result.get('code') == '0':
                return result.get('data', [])
            return []
        except Exception as e:
            return []

    def get_klines_batch(self, symbols: List[str], bars: List[str], 
                         limit: int = 50) -> Dict[str, Dict[str, List]]:
        """批量并发获取多个交易对的K线数据"""
        def fetch_single(inst_id: str) -> tuple:
            klines = {}
            for bar in bars:
                klines[bar] = self.get_klines(inst_id, bar, limit)
            return inst_id, klines
        
        results = {}
        with ThreadPoolExecutor(max_workers=25) as ex:
            futures = {ex.submit(fetch_single, s): s for s in symbols}
            for future in as_completed(futures):
                inst_id, klines = future.result()
                results[inst_id] = klines
        return results

    def parse_ticker(self, ticker: Dict) -> ScannerSymbol:
        """
        解析行情数据为 ScannerSymbol

        Args:
            ticker: 行情数据

        Returns:
            ScannerSymbol 实例
        """
        symbol = ScannerSymbol(
            inst_id=ticker.get('instId', ''),
            last_price=float(ticker.get('last', 0)),
            # vol24h 是 USDT 单位成交量，volCcy24h 是币单位
            volume_24h=float(ticker.get('vol24h', 0)),
            price_change_24h=float(ticker.get('sodUtc8', 0)),  # 需要计算涨跌幅
            high_24h=float(ticker.get('high24h', 0)),
            low_24h=float(ticker.get('low24h', 0)),
        )
        # 初始化 extra_data 用于存储 K 线数据
        symbol.extra_data['klines'] = {}
        return symbol

    def scan_once(self, symbols: List[str] = None) -> List[Dict]:
        """
        执行一次扫描

        Args:
            symbols: 要扫描的交易对列表，如果为 None 则获取所有合约

        Returns:
            扫描结果列表
        """
        if not self.strategy:
            raise ValueError("未设置扫描策略")

        # 设置运行状态
        self.is_running = True

        # 获取交易对列表
        if symbols is None:
            symbols = self.get_contract_symbols()

        if not symbols:
            return []

        results = []
        total = len(symbols)
        scanned = 0
        passed = 0

        # 通知扫描开始
        if self.on_scan_start:
            self.on_scan_start(total)

        # 检查是否是排行榜类型的扫描策略
        from src.scanner.base_scanner import ScannerSymbol
        is_ranking_scanner = hasattr(self.strategy, 'scan_all_symbols')
        
        # 检测是否是高级版单边趋势扫描器(需要K线数据)
        is_advanced_scanner = hasattr(self.strategy, '_analyze_unilateral_trend_advanced')

        # 如果是排行榜扫描器，收集所有数据后一次性处理
        if is_ranking_scanner:
            all_symbols_data = []
            
            # 快速获取所有行情数据（单次API请求）
            all_tickers = self.get_all_tickers()
            ticker_map = {t['instId']: t for t in all_tickers}
            
            # 预过滤：只保留高流动性币种（减少后续处理）
            min_volume = 500000  # 50万 USDT 成交量门槛
            filtered_symbols = [
                inst_id for inst_id in symbols 
                if inst_id in ticker_map and float(ticker_map[inst_id].get('vol24h', 0)) >= min_volume
            ]
            
            # 批量解析行情数据
            for inst_id in filtered_symbols:
                ticker = ticker_map.get(inst_id)
                if not ticker:
                    continue
                symbol = self.parse_ticker(ticker)
                open_24h = float(ticker.get('open24h', 0))
                if open_24h > 0:
                    symbol.price_change_24h = ((symbol.last_price - open_24h) / open_24h) * 100
                all_symbols_data.append(symbol)
            
            scanned = len(all_symbols_data)
            
            # 如果是高级版扫描器，并发获取K线数据
            if is_advanced_scanner and all_symbols_data:
                inst_ids = [s.inst_id for s in all_symbols_data]
                klines_batch = self.get_klines_batch(inst_ids, ['5m', '15m', '1H', '4H'], limit=50)
                for symbol in all_symbols_data:
                    symbol.extra_data['klines'] = klines_batch.get(symbol.inst_id, {})
            
            # 进度通知
            for i, symbol in enumerate(all_symbols_data):
                if self.on_scan_progress:
                    self.on_scan_progress(i + 1, scanned, {
                        'symbol': symbol.inst_id,
                        'last_price': symbol.last_price,
                        'volume_24h': symbol.volume_24h,
                        'price_change_24h': symbol.price_change_24h,
                        'progress_only': True
                    })

            # 所有数据收集完毕，调用排行榜扫描方法
            try:
                ranking_result = self.strategy.scan_all_symbols(all_symbols_data)
                
                # 将排行榜结果包装成单个结果对象返回
                results = [ranking_result]
                passed = len(ranking_result.get('top_gainers', [])) + len(ranking_result.get('top_losers', []))
                scanned = len(all_symbols_data)
            except Exception as e:
                print(f"排行榜扫描失败: {e}")
                import traceback
                traceback.print_exc()
                if self.on_scan_error:
                    self.on_scan_error("排行榜", str(e))
        else:
            # 普通扫描逻辑
            for i, inst_id in enumerate(symbols):
                if not self.is_running:
                    break

                try:
                    # 获取行情数据
                    ticker = self.get_ticker_data(inst_id)
                    if not ticker:
                        continue

                    # 解析行情
                    symbol = self.parse_ticker(ticker)

                    # 计算涨跌幅（使用 open24h 和 last）
                    open_24h = float(ticker.get('open24h', 0))
                    if open_24h > 0:
                        symbol.price_change_24h = ((symbol.last_price - open_24h) / open_24h) * 100

                    # 如果是多周期扫描策略，获取 K 线数据
                    klines_data = {}
                    if hasattr(self.strategy, '_check_daily_breakout'):
                        # 需要多周期数据
                        klines_data = {
                            '1D': self.get_klines(inst_id, bar='1D', limit=50),
                            '1H': self.get_klines(inst_id, bar='1H', limit=100),
                            '3m': self.get_klines(inst_id, bar='3m', limit=100),
                        }
                        symbol.extra_data['klines'] = klines_data

                    # 执行扫描
                    result = self.strategy.scan_symbol(symbol)
                    result['last_price'] = symbol.last_price
                    result['volume_24h'] = symbol.volume_24h
                    result['price_change_24h'] = symbol.price_change_24h
                    result['high_24h'] = symbol.high_24h
                    result['low_24h'] = symbol.low_24h

                    results.append(result)
                    scanned += 1

                    if result['passed']:
                        passed += 1

                    # 通知进度
                    if self.on_scan_progress:
                        self.on_scan_progress(i + 1, total, result)

                    # 避免请求过快
                    time.sleep(0.05)

                except Exception as e:
                    print(f"扫描 {inst_id} 失败: {e}")
                    if self.on_scan_error:
                        self.on_scan_error(inst_id, str(e))

        # 保存结果
        with self._lock:
            self.last_results = results
            self.last_scan_time = datetime.now()
            self.total_scanned = scanned
            self.total_passed = passed

        # 通知扫描完成
        if self.on_scan_complete:
            self.on_scan_complete(scanned, passed, results)

        return results

    def start_auto_scan(self):
        """
        启动自动扫描（后台线程）
        """
        if self.is_running:
            return

        self.is_running = True
        self._scan_thread = threading.Thread(target=self._auto_scan_loop, daemon=True)
        self._scan_thread.start()

    def stop_auto_scan(self):
        """
        停止自动扫描
        """
        self.is_running = False
        if self._scan_thread:
            self._scan_thread.join(timeout=5)
            self._scan_thread = None

    def _auto_scan_loop(self):
        """
        自动扫描循环
        """
        while self.is_running:
            try:
                self.scan_once()
            except Exception as e:
                print(f"自动扫描异常: {e}")
                if self.on_scan_error:
                    self.on_scan_error("引擎", str(e))

            # 等待下一次扫描
            for _ in range(self.scan_interval):
                if not self.is_running:
                    break
                time.sleep(1)

    def get_results(self) -> List[Dict]:
        """
        获取最新扫描结果

        Returns:
            扫描结果列表
        """
        with self._lock:
            return self.last_results.copy()

    def get_status(self) -> Dict:
        """
        获取引擎状态

        Returns:
            状态字典
        """
        return {
            'is_running': self.is_running,
            'scan_interval': self.scan_interval,
            'last_scan_time': self.last_scan_time,
            'total_scanned': self.total_scanned,
            'total_passed': self.total_passed,
            'has_strategy': self.strategy is not None,
        }
