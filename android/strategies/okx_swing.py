import os
import importlib.util

_path = os.path.join(os.path.dirname(__file__), 'OKX小时线波段共振策略.py')
_spec = importlib.util.spec_from_file_location('okx_swing_impl', _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
OKXHourSwingScanner = _mod.OKXHourSwingScanner
