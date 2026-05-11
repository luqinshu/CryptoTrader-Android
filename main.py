import importlib.util
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ANDROID_DIR = os.path.join(PROJECT_ROOT, "android")
ANDROID_MAIN = os.path.join(ANDROID_DIR, "main.py")

if ANDROID_DIR not in sys.path:
    sys.path.insert(0, ANDROID_DIR)

spec = importlib.util.spec_from_file_location("android_main", ANDROID_MAIN)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)
CryptoScannerApp = module.CryptoScannerApp


if __name__ == "__main__":
    CryptoScannerApp().run()
