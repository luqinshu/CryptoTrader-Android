[app]
title = CryptoScannerPro
package.name = cryptoscanner
package.domain = org.cryptotrader
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,json
version = 1.0.0

# 核心依赖 (必须包含 pandas, numpy, ta 等)
requirements = python3,kivy==2.2.1,kivymd==1.1.1,requests,urllib3,chardet,certifi,idna,pandas,numpy,ta,okx

orientation = portrait
fullscreen = 0
android.permissions = INTERNET, WRITE_EXTERNAL_STORAGE

# 安卓相关配置
android.api = 31
android.minapi = 21
android.sdk = 31
android.ndk = 25b
android.arch = armeabi-v7a, arm64-v8a

[buildozer]
log_level = 2
warn_on_root = 1
