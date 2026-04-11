[app]
# 应用基本信息
title = CryptoScannerPro
package.name = cryptoscannerpro
package.domain = org.cryptotrader
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,json,ttf
source.exclude_dirs = tests,venv,.git,__pycache__,*.pyc
source.exclude_patterns = LICENSE,*.md

# 版本
version = 1.0.0

# 应用入口
entrypoint = main_android.py

# 核心依赖 (Android 打包必须包含)
# python3: Python 解释器
# kivy: GUI 框架
# kivymd: Material Design 组件
# pandas,numpy: 数据处理
# ta: 技术指标库
# requests,urllib3,chardet,certifi,idna: HTTP 请求
requirements = python3,kivy==2.2.1,kivymd==1.1.1,pandas,numpy,ta,requests,urllib3,chardet,certifi,idna

# 屏幕方向
orientation = portrait

# 全屏模式 (0=不全屏, 1=全屏)
fullscreen = 0

# Android 权限
android.permissions = INTERNET, ACCESS_NETWORK_STATE

# Android 配置
android.api = 33
android.minapi = 21
android.sdk = 33
android.ndk = 25b
android.archs = arm64-v8a, armeabi-v7a

# 不允许打包过大的文件
android.add_jars = .

# 不允许自动旋转
android.windowOrientation = portrait

# 日志级别
log_level = 2

[buildozer]
# 日志级别
log_level = 2

# 警告根用户
warn_on_root = 1

# 工作目录
build_dir = .buildozer

# bin 目录
bin_dir = bin
