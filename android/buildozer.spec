[app]
title = CryptoScanner Pro
package.name = cryptoscanner
package.domain = org.cryptotrader
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,json,ttf,ttc,ini,gif
source.exclude_dirs = tests,venv,venv313,.git,__pycache__,.claude,.venv,.buildozer,node_modules,reports,data,.pytest_cache,.github,screenshots

version = 1.0.0

requirements = python3,kivy==2.3.0,android

orientation = portrait
fullscreen = 0

android.permissions = INTERNET, ACCESS_NETWORK_STATE, ACCESS_WIFI_STATE

android.api = 33
android.minapi = 26
android.sdk = 33
android.ndk = 25b
android.archs = arm64-v8a

android.windowOrientation = portrait
android.density = 480
android.screen = portrait
android.use_fullscreen = False
android.display_cutout = True
android.allow_backup = False
android.accept_sdk_license = True

p4a.branch = master
# p4a.local_recipes = ./p4a_recipes

# Release signing (uncomment and set env vars to build release)
# android.release_artifact = apk
# p4a.release = %(source.dir)s/keystore/cryptoscanner.keystore
# p4a.release_keyalias = cryptoscanner

[buildozer]
log_level = 2
warn_on_root = 1
build_dir = .buildozer
bin_dir = bin
