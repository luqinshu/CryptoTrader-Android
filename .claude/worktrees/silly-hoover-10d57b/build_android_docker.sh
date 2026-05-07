#!/bin/bash

#############################################################################
# CryptoScanner Pro - 使用Docker打包Android APK
# 解决Python版本兼容问题
#############################################################################

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     CryptoScanner Pro - Docker Android 打包工具      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# 检查Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker 未安装，请先安装 Docker Desktop"
    echo ""
    echo "  macOS 安装方法:"
    echo "    brew install --cask docker"
    echo ""
    exit 1
fi

print_success "Docker 已安装"

# 检查项目文件
if [ ! -f "main_android.py" ] || [ ! -f "buildozer_android.spec" ]; then
    print_error "请在项目根目录运行此脚本"
    exit 1
fi

print_success "项目文件检查通过"
echo ""

# 开始打包
print_info "开始 Docker 打包..."
print_info "首次运行会下载 Android SDK/NDK（约 2GB），请耐心等待..."
echo ""

# 使用官方 buildozer Docker 镜像
docker run --rm -v "$(pwd)":/home/user/hostcwd \
  -v buildozer-cache:/home/user/.buildozer \
  -e BUILDOZER_ALLOW_ROOT=1 \
  --workdir /home/user/hostcwd \
  python:3.10-slim \
  bash -c "
    set -e
    echo '正在安装依赖...'
    apt-get update
    apt-get install -y git cmake openjdk-17-jdk-headless \
      build-essential autoconf automake libtool pkg-config
    
    pip install buildozer cython
    
    echo '开始打包 APK...'
    buildozer -v android debug
    
    echo ''
    echo '打包完成！APK 文件位置:'
    ls -lh bin/*.apk 2>/dev/null || echo '未找到 APK 文件'
  "

echo ""
print_info "========================================="
print_info "  打包完成"
print_info "========================================="
echo ""
print_success "APK 文件位于: bin/"
echo ""
print_info "安装到手机:"
echo "  adb install bin/CryptoScannerPro-1.0.0-debug.apk"
echo ""
