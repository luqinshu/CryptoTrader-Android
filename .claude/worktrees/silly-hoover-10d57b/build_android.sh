#!/bin/bash

#############################################################################
# CryptoScanner Pro - Android 一键打包脚本
# 使用方法: chmod +x build_android.sh && ./build_android.sh
#############################################################################

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 标题
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     CryptoScanner Pro - Android APK 打包工具         ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# 检查是否在项目目录
if [ ! -f "main_android.py" ]; then
    print_error "请在项目根目录运行此脚本！"
    exit 1
fi

# 检查 Python
if ! command -v python3 &> /dev/null; then
    print_error "未找到 Python3，请先安装 Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
print_success "Python 版本: $PYTHON_VERSION"

# 检查 Buildozer
if ! command -v buildozer &> /dev/null; then
    print_warning "未检测到 Buildozer，正在安装..."
    
    pip3 install --user buildozer cython
    
    # 添加到 PATH
    export PATH="$HOME/Library/Python/3.9/bin:$PATH" 2>/dev/null || true
    export PATH="$HOME/.local/bin:$PATH" 2>/dev/null || true
    
    if command -v buildozer &> /dev/null; then
        print_success "Buildozer 安装成功"
    else
        print_error "Buildozer 安装失败，请手动安装"
        exit 1
    fi
else
    BUILDOZER_VERSION=$(buildozer --version 2>/dev/null || echo "unknown")
    print_success "Buildozer 已安装: $BUILDOZER_VERSION"
fi

echo ""
print_info "========================================="
print_info "  请选择打包方式"
print_info "========================================="
echo ""
echo "  1) 标准打包（推荐）"
echo "  2) 清理后重新打包"
echo "  3) 仅检查环境（不打包）"
echo ""

read -p "请选择 [1-3]: " choice

case $choice in
    1)
        print_info "开始标准打包..."
        CLEAN=0
        ;;
    2)
        print_warning "将清理所有缓存后重新打包"
        read -p "确认？[y/N]: " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            print_info "清理缓存..."
            buildozer android clean
            rm -rf .buildozer
            rm -rf bin
            print_success "清理完成"
        else
            print_info "已取消清理"
        fi
        CLEAN=0
        ;;
    3)
        print_info "检查环境..."
        echo ""
        
        # 检查必要文件
        FILES=("main_android.py" "buildozer_android.spec" "requirements.txt")
        for file in "${FILES[@]}"; do
            if [ -f "$file" ]; then
                print_success "✓ $file 存在"
            else
                print_error "✗ $file 不存在"
            fi
        done
        
        echo ""
        
        # 检查依赖包
        DEPS=("pandas" "numpy" "ta" "requests" "kivymd")
        for dep in "${DEPS[@]}"; do
            if python3 -c "import $dep" 2>/dev/null; then
                print_success "✓ $dep 已安装"
            else
                print_warning "✗ $dep 未安装（打包时会自动包含）"
            fi
        done
        
        echo ""
        print_success "环境检查完成"
        exit 0
        ;;
    *)
        print_error "无效选择"
        exit 1
        ;;
esac

echo ""
print_info "========================================="
print_info "  开始打包 APK"
print_info "========================================="
echo ""

print_info "使用配置文件: buildozer_android.spec"
print_info "入口文件: main_android.py"
echo ""

# 开始打包
START_TIME=$(date +%s)

print_warning "首次打包需要下载 Android SDK/NDK（约2GB），请耐心等待..."
echo ""

# 执行打包
if buildozer -v android debug; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    MINUTES=$((DURATION / 60))
    SECONDS=$((DURATION % 60))
    
    echo ""
    print_success "╔════════════════════════════════════════╗"
    print_success "║        APK 打包成功！🎉                ║"
    print_success "╚════════════════════════════════════════╝"
    echo ""
    print_info "耗时: ${MINUTES}分${SECONDS}秒"
    echo ""
    
    # 查找 APK 文件
    APK_FILE=$(find bin -name "*.apk" -type f 2>/dev/null | head -n 1)
    
    if [ -n "$APK_FILE" ]; then
        APK_SIZE=$(du -h "$APK_FILE" | cut -f1)
        print_success "APK 位置: $APK_FILE"
        print_success "APK 大小: $APK_SIZE"
        echo ""
        
        # 安装提示
        print_info "========================================="
        print_info "  安装指南"
        print_info "========================================="
        echo ""
        echo "  方式1: 使用 adb 安装"
        echo "    adb install $APK_FILE"
        echo ""
        echo "  方式2: 手动传输到手机"
        echo "    1. 将 APK 文件复制到手机"
        echo "    2. 在文件管理器中点击安装"
        echo "    3. 允许安装未知来源应用"
        echo ""
        echo "  方式3: 启动临时 HTTP 服务器"
        echo "    cd bin && python3 -m http.server 8000"
        echo "    手机浏览器访问: http://YOUR_IP:8000"
        echo ""
    else
        print_warning "未找到 APK 文件，请检查构建日志"
    fi
    
else
    echo ""
    print_error "╔════════════════════════════════════════╗"
    print_error "║        APK 打包失败！❌                ║"
    print_error "╚════════════════════════════════════════╝"
    echo ""
    print_error "请查看上方错误信息"
    echo ""
    print_info "常见问题排查："
    echo "  1. 确保网络连接正常"
    echo "  2. 清理缓存后重试: buildozer android clean"
    echo "  3. 检查 buildozer_android.spec 配置"
    echo "  4. 查看详细日志: cat .buildozer/android/platform/build/build.log"
    echo ""
    exit 1
fi

echo ""
print_info "========================================="
print_info "  后续操作"
print_info "========================================="
echo ""
echo "  查看完整打包日志："
echo "    cat .buildozer/android/platform/build/build.log"
echo ""
echo "  重新打包（不清理缓存）："
echo "    buildozer android debug"
echo ""
echo "  完全清理后重新打包："
echo "    buildozer android clean && buildozer android debug"
echo ""
print_success "完成！"
echo ""
