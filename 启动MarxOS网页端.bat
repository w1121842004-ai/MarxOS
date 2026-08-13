@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem macOS/Windows：torch 与 Milvus Lite（FAISS 实现的 HNSW）共用 libomp 时，
rem 多线程检索可能段错误；固定单线程，app.py 内部也会兜底设置。
set "OMP_NUM_THREADS=1"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PYTHON=venv\Scripts\python.exe"
) else (
    echo MarxOS 启动失败：未找到 .venv\Scripts\python.exe 或 venv\Scripts\python.exe。
    echo 请先创建虚拟环境并安装 requirements.txt。
    exit /b 1
)

rem web_app.py 会加载 .env，未配置项统一使用 marxos\config\settings.py 的默认值。
"%PYTHON%" web_app.py
