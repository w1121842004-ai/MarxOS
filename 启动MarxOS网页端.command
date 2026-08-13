#!/bin/zsh
cd "$(dirname "$0")"

# macOS ARM: torch + Milvus Lite (FAISS-backed HNSW) 共用 libomp，多线程会在
# 检索时段错误；启动前固定单线程，app.py 内部也会兜底设置。
export OMP_NUM_THREADS=1

if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
elif [[ -x "venv/bin/python" ]]; then
  PYTHON="venv/bin/python"
else
  print -u2 "MarxOS 启动失败：未找到 .venv/bin/python 或 venv/bin/python。"
  print -u2 "请先创建虚拟环境并安装 requirements.txt。"
  exit 1
fi

# web_app.py 会加载 .env，未配置项统一使用 marxos/config/settings.py 的默认值。
exec "$PYTHON" web_app.py
