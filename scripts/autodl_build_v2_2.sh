#!/usr/bin/env bash
# AutoDL 全集 corpus-v2.2 一键构建脚本。
#
# 前置条件（见 docs/autodl_guide.md）：
#   1. 已 git clone 本仓库到 ~/MarxOS
#   2. 已上传 data/ocr_cache_text_layer、data/ocr_cache、rag/ 到仓库 data/、rag/ 目录
#   3. 已创建 venv 并安装 requirements（GPU 版 torch）
#
# 产物（构建完成后用 scp 回传本机）：
#   data/artifacts/corpus_v2_2/         段落/富化/子块/BM25 统计/manifest
#   data/milvus_lite/marxos_corpus_v2_2.db/  Milvus Lite 向量库
#
# 用法：bash scripts/autodl_build_v2_2.sh [--cuda|--cpu] [--batch N]

set -euo pipefail

DEVICE="cuda"
BATCH=128
for arg in "$@"; do
  case "$arg" in
    --cpu) DEVICE="cpu" ;;
    --cuda) DEVICE="cuda" ;;
    --batch) shift; BATCH="$1" ;;
  esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CONFIG="config/rebuild_v2_2.json"
ART="data/artifacts/corpus_v2_2"

echo "== 设备: ${DEVICE} 批大小: ${BATCH} =="

# 0. 环境
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r requirements.txt

# 1. preflight
python scripts/rebuild_corpus_v2.py preflight --config "$CONFIG"

# 2. pages（页面记录冻结，含 page_number_v2 页码）
python scripts/rebuild_corpus_v2.py pages --config "$CONFIG"

# 3. paragraphs（段落检测 + 全量审计）
python scripts/rebuild_corpus_v2.py paragraphs --config "$CONFIG"

# 4. enrich（确定性书目富化）
python scripts/rebuild_corpus_v2.py enrich --config "$CONFIG"

# 5. retrieval-units（semantic child 320/64）
python scripts/build_retrieval_records_v2.py \
  --input "$ART/paragraph_records_enriched.jsonl" \
  --output "$ART/semantic_child_records.jsonl" \
  --summary "$ART/retrieval_build_summary.json" \
  --chunk-size 320 --chunk-overlap 64

# 6. index（全量 embedding + BM25 统计 + Milvus 写入，checkpoint 续跑）
python scripts/build_milvus_v2.py \
  --input "$ART/semantic_child_records.jsonl" \
  --uri "data/milvus_lite/marxos_corpus_v2_2.db" \
  --collection "marxos_passages_v2_2" \
  --bm25-stats "$ART/bm25_stats.json" \
  --checkpoint "$ART/milvus_v2_2_checkpoint.json" \
  --embedding-model BAAI/bge-m3 \
  --dim 1024 --batch-size "$BATCH" --device "$DEVICE"

# 7. validate（schema/行数/双 hash/lineage/hybrid probe）
python scripts/validate_milvus_v2.py \
  --child-records "$ART/semantic_child_records.jsonl" \
  --parent-records "$ART/paragraph_records_enriched.jsonl" \
  --uri "$ROOT/data/milvus_lite/marxos_corpus_v2_2.db" \
  --collection "marxos_passages_v2_2" \
  --bm25-stats "$ART/bm25_stats.json"

# 8. manifest
python scripts/rebuild_corpus_v2.py manifest --config "$CONFIG"

echo "== 构建完成。回传 data/artifacts/corpus_v2_2/ 与 data/milvus_lite/marxos_corpus_v2_2.db/ 到本机 =="
