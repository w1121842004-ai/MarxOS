#!/bin/zsh
cd "$(dirname "$0")"
export MARXOS_VECTOR_BACKEND=milvus
export MARXOS_EMBEDDING_MODEL=BAAI/bge-m3
export MARXOS_EMBEDDING_DEVICE=mps
export MILVUS_URI=./data/milvus_lite/marxos_text_layer_bgem3.db
export MILVUS_COLLECTION=marxos_text_layer_bgem3
export MILVUS_SPARSE_PROVIDER=lexical
export MARXOS_MILVUS_HYBRID=1
export BGE_M3_MAX_LENGTH=512
export EXACT_QUOTE_LOOKUP_TIMEOUT_SEC=3
export EXACT_QUOTE_GLOBAL_FALLBACK=0
export OCR_CACHE_DIR=data/ocr_cache_text_layer
export PARAGRAPH_CACHE_PATH=data/paragraph_cache_text_layer.jsonl
export SEMANTIC_LIGHT_SPARSE_INDEX_PATH=data/sparse_paragraph_index_text_layer.pkl
exec .venv/bin/python web_app.py
