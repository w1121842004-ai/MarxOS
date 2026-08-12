@echo off
chcp 65001 >nul
cd /d C:\Users\Administrator\Desktop\MarxOS
set MARXOS_VECTOR_BACKEND=milvus
set MARXOS_EMBEDDING_MODEL=BAAI/bge-m3
set MARXOS_EMBEDDING_DEVICE=cpu
set MILVUS_URI=.\data\milvus_lite\marxos_text_layer_bgem3.db
set MILVUS_COLLECTION=marxos_text_layer_bgem3
set MILVUS_SPARSE_PROVIDER=lexical
set MARXOS_MILVUS_HYBRID=1
set BGE_M3_MAX_LENGTH=512
set EXACT_QUOTE_LOOKUP_TIMEOUT_SEC=3
set EXACT_QUOTE_GLOBAL_FALLBACK=0
set OCR_CACHE_DIR=data\ocr_cache_text_layer
set PARAGRAPH_CACHE_PATH=data\paragraph_cache_text_layer.jsonl
set SEMANTIC_LIGHT_SPARSE_INDEX_PATH=data\sparse_paragraph_index_text_layer.pkl
venv\Scripts\python.exe web_app.py
