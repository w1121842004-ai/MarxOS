from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CANONICAL_DEFAULTS = {
    "MARXOS_CORPUS_PROFILE": "me_full_v2",
    "MARXOS_RETRIEVAL_PROFILE": "milvus_bgem3_v2",
    "MARXOS_ANSWER_PROFILE": "deepseek_default",
    "MARXOS_VECTOR_BACKEND": "milvus",
    "MARXOS_EMBEDDING_MODEL": "BAAI/bge-m3",
    "MARXOS_EMBEDDING_DEVICE": "cpu",
    "MILVUS_URI": "./data/milvus_lite/marxos_corpus_v2.db",
    "MILVUS_COLLECTION": "marxos_passages_v2",
    "MILVUS_SPARSE_PROVIDER": "bm25",
    "MARXOS_BM25_STATS_PATH": "data/artifacts/corpus_v2/bm25_stats_v2_1.json",
    "MILVUS_HYBRID_SEARCH": "1",
    "OCR_CACHE_DIR": "data/ocr_cache_text_layer",
    "PARAGRAPH_CACHE_PATH": "data/artifacts/corpus_v2/paragraph_records_enriched_v2_1.jsonl",
    "SEMANTIC_LIGHT_SPARSE_INDEX_PATH": "data/sparse_paragraph_index_text_layer.pkl",
}


def parse_env_example() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, value = stripped.split("=", 1)
            values[key] = value
    return values


class LaunchContractTests(unittest.TestCase):
    def test_env_example_freezes_canonical_defaults(self) -> None:
        values = parse_env_example()
        for key, expected in CANONICAL_DEFAULTS.items():
            self.assertEqual(values.get(key), expected, key)

    def test_macos_launcher_delegates_defaults_to_shared_config(self) -> None:
        text = (ROOT / "启动MarxOS网页端.command").read_text(encoding="utf-8")
        self.assertIn('cd "$(dirname "$0")"', text)
        for key in CANONICAL_DEFAULTS:
            self.assertNotRegex(text, rf"export\s+{re.escape(key)}=", key)
        self.assertIn('exec "$PYTHON" web_app.py', text)

    def test_windows_launcher_delegates_defaults_to_shared_config(self) -> None:
        text = (ROOT / "启动MarxOS网页端.bat").read_text(encoding="utf-8")
        self.assertIn('cd /d "%~dp0"', text)
        self.assertNotIn("C:\\Users\\", text)
        for key in CANONICAL_DEFAULTS:
            self.assertNotRegex(text, rf"(?im)^set\s+{re.escape(key)}=", key)
        self.assertIn('"%PYTHON%" web_app.py', text)


if __name__ == "__main__":
    unittest.main()
