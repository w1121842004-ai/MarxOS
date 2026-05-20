from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app
from langchain_core.documents import Document


CASES = [
    {
        "id": "mea04_dialectics_finality",
        "query": "在辩证法哲学看来，不存在任何最终的东西、绝对的东西、神圣的东西",
        "source": "mea04.pdf",
        "page_span": [287, 288],
        "expected_printed_page": 270,
    },
]


def run_case(case: dict) -> dict:
    doc = Document(
        page_content=case["query"],
        metadata={
            "source": case["source"],
            "page": case["page_span"][0],
            "pdf_page": case["page_span"][0],
            "page_span": case["page_span"],
        },
    )
    refined = app.refine_doc_citation_page_for_query(doc, case["query"])
    metadata = app.normalize_metadata(refined.metadata)
    actual = metadata.get("printed_page") or metadata.get("citation_page")
    return {
        "id": case["id"],
        "status": "pass" if actual == case["expected_printed_page"] else "fail",
        "expected_printed_page": case["expected_printed_page"],
        "actual_page": actual,
        "metadata": metadata,
    }


def main() -> int:
    results = [run_case(case) for case in CASES]
    for result in results:
        print(
            f"[{result['status'].upper()}] {result['id']} "
            f"expected={result['expected_printed_page']} actual={result['actual_page']}"
        )
    report = ROOT_DIR / "logs" / "citation_page_eval.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if any(result["status"] == "fail" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
