from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import app


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


DATASET_PATH = ROOT_DIR / "eval_dataset.json"
REPORT_PATH = ROOT_DIR / "logs" / "eval_dataset_results.json"
NEGATIVE_EXPECTATIONS = {
    "\u4e0d\u5b58\u5728",
    "\u65e0",
    "\u4e0d\u662f\u9a6c\u514b\u601d\u76f4\u63a5\u539f\u8bdd",
    "\u4e0d\u662f\u539f\u6587\u8868\u8fbe",
}
REFUSAL_MARKERS = {
    "\u4e0d\u652f\u6301",
    "\u4e0d\u662f",
    "\u4e0d\u5e94",
    "\u4e0d\u80fd",
    "\u672a\u80fd",
    "\u4e0d\u8981",
}
FABRICATION_RISK_MARKERS = {
    "\u63d0\u51fa\u4e8e\u300a",
    "\u7b2c1\u9875",
    "\u7b2c2\u9875",
    "\u7b2c3\u9875",
    "\u7b2c4\u9875",
    "\u7b2c5\u9875",
    "\u5377\u7b2c",
    "pdf",
}


def normalize(text: object) -> str:
    text = str(text or "")
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", text).lower()


def expected_items(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    return [str(value)]


def expected_tokens(work: str) -> list[str]:
    normalized = normalize(work)
    if not normalized or normalized in {normalize(item) for item in NEGATIVE_EXPECTATIONS}:
        return []

    # Composite expectations often include section hints. Treat spaces as all-of
    # constraints, while still allowing the full title to be found as one phrase.
    parts = [part for part in re.split(r"\s+", work.strip()) if part]
    tokens = [normalize(part) for part in parts if normalize(part)]
    return tokens or [normalized]


def doc_text(doc) -> str:
    metadata = app.normalize_metadata(doc.metadata)
    fields = [
        metadata.get("book"),
        metadata.get("article"),
        metadata.get("section"),
        metadata.get("classic_title"),
        metadata.get("classic_author"),
        metadata.get("classic_work_type"),
        metadata.get("source"),
        doc.page_content,
    ]
    return "\n".join(str(field or "") for field in fields)


def summarize_doc(doc) -> dict:
    metadata = app.normalize_metadata(doc.metadata)
    return {
        "source": metadata.get("source"),
        "book": metadata.get("book"),
        "article": metadata.get("article") or metadata.get("section"),
        "citation_page": metadata.get("citation_page"),
        "citation_page_type": metadata.get("citation_page_type"),
        "match_type": metadata.get("match_type"),
        "snippet": " ".join(str(doc.page_content or "").split())[:180],
    }


def retrieve(db, question: str, top_k: int):
    return app.retrieve_documents(question, db, k=top_k)


def evaluate_case(case: dict, docs: list) -> dict:
    question_id = case.get("id")
    expected = expected_items(case.get("expected_work"))
    hard_negative = expected_items(case.get("hard_negative"))
    combined = normalize("\n".join(doc_text(doc) for doc in docs))

    expected_norms = [normalize(item) for item in expected]
    is_negative = any(item in {normalize(x) for x in NEGATIVE_EXPECTATIONS} for item in expected_norms)

    if is_negative:
        answer = app.run_query(case.get("question") or "")
        answer_norm = normalize(answer)
        has_refusal = any(normalize(marker) in answer_norm for marker in REFUSAL_MARKERS)
        risky_citation = any(normalize(marker) in answer_norm for marker in FABRICATION_RISK_MARKERS)
        status = "pass" if has_refusal and not risky_citation else "fail"
        reasons = []
        if not has_refusal:
            reasons.append("negative_answer_missing_refusal")
        if risky_citation:
            reasons.append("negative_answer_may_fabricate_citation")
        answer_preview = answer
    else:
        missed = []
        for work in expected:
            tokens = expected_tokens(work)
            if tokens and not all(token in combined for token in tokens):
                missed.append(work)

        hard_hits = [
            item for item in hard_negative
            if normalize(item) and normalize(item) in combined
        ]

        reasons = []
        if missed:
            reasons.append("missing_expected_work: " + "; ".join(missed))
        if hard_hits:
            reasons.append("hard_negative_hit: " + "; ".join(hard_hits))

        status = "pass" if not reasons else "fail"
        answer_preview = None

    return {
        "id": question_id,
        "question": case.get("question"),
        "expected_work": case.get("expected_work"),
        "expected_author": case.get("expected_author"),
        "status": status,
        "reasons": reasons,
        "answer_preview": answer_preview,
        "top_docs": [summarize_doc(doc) for doc in docs],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--report", default=str(REPORT_PATH))
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    with dataset_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    db = app.load_vectorstore()
    results = []
    for index, case in enumerate(dataset, start=1):
        question = case["question"]
        docs = retrieve(db, question, args.top_k)
        result = evaluate_case(case, docs)
        results.append(result)
        print(f"[{result['status'].upper():11}] {case.get('id', index):>2} {question}")
        if result["reasons"]:
            for reason in result["reasons"]:
                print(f"  - {reason}")
        first = result["top_docs"][0] if result["top_docs"] else {}
        print(f"  top1: {first.get('source')} | {first.get('article')} | page={first.get('citation_page')}")

    counts = {
        "pass": sum(1 for item in results if item["status"] == "pass"),
        "fail": sum(1 for item in results if item["status"] == "fail"),
        "unsupported": sum(1 for item in results if item["status"] == "unsupported"),
    }
    summary = {"total": len(results), **counts}

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    print("\nSummary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")
    return 1 if counts["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
