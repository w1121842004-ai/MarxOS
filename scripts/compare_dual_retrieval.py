from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app  # noqa: E402
from scripts.evaluate_retrieval import CONCEPT_QUESTIONS, concept_match_passed, format_metadata  # noqa: E402


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


@dataclass
class DualResult:
    index: int
    question: str
    chunk_passed: bool
    paragraph_passed: bool
    same_top_source: bool
    same_top_page: bool


def page_key(metadata: dict) -> object:
    return (
        metadata.get("citation_page")
        or metadata.get("printed_page")
        or metadata.get("page")
        or metadata.get("pdf_page")
    )


def compare(k: int = 3, show: int = 3) -> None:
    if not app.paragraph_vectorstore_exists():
        raise FileNotFoundError(
            f"paragraph vectorstore not found: {app.PARAGRAPH_VECTORSTORE_DIR}. "
            "Run scripts/build_paragraph_vectorstore.py first."
        )

    chunk_db = app.load_vectorstore()
    paragraph_db = app.load_paragraph_vectorstore()
    results = []

    for index, item in enumerate(CONCEPT_QUESTIONS, start=1):
        dual = app.retrieve_dual_documents(item.question, chunk_db, paragraph_db, k=k)
        chunk_docs = dual["chunk"]
        paragraph_docs = dual["paragraph"]
        chunk_ok, chunk_reason = concept_match_passed(item, chunk_docs)
        paragraph_ok, paragraph_reason = concept_match_passed(item, paragraph_docs)

        chunk_top = chunk_docs[0].metadata if chunk_docs else {}
        paragraph_top = paragraph_docs[0].metadata if paragraph_docs else {}
        same_top_source = chunk_top.get("source") == paragraph_top.get("source")
        same_top_page = page_key(chunk_top) == page_key(paragraph_top)

        results.append(
            DualResult(
                index=index,
                question=item.question,
                chunk_passed=chunk_ok,
                paragraph_passed=paragraph_ok,
                same_top_source=same_top_source,
                same_top_page=same_top_page,
            )
        )

        print(f"\n===== #{index} {item.question} =====")
        print(f"chunk: {chunk_ok} ({chunk_reason})")
        for rank, doc in enumerate(chunk_docs[:show], start=1):
            print(f"  C{rank}. {format_metadata(doc.metadata)}")
            print(f"      {app.compact_preview(doc.page_content, limit=120)}")

        print(f"paragraph: {paragraph_ok} ({paragraph_reason})")
        for rank, doc in enumerate(paragraph_docs[:show], start=1):
            print(f"  P{rank}. {format_metadata(doc.metadata)}")
            print(f"      {app.compact_preview(doc.page_content, limit=120)}")

    total = len(results)
    chunk_passed = sum(result.chunk_passed for result in results)
    paragraph_passed = sum(result.paragraph_passed for result in results)
    same_source = sum(result.same_top_source for result in results)
    same_page = sum(result.same_top_page for result in results)

    print("\n===== DUAL RETRIEVAL SUMMARY =====")
    print(f"concept questions: {total}")
    print(f"chunk passed: {chunk_passed}/{total}")
    print(f"paragraph passed: {paragraph_passed}/{total}")
    print(f"same top source: {same_source}/{total}")
    print(f"same top page: {same_page}/{total}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare chunk and paragraph retrieval on concept questions.")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--show", type=int, default=3)
    args = parser.parse_args()
    compare(k=args.k, show=args.show)


if __name__ == "__main__":
    main()
