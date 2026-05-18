from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import (
    clean_article_title,
    is_noisy_article_title,
    normalize_for_match,
    retrieve_documents,
)
from scripts.evaluate_retrieval import build_questions, format_metadata, load_vectorstore


ARTICLE_RED_FLAGS = [
    "目录",
    "目次",
    "索引",
    "注释",
    "编者注",
    "说明",
    "序言",
    "选编说明",
]


def article_flags(metadata: dict) -> list[str]:
    article = clean_article_title(metadata.get("section") or metadata.get("article"))
    article_norm = normalize_for_match(article)
    flags = []

    if not article:
        flags.append("empty_article")
    if is_noisy_article_title(article):
        flags.append("noisy_article")
    for marker in ARTICLE_RED_FLAGS:
        if normalize_for_match(marker) in article_norm:
            flags.append(f"marker:{marker}")
    if metadata.get("classic_title") and metadata.get("article") == metadata.get("classic_title"):
        flags.append("canonical_classic_title")

    return flags


def main() -> None:
    db = load_vectorstore()
    concept_questions = [
        question for question in build_questions()
        if question.group == "concept"
    ]
    flagged = []

    for index, question in enumerate(concept_questions, start=1):
        docs = retrieve_documents(question.question, db, k=3)
        print(f"\n===== concept {index}: {question.question} =====")
        for rank, doc in enumerate(docs, start=1):
            flags = article_flags(doc.metadata)
            flag_text = ", ".join(flags) if flags else "ok"
            print(f"[{rank}] flags={flag_text}; {format_metadata(doc.metadata)}")
            if rank == 1 and any(flag != "canonical_classic_title" for flag in flags):
                flagged.append((question.question, rank, flags, doc.metadata))

    print("\n===== SUMMARY =====")
    print(f"Audited concept questions: {len(concept_questions)}")
    if not flagged:
        print("No suspicious concept Top1 article metadata found.")
        return

    print("Suspicious concept Top1 article metadata:")
    for question, rank, flags, metadata in flagged:
        print(f"- {question} rank={rank} flags={','.join(flags)} source={metadata.get('source')} article={metadata.get('article')}")


if __name__ == "__main__":
    main()
