from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag.exact_quote_lookup import exact_quote_lookup, normalize_quote
from scripts.evaluate_retrieval import build_questions


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


METADATA_MARKERS = [
    "目录",
    "目次",
    "索引",
    "注释",
    "编者注",
    "说明",
    "序言",
    "选编说明",
]

LEAD_ANNOTATION_MARKERS = [
    "重要著作",
    "基本理论的重要著作",
    "本文节选自",
    "选编说明",
    "编者注",
]


@dataclass
class AuditFinding:
    index: int
    question: str
    source: str
    page: int | None
    article: str
    flags: list[str]


def detect_annotation_flags(metadata: dict, content: str) -> list[str]:
    article = str(metadata.get("section") or metadata.get("article") or "")
    article_norm = normalize_quote(article)
    lead_norm = normalize_quote(content[:260])
    flags = []

    for marker in METADATA_MARKERS:
        if normalize_quote(marker) in article_norm:
            flags.append(f"metadata:{marker}")

    for marker in LEAD_ANNOTATION_MARKERS:
        if normalize_quote(marker) in lead_norm:
            flags.append(f"lead:{marker}")

    return flags


def audit_core_quote_top1() -> list[AuditFinding]:
    findings = []
    quote_questions = [item for item in build_questions() if item.group == "core_quote"]

    for index, item in enumerate(quote_questions, start=1):
        docs = exact_quote_lookup(item.question, limit=3)
        if not docs:
            findings.append(
                AuditFinding(index, item.question, "", None, "", ["no_exact_quote_hit"])
            )
            continue

        top_doc = docs[0]
        metadata = top_doc.metadata
        flags = detect_annotation_flags(metadata, top_doc.page_content)
        if flags:
            findings.append(
                AuditFinding(
                    index=index,
                    question=item.question,
                    source=str(metadata.get("source") or ""),
                    page=metadata.get("pdf_page"),
                    article=str(metadata.get("article") or metadata.get("section") or ""),
                    flags=flags,
                )
            )

    return findings


def main() -> int:
    quote_questions = [item for item in build_questions() if item.group == "core_quote"]
    findings = audit_core_quote_top1()

    print(f"Audited core_quote Top1: {len(quote_questions)}")
    if not findings:
        print("No annotation-like Top1 hits found.")
        return 0

    print(f"Findings: {len(findings)}")
    for finding in findings:
        print(
            f"- #{finding.index} {finding.source} page={finding.page} "
            f"flags={','.join(finding.flags)} article={finding.article}"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
