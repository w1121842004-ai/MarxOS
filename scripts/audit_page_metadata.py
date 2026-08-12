from __future__ import annotations

import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", "vectorstore/marx_reader"))
INDEX_PKL = VECTORSTORE_DIR / "index.pkl"


def load_docs():
    with INDEX_PKL.open("rb") as f:
        docstore, _ = pickle.load(f)

    return list(docstore._dict.values())


def as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def compact(text, limit=80):
    text = " ".join(str(text or "").split())
    return text[:limit] + ("..." if len(text) > limit else "")


def audit_docs(docs):
    by_source = defaultdict(list)
    for doc in docs:
        by_source[doc.metadata.get("source") or "unknown"].append(doc)

    print(f"vectorstore: {VECTORSTORE_DIR}")
    print(f"documents: {len(docs)}")
    print(f"sources: {len(by_source)}")
    print()

    total_printed = sum(1 for doc in docs if doc.metadata.get("printed_page") is not None)
    total_pdf_only = len(docs) - total_printed
    print(f"printed_page docs: {total_printed}")
    print(f"pdf-only docs: {total_pdf_only}")
    print()

    suspicious = []
    source_stats = []

    for source, source_docs in sorted(by_source.items()):
        rows = []
        for doc in source_docs:
            metadata = doc.metadata
            pdf_page = as_int(metadata.get("pdf_page"))
            printed_page = as_int(metadata.get("printed_page"))
            if pdf_page is None:
                continue
            rows.append((pdf_page, printed_page, doc))

        rows.sort(key=lambda item: item[0])

        printed_rows = [(pdf, printed, doc) for pdf, printed, doc in rows if printed is not None]
        offsets = [pdf - printed for pdf, printed, _ in printed_rows]
        unique_offsets = sorted(set(offsets))
        source_stats.append(
            (
                source,
                len(rows),
                len(printed_rows),
                len(rows) - len(printed_rows),
                min(offsets) if offsets else None,
                max(offsets) if offsets else None,
                len(unique_offsets),
            )
        )

        previous_pdf = None
        previous_printed = None
        for pdf_page, printed_page, doc in printed_rows:
            offset = pdf_page - printed_page
            reason = None

            if offset < -5 or offset > 140:
                reason = f"offset_out_of_range:{offset}"
            elif previous_pdf is not None and previous_printed is not None:
                pdf_delta = pdf_page - previous_pdf
                printed_delta = printed_page - previous_printed
                if pdf_delta > 0 and printed_delta < -3:
                    reason = f"printed_page_reversal:{previous_printed}->{printed_page}"
                elif pdf_delta > 0 and printed_delta > pdf_delta + 8:
                    reason = f"printed_page_jump:{previous_printed}->{printed_page}"

            if reason:
                suspicious.append((source, pdf_page, printed_page, offset, reason, doc))

            previous_pdf = pdf_page
            previous_printed = printed_page

    print("source summary:")
    print("source docs printed pdf_only offset_min offset_max offset_variants")
    for row in source_stats:
        print("%s %s %s %s %s %s %s" % row)
    print()

    print(f"suspicious printed-page rows: {len(suspicious)}")
    for source, pdf_page, printed_page, offset, reason, doc in suspicious[:80]:
        metadata = doc.metadata
        print(
            f"- {source} pdf={pdf_page} printed={printed_page} offset={offset} "
            f"reason={reason} article={compact(metadata.get('article'), 50)} "
            f"text={compact(doc.page_content, 90)}"
        )


def main():
    if not INDEX_PKL.exists():
        raise FileNotFoundError(INDEX_PKL)

    audit_docs(load_docs())


if __name__ == "__main__":
    main()
