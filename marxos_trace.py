from __future__ import annotations

import sys


def compact_preview(text, clean_text, limit=180):
    text = " ".join(clean_text(text, "").split())
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def print_trace_line(text=""):
    print(text, file=sys.stderr)


def print_query_trace(query, query_intent):
    print_trace_line("\n===== MarxOS Trace =====")
    print_trace_line(f"query: {query}")
    print_trace_line(f"intent: {query_intent}")


def print_constraints_trace(constraints):
    if not constraints:
        print_trace_line("routing_constraints: none")
        return

    print_trace_line("routing_constraints:")
    print_trace_line(f"- title: {constraints.get('title')}")
    print_trace_line(f"- sources: {sorted(constraints.get('sources') or [])}")
    print_trace_line(f"- page_ranges: {constraints.get('page_ranges') or {}}")


def print_docs_trace(docs, normalize_metadata, format_citation, compact_preview_fn, label="retrieved_docs"):
    print_trace_line(f"{label}: {len(docs)}")

    for index, doc in enumerate(docs, start=1):
        metadata = normalize_metadata(doc.metadata)
        print_trace_line(f"\n[{index}]")
        print_trace_line(
            "metadata: "
            f"book={metadata.get('book')}, article={metadata.get('article')}, "
            f"section={metadata.get('section')}, source={metadata.get('source')}, "
            f"page={metadata.get('page')}, printed_page={metadata.get('printed_page')}, "
            f"pdf_page={metadata.get('pdf_page')}, citation_page={metadata.get('citation_page')}, "
            f"citation_page_type={metadata.get('citation_page_type')}"
        )
        print_trace_line(
            "standard_metadata: "
            f"series={metadata.get('series')}, volume={metadata.get('volume')}, "
            f"publisher={metadata.get('publisher')}, publication_year={metadata.get('publication_year')}, "
            f"source_file={metadata.get('source_file')}"
        )
        if metadata.get("match_type"):
            print_trace_line(
                f"match: type={metadata.get('match_type')}, confidence={metadata.get('confidence')}, "
                f"lookup_scope={metadata.get('lookup_scope')}"
            )
        print_trace_line(f"sentence_citation: {format_citation(metadata, include_article=False)}")
        print_trace_line(f"paragraph_citation: {format_citation(metadata, include_article=True)}")
        print_trace_line(f"preview: {compact_preview_fn(doc.page_content)}")


def print_prompt_trace(prompt, compact_preview_fn):
    print_trace_line("\nprompt_preview:")
    print_trace_line(compact_preview_fn(prompt, limit=500))
    print_trace_line("===== End Trace =====\n")


def build_trace_only_answer(query_intent, docs, prompt, normalize_metadata, compact_preview_fn, paragraph_docs=None):
    paragraph_docs = paragraph_docs or []
    lines = [
        "已完成 TRACE_ONLY 调试运行，未调用 DeepSeek。",
        f"intent: {query_intent}",
        f"retrieved_docs: {len(docs)}",
        "",
        "Top chunks:",
    ]

    for index, doc in enumerate(docs, start=1):
        metadata = normalize_metadata(doc.metadata)
        lines.append(
            f"{index}. source={metadata.get('source')}, article={metadata.get('article')}, "
            f"page={metadata.get('page')}, pdf_page={metadata.get('pdf_page')}, "
            f"citation_page={metadata.get('citation_page')}, type={metadata.get('citation_page_type')}"
        )
        lines.append(f"   preview: {compact_preview_fn(doc.page_content, limit=120)}")

    if paragraph_docs:
        lines.extend(["", "Top paragraphs:"])
        for index, doc in enumerate(paragraph_docs, start=1):
            metadata = normalize_metadata(doc.metadata)
            lines.append(
                f"{index}. source={metadata.get('source')}, article={metadata.get('article')}, "
                f"page={metadata.get('page')}, pdf_page={metadata.get('pdf_page')}, "
                f"citation_page={metadata.get('citation_page')}, type={metadata.get('citation_page_type')}"
            )
            lines.append(f"   preview: {compact_preview_fn(doc.page_content, limit=120)}")

    lines.extend(["", "Prompt preview:", compact_preview_fn(prompt, limit=700)])
    return "\n".join(lines)
