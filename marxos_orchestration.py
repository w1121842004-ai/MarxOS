from __future__ import annotations


def prepare_query_request(
    query,
    route_query,
    clean_text,
    is_unreadable_query,
    answer_unsupported_claim,
    classify_query,
):
    query = clean_text(query, "")
    route_query = clean_text(route_query or query, "")
    if is_unreadable_query(route_query):
        return {
            "early_answer": (
                "未能读取到可用的中文问题。"
                "如果是在 PowerShell 中通过管道或重定向输入，"
                "请先运行 `chcp 65001`，或在交互式提示中直接输入问题。"
            )
        }

    unsupported_answer = answer_unsupported_claim(route_query)
    if unsupported_answer:
        return {"early_answer": unsupported_answer}

    query_intent = classify_query(route_query)
    if query != route_query and query_intent == "quote_lookup" and "《" not in route_query and "》" not in route_query:
        query_intent = "rag_answer"

    return {
        "query": query,
        "route_query": route_query,
        "query_intent": query_intent,
    }


def maybe_answer_local_lookup(
    query,
    route_query,
    query_intent,
    trace,
    trace_only,
    answer_bibliographic_query,
    extract_bibliographic_title,
    answer_quote_query,
    print_trace_line,
):
    if query_intent == "bibliographic_lookup":
        bibliographic_answer = answer_bibliographic_query(route_query)
        if trace or trace_only:
            print_trace_line("search_path: local article map / core classics")
            print_trace_line(f"bibliographic_answer_found: {bool(bibliographic_answer)}")
            print_trace_line("===== End Trace =====\n")
        if bibliographic_answer:
            return bibliographic_answer
        title = extract_bibliographic_title(route_query)
        return f"未能在当前核心书目表中确认《{title}》。"

    if query_intent == "quote_lookup":
        if trace or trace_only:
            print_trace_line("search_path: exact OCR quote lookup")
        answer = answer_quote_query(query, trace=trace or trace_only)
        if trace or trace_only:
            print_trace_line("===== End Trace =====\n")
        return answer

    return ""


def collect_retrieval_materials(
    query,
    route_query,
    query_intent,
    constraints,
    paragraph_vectorstore_dir,
    trace,
    trace_only,
    topic_info_from_constraints,
    set_last_topic_info,
    print_trace_line,
    print_constraints_trace,
    load_vectorstore,
    retrieve_documents,
    paragraph_vectorstore_exists,
    load_paragraph_vectorstore,
    filter_paragraph_docs_by_text_overlap,
    merge_prefer_paragraph_docs,
    refine_docs_citation_pages_for_query,
    evidence_from_docs,
    is_topic_view_list_query,
):
    set_last_topic_info(topic_info_from_constraints(constraints))
    if trace or trace_only:
        print_trace_line("search_path: FAISS vector similarity search -> rule rerank -> DeepSeek")
        print_constraints_trace(constraints)

    db = load_vectorstore()
    retrieve_k = 12 if query_intent == "rag_answer" else 5
    docs = retrieve_documents(query, db, k=retrieve_k)
    paragraph_docs_for_answer = []
    topic_list_query = query_intent == "rag_answer" and is_topic_view_list_query(route_query, constraints)
    paragraph_store_ready = paragraph_vectorstore_exists()

    if paragraph_store_ready and not topic_list_query:
        paragraph_db_for_answer = load_paragraph_vectorstore()
        paragraph_docs_for_answer = filter_paragraph_docs_by_text_overlap(
            query,
            retrieve_documents(query, paragraph_db_for_answer, k=max(retrieve_k * 3, 12)),
            limit=retrieve_k,
        )
        docs = merge_prefer_paragraph_docs(paragraph_docs_for_answer, docs, retrieve_k)

    docs = refine_docs_citation_pages_for_query(docs, route_query)

    paragraph_docs = []
    if trace or trace_only:
        if paragraph_store_ready:
            paragraph_docs = paragraph_docs_for_answer[:5]
        elif query_intent != "rag_answer" or not topic_list_query:
            print_trace_line(f"paragraph_vectorstore_missing: {paragraph_vectorstore_dir}")

    return {
        "docs": docs,
        "evidence": evidence_from_docs(docs),
        "paragraph_docs": paragraph_docs,
    }


def maybe_answer_local_view_query(
    query,
    route_query,
    query_intent,
    constraints,
    docs,
    evidence,
    set_last_evidence,
    filter_evidence_to_answer,
    audit_answer_citations,
    is_topic_view_list_query,
    build_topic_view_list_answer,
    topic_answer_evidence,
    is_view_list_query,
    build_strict_title_view_list_answer,
    strict_title_answer_evidence,
):
    if constraints.get("strict_title") and not docs:
        title = constraints.get("title") or "该文"
        answer = (
            f"当前语料库未检索到《{title}》的正文页段，因此本轮不输出跨篇替代性引文。"
            "请先补齐该文在本地库中的页段映射或OCR文本后再回答。"
        )
        set_last_evidence([], {"ok": True, "issues": [], "evidence_count": 0, "answer": answer})
        return answer

    if query_intent == "rag_answer" and is_topic_view_list_query(route_query, constraints):
        topic_view_answer = build_topic_view_list_answer(route_query, constraints, evidence, limit=min(10, len(evidence)))
        if topic_view_answer:
            answer_evidence = topic_answer_evidence(evidence, constraints, limit=min(10, len(evidence)))
            display_evidence = filter_evidence_to_answer(
                topic_view_answer,
                answer_evidence,
                fallback_limit=min(10, len(answer_evidence)),
            )
            audit = audit_answer_citations(topic_view_answer, display_evidence)
            set_last_evidence(display_evidence, audit)
            return audit["answer"]

    if query_intent == "rag_answer" and constraints.get("strict_title") and is_view_list_query(route_query):
        title_view_answer = build_strict_title_view_list_answer(
            route_query,
            constraints,
            evidence,
            limit=min(8, len(evidence)),
        )
        if title_view_answer:
            answer_evidence = strict_title_answer_evidence(
                route_query,
                constraints,
                evidence,
                limit=min(8, len(evidence)),
            )
            display_evidence = filter_evidence_to_answer(
                title_view_answer,
                answer_evidence,
                fallback_limit=min(8, len(answer_evidence)),
            )
            audit = audit_answer_citations(title_view_answer, display_evidence)
            set_last_evidence(display_evidence, audit)
            return audit["answer"]

    return ""
