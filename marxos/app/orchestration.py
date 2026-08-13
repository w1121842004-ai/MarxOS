from __future__ import annotations

import json
import sys
import time

from langchain_core.documents import Document

from marxos import query_planner

from langchain_core.documents import Document


def _doc_page_anchor(doc):
    metadata = doc.metadata or {}
    return (
        metadata.get("printed_page")
        or metadata.get("citation_page")
        or metadata.get("pdf_page")
        or metadata.get("page")
    )


def _doc_identity(doc):
    metadata = doc.metadata or {}
    content = str(doc.page_content or "").strip()
    return (
        metadata.get("source"),
        metadata.get("paragraph_id") or metadata.get("pdf_page") or metadata.get("page"),
        metadata.get("printed_page") or metadata.get("citation_page"),
        metadata.get("article") or metadata.get("section"),
        content[:120],
    )


def _tag_docs(docs, path):
    tagged = []
    for doc in docs or []:
        metadata = dict(doc.metadata or {})
        metadata.setdefault("crag_path", path)
        tagged.append(Document(page_content=doc.page_content, metadata=metadata))
    return tagged


def _merge_ranked_docs(*doc_groups):
    merged = []
    seen = set()
    for docs in doc_groups:
        for doc in docs or []:
            key = _doc_identity(doc)
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)
    return merged


def assess_retrieval_quality(query_intent, docs, evidence, constraints, strategy=None):
    docs = docs or []
    evidence = evidence or []
    issues = []
    score = 0

    if not docs:
        issues.append("no_docs")
        return {"ok": False, "score": 0, "issues": issues}

    score += min(len(docs), 5) * 10
    score += min(len(evidence), 4) * 8

    anchored_docs = [doc for doc in docs if _doc_page_anchor(doc) is not None]
    if anchored_docs:
        score += min(len(anchored_docs), 4) * 7
    else:
        issues.append("no_page_anchor")

    match_types = {str((doc.metadata or {}).get("match_type") or "") for doc in docs}
    if "exact_quote" in match_types:
        score += 25
    if "cache_backstop" in match_types:
        score += 8
    if "paragraph_vector_candidate" in match_types:
        score += 6

    locator_only = bool(docs) and match_types.issubset({"locator_backstop"})
    if locator_only:
        score -= 35
        issues.append("locator_only")

    sources = {
        (doc.metadata or {}).get("source")
        for doc in docs
        if (doc.metadata or {}).get("source")
    }
    if constraints.get("topic_id") or constraints.get("min_distinct_sources"):
        if len(sources) >= max(2, int(constraints.get("min_distinct_sources") or 0)):
            score += 12
        else:
            issues.append("insufficient_source_diversity")

    if constraints.get("strict_title"):
        strict_hits = [
            doc for doc in docs
            if (doc.metadata or {}).get("classic_title")
            or (doc.metadata or {}).get("locator_title")
        ]
        if strict_hits:
            score += 10
        else:
            issues.append("missing_strict_title_binding")

    if query_intent == "quote_lookup" and "exact_quote" not in match_types:
        score -= 20
        issues.append("missing_exact_quote_hit")

    if query_intent == "concept_explain" and not anchored_docs:
        score -= 10

    threshold = 45
    if query_intent == "quote_lookup":
        threshold = 55
    elif constraints.get("strict_title") or constraints.get("topic_id"):
        threshold = 52
    # Allow strategy to override the threshold (e.g. comparison lowers it)
    if strategy and getattr(strategy, "crag_threshold_override", None) is not None:
        threshold = strategy.crag_threshold_override

    return {
        "ok": score >= threshold and not locator_only,
        "score": score,
        "issues": issues,
        "threshold": threshold,
        "source_count": len(sources),
        "anchored_count": len(anchored_docs),
    }


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
    if query_intent == "chitchat":
        return {
            "query": query,
            "route_query": route_query,
            "query_intent": query_intent,
            "early_answer": answer_chitchat_query(route_query),
        }
    if query != route_query and query_intent == "quote_lookup" and "《" not in route_query and "》" not in route_query:
        query_intent = "rag_answer"

    return {
        "query": query,
        "route_query": route_query,
        "query_intent": query_intent,
    }


def is_chitchat_query(query):
    normalized = "".join(str(query or "").strip().lower().split())
    if not normalized:
        return False
    greetings = {
        "你好",
        "您好",
        "哈喽",
        "嗨",
        "hi",
        "hello",
        "hey",
        "在吗",
        "在不在",
        "早上好",
        "上午好",
        "中午好",
        "下午好",
        "晚上好",
    }
    identity_queries = {
        "你是谁",
        "你是啥",
        "你是什么",
        "你能做什么",
        "你可以做什么",
        "你可以作甚",
        "你能作甚",
        "你作甚",
        "你能干什么",
        "你可以干什么",
        "你能干啥",
        "你可以干啥",
        "你能做啥",
        "你可以做啥",
        "你有什么用",
        "你有什么功能",
        "你有哪些功能",
        "你会什么",
        "介绍一下你自己",
    }
    return normalized in greetings or normalized in identity_queries


def answer_chitchat_query(query):
    normalized = "".join(str(query or "").strip().lower().split())
    if normalized in {"你是谁", "你是啥", "你是什么", "介绍一下你自己"}:
        return "我是 MarxOS，一个面向马克思、恩格斯经典文本的本地检索问答助手。"
    if normalized in {
        "你能做什么",
        "你可以做什么",
        "你可以作甚",
        "你能作甚",
        "你作甚",
        "你能干什么",
        "你可以干什么",
        "你能干啥",
        "你可以干啥",
        "你能做啥",
        "你可以做啥",
        "你有什么用",
        "你有什么功能",
        "你有哪些功能",
        "你会什么",
    }:
        return "我可以帮助你查找原著出处、解释马克思主义概念、梳理文本论证，并围绕经典文献做学术分析。"
    return "你好。我是 MarxOS，可以帮你查找原著出处、解释概念或分析马克思主义经典文本。"


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
    force_corrective=False,
    query_plan=None,
    performance=None,
    strategy=None,
):
    performance = performance or {}

    # Apply intent strategy overrides to performance dict (NEW)
    if strategy:
        from marxos.config.retrieval_strategies import apply_strategy
        performance = apply_strategy(performance, strategy)

    retrieval_started = time.perf_counter()

    def log_retrieval_phase(phase, started, **extra):
        payload = {
            "event": "marxos_retrieval_timing",
            "phase": phase,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "total_ms": int((time.perf_counter() - retrieval_started) * 1000),
        }
        payload.update(extra)
        try:
            print(json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)
        except BrokenPipeError:
            pass
        except UnicodeEncodeError:
            try:
                print(json.dumps(payload, ensure_ascii=True), file=sys.stderr, flush=True)
            except BrokenPipeError:
                pass

    def retrieve_documents_for_mode(query_text, db, k, variant_retrieval=False):
        try:
            return retrieve_documents(
                query_text,
                db,
                k=k,
                performance=performance,
                strategy=strategy,
                variant_retrieval=variant_retrieval,
            )
        except TypeError as exc:
            if "performance" not in str(exc) and "strategy" not in str(exc) and "variant_retrieval" not in str(exc):
                raise
            return retrieve_documents(query_text, db, k=k)

    def doc_key(doc):
        metadata = doc.metadata or {}
        return (
            metadata.get("source"),
            metadata.get("page"),
            metadata.get("printed_page"),
            metadata.get("citation_page"),
            metadata.get("article") or metadata.get("section"),
            str(doc.page_content or "")[:120],
        )

    def rrf_merge_doc_lists(doc_lists, limit):
        ranked = {}
        order = {}
        for list_index, docs_for_query in enumerate(doc_lists):
            for rank, doc in enumerate(docs_for_query, start=1):
                key = doc_key(doc)
                if key not in ranked:
                    ranked[key] = {
                        "score": 0.0,
                        "doc": Document(page_content=doc.page_content, metadata=dict(doc.metadata or {})),
                    }
                    order[key] = len(order)
                ranked[key]["score"] += 1 / (60 + rank)
                metadata = ranked[key]["doc"].metadata
                metadata["planner_rrf_score"] = round(ranked[key]["score"], 6)
                metadata.setdefault("planner_first_list", list_index)
                metadata.setdefault("planner_first_rank", rank)
        merged = sorted(
            ranked.values(),
            key=lambda item: (item["score"], -order[doc_key(item["doc"])]),
            reverse=True,
        )
        return [item["doc"] for item in merged[:limit]]

    def retrieve_with_plan(db, k):
        if not query_plan or query_intent in {"quote_lookup", "bibliographic_lookup"}:
            return _tag_docs(retrieve_documents_for_mode(query, db, k), "initial_chunk")

        retrieval_queries = query_plan.get("retrieval_queries") or [query]
        if not performance.get("planner_multi_query", True) or len(retrieval_queries) <= 1:
            return _tag_docs(retrieve_documents_for_mode(retrieval_queries[0], db, k), "initial_chunk")

        per_query_k = max(k, min(k * 2, 16))
        doc_lists = []
        for index, variant in enumerate(retrieval_queries):
            doc_lists.append(
                retrieve_documents_for_mode(
                    variant,
                    db,
                    per_query_k,
                    variant_retrieval=index > 0,
                )
            )
        return _tag_docs(rrf_merge_doc_lists(doc_lists, limit=max(k * 2, 12)), "planner_rrf_chunk")[:k]

    def build_state(selected_docs, selected_paragraph_docs, report, path):
        phase_started = time.perf_counter()
        if performance.get("citation_page_refinement", True):
            selected_docs = refine_docs_citation_pages_for_query(selected_docs, route_query)
        log_retrieval_phase(
            f"{path}_refine_citation_pages",
            phase_started,
            enabled=bool(performance.get("citation_page_refinement", True)),
            doc_count=len(selected_docs or []),
        )
        phase_started = time.perf_counter()
        selected_evidence = evidence_from_docs(selected_docs)
        log_retrieval_phase(
            f"{path}_evidence_from_docs",
            phase_started,
            evidence_count=len(selected_evidence or []),
        )
        return {
            "docs": selected_docs,
            "evidence": selected_evidence,
            "paragraph_docs": selected_paragraph_docs[:5] if (trace or trace_only) else [],
            "crag_report": {**report, "path": path},
        }

    set_last_topic_info(topic_info_from_constraints(constraints))
    if trace or trace_only:
        print_trace_line("search_path: vector similarity search -> rule rerank -> DeepSeek")
        print_constraints_trace(constraints)

    phase_started = time.perf_counter()
    db = load_vectorstore()
    log_retrieval_phase("load_vectorstore", phase_started)
    topic_list_query = query_intent != "deep_analysis" and is_topic_view_list_query(route_query, constraints)
    retrieve_k = int(
        performance.get(
            "rag_retrieve_k" if query_intent == "rag_answer" else "retrieve_k",
            12 if query_intent == "rag_answer" else 5,
        )
    )
    if topic_list_query:
        retrieve_k = max(retrieve_k, 10)
    phase_started = time.perf_counter()
    docs = retrieve_with_plan(db, retrieve_k)
    log_retrieval_phase(
        "initial_chunk_retrieve",
        phase_started,
        doc_count=len(docs or []),
        retrieve_k=retrieve_k,
        multi_query=bool(performance.get("planner_multi_query", True)),
    )
    paragraph_docs_for_answer = []
    phase_started = time.perf_counter()
    paragraph_store_ready = paragraph_vectorstore_exists()
    log_retrieval_phase("paragraph_store_check", phase_started, ready=bool(paragraph_store_ready))

    if performance.get("paragraph_retrieval", True) and paragraph_store_ready and not topic_list_query:
        phase_started = time.perf_counter()
        paragraph_db_for_answer = load_paragraph_vectorstore()
        paragraph_docs_for_answer = filter_paragraph_docs_by_text_overlap(
            query,
            retrieve_documents_for_mode(
                query_plan.get("standalone_query", query) if query_plan else query,
                paragraph_db_for_answer,
                max(retrieve_k * 3, 12),
            ),
            limit=retrieve_k,
        )
        paragraph_docs_for_answer = _tag_docs(paragraph_docs_for_answer, "initial_paragraph")
        docs = merge_prefer_paragraph_docs(paragraph_docs_for_answer, docs, retrieve_k)
        log_retrieval_phase(
            "initial_paragraph_retrieve",
            phase_started,
            paragraph_count=len(paragraph_docs_for_answer or []),
            doc_count=len(docs or []),
        )

    phase_started = time.perf_counter()
    phase_started = time.perf_counter()
    initial_evidence_for_assess_docs = docs
    if performance.get("citation_page_refinement", True):
        initial_evidence_for_assess_docs = refine_docs_citation_pages_for_query(docs, route_query)
    initial_evidence_for_assess = evidence_from_docs(initial_evidence_for_assess_docs)
    log_retrieval_phase(
        "initial_assess_materials",
        phase_started,
        refinement_enabled=bool(performance.get("citation_page_refinement", True)),
        evidence_count=len(initial_evidence_for_assess or []),
    )

    initial_state = build_state(
        docs,
        paragraph_docs_for_answer,
        assess_retrieval_quality(
            query_intent,
            docs,
            initial_evidence_for_assess,
            constraints,
            strategy=strategy,
        ),
        "initial",
    )
    log_retrieval_phase(
        "initial_assess",
        phase_started,
        score=int(initial_state["crag_report"].get("score") or 0),
        ok=bool(initial_state["crag_report"].get("ok", False)),
    )

    best_state = initial_state
    report = initial_state["crag_report"]

    should_correct = performance.get("corrective_retrieval", True) and (
        force_corrective or (
            not report.get("ok")
            and query_intent not in {"bibliographic_lookup", "quote_lookup"}
        ) or (
            query_intent == "quote_lookup" and not report.get("ok")
        )
    )

    if should_correct:
        phase_started = time.perf_counter()
        corrective_chunk_docs = _tag_docs(
            retrieve_documents_for_mode(route_query, db, max(retrieve_k * 2, 8)),
            "corrective_route_chunk",
        )
        log_retrieval_phase(
            "corrective_chunk_retrieve",
            phase_started,
            doc_count=len(corrective_chunk_docs or []),
        )
        corrective_paragraph_docs = []
        if paragraph_store_ready and not topic_list_query:
            phase_started = time.perf_counter()
            paragraph_db_for_answer = paragraph_db_for_answer if paragraph_docs_for_answer else load_paragraph_vectorstore()
            corrective_paragraph_docs = filter_paragraph_docs_by_text_overlap(
                route_query,
                retrieve_documents_for_mode(route_query, paragraph_db_for_answer, max(retrieve_k * 4, 16)),
                limit=max(retrieve_k, 8),
            )
            corrective_paragraph_docs = _tag_docs(corrective_paragraph_docs, "corrective_paragraph")
            log_retrieval_phase(
                "corrective_paragraph_retrieve",
                phase_started,
                paragraph_count=len(corrective_paragraph_docs or []),
            )

        phase_started = time.perf_counter()
        merged_docs = _merge_ranked_docs(
            corrective_paragraph_docs,
            corrective_chunk_docs,
            paragraph_docs_for_answer,
            docs,
        )
        if corrective_paragraph_docs:
            merged_docs = merge_prefer_paragraph_docs(corrective_paragraph_docs, merged_docs, max(retrieve_k * 2, 10))
        else:
            merged_docs = merged_docs[: max(retrieve_k * 2, 10)]

        phase_started = time.perf_counter()
        corrective_evidence_for_assess_docs = merged_docs
        if performance.get("citation_page_refinement", True):
            corrective_evidence_for_assess_docs = refine_docs_citation_pages_for_query(merged_docs, route_query)
        corrective_evidence_for_assess = evidence_from_docs(corrective_evidence_for_assess_docs)
        log_retrieval_phase(
            "corrective_assess_materials",
            phase_started,
            refinement_enabled=bool(performance.get("citation_page_refinement", True)),
            evidence_count=len(corrective_evidence_for_assess or []),
        )

        corrective_state = build_state(
            merged_docs,
            corrective_paragraph_docs or paragraph_docs_for_answer,
            assess_retrieval_quality(
                query_intent,
                merged_docs,
                corrective_evidence_for_assess,
                constraints,
                strategy=strategy,
            ),
            "forced_corrective" if force_corrective else "corrective",
        )
        log_retrieval_phase(
            "corrective_assess",
            phase_started,
            score=int(corrective_state["crag_report"].get("score") or 0),
            ok=bool(corrective_state["crag_report"].get("ok", False)),
            doc_count=len(merged_docs or []),
        )
        if corrective_state["crag_report"]["score"] >= best_state["crag_report"]["score"]:
            best_state = corrective_state
            report = corrective_state["crag_report"]

    if trace or trace_only:
        if report.get("path") == "corrective":
            print_trace_line(
                f"crag: corrective retrieval engaged (score={report.get('score')}, issues={','.join(report.get('issues') or []) or 'none'})"
            )
        else:
            print_trace_line(
                f"crag: initial retrieval accepted (score={report.get('score')}, issues={','.join(report.get('issues') or []) or 'none'})"
            )
        if not paragraph_store_ready and (query_intent != "rag_answer" or not topic_list_query):
            print_trace_line(f"paragraph_vectorstore_missing: {paragraph_vectorstore_dir}")

    return best_state


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
    # deep_analysis: always skip local shortcuts → go to LLM
    if query_intent == "deep_analysis":
        return None

    if constraints.get("strict_title") and not docs:
        title = constraints.get("title") or "该文"
        answer = (
            f"当前语料库未检索到《{title}》的正文页段，因此本轮不输出跨篇替代性引文。"
            "请先补齐该文在本地库中的页段映射或OCR文本后再回答。"
        )
        set_last_evidence([], {"ok": True, "issues": [], "evidence_count": 0, "answer": answer})
        return answer

    if query_intent != "deep_analysis" and is_topic_view_list_query(route_query, constraints):
        topic_view_answer = build_topic_view_list_answer(route_query, constraints, evidence, limit=min(10, len(evidence)))
        if topic_view_answer:
            answer_evidence = evidence[: min(10, len(evidence))]
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
