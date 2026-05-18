from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag.core_classics import classic_entries_for_query, load_core_classics
from rag.exact_quote_lookup import exact_quote_lookup
from app import retrieve_documents


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTORSTORE_DIR = Path("vectorstore/marx_reader_core")
ARTICLE_MAP_PATH = Path("rag/article_map_core.json")


@dataclass(frozen=True)
class EvalQuestion:
    group: str
    question: str
    target_title: str | None = None
    expected_sources: tuple[str, ...] = ()
    expected_article_terms: tuple[str, ...] = ()
    expected_content_terms: tuple[str, ...] = ()


@dataclass
class EvalResult:
    index: int
    group: str
    passed: bool
    reason: str


BASE_QUESTIONS = [
    EvalQuestion("core_title", "《共产党宣言》收录在哪一卷，从哪一页开始？", "共产党宣言"),
    EvalQuestion("core_title", "《共产党宣言》收录在哪里？", "共产党宣言"),
    EvalQuestion("core_title", "《资本论》第一卷从哪一页开始？", "资本论"),
    EvalQuestion("core_title", "《哥达纲领批判》收录在哪一卷？", "哥达纲领批判"),
    EvalQuestion("core_title", "《黑格尔法哲学批判》导言收录在哪一卷？", "黑格尔法哲学批判导言"),
    EvalQuestion("core_title", "《1844年经济学哲学手稿》从哪一页开始？", "1844年经济学哲学手稿"),
    EvalQuestion("core_title", "《德意志意识形态》收录在哪一卷？", "德意志意识形态"),
    EvalQuestion("core_title", "《路易·波拿巴的雾月十八日》收录在哪一卷？", "路易·波拿巴的雾月十八日"),
    EvalQuestion("core_title", "《法兰西内战》从哪一页开始？", "法兰西内战"),
    EvalQuestion("core_title", "《反杜林论》收录在哪一卷？", "反杜林论"),
    EvalQuestion("core_title", "《社会主义从空想到科学的发展》收录在哪一卷？", "社会主义从空想到科学的发展"),
    EvalQuestion("core_title", "《家庭、私有制和国家的起源》收录在哪一卷？", "家庭、私有制和国家的起源"),
    EvalQuestion("core_title", "《自然辩证法》从哪一页开始？", "自然辩证法"),
    EvalQuestion("core_title", "《关于费尔巴哈的提纲》收录在哪一卷？", "关于费尔巴哈的提纲"),
    EvalQuestion("core_title", "《雇佣劳动与资本》收录在哪一卷？", "雇佣劳动与资本"),
    EvalQuestion("core_title", "《工资、价格和利润》收录在哪一卷？", "工资、价格和利润"),
    EvalQuestion("core_quote", "“一个幽灵，共产主义的幽灵，在欧洲游荡。”出自哪里？"),
    EvalQuestion("core_quote", "“至今一切社会的历史都是阶级斗争的历史。”出自哪里？"),
    EvalQuestion("core_quote", "“哲学家们只是用不同的方式解释世界，问题在于改变世界。”出自哪一页？"),
    EvalQuestion("core_quote", "“宗教是人民的鸦片。”出自哪里？"),
    EvalQuestion("core_quote", "“劳动首先是人和自然之间的过程。”这句话出自哪里？"),
    EvalQuestion("core_quote", "“资本不是物，而是一定的、社会的、属于一定历史社会形态的生产关系。”出自哪里？"),
    EvalQuestion("core_quote", "“各尽所能，按需分配。”出自哪里？"),
    EvalQuestion("core_quote", "“国家是社会在一定发展阶段上的产物。”出自哪里？"),
    EvalQuestion("core_quote", "“自由是对必然的认识。”出自哪里？"),
    EvalQuestion("core_quote", "“工人没有祖国。”出自哪里？"),
    EvalQuestion("negative", "《一个不存在的马克思著作标题》收录在哪一卷？"),
    EvalQuestion("negative", "请给出“这是一句随便编造的引文”的准确页码。"),
]


CONCEPT_QUESTIONS = [
    EvalQuestion(
        "concept",
        "资本是什么？",
        expected_sources=("mea01.pdf", "mes02.pdf", "mea07.pdf"),
        expected_article_terms=("资本",),
        expected_content_terms=("资本",),
    ),
    EvalQuestion(
        "concept",
        "劳动过程是什么？",
        expected_sources=("mes02.pdf", "mea05.pdf"),
        expected_article_terms=("劳动过程",),
        expected_content_terms=("劳动", "过程"),
    ),
    EvalQuestion(
        "concept",
        "私有制与家庭起源有什么关系？",
        expected_sources=("mea04.pdf", "mes04.pdf"),
        expected_article_terms=("家庭", "私有制", "国家"),
        expected_content_terms=("家庭", "私有制"),
    ),
    EvalQuestion(
        "concept",
        "唯物辩证法应该查哪些原文？",
        expected_sources=("mes03.pdf", "mea09.pdf"),
        expected_article_terms=("反杜林论", "自然辩证法"),
    ),
    EvalQuestion(
        "concept",
        "阶级斗争是什么意思？",
        expected_sources=("mes01.pdf", "mea02.pdf"),
        expected_content_terms=("阶级", "斗争"),
    ),
    EvalQuestion(
        "concept",
        "国家的起源是什么？",
        expected_sources=("mea04.pdf", "mes04.pdf"),
        expected_content_terms=("国家", "起源"),
    ),
    EvalQuestion(
        "concept",
        "剩余价值是什么？",
        expected_sources=("mea05.pdf", "mes02.pdf"),
        expected_content_terms=("剩余价值",),
    ),
    EvalQuestion(
        "concept",
        "异化劳动是什么？",
        expected_sources=("mea01.pdf", "mes01.pdf"),
        expected_article_terms=("异化劳动", "外化劳动"),
        expected_content_terms=("异化", "劳动"),
    ),
    EvalQuestion(
        "concept",
        "费尔巴哈提纲讲实践吗？",
        expected_sources=("mes01.pdf", "mea01.pdf"),
        expected_content_terms=("实践",),
    ),
]


NEGATIVE_QUESTIONS = [
    EvalQuestion("negative", "请定位“马克思在火星殖民地经济学手稿”这篇文章。"),
    EvalQuestion("negative", "《不存在的剩余时间论》收录在哪一卷？"),
    EvalQuestion("negative", "“资本是一只会唱歌的机器”出自哪一页？"),
    EvalQuestion("negative", "请给出《量子共产主义宣言》的准确印刷页。"),
    EvalQuestion("negative", "“阶级斗争已经由机器人自动解决”这句原文在哪？"),
    EvalQuestion("negative", "《恩格斯论互联网平台经济》在哪一卷？"),
]


def generated_core_title_questions() -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []

    for classic in load_core_classics():
        title = classic.get("title", "")
        if not title:
            continue

        questions.extend(
            [
                EvalQuestion("core_title", f"《{title}》在核心库中的准确位置是什么？", title),
                EvalQuestion("core_title", f"{title} 的印刷页范围是多少？", title),
            ]
        )

        aliases = [
            alias for alias in (classic.get("aliases") or [])
            if alias and alias != title
        ]
        if aliases:
            questions.append(
                EvalQuestion("core_title", f"《{aliases[0]}》对应哪一卷和页码？", title)
            )

    return questions


def generated_core_quote_questions() -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []

    for classic in load_core_classics():
        for quote in classic.get("quotes") or []:
            if quote:
                questions.append(EvalQuestion("core_quote", f"请定位原文：“{quote}”。"))

    return questions


def build_questions() -> list[EvalQuestion]:
    questions = (
        BASE_QUESTIONS
        + generated_core_title_questions()
        + generated_core_quote_questions()
        + CONCEPT_QUESTIONS
        + NEGATIVE_QUESTIONS
    )
    seen = set()
    unique_questions = []

    for question in questions:
        key = (question.group, question.question, question.target_title)
        if key in seen:
            continue
        seen.add(key)
        unique_questions.append(question)

    return unique_questions


def normalize_for_match(text: str) -> str:
    text = str(text or "")
    return re.sub(r"[《》“”\"'（）()\[\]［］，。；：、\s·\-.—–]", "", text).lower()


def load_article_map() -> dict:
    with ARTICLE_MAP_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_title_matches(article_map: dict, title: str, limit: int = 5) -> list[tuple[str, dict, dict]]:
    core_entries = classic_entries_for_query(title)
    if core_entries:
        matches = []
        for entry in core_entries[:limit]:
            source = entry["source"]
            payload = article_map.get(source, {"book": source, "entries": []})
            matches.append(
                (
                    source,
                    payload,
                    {
                        "title": entry.get("article") or entry.get("classic_title"),
                        "start_printed_page": entry["start_page"],
                        "end_printed_page": entry["end_page"],
                        "level": 0,
                        "parent": f"core_classic:{entry.get('classic_id')}",
                        "classic_author": entry.get("classic_author"),
                        "classic_work_year": entry.get("classic_work_year"),
                        "classic_work_type": entry.get("classic_work_type"),
                        "entry_type": entry.get("entry_type"),
                    },
                )
            )
        return matches

    normalized_title = normalize_for_match(title)
    matches = []

    for source, payload in article_map.items():
        for entry in payload.get("entries", []):
            normalized_entry = normalize_for_match(entry.get("title", ""))
            if not normalized_entry:
                continue

            if normalized_entry == normalized_title:
                score = 0
            elif normalized_title in normalized_entry or normalized_entry in normalized_title:
                score = 1
            else:
                continue

            suspicious = any(term in entry.get("title", "") for term in ["注释", "索引", "年表", "扉页", "第一页"])
            if suspicious:
                score += 5

            matches.append((score, source, payload, entry))

    matches.sort(
        key=lambda item: (
            item[0],
            item[3].get("level", 9),
            item[3].get("start_printed_page", 99999),
            item[1],
        )
    )

    return [(source, payload, entry) for _, source, payload, entry in matches[:limit]]


def clean_preview(text: str, limit: int = 180) -> str:
    text = " ".join(str(text).split())
    return text[:limit] + ("..." if len(text) > limit else "")


def format_metadata(metadata: dict) -> str:
    fields = [
        ("book", metadata.get("book")),
        ("article", metadata.get("article")),
        ("section", metadata.get("section")),
        ("page", metadata.get("page")),
        ("printed_page", metadata.get("printed_page")),
        ("pdf_page", metadata.get("pdf_page")),
        ("citation_page", metadata.get("citation_page")),
        ("citation_page_type", metadata.get("citation_page_type")),
        ("source", metadata.get("source")),
        ("match_type", metadata.get("match_type")),
        ("confidence", metadata.get("confidence")),
        ("lookup_scope", metadata.get("lookup_scope")),
        ("classic_author", metadata.get("classic_author")),
        ("classic_work_year", metadata.get("classic_work_year")),
        ("classic_work_type", metadata.get("classic_work_type")),
        ("entry_type", metadata.get("entry_type")),
    ]
    return ", ".join(f"{key}={value}" for key, value in fields if value not in (None, ""))


def load_vectorstore() -> FAISS:
    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            f"Vectorstore not found: {VECTORSTORE_DIR}. Build it before running evaluation."
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def page_in_entry(metadata: dict, entry: dict) -> bool:
    if metadata.get("source") != entry["source"]:
        return False

    try:
        page = int(metadata.get("printed_page") or metadata.get("page"))
    except (TypeError, ValueError):
        return False

    return entry["start_page"] <= page <= entry["end_page"]


def rerank_with_core_classic(query: str, docs: list, limit: int) -> list:
    entries = classic_entries_for_query(query)

    if not entries:
        return docs[:limit]

    ranked = []
    for doc in docs:
        score = 0
        for entry in entries:
            if doc.metadata.get("source") == entry["source"]:
                score += 50
            if page_in_entry(doc.metadata, entry):
                score += 100 - entry.get("priority", 99)
        ranked.append((score, doc))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [doc for score, doc in ranked[:limit]]


def title_match_passed(item: EvalQuestion, matches: list[tuple[str, dict, dict]]) -> tuple[bool, str]:
    if not matches:
        return False, "no title matches"

    expected_entries = classic_entries_for_query(item.target_title or item.question)
    if not expected_entries:
        return True, "title match found"

    source, _payload, entry = matches[0]
    expected = expected_entries[0]
    expected_start = expected.get("start_page")
    expected_end = expected.get("end_page")
    actual_start = entry.get("start_printed_page")
    actual_end = entry.get("end_printed_page")

    if source != expected.get("source"):
        return False, f"top source {source}, expected {expected.get('source')}"

    if actual_start != expected_start or actual_end != expected_end:
        return (
            False,
            f"top pages {actual_start}-{actual_end}, expected {expected_start}-{expected_end}",
        )

    return True, f"top source/page ok: {source} {actual_start}-{actual_end}"


def quote_match_passed(docs: list) -> tuple[bool, str]:
    if not docs:
        return False, "no quote results"

    top_metadata = docs[0].metadata
    if top_metadata.get("match_type") != "exact_quote":
        return False, f"top match_type {top_metadata.get('match_type')}"

    if float(top_metadata.get("confidence") or 0) < 1.0:
        return False, f"top confidence {top_metadata.get('confidence')}"

    source = top_metadata.get("source")
    page = top_metadata.get("citation_page") or top_metadata.get("page")
    return True, f"exact quote top hit: {source} page {page}"


def concept_match_passed(item: EvalQuestion, docs: list) -> tuple[bool, str]:
    if not docs:
        return False, "no concept results"

    top_doc = docs[0]
    metadata = top_doc.metadata
    source = metadata.get("source")
    article = str(metadata.get("section") or metadata.get("article") or "")
    content = str(top_doc.page_content or "")
    article_norm = normalize_for_match(article)
    content_norm = normalize_for_match(content)

    if item.expected_sources and source not in item.expected_sources:
        return False, f"top source {source}, expected one of {item.expected_sources}"

    if item.expected_article_terms:
        article_terms = [normalize_for_match(term) for term in item.expected_article_terms]
        if not any(term and term in article_norm for term in article_terms):
            return False, f"top article {article!r} lacks {item.expected_article_terms}"

    if item.expected_content_terms:
        content_terms = [normalize_for_match(term) for term in item.expected_content_terms]
        missing_terms = [
            raw_term for raw_term, term in zip(item.expected_content_terms, content_terms)
            if term and term not in content_norm
        ]
        if missing_terms:
            return False, f"top content lacks {tuple(missing_terms)}"

    page = metadata.get("citation_page") or metadata.get("printed_page") or metadata.get("page")
    return True, f"concept top hit: {source} page {page}"


def print_summary(results: list[EvalResult]) -> None:
    passed = sum(1 for result in results if result.passed)
    total = len(results)
    print("\n===== SUMMARY =====")
    print(f"Passed: {passed}/{total}")

    by_group: dict[str, list[EvalResult]] = {}
    for result in results:
        by_group.setdefault(result.group, []).append(result)

    for group, group_results in by_group.items():
        group_passed = sum(1 for result in group_results if result.passed)
        print(f"{group}: {group_passed}/{len(group_results)}")

    failed = [result for result in results if not result.passed]
    if not failed:
        print("All evaluation cases passed.")
        return

    print("\nFailures:")
    for result in failed:
        print(f"- #{result.index} {result.group}: {result.reason}")


def evaluate(k: int = 3) -> None:
    article_map = load_article_map()
    db = None
    results: list[EvalResult] = []
    questions = build_questions()

    for index, item in enumerate(questions, start=1):
        print(f"\n===== {index}. {item.group} =====")
        print(f"Q: {item.question}")

        if item.group == "core_title":
            matches = find_title_matches(article_map, item.target_title or item.question, limit=k)
            if not matches:
                print("No article-map matches.")
                results.append(EvalResult(index, item.group, False, "no article-map matches"))
                continue

            for rank, (source, payload, entry) in enumerate(matches, start=1):
                print(
                    f"\n[{rank}] book={payload.get('book')}, source={source}, "
                    f"title={entry.get('title')}, pages={entry.get('start_printed_page')}-{entry.get('end_printed_page')}, "
                    f"level={entry.get('level')}, parent={entry.get('parent')}"
                )

            passed, reason = title_match_passed(item, matches)
            results.append(EvalResult(index, item.group, passed, reason))
            continue

        if item.group == "negative":
            exact_docs = exact_quote_lookup(item.question, limit=k)
            if exact_docs:
                print("Unexpected exact quote match.")
                docs = exact_docs
                for rank, doc in enumerate(docs, start=1):
                    print(f"\n[{rank}] {format_metadata(doc.metadata)}")
                    print(clean_preview(doc.page_content))
                results.append(EvalResult(index, item.group, False, "unexpected exact quote match"))
            else:
                print("No trusted answer; vector candidates suppressed.")
                results.append(EvalResult(index, item.group, True, "suppressed"))
            continue

        if db is None:
            db = load_vectorstore()

        if item.group == "concept":
            docs = retrieve_documents(item.question, db, k=k)
            if not docs:
                print("No retrieval results.")
                results.append(EvalResult(index, item.group, False, "no retrieval results"))
                continue

            for rank, doc in enumerate(docs, start=1):
                print(f"\n[{rank}] {format_metadata(doc.metadata)}")
                print(clean_preview(doc.page_content))

            passed, reason = concept_match_passed(item, docs)
            results.append(EvalResult(index, item.group, passed, reason))
            continue

        exact_docs = exact_quote_lookup(item.question, limit=k)
        if exact_docs:
            docs = exact_docs
        else:
            if item.group == "core_quote":
                print("No exact quote match; showing vector candidates only.")
            fetch_k = 120 if classic_entries_for_query(item.question) else k
            docs = db.similarity_search(item.question, k=fetch_k)
            docs = rerank_with_core_classic(item.question, docs, k)
            if item.group == "core_quote":
                for doc in docs:
                    doc.metadata["match_type"] = "vector_candidate"
                    doc.metadata["confidence"] = 0.0
        if not docs:
            print("No retrieval results.")
            results.append(EvalResult(index, item.group, False, "no retrieval results"))
            continue

        for rank, doc in enumerate(docs, start=1):
            print(f"\n[{rank}] {format_metadata(doc.metadata)}")
            print(clean_preview(doc.page_content))

        if item.group == "core_quote":
            passed, reason = quote_match_passed(docs)
            results.append(EvalResult(index, item.group, passed, reason))

    print_summary(results)


if __name__ == "__main__":
    evaluate()
