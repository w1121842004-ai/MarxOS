from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
VECTORSTORE_PKL = ROOT / "vectorstore" / "marx_reader_core" / "index.pkl"
DEFAULT_OUTPUT = ROOT / "eval_dataset_me_200.json"

QUESTION_TYPES = [
    ("引文出处类", "给出一句原文或近似表述，要求回答出处、含义和上下文。"),
    ("原文解释类", "要求解释片段中的关键概念或论证。"),
    ("理论分析类", "围绕一个理论问题展开分析，必须依据片段。"),
    ("综合比较类", "把多个片段或多个主题联系起来综合说明。"),
    ("现实阐释类", "要求在不脱离原著的前提下，用原理分析现实问题。"),
    ("论文写作类", "要求给出论文题目、论点、结构或一段学术性写作。"),
    ("辨析反驳类", "要求辨析常见误读，并依据原文作出反驳。"),
    ("研究综述类", "要求围绕主题形成可用于研究的概括性答案。"),
]


def clean_text(text: str, limit: int = 620) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text[:limit]


def load_documents() -> list:
    if not VECTORSTORE_PKL.exists():
        raise FileNotFoundError(f"vectorstore pkl not found: {VECTORSTORE_PKL}")

    with VECTORSTORE_PKL.open("rb") as f:
        docstore, _index_to_id = pickle.load(f)

    docs = list(docstore._dict.values())
    docs = [
        doc
        for doc in docs
        if str((doc.metadata or {}).get("source") or "").startswith("me")
        and len(clean_text(doc.page_content, limit=10000)) >= 160
    ]
    if not docs:
        raise RuntimeError("no usable Marx/Engels documents found in vectorstore")
    return docs


def source_label(doc) -> str:
    meta = doc.metadata or {}
    parts = [
        str(meta.get("source") or ""),
        str(meta.get("article") or meta.get("section") or "").strip(),
    ]
    page = meta.get("citation_page") or meta.get("printed_page") or meta.get("pdf_page")
    if page is not None:
        parts.append(f"页码:{page}")
    return " | ".join(part for part in parts if part)


def pick_excerpt_cards(docs: list, batch_index: int, cards_per_batch: int) -> list[dict]:
    rng = random.Random(20260615 + batch_index)
    by_source: dict[str, list] = {}
    for doc in docs:
        source = (doc.metadata or {}).get("source") or "unknown"
        by_source.setdefault(source, []).append(doc)

    sources = sorted(by_source)
    rng.shuffle(sources)
    selected = []
    for source in sources:
        candidates = by_source[source]
        doc = rng.choice(candidates)
        selected.append(
            {
                "source": source_label(doc),
                "text": clean_text(doc.page_content),
            }
        )
        if len(selected) >= cards_per_batch:
            break

    while len(selected) < cards_per_batch:
        doc = rng.choice(docs)
        selected.append({"source": source_label(doc), "text": clean_text(doc.page_content)})

    return selected


def build_prompt(batch_index: int, count: int, existing_queries: list[str], cards: list[dict]) -> str:
    type_lines = "\n".join(f"- {name}: {desc}" for name, desc in QUESTION_TYPES)
    excerpt_lines = "\n\n".join(
        f"[EXCERPT-{i}]\n出处: {card['source']}\n原文: {card['text']}"
        for i, card in enumerate(cards, start=1)
    )
    existing = "\n".join(f"- {query}" for query in existing_queries[-60:])
    return f"""
你是一名马克思恩格斯经典著作研究者，正在为本地 RAG 系统构造高质量测评集。

请只依据下面给出的马恩全集原文材料，生成 {count} 道中文企业级测评题。题型要覆盖：
{type_lines}

硬性要求：
1. 输出必须是严格 JSON 对象，格式为 {{"items":[...]}}，不要输出 Markdown。
2. 每个 item 必须包含且只包含以下字段：
   id, query, answer, question_type, difficulty, discipline, evaluation_mode, source_scope,
   expected_work, expected_author, expected_citations, evaluation_points, rubric, tags
3. id 使用本批次内临时编号，例如 "draft_{batch_index}_01"；最终脚本会重写为稳定 id。
4. question_type 只能使用：
   quote_lookup, source_context, concept_explain, analysis, synthesis,
   paper_writing, citation_verification, misconception_rebuttal, research_review
5. difficulty 只能使用：easy, medium, hard。
6. discipline 只能使用：
   philosophy, political_economy, scientific_socialism, history,
   politics_state, party_labor, letters, methodology, culture_religion
7. evaluation_mode 只能使用：
   strict_citation, source_required, source_preferred, judge_only
   quote_lookup/source_context/citation_verification 使用 strict_citation；
   concept_explain/analysis/misconception_rebuttal 使用 source_required；
   synthesis/research_review/paper_writing 使用 source_preferred。
8. source_scope 是字符串数组，至少包含一个 PDF 文件名，例如 ["me23.pdf"]。
9. expected_citations 是数组，每项包含：
   source, article, citation_page, quote
   citation_page 不确定时可为 null；quote 必须是材料中的短片段，不要超过 50 个汉字。
10. evaluation_points 是 3-6 条评分点。
11. rubric 是对象，必须包含 retrieval, citation, reasoning, faithfulness 四个键。
12. tags 是 2-6 个短标签。
13. query 必须像真实用户会问的问题，严禁出现“材料1”“材料2”“材料中”“根据材料”“上述材料”“片段”等字样。
14. answer 必须结合原文作答，允许概括，但不得编造材料之外的具体出处。
15. answer 里应尽量点明所依据的卷册/篇名/页码信息；如果材料未给篇名，就写 PDF 来源和页码。
16. 引文类问题的 answer 要给出出处和简短解释；论文写作类问题的 answer 可以给出论点和提纲。
17. 避免与已有问题重复，避免空泛教科书式问答。
18. 本批次编号是 {batch_index}，请让问题主题分散。

已有问题片段，避免重复：
{existing or "无"}

原文材料：
{excerpt_lines}
""".strip()


REQUIRED_FIELDS = [
    "id",
    "query",
    "answer",
    "question_type",
    "difficulty",
    "discipline",
    "evaluation_mode",
    "source_scope",
    "expected_work",
    "expected_author",
    "expected_citations",
    "evaluation_points",
    "rubric",
    "tags",
]

QUESTION_TYPE_VALUES = {
    "quote_lookup",
    "source_context",
    "concept_explain",
    "analysis",
    "synthesis",
    "paper_writing",
    "citation_verification",
    "misconception_rebuttal",
    "research_review",
}
DIFFICULTY_VALUES = {"easy", "medium", "hard"}
DISCIPLINE_VALUES = {
    "philosophy",
    "political_economy",
    "scientific_socialism",
    "history",
    "politics_state",
    "party_labor",
    "letters",
    "methodology",
    "culture_religion",
}
EVALUATION_MODE_VALUES = {
    "strict_citation",
    "source_required",
    "source_preferred",
    "judge_only",
}
STRICT_TYPES = {"quote_lookup", "source_context", "citation_verification"}
SOURCE_REQUIRED_TYPES = {"concept_explain", "analysis", "misconception_rebuttal"}
SOURCE_PREFERRED_TYPES = {"synthesis", "research_review", "paper_writing"}
BAD_QUERY_MARKERS = (
    "材料",
    "根据材料",
    "上述材料",
    "片段",
    "摘录",
    "EXCERPT",
    "excerpt",
)


def default_evaluation_mode(question_type: str) -> str:
    if question_type in STRICT_TYPES:
        return "strict_citation"
    if question_type in SOURCE_REQUIRED_TYPES:
        return "source_required"
    if question_type in SOURCE_PREFERRED_TYPES:
        return "source_preferred"
    return "source_required"


def normalize_item(item: dict, fallback_id: str) -> dict | None:
    query = str(item.get("query") or "").strip()
    answer = str(item.get("answer") or "").strip()
    if not query or not answer:
        return None
    if any(marker in query for marker in BAD_QUERY_MARKERS):
        return None

    question_type = str(item.get("question_type") or "analysis").strip()
    if question_type not in QUESTION_TYPE_VALUES:
        question_type = "analysis"

    difficulty = str(item.get("difficulty") or "medium").strip()
    if difficulty not in DIFFICULTY_VALUES:
        difficulty = "medium"

    discipline = str(item.get("discipline") or "philosophy").strip()
    if discipline not in DISCIPLINE_VALUES:
        discipline = "philosophy"

    evaluation_mode = str(item.get("evaluation_mode") or "").strip()
    if evaluation_mode not in EVALUATION_MODE_VALUES:
        evaluation_mode = default_evaluation_mode(question_type)

    source_scope = item.get("source_scope")
    if not isinstance(source_scope, list):
        source_scope = []
    source_scope = [str(value).strip() for value in source_scope if str(value).strip()]

    citations = item.get("expected_citations")
    if not isinstance(citations, list):
        citations = []
    normalized_citations = []
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        source = str(citation.get("source") or "").strip()
        if source and source not in source_scope:
            source_scope.append(source)
        page = citation.get("citation_page")
        try:
            page = int(page) if page not in (None, "") else None
        except (TypeError, ValueError):
            page = None
        normalized_citations.append(
            {
                "source": source,
                "article": str(citation.get("article") or "").strip(),
                "citation_page": page,
                "quote": str(citation.get("quote") or "").strip()[:80],
            }
        )

    evaluation_points = item.get("evaluation_points")
    if not isinstance(evaluation_points, list):
        evaluation_points = []
    evaluation_points = [
        str(point).strip()
        for point in evaluation_points
        if str(point).strip()
    ][:6]

    rubric = item.get("rubric")
    if not isinstance(rubric, dict):
        rubric = {}
    rubric = {
        "retrieval": str(rubric.get("retrieval") or "应检索到 source_scope 或 expected_citations 指向的原文材料。").strip(),
        "citation": str(rubric.get("citation") or "引用信息应与 expected_citations 基本一致。").strip(),
        "reasoning": str(rubric.get("reasoning") or "应围绕问题展开解释、分析或综合，而非只罗列概念。").strip(),
        "faithfulness": str(rubric.get("faithfulness") or "不得引入原文材料无法支持的判断。").strip(),
    }

    tags = item.get("tags")
    if not isinstance(tags, list):
        tags = []
    tags = [str(tag).strip() for tag in tags if str(tag).strip()][:6]

    return {
        "id": str(item.get("id") or fallback_id).strip() or fallback_id,
        "query": query,
        "answer": answer,
        "question_type": question_type,
        "difficulty": difficulty,
        "discipline": discipline,
        "evaluation_mode": evaluation_mode,
        "source_scope": source_scope,
        "expected_work": str(item.get("expected_work") or "").strip(),
        "expected_author": str(item.get("expected_author") or "").strip(),
        "expected_citations": normalized_citations,
        "evaluation_points": evaluation_points,
        "rubric": rubric,
        "tags": tags,
    }


def parse_items(raw: str, id_prefix: str) -> list[dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
    data = json.loads(raw)
    items = data["items"] if isinstance(data, dict) else data
    cleaned = []
    for index, item in enumerate(items, start=1):
        normalized = normalize_item(item, f"{id_prefix}_{index:03d}")
        if normalized:
            cleaned.append(normalized)
    return cleaned


def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected list JSON at {path}")
    items = []
    for index, item in enumerate(data, start=1):
        normalized = normalize_item(item, f"me_eval_{index:03d}")
        if normalized:
            items.append(normalized)
    return items


def save_items(path: Path, items: list[dict]) -> None:
    for index, item in enumerate(items, start=1):
        item["id"] = f"me_eval_{index:03d}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a 200-item Marx/Engels QA eval dataset.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--cards-per-batch", type=int, default=14)
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    parser.add_argument("--overwrite", action="store_true", help="Ignore any existing output and regenerate from scratch.")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output

    items = [] if args.overwrite else load_existing(output)
    docs = load_documents()
    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )

    seen_queries = {item["query"] for item in items}
    batch_index = len(items) // max(args.batch_size, 1)
    print(f"loaded docs={len(docs)} existing_items={len(items)} output={output}", flush=True)

    while len(items) < args.target:
        batch_index += 1
        remaining = args.target - len(items)
        count = min(args.batch_size, remaining)
        cards = pick_excerpt_cards(docs, batch_index, args.cards_per_batch)
        prompt = build_prompt(batch_index, count, [item["query"] for item in items], cards)

        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": "你只输出严格 JSON，不输出 Markdown，不输出解释。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.45,
            max_tokens=6000,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        new_items = parse_items(raw, f"draft_{batch_index}")

        added = 0
        for item in new_items:
            if item["query"] in seen_queries:
                continue
            items.append(item)
            seen_queries.add(item["query"])
            added += 1
            if len(items) >= args.target:
                break

        save_items(output, items)
        print(
            f"batch={batch_index} requested={count} parsed={len(new_items)} added={added} total={len(items)}",
            flush=True,
        )
        if added == 0:
            raise RuntimeError("model returned no new usable items")

    save_items(output, items[: args.target])
    print(f"done: {output} items={args.target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
