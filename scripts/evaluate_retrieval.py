from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTORSTORE_DIR = Path("vectorstore/marx_knowledge_base")
ARTICLE_MAP_PATH = Path("rag/article_map.json")


@dataclass(frozen=True)
class EvalQuestion:
    group: str
    question: str
    target_title: str | None = None


QUESTIONS = [
    EvalQuestion("core_title", "《共产党宣言》收录在哪一卷，从哪一页开始？", "共产党宣言"),
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


def normalize_for_match(text: str) -> str:
    text = str(text or "")
    return re.sub(r"[《》“”\"'（）()\[\]［］，。；：、\s·\-.—–]", "", text).lower()


def load_article_map() -> dict:
    with ARTICLE_MAP_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_title_matches(article_map: dict, title: str, limit: int = 5) -> list[tuple[str, dict, dict]]:
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
        ("source", metadata.get("source")),
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


def evaluate(k: int = 3) -> None:
    article_map = load_article_map()
    db = None

    for index, item in enumerate(QUESTIONS, start=1):
        print(f"\n===== {index}. {item.group} =====")
        print(f"Q: {item.question}")

        if item.group == "core_title":
            matches = find_title_matches(article_map, item.target_title or item.question, limit=k)
            if not matches:
                print("No article-map matches.")
                continue

            for rank, (source, payload, entry) in enumerate(matches, start=1):
                print(
                    f"\n[{rank}] book={payload.get('book')}, source={source}, "
                    f"title={entry.get('title')}, pages={entry.get('start_printed_page')}-{entry.get('end_printed_page')}, "
                    f"level={entry.get('level')}, parent={entry.get('parent')}"
                )
            continue

        if db is None:
            db = load_vectorstore()

        docs = db.similarity_search(item.question, k=k)
        if not docs:
            print("No retrieval results.")
            continue

        for rank, doc in enumerate(docs, start=1):
            print(f"\n[{rank}] {format_metadata(doc.metadata)}")
            print(clean_preview(doc.page_content))


if __name__ == "__main__":
    evaluate()
