from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTORSTORE_DIR = Path("vectorstore/marx_knowledge_base")


@dataclass(frozen=True)
class EvalQuestion:
    group: str
    question: str


QUESTIONS = [
    EvalQuestion("bibliographic_lookup", "《反杜林论》收录在哪一卷？"),
    EvalQuestion("bibliographic_lookup", "《自然辩证法》从哪一页开始？"),
    EvalQuestion("quote_lookup", "“劳动首先是人和自然之间的过程”这句话出自哪里？"),
    EvalQuestion("quote_lookup", "“哲学家们只是用不同的方式解释世界，问题在于改变世界”出自哪一页？"),
    EvalQuestion("concept_explain", "什么是剩余价值？"),
    EvalQuestion("concept_explain", "什么是异化劳动？"),
    EvalQuestion("theory_analysis", "如何用马克思主义分析平台经济中的劳动关系？"),
    EvalQuestion("theory_analysis", "为什么马克思说资本不是物，而是一种社会关系？"),
    EvalQuestion("negative", "《一个不存在的马克思著作标题》收录在哪一卷？"),
    EvalQuestion("negative", "请给出“这是一句随便编造的引文”的准确页码。"),
]


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
    db = load_vectorstore()

    for index, item in enumerate(QUESTIONS, start=1):
        print(f"\n===== {index}. {item.group} =====")
        print(f"Q: {item.question}")

        docs = db.similarity_search(item.question, k=k)
        if not docs:
            print("No retrieval results.")
            continue

        for rank, doc in enumerate(docs, start=1):
            print(f"\n[{rank}] {format_metadata(doc.metadata)}")
            print(clean_preview(doc.page_content))


if __name__ == "__main__":
    evaluate()
