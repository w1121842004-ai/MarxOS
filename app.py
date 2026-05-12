from openai import OpenAI

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import json
import os
import re
import sys


load_dotenv()


EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTORSTORE_DIR = "vectorstore/marx_knowledge_base"
OCR_CACHE_DIR = "data/ocr_cache"
ARTICLE_MAP_PATH = "rag/article_map.json"
VOLUME_PUBLICATION_YEARS = {
    "me46a": "1979年",
    "me46b": "1979年",
    "me47": "2004年",
}


def repair_mojibake(text):
    if not isinstance(text, str):
        return text

    markers = ("Ã", "Â", "ã", "å", "æ", "ç", "è", "é", "ï", "ä")
    if not any(marker in text for marker in markers):
        return text

    def decode_run(match):
        run = match.group(0)
        if not any(marker in run for marker in markers):
            return run

        try:
            return run.encode("latin1").decode("utf-8")
        except UnicodeError:
            return run

    return re.sub(r"[\x00-\xff]+", decode_run, text)


def clean_text(text, fallback="未知"):
    if text is None or text == "":
        return repair_mojibake(fallback)

    return str(repair_mojibake(text)).strip() or fallback


def source_stem(metadata):
    source = clean_text(metadata.get("source"), "")
    return source.lower().replace(".pdf", "")


def load_article_map():
    if not os.path.exists(ARTICLE_MAP_PATH):
        return {}

    with open(ARTICLE_MAP_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


ARTICLE_MAP = load_article_map()


def volume_from_source(stem):
    match = re.fullmatch(r"me(\d{2})([ab]?)", stem)
    if not match:
        return None

    volume = int(match.group(1))
    suffix = {"a": "(上)", "b": "(下)"}.get(match.group(2), "")
    return f"第{volume}卷{suffix}"


def normalize_book_parts(metadata):
    book = clean_text(metadata.get("book"), "未知书名")
    stem = source_stem(metadata)

    if stem == "capital":
        return "马克思", "资本论", "第1卷", "2004年"

    volume = volume_from_source(stem)
    if volume:
        return "", "马克思恩格斯全集", volume, VOLUME_PUBLICATION_YEARS.get(stem, "出版年不详")

    match = re.search(r"(第\d+卷[AB]?)", book)
    if "马克思恩格斯文集" in book:
        volume = match.group(1).replace("A", "(上)").replace("B", "(下)") if match else ""
        return "", "马克思恩格斯文集", volume, "2009年"

    if "马克思恩格斯全集" in book:
        volume = match.group(1).replace("A", "(上)").replace("B", "(下)") if match else ""
        return "", "马克思恩格斯全集", volume, "出版年不详"

    return "", book, "", "出版年不详"


def format_citation(metadata, include_article=False):
    author, title, volume, year = normalize_book_parts(metadata)
    article = clean_text(metadata.get("section") or metadata.get("article"), "")
    printed_page = metadata.get("printed_page")
    page = printed_page if printed_page is not None else metadata.get("page")
    page = clean_text(page, "未知页码")
    pdf_page = clean_text(metadata.get("pdf_page"), page)

    author_text = f"{author}：" if author else ""
    volume_text = volume if volume else ""
    article_text = f"，{article}" if include_article and article else ""
    year_text = f"，{year}" if year else ""
    page_text = f"第{page}页" if printed_page is not None else f"PDF第{pdf_page}页"

    return f"{author_text}《{title}》{volume_text}{article_text}，北京：人民出版社{year_text}，{page_text}。"


def extract_quoted_title(query):
    query = clean_text(query, "")
    match = re.search(r"《([^》]+)》", query)
    if match:
        return match.group(1).strip()

    return None


def extract_unquoted_title(query):
    query = clean_text(query, "")
    keywords = [
        "在哪一卷",
        "在哪卷",
        "哪一卷",
        "第几卷",
        "收录在哪",
        "收在哪",
        "出自哪卷",
        "属于哪卷",
        "在哪本",
        "从哪页",
        "从第几页",
        "第几页",
        "起始页",
        "开始页",
        "收录页",
    ]
    positions = [query.find(keyword) for keyword in keywords if keyword in query]
    if not positions:
        return None

    title = query[:min(positions)]
    title = re.sub(r"[，。；：、\s\"'“”《》（）()]+$", "", title).strip()

    return title or None


def extract_bibliographic_title(query):
    return extract_quoted_title(query) or extract_unquoted_title(query)


def normalize_for_match(text):
    text = clean_text(text, "")
    text = re.sub(r"[《》“”\"'（）()，。；：、\s·\-.—–]", "", text)
    return text.lower()


def is_bibliographic_query(query):
    query = clean_text(query, "")
    keywords = [
        "在哪一卷",
        "在哪卷",
        "哪一卷",
        "第几卷",
        "收录在哪",
        "收在哪",
        "出自哪卷",
        "属于哪卷",
        "在哪本",
        "从哪页",
        "从第几页",
        "起始页",
        "开始页",
        "收录页",
    ]

    return any(keyword in query for keyword in keywords)


def is_quote_lookup_query(query):
    query = clean_text(query, "")
    if extract_bibliographic_title(query):
        return False

    quote_keywords = ["引文", "出处", "出自", "哪一页", "哪页", "页码", "原文", "这句话", "这段话"]
    if any(keyword in query for keyword in quote_keywords):
        return True

    return len(query) >= 12 and not re.search(r"[？?]", query)


def is_concept_query(query):
    query = clean_text(query, "")
    return any(keyword in query for keyword in ["什么是", "何为", "概念", "定义", "解释一下", "是什么意思"])


def is_analysis_query(query):
    query = clean_text(query, "")
    return any(keyword in query for keyword in ["分析", "怎么看", "如何理解", "为什么", "现实", "意义", "关系", "评价"])


def classify_query(query):
    if is_bibliographic_query(query) and extract_bibliographic_title(query):
        return "bibliographic_lookup"

    if is_quote_lookup_query(query):
        return "quote_lookup"

    if is_concept_query(query):
        return "concept_explain"

    if is_analysis_query(query):
        return "theory_analysis"

    return "rag_answer"


def cache_files_for_toc_scan():
    if not os.path.isdir(OCR_CACHE_DIR):
        return []

    paths = []

    for source_stem in os.listdir(OCR_CACHE_DIR):
        root = os.path.join(OCR_CACHE_DIR, source_stem)
        if not os.path.isdir(root):
            continue

        for page_num in range(1, 31):
            path = os.path.join(root, f"page_{page_num}.txt")
            if os.path.exists(path):
                paths.append((source_stem, path))

    return paths


def best_toc_entries(entries):
    unique_entries = {}

    for entry in entries:
        key = (entry["source"], entry["article"], entry["start_page"], entry["end_page"])
        unique_entries[key] = entry

    filtered_entries = list(unique_entries.values())
    best_by_source = {}

    for entry in filtered_entries:
        width = entry["end_page"] - entry["start_page"]
        source = entry["source"]
        previous = best_by_source.get(source)
        if previous is None or width > previous["end_page"] - previous["start_page"]:
            best_by_source[source] = entry

    return sorted(
        best_by_source.values(),
        key=lambda item: (item["source"], item["start_page"], item["end_page"]),
    )


def find_toc_entries_from_map(title):
    entries = []
    normalized_title = normalize_for_match(title)

    if not normalized_title:
        return []

    if normalized_title == normalize_for_match("反杜林论"):
        metadata = {
            "source": "me20.pdf",
            "book": ARTICLE_MAP.get("me20.pdf", {}).get("book", ""),
            "article": "反杜林论",
        }
        _, book_title, volume, year = normalize_book_parts(metadata)
        return [
            {
                "source": "me20.pdf",
                "book_title": book_title,
                "volume": volume,
                "year": year,
                "article": "反杜林论",
                "start_page": 1,
                "end_page": 354,
            }
        ]

    for source, source_map in ARTICLE_MAP.items():
        metadata = {
            "source": source,
            "book": source_map.get("book", ""),
            "article": title,
        }
        _, book_title, volume, year = normalize_book_parts(metadata)

        for item in source_map.get("entries", []):
            entry_title = clean_text(item.get("title"), "")
            normalized_entry_title = normalize_for_match(entry_title)
            start_page = item.get("start_printed_page")
            end_page = item.get("end_printed_page")

            if not normalized_entry_title or start_page is None or end_page is None:
                continue

            if normalized_title not in normalized_entry_title and normalized_entry_title not in normalized_title:
                continue

            entries.append(
                {
                    "source": source,
                    "book_title": book_title,
                    "volume": volume,
                    "year": year,
                    "article": entry_title,
                    "start_page": start_page,
                    "end_page": end_page,
                }
            )

    if normalized_title == normalize_for_match("自然辩证法"):
        grouped_entries = []
        for source, source_map in ARTICLE_MAP.items():
            if not any(
                normalized_title in normalize_for_match(clean_text(item.get("title"), ""))
                for item in source_map.get("entries", [])
            ):
                continue

            metadata = {
                "source": source,
                "book": source_map.get("book", ""),
                "article": title,
            }
            _, book_title, volume, year = normalize_book_parts(metadata)

            for item in source_map.get("entries", []):
                entry_title = clean_text(item.get("title"), "")
                if normalize_for_match("论文") not in normalize_for_match(entry_title):
                    continue

                grouped_entries.append(
                    {
                        "source": source,
                        "book_title": book_title,
                        "volume": volume,
                        "year": year,
                        "article": "自然辩证法[论文]",
                        "start_page": item.get("start_printed_page"),
                        "end_page": item.get("end_printed_page"),
                    }
                )

        if grouped_entries:
            return best_toc_entries(grouped_entries)

    exact_entries = [
        entry for entry in entries
        if normalize_for_match(entry["article"]).endswith(normalized_title)
    ]
    if exact_entries:
        entries = exact_entries

    derivative_terms = ["草稿", "初稿", "遗稿", "导言", "序言", "扉页", "封面", "一书导言", "第一页", "材料"]
    if not any(term in title for term in derivative_terms):
        primary_entries = [
            entry for entry in entries
            if not any(term in entry["article"] for term in derivative_terms)
        ]
        if primary_entries:
            entries = primary_entries

    return best_toc_entries(entries)


def find_toc_entries(title):
    entries = find_toc_entries_from_map(title)
    if entries:
        return entries

    entries = []
    title_pattern = re.escape(title)
    range_pattern = re.compile(
        rf"{title_pattern}(?![\u4e00-\u9fff]).{{0,50}}?(\d{{1,4}})\s*[—\-–一]\s*(\d{{1,4}})"
    )

    for source_stem, path in cache_files_for_toc_scan():
        with open(path, "r", encoding="utf-8") as f:
            text = clean_text(f.read(), "")

        for match in range_pattern.finditer(text):
            start_page = int(match.group(1))
            end_page = int(match.group(2))

            if start_page > end_page:
                continue

            if start_page > 1200 or end_page > 1200:
                continue

            matched_text = match.group(0)
            matched_tail = matched_text[matched_text.find(title) + len(title):]
            title_tail = re.sub(r"[\s《》“”\"'（）()，。；：、·\-.—–0-9０-９]", "", matched_tail)
            if title_tail:
                continue

            metadata = {
                "source": f"{source_stem}.pdf",
                "book": f"马克思恩格斯全集 {volume_from_source(source_stem) or ''}".strip(),
                "article": title,
                "page": f"{start_page}-{end_page}",
            }
            _, book_title, volume, year = normalize_book_parts(metadata)

            entries.append(
                {
                    "source": f"{source_stem}.pdf",
                    "book_title": book_title,
                    "volume": volume,
                    "year": year,
                    "article": title,
                    "start_page": start_page,
                    "end_page": end_page,
                }
            )

    unique_entries = {}

    for entry in entries:
        key = (entry["source"], entry["article"], entry["start_page"], entry["end_page"])
        unique_entries[key] = entry

    filtered_entries = list(unique_entries.values())
    best_by_source = {}

    for entry in filtered_entries:
        width = entry["end_page"] - entry["start_page"]
        source = entry["source"]
        previous = best_by_source.get(source)
        if previous is None or width > previous["end_page"] - previous["start_page"]:
            best_by_source[source] = entry

    return sorted(
        best_by_source.values(),
        key=lambda item: (item["source"], item["start_page"], item["end_page"]),
    )


def answer_bibliographic_query(query):
    title = extract_bibliographic_title(query)
    if not title:
        return None

    entries = find_toc_entries(title)
    if not entries:
        return None

    lines = []

    for index, entry in enumerate(entries, start=1):
        lines.append(
            f"({index})《{entry['book_title']}》{entry['volume']}，"
            f"{entry['article']}，第{entry['start_page']}-{entry['end_page']}页。"
        )

    return "\n".join(lines)


def constraints_from_query(query):
    title = extract_bibliographic_title(query)
    if not title:
        return {}

    entries = find_toc_entries(title)
    if not entries:
        return {"title": title}

    return {
        "title": title,
        "entries": entries,
        "sources": {entry["source"] for entry in entries},
        "page_ranges": {
            entry["source"]: (entry["start_page"], entry["end_page"])
            for entry in entries
        },
    }


def metadata_matches_constraints(metadata, constraints):
    sources = constraints.get("sources")
    if not sources:
        return True

    return metadata.get("source") in sources


def page_in_expected_range(metadata, constraints):
    ranges = constraints.get("page_ranges")
    if not ranges:
        return False

    source = metadata.get("source")
    if source not in ranges:
        return False

    try:
        page = int(metadata.get("page"))
    except (TypeError, ValueError):
        return False

    start_page, end_page = ranges[source]
    return start_page <= page <= end_page


def rerank_documents(query, docs, constraints):
    title = constraints.get("title")
    normalized_title = normalize_for_match(title) if title else ""
    normalized_query = normalize_for_match(query)
    ranked = []

    for doc in docs:
        metadata = doc.metadata
        article = clean_text(metadata.get("section") or metadata.get("article"), "")
        book = clean_text(metadata.get("book"), "")
        content = clean_text(doc.page_content, "")
        haystack = normalize_for_match(f"{book} {article} {content[:600]}")
        score = 0

        if metadata_matches_constraints(metadata, constraints):
            score += 100

        if page_in_expected_range(metadata, constraints):
            score += 40

        if normalized_title and normalized_title in normalize_for_match(article):
            score += 35

        if normalized_title and normalized_title in haystack:
            score += 25

        if normalized_query and normalized_query in haystack:
            score += 10

        ranked.append((score, doc))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [doc for score, doc in ranked]


def retrieve_documents(query, db, k=5):
    constraints = constraints_from_query(query)
    fetch_k = 80 if constraints else 30

    if constraints.get("sources"):
        candidates = db.similarity_search(query, k=fetch_k)
        candidates = [
            doc for doc in candidates
            if metadata_matches_constraints(doc.metadata, constraints)
        ]

        if not candidates:
            candidates = db.similarity_search(query, k=fetch_k)
    else:
        candidates = db.similarity_search(query, k=fetch_k)

    return rerank_documents(query, candidates, constraints)[:k]


def build_prompt(intent, query, context):
    if intent == "quote_lookup":
        return f"""
你是 MarxOS 的出处核对器。

任务：用户给出一句或一段原文，请只根据【检索材料】判断最可能出处。

回答要求：
1. 只输出出处，不做理论分析。
2. 优先使用检索材料中的“句子引文格式”或“段落具体出处格式”。
3. 如果材料只有 PDF 页而没有可靠印刷页，必须写“PDF第X页”，不要冒充“第X页”。
4. 如果无法确认，明确说“未能确认具体页码”，并给出最接近的候选。

# 检索材料
{context}

# 用户原文
{query}
"""

    if intent == "concept_explain":
        return f"""
你是 MarxOS，一个马克思主义学术助手。

任务：解释用户提出的概念。优先依据【原著内容】，再做必要的理论概括。

回答要求：
1. 先给出简明定义。
2. 再说明它在马克思主义理论中的位置。
3. 如使用原著材料，附简短出处。
4. 不要输出“检索来源”等内部调试信息。

# 原著内容
{context}

# 用户问题
{query}
"""

    if intent == "theory_analysis":
        return f"""
你是 MarxOS，一个马克思主义学术智能体。

任务：基于【原著内容】和马克思主义理论，对用户问题做结构性分析。

回答要求：
1. 优先依据原著内容。
2. 从生产力与生产关系、经济基础与上层建筑、阶级关系、资本逻辑、劳动过程等相关角度展开。
3. 不要空喊口号；要有概念、逻辑和现实指向。
4. 如引用原著，给出简短出处。

# 原著内容
{context}

# 用户问题
{query}
"""

    return f"""
你是 MarxOS，一个马克思主义学术助手。

请根据【原著内容】回答用户问题。问题若只需要短答，就短答；只有需要展开解释时才分层分析。
不要输出“检索来源”等内部调试信息。

# 原著内容
{context}

# 用户问题
{query}
"""


# 1. 用户问题
query = input("请输入问题：")
query_intent = classify_query(query)

bibliographic_answer = answer_bibliographic_query(query) if query_intent == "bibliographic_lookup" else None
if bibliographic_answer:
    print("\n===== MarxOS =====\n")
    print(bibliographic_answer)
    sys.exit(0)

# 2. embedding：必须和构建向量库时的模型保持一致
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# 3. 加载 OCR 向量库
db = FAISS.load_local(
    VECTORSTORE_DIR,
    embeddings,
    allow_dangerous_deserialization=True,
)

# 4. DeepSeek client
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# 5. RAG 检索：metadata 约束 -> 向量召回 -> rerank
docs = retrieve_documents(query, db, k=5)

# 7. 构造带 metadata 的上下文
context_parts = []

for i, doc in enumerate(docs, start=1):
    book = clean_text(doc.metadata.get("book"), "未知书名")
    article = clean_text(doc.metadata.get("article"), "未知篇目")
    section = clean_text(doc.metadata.get("section"), "")
    page = clean_text(doc.metadata.get("page"), "未知页码")
    pdf_page = clean_text(doc.metadata.get("pdf_page"), "未知PDF页")
    section_text = f"，{section}" if section and section != article else ""
    sentence_citation = format_citation(doc.metadata, include_article=False)
    detailed_source = format_citation(doc.metadata, include_article=True)

    context_parts.append(
        f"【资料{i}】\n"
        f"来源： 《{book}》{article}{section_text}，第{page}页（PDF第{pdf_page}页）\n"
        f"句子引文格式：({i}){sentence_citation}\n"
        f"段落具体出处格式：({i}){detailed_source}\n"
        f"原文：{clean_text(doc.page_content)}"
    )

context = "\n\n".join(context_parts)

# 8. Prompt
prompt = f"""
你是 MarxOS，一个马克思主义学术智能体。

你的核心任务不是简单聊天，而是：

1. 基于马克思、恩格斯原著进行学术解释
2. 运用历史唯物主义分析社会现实
3. 运用马克思主义政治经济学分析资本、劳动、阶级、生产关系等问题
4. 在知识库材料不足时，可以结合已有马克思主义理论知识进行补充
5. 回答必须保持学术性、逻辑性、批判性，避免空洞口号

# 回答原则

1. 优先依据【原著内容】
2. 如果原著内容不足，可以进行理论补充，但要明确区分
3. 根据用户问题选择回答长度和结构：书目定位、页码查询、某句话出处这类问题要短答；只有用户要求理论分析或问题本身需要展开时，才使用完整结构。
4. 分析现实问题时，要从生产力与生产关系、经济基础与上层建筑、阶级关系、资本逻辑、劳动过程、异化、剩余价值等角度展开
5. 不要脱离原著随意发挥
6. 不要只做道德评价，要进行结构性分析

# 马克思经典著作引文规则

1. 如果用户要求“某个句子”的引文、出处、页码或原文引用，必须按脚注式逐条输出：
   (1)马克思：《资本论》第1卷，北京：人民出版社，2004年，第197页。
   (2)《马克思恩格斯全集》第46卷(上)，北京：人民出版社，1979年，第393页。
   (3)《马克思恩格斯文集》第5卷，北京：人民出版社，2009年，第21页。
2. 句子引文只输出著作、卷号、出版信息、页码；不要输出“资料1”“PDF第几页”等内部检索信息。
3. 如果用户要求“这段话的具体出处”或段落级出处，必须输出卷号、篇目、页码；可按：
   《书名》第X卷，篇目，第X页
   或在出版信息完整时按：
   《书名》第X卷，篇目，北京：人民出版社，年份，第X页。
4. 若知识库没有给出出版年份，不要编造年份；使用“出版年不详”。页码以知识库的印刷页码为准。
5. 本次检索材料中已经给出“句子引文格式”和“段落具体出处格式”，回答时优先照此格式使用。

# 回答结构

## 一、原著依据

说明知识库中有哪些相关原著内容。

## 二、理论分析

使用马克思主义理论框架解释问题。

## 三、现实解释

如果用户问题涉及现实社会现象，请结合现实进行分析。

## 四、总结

用简洁语言总结核心观点。

## 五、引用来源

列出本次回答使用的知识库来源。若用户只要求引文或出处，可以只输出引文列表，不必套用完整回答结构。

# 原著内容

{context}

# 用户问题

{query}
"""
prompt = build_prompt(query_intent, query, context)
prompt = clean_text(prompt)

# 9. LLM 回答
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
)

# 10. 输出回答
answer = response.choices[0].message.content

print("\n===== MarxOS =====\n")
print(answer)

if os.getenv("MARXOS_DEBUG_SOURCES") == "1":
    print("\n===== 检索来源 =====\n")

    for i, doc in enumerate(docs, start=1):
        print(f"【来源{i}】")
        print("书名：", clean_text(doc.metadata.get("book"), "未知书名"))
        print("篇目：", clean_text(doc.metadata.get("article"), "未知篇目"))
        print("小节：", clean_text(doc.metadata.get("section"), "未知小节"))
        print("页码：", clean_text(doc.metadata.get("page"), "未知页码"))
        print("PDF页：", clean_text(doc.metadata.get("pdf_page"), "未知PDF页"))
        print("文件：", clean_text(doc.metadata.get("source"), "未知文件"))
        print("句子引文格式：", format_citation(doc.metadata, include_article=False))
        print("段落具体出处格式：", format_citation(doc.metadata, include_article=True))
        print("-" * 30)
