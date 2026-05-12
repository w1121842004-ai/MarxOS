from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# 1. 加载 embedding 模型
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# 2. 加载本地 FAISS 向量库
vectorstore = FAISS.load_local(
    "vectorstore/capital_ocr",
    embeddings,
    allow_dangerous_deserialization=True
)


# 3. 测试问题
query = "什么是剩余价值？"


# 4. 相似度检索
docs = vectorstore.similarity_search(query, k=3)


# 5. 打印检索结果
print("\n===== 检索结果 =====\n")

for i, doc in enumerate(docs, start=1):
    print(f"【结果 {i}】")
    print("书名：", doc.metadata.get("book"))
    print("卷数：", doc.metadata.get("volume"))
    print("页码：", doc.metadata.get("page"))
    print("OCR：", doc.metadata.get("ocr"))
    print("\n原文片段：")
    print(doc.page_content)
    print("\n" + "-" * 50 + "\n")