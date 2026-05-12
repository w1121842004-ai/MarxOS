import os
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


load_dotenv()


# 1. 读取 OCR 输出文本
with open("output.txt", "r", encoding="utf-8") as f:
    text = f.read()


# 2. 把 OCR 文本包装成 Document，并加入 metadata
doc = Document(
    page_content=text,
    metadata={
        "book": "资本论",
        "volume": "第一卷",
        "page": 208,
        "source": "capital.pdf",
        "ocr": True
    }
)


# 3. 切分 chunk
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents([doc])

print("切分后的 chunk 数量：", len(chunks))

print("\n===== 第一个 chunk 示例 =====")
print(chunks[0].page_content)

print("\n===== 第一个 chunk 的 metadata =====")
print(chunks[0].metadata)


# 4. 创建 embedding 模型
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# 5. 建立 FAISS 向量库
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)


# 6. 保存向量库
vectorstore.save_local("vectorstore/capital_ocr")

print("\n向量库保存完成：vectorstore/capital_ocr")
