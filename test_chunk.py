from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# embedding
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 加载知识库
db = FAISS.load_local(
    "vectorstore/marx_knowledge_base",
    embeddings,
    allow_dangerous_deserialization=True
)

# 查看前几个 chunk
docs = db.similarity_search("劳动过程是如何生产剩余价值的", k=5)

for i, doc in enumerate(docs):

    print(f"\n===== Chunk {i+1} =====\n")

    print(doc.page_content)

    print("\nmetadata:")

    print(doc.metadata)
