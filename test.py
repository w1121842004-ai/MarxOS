from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

query = "阶级"

docs = db.similarity_search(query, k=2)

for i, doc in enumerate(docs, start=1):
    print(f"\n===== 结果 {i} =====")
    print(doc.page_content)
    print("来源信息：", doc.metadata)