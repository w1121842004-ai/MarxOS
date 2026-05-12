"""
Legacy entrypoint.

MarxOS now builds the knowledge base in two explicit steps:

1. PDF -> OCR -> txt cache
   python rag/ocr_to_cache.py

2. txt cache -> metadata -> chunks -> FAISS
   python rag/build_vectorstore_from_cache.py
"""


def main():
    print("MarxOS 知识库构建已经拆成两个步骤：")
    print("")
    print("1. 先运行 OCR 缓存：")
    print("   python rag/ocr_to_cache.py")
    print("")
    print("2. 再从缓存构建 FAISS：")
    print("   python rag/build_vectorstore_from_cache.py")


if __name__ == "__main__":
    main()
