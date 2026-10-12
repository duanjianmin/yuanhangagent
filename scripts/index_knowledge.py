"""知识库入库：清洗、分块、向量化，写入 Chroma。可以重复运行，每次都会重建集合。

用法（在 project 目录下）：python -m scripts.index_knowledge
"""
import chromadb

from app import config
from app.rag.ingest import load_chunks
from app.rag.retriever import embed


def main() -> None:
    chunks = load_chunks()
    vectors = embed([c.indexed_text for c in chunks])

    client = chromadb.HttpClient(host=config.CHROMA_HOST, port=config.CHROMA_PORT)
    try:
        client.delete_collection(config.CHROMA_COLLECTION)
    except Exception:
        pass                                    # 第一次运行时集合还不存在
    collection = client.create_collection(
        config.CHROMA_COLLECTION,
        configuration={"hnsw": {"space": "cosine"}},
    )
    collection.add(
        ids=[c.id for c in chunks],
        embeddings=vectors,
        documents=[c.text for c in chunks],
        metadatas=[{"doc": c.doc, "title": c.title} for c in chunks],
    )
    print(f"已写入 {collection.count()} 个块，向量维度 {len(vectors[0])}")


if __name__ == "__main__":
    main()
