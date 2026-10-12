"""检索：向量检索（Chroma）+ BM25，RRF 融合，再用重排序服务精排。"""
import math
import re
from collections import Counter
from dataclasses import dataclass

import chromadb
import httpx

from app import config
from app.rag.ingest import Chunk, load_chunks

# bge 系列的中文模型建议在“查询”之前加上这句指令，文档不加。
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*|[一-鿿]+")


# ---------- 两个模型服务 ----------

# 复用同一个客户端。每次调用都新建客户端（例如直接用 httpx.post）要重新初始化、重新建立连接，
# 实测每个请求会多出约 300 毫秒，而模型服务本身的推理只需要 10 毫秒左右。
_http = httpx.Client(timeout=60)

def embed(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """调用 Embedding 服务。返回的向量已经归一化。"""
    if is_query:
        texts = [QUERY_INSTRUCTION + t for t in texts]
    vectors = []
    for start in range(0, len(texts), 16):           # 服务端对单次请求的条数有限制
        response = _http.post(f"{config.EMBED_URL}/embed", json={"inputs": texts[start:start + 16]})
        response.raise_for_status()
        vectors.extend(response.json())
    return vectors


def rerank_scores(query: str, texts: list[str]) -> list[float]:
    """调用重排序服务，返回与 texts 顺序一致的相关性分数（0 到 1）。"""
    response = _http.post(f"{config.RERANK_URL}/rerank", json={"query": query, "texts": texts})
    response.raise_for_status()
    scores = [0.0] * len(texts)
    for item in response.json():
        scores[item["index"]] = item["score"]
    return scores


# ---------- BM25 ----------

def tokenize(text: str) -> list[str]:
    """英文、数字和编号（如 e-809、hr-01）整体作为一个词；中文按相邻两个字切分。"""
    tokens = []
    for part in TOKEN.findall(text.lower()):
        if part[0].isascii():
            tokens.append(part)
        elif len(part) == 1:
            tokens.append(part)
        else:
            tokens.extend(part[i:i + 2] for i in range(len(part) - 1))
    return tokens


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_length = sum(self.lengths) / len(self.lengths)
        df = Counter(term for d in self.docs for term in d)
        n = len(texts)
        self.idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        result = []
        for doc, length in zip(self.docs, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_length)
            result.append(sum(self.idf.get(t, 0.0) * doc[t] * (self.k1 + 1) / (doc[t] + norm)
                              for t in terms if t in doc))
        return result


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """倒数排名融合：只看名次，不看分数。"""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


# ---------- 检索器 ----------

@dataclass
class Hit:
    chunk: Chunk
    score: float       # 重排序分数；没有经过重排序时为 0


class Retriever:
    """mode 取 "vector"、"bm25"、"hybrid"、"hybrid_rerank"，后三种用于评估时的对比。"""

    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks or load_chunks()
        self.by_id = {c.id: c for c in self.chunks}
        self.bm25 = BM25([c.indexed_text for c in self.chunks])
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            client = chromadb.HttpClient(host=config.CHROMA_HOST, port=config.CHROMA_PORT)
            self._collection = client.get_collection(config.CHROMA_COLLECTION)
        return self._collection

    def vector_ids(self, query: str, k: int) -> list[str]:
        vector = embed([query], is_query=True)[0]
        return self.collection.query(query_embeddings=[vector], n_results=k)["ids"][0]

    def bm25_ids(self, query: str, k: int) -> list[str]:
        scores = self.bm25.scores(query)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i].id for i in order[:k] if scores[i] > 0]

    def search(self, query: str, mode: str = "hybrid_rerank",
               top_k: int = config.RERANK_TOP_K) -> list[Hit]:
        if mode == "vector":
            ids = self.vector_ids(query, top_k)
        elif mode == "bm25":
            ids = self.bm25_ids(query, top_k)
        else:
            ids = rrf([self.vector_ids(query, config.RECALL_K),
                       self.bm25_ids(query, config.RECALL_K)])
        if mode != "hybrid_rerank":
            return [Hit(self.by_id[i], 0.0) for i in ids[:top_k]]

        candidates = [self.by_id[i] for i in ids]
        scores = rerank_scores(query, [c.indexed_text for c in candidates])
        hits = sorted((Hit(c, s) for c, s in zip(candidates, scores)),
                      key=lambda h: h.score, reverse=True)
        return hits[:top_k]


def format_context(hits: list[Hit]) -> str:
    """把检索结果拼成带编号和出处的资料，供模型引用。"""
    return "\n\n".join(f"[{n}] 《{h.chunk.doc}》{h.chunk.title}\n{h.chunk.text}"
                       for n, h in enumerate(hits, start=1))
