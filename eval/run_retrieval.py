"""检索评估：比较四种检索方式的 Recall@1、Recall@3 和 MRR。不调用大模型。

用法（在 project 目录下，三个 Docker 服务已启动并且知识库已入库）：
    python -m eval.run_retrieval
"""
import json
from pathlib import Path

from app.rag.retriever import Retriever

DATASET = Path(__file__).parent / "datasets" / "retrieval.jsonl"
REPORT = Path(__file__).parent / "reports" / "retrieval.json"
MODES = ["bm25", "vector", "hybrid", "hybrid_rerank"]
LABELS = {"bm25": "BM25", "vector": "向量检索", "hybrid": "混合（RRF）", "hybrid_rerank": "混合 + 重排序"}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def first_rank(ids: list[str], expect: list[str]) -> int:
    """期望的块第一次出现的名次（从 1 开始），没有出现时为 0。"""
    return next((rank for rank, i in enumerate(ids, start=1) if i in expect), 0)


def metrics(ranks: list[int]) -> dict:
    n = len(ranks)
    return {"recall@1": sum(r == 1 for r in ranks) / n,
            "recall@3": sum(0 < r <= 3 for r in ranks) / n,
            "mrr": sum(1 / r for r in ranks if r) / n}


def main() -> None:
    samples, retriever = load(DATASET), Retriever()
    ranks = {mode: [] for mode in MODES}
    for sample in samples:
        for mode in MODES:
            ids = [h.chunk.id for h in retriever.search(sample["q"], mode, top_k=10)]
            ranks[mode].append(first_rank(ids, sample["expect"]))

    print(f"查询 {len(samples)} 条，知识库 {len(retriever.chunks)} 个块\n")
    print(f"{'检索方式':<14}{'Recall@1':>10}{'Recall@3':>10}{'MRR':>8}")
    report = {"samples": len(samples), "overall": {}, "by_kind": {}, "misses": {}}
    for mode in MODES:
        m = metrics(ranks[mode])
        report["overall"][mode] = {k: round(v, 3) for k, v in m.items()}
        print(f"{LABELS[mode]:<14}{m['recall@1']:>10.2f}{m['recall@3']:>10.2f}{m['mrr']:>8.3f}")

    kinds = sorted({s["kind"] for s in samples})
    print("\n按问题类型的 Recall@3：")
    print(f"{'类型':<12}{'条数':>4}" + "".join(f"{LABELS[m]:>14}" for m in MODES))
    for kind in kinds:
        index = [i for i, s in enumerate(samples) if s["kind"] == kind]
        row = {m: metrics([ranks[m][i] for i in index])["recall@3"] for m in MODES}
        report["by_kind"][kind] = {"samples": len(index), **{m: round(v, 3) for m, v in row.items()}}
        print(f"{kind:<12}{len(index):>4}" + "".join(f"{row[m]:>16.2f}" for m in MODES))

    print("\n最终方式（混合 + 重排序）没有排在第一位的查询：")
    for sample, rank in zip(samples, ranks["hybrid_rerank"]):
        if rank != 1:
            top = retriever.search(sample["q"], "hybrid_rerank", top_k=1)[0]
            report["misses"][sample["q"]] = {"rank": rank, "top": top.chunk.id}
            print(f"  名次 {rank or '未召回'}：{sample['q']} → 第一位是 {top.chunk.id}（期望 {sample['expect'][0]}）")

    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
