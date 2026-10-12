"""意图识别评估：准确率、耗时和费用。

默认配置下路由模型和主力模型是同一个，只评估一次。想比较另一个模型时，运行之前用环境变量
把路由模型指向它（ROUTER_BASE_URL、ROUTER_MODEL、ROUTER_API_KEY，以及三个 ROUTER_PRICE_*），
脚本会用同一份数据、同一份提示词把两个模型各测一遍。

用法（在 project 目录下）：python -m eval.run_intent [标签]
"""
import asyncio
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from langchain_core.messages import HumanMessage

from app import config, llm, prompts
from app.graph import parse_intent

DATASET = Path(__file__).parent / "datasets" / "intent.jsonl"
LABEL = sys.argv[1] if len(sys.argv) > 1 else "latest"        # python -m eval.run_intent [标签]
REPORT = Path(__file__).parent / "reports" / f"intent-{LABEL}.json"
NAMES = {"router": config.ROUTER_MODEL, "worker": config.WORKER_MODEL}


async def classify(role: str, sample: dict) -> dict:
    history = sample.get("history", "")
    prompt = prompts.INTENT_PROMPT.format(
        history=f"最近的对话：\n{history}\n\n" if history else "", message=sample["message"])
    try:
        reply, record = await llm.call(role, [HumanMessage(prompt)], "classify")
        return {**record, "predicted": parse_intent(reply.content), "raw": reply.content.strip()[:60]}
    except Exception as error:
        return {"predicted": "error", "raw": f"{type(error).__name__}", "seconds": 0.0,
                "input_tokens": 0, "output_tokens": 0, "cost": 0.0}


async def evaluate(role: str, samples: list[dict]) -> dict:
    # 并发数由 llm.call 里的信号量控制
    results = await asyncio.gather(*(classify(role, s) for s in samples))
    wrong = [{"message": s["message"], "expected": s["intent"], "predicted": r["predicted"], "raw": r["raw"]}
             for s, r in zip(samples, results) if r["predicted"] != s["intent"]]
    per_class = {}
    for intent in prompts.INTENTS:
        index = [i for i, s in enumerate(samples) if s["intent"] == intent]
        per_class[intent] = f"{sum(results[i]['predicted'] == intent for i in index)}/{len(index)}"
    seconds = [r["seconds"] for r in results]
    return {
        "model": NAMES[role], "samples": len(samples),
        "accuracy": round(1 - len(wrong) / len(samples), 3),
        "per_class": per_class,
        "confusion": dict(Counter(f"{w['expected']}→{w['predicted']}" for w in wrong)),
        "avg_seconds": round(statistics.mean(seconds), 2), "max_seconds": round(max(seconds), 2),
        "avg_input_tokens": round(statistics.mean(r["input_tokens"] for r in results)),
        "avg_output_tokens": round(statistics.mean(r["output_tokens"] for r in results)),
        "cost_per_1000": round(sum(r["cost"] for r in results) / len(samples) * 1000, 3),
        "wrong": wrong,
    }


async def main() -> None:
    samples = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = {}
    for role in config.distinct_roles():
        report[role] = result = await evaluate(role, samples)
        print(f"== {role}：{result['model']}")
        print(f"   准确率 {result['accuracy']:.1%}（{result['samples']} 条）  各类别 {result['per_class']}")
        print(f"   单次耗时 平均 {result['avg_seconds']}s，最长 {result['max_seconds']}s"
              f"（并发 {config.ROUTER_CONCURRENCY if role == 'router' else config.WORKER_CONCURRENCY}）")
        print(f"   词元 平均输入 {result['avg_input_tokens']}，输出 {result['avg_output_tokens']}；"
              f"每 1000 次的费用约 {result['cost_per_1000']} 元")
        for w in result["wrong"]:
            print(f"   错：{w['message']} | 期望 {w['expected']}，得到 {w['predicted']}（{w['raw']}）")
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
