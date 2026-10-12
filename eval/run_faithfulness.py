"""忠实度评估：回答中的每一句是否都有资料作为依据。用模型评判，使用之前先校准。

第一步：在人工标注的校准集上测试评判模型，得到它与人工标注的一致率。
        路由模型和主力模型配置成不同的模型时，两个都测，优先使用路由模型（与被评者不同）。
第二步：用一致率达标的评判模型，评判端到端评估中制度问答的真实回答。

用法（在 project 目录下，先运行过 eval.run_e2e）：python -m eval.run_faithfulness [端到端报告的标签]
"""
import asyncio
import json
import re
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage

from app import config, llm
from app.graph import strip_json
from app.rag.ingest import load_chunks

HERE = Path(__file__).parent
LABEL = sys.argv[1] if len(sys.argv) > 1 else "latest"
THRESHOLD = 0.9                      # 与人工标注的一致率达到这个值，评判模型才可以使用
NAMES = {"router": config.ROUTER_MODEL, "worker": config.WORKER_MODEL}

JUDGE_PROMPT = """你是严格的审核员。判断“回答”中的内容是否全部能从“资料”中找到依据。

判断标准：
- faithful：回答中的每一个事实（数字、期限、条件、流程）都与资料一致。换一种说法、只回答资料的一部分、说明资料中没有相关规定，都算 faithful。
- unfaithful：回答中有任何一个事实与资料矛盾，或者资料中根本没有提到。

只关心回答是否有依据，不关心回答是否完整、是否礼貌。回答末尾的 [1] 这类标注是出处编号，不需要核对。

先逐条核对回答中的事实，再给出结论。只输出 JSON：{{"reason": "一句话说明", "verdict": "faithful 或 unfaithful"}}

资料：
{context}

问题：{question}

回答：{answer}"""


def parse_verdict(text: str) -> str:
    try:
        return "unfaithful" if "unfaithful" in json.loads(strip_json(text))["verdict"].lower() else "faithful"
    except (ValueError, KeyError, AttributeError, TypeError):
        found = re.findall(r"unfaithful|faithful", text.lower())     # JSON 不合法时退而求其次
        return found[-1] if found else "invalid"


async def judge(role: str, context: str, question: str, answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(context=context, question=question, answer=answer)
    try:
        reply, record = await llm.call(role, [HumanMessage(prompt)], "judge")
        return {"verdict": parse_verdict(reply.content), "raw": reply.content.strip()[:200],
                "seconds": record["seconds"], "cost": record["cost"]}
    except Exception as error:
        return {"verdict": "invalid", "raw": type(error).__name__, "seconds": 0.0, "cost": 0.0}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def calibrate(role: str, cases: list[dict], chunks: dict) -> dict:
    contexts = ["\n\n".join(chunks[i].text for i in case["chunks"]) for case in cases]
    results = await asyncio.gather(*(judge(role, ctx, c["question"], c["answer"])
                                     for ctx, c in zip(contexts, cases)))
    pairs = [(c["label"], r["verdict"]) for c, r in zip(cases, results)]
    wrong = [{"question": c["question"], "answer": c["answer"], "label": c["label"],
              "verdict": r["verdict"], "raw": r["raw"]}
             for c, r in zip(cases, results) if c["label"] != r["verdict"]]
    return {
        "model": NAMES[role], "cases": len(cases),
        "agreement": round(sum(a == b for a, b in pairs) / len(pairs), 3),
        "missed_unfaithful": sum(a == "unfaithful" and b == "faithful" for a, b in pairs),   # 漏判：最危险
        "false_alarm": sum(a == "faithful" and b == "unfaithful" for a, b in pairs),         # 误判
        "invalid": sum(b == "invalid" for _, b in pairs),
        "avg_seconds": round(sum(r["seconds"] for r in results) / len(results), 2),
        "cost": round(sum(r["cost"] for r in results), 5),
        "wrong": wrong,
    }


async def main() -> None:
    chunks = {c.id: c for c in load_chunks()}
    cases = load(HERE / "datasets" / "faithfulness_calibration.jsonl")
    report = {"calibration": {}}
    print(f"校准集 {len(cases)} 条（faithful {sum(c['label'] == 'faithful' for c in cases)} 条，"
          f"unfaithful {sum(c['label'] == 'unfaithful' for c in cases)} 条）")
    roles = config.distinct_roles()
    for role in roles:
        report["calibration"][role] = result = await calibrate(role, cases, chunks)
        print(f"== 评判模型 {result['model']}：一致率 {result['agreement']:.1%}，漏判 {result['missed_unfaithful']}，"
              f"误判 {result['false_alarm']}，无效输出 {result['invalid']}，平均 {result['avg_seconds']}s，"
              f"费用 {result['cost']} 元")
        for w in result["wrong"]:
            print(f"   人工 {w['label']}，评判 {w['verdict']}：{w['answer'][:40]}… | {w['raw'][:80]}")

    # 被评判的回答由主力模型生成。路由模型配置成另一个模型时，优先用它来评判，
    # 可以减轻“自己评自己”的偏好；一致率不达标时才换成主力模型。
    # 默认配置下两个角色是同一个模型，评判者与被评者相同，解读结果时要考虑这一点。
    qualified = [r for r in roles if report["calibration"][r]["agreement"] >= THRESHOLD]
    if not qualified:
        print(f"\n评判模型的一致率低于 {THRESHOLD:.0%}，不能用于评估。")
        return
    role = qualified[0]
    report["judge"] = NAMES[role]

    e2e_path = HERE / "reports" / f"e2e-{LABEL}.json"
    flagged = {s["id"] for s in load(HERE / "datasets" / "e2e.jsonl") if s.get("faithfulness")}
    answers = [r for r in json.loads(e2e_path.read_text(encoding="utf-8"))["results"] if r["id"] in flagged]
    verdicts = await asyncio.gather(*(
        judge(role, "\n\n".join(chunks[h["id"]].text for h in r["hits"]), r["query"] or r["question"], r["answer"])
        for r in answers))
    report["answers"] = [{"id": r["id"], "question": r["question"], "answer": r["answer"],
                          "verdict": v["verdict"], "raw": v["raw"]} for r, v in zip(answers, verdicts)]
    faithful = sum(v["verdict"] == "faithful" for v in verdicts)
    report["faithfulness"] = round(faithful / len(verdicts), 3) if verdicts else None
    print(f"\n用 {NAMES[role]} 评判 e2e-{LABEL} 中的 {len(verdicts)} 条制度问答：忠实 {faithful} 条")
    for item in report["answers"]:
        if item["verdict"] != "faithful":
            print(f"   {item['id']} {item['verdict']}：{item['answer'][:60]}… | {item['raw'][:100]}")

    path = HERE / "reports" / f"faithfulness-{LABEL}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"报告已保存：{path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
