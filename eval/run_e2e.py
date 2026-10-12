"""端到端评估：在数据集上运行真实的助手，从轨迹和结果两个层次检查，全部用确定性的规则。

用法（在 project 目录下）：
    python -m eval.run_e2e --label baseline               # 运行并保存为 reports/e2e-baseline.json
    python -m eval.run_e2e --label v2 --compare baseline  # 运行并与基线逐条比较
    python -m eval.run_e2e --label passk --repeat 3       # 只对标了 repeat 的样本重复运行，计算 pass^3
"""
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from app import config, db
from app.graph import build_graph
from app.observability import read_traces
from app.service import Assistant
from app.tools import ToolBox

HERE = Path(__file__).parent
WRITE_TABLES = ("leave_requests", "trip_requests", "tickets", "room_bookings")


def load(name: str) -> list[dict]:
    path = HERE / "datasets" / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def table_counts() -> dict[str, int]:
    with db.connect() as connection:
        return {t: connection.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in WRITE_TABLES}


def row_exists(table: str, where: dict) -> bool:
    condition = " AND ".join(f"{column} = ?" for column in where)
    with db.connect() as connection:
        return connection.execute(f"SELECT 1 FROM {table} WHERE {condition}", tuple(where.values())).fetchone() is not None


def contains(answer: str, fact) -> bool:
    """fact 是字符串时必须出现；是列表时出现其中任意一个即可。比较时忽略空格。"""
    text = answer.replace(" ", "")
    options = [fact] if isinstance(fact, str) else fact
    return any(option.replace(" ", "") in text for option in options)


async def run_sample(assistant: Assistant, sample: dict, tag: str) -> dict:
    """在一份全新的数据上运行一个样本，返回检查结果。"""
    db.init_db()
    before = table_counts()
    session_id = f"{sample['id']}-{tag}"
    started = time.perf_counter()
    answer, error = "", ""
    for turn in sample["turns"]:
        events = [e async for e in assistant.chat(sample["emp_id"], session_id, turn)]
        while events[-1]["type"] == "confirm":            # 按样本的设定同意或拒绝
            events = [e async for e in assistant.confirm(sample["emp_id"], session_id, sample.get("confirm", True))]
        if events[-1]["type"] == "error":
            error = events[-1]["message"]
            break
        answer = events[-1]["answer"]
    seconds = time.perf_counter() - started
    traces = [t for t in read_traces() if t["session_id"] == session_id]
    last = traces[-1] if traces else {}
    called = [c for t in traces for c in t["tool_calls"]]
    names = [c["name"] for c in called]
    after = table_counts()

    problems = []
    # ---- 轨迹层 ----
    for tool in sample.get("tools", []):
        if tool not in names:
            problems.append(f"轨迹：没有调用 {tool}")
    for tool in sample.get("forbidden_tools", []):
        if (tool == "*" and names) or tool in names:
            problems.append(f"轨迹：不应调用工具，实际调用了 {names}")
    keys = [json.dumps([c["name"], c["args"]], sort_keys=True, ensure_ascii=False) for c in called]
    if len(keys) != len(set(keys)):
        problems.append("轨迹：用相同的参数重复调用了同一个工具")
    if last.get("finish") == "max_steps":
        problems.append("轨迹：达到步数上限")
    trajectory_ok = not problems
    # ---- 结果层 ----
    if error:
        problems.append(f"结果：出错 {error}")
    for fact in sample.get("facts", []):
        if not contains(answer, fact):
            problems.append(f"结果：回答中缺少 {fact}")
    for text in sample.get("forbidden", []):
        if contains(answer, text):
            problems.append(f"结果：回答中出现了不应出现的 {text}")
    rows = sample.get("rows", [])
    for row in rows:
        if not row_exists(row["table"], row["where"]):
            problems.append(f"结果：数据库中没有期望的记录 {row['table']} {row['where']}")
    new_rows = sum(after.values()) - sum(before.values())
    if new_rows != len(rows):
        problems.append(f"结果：期望新增 {len(rows)} 条记录，实际新增 {new_rows} 条")
    cost = sum(t["cost"] for t in traces)
    if sample.get("light") and any(t["usage"]["worker"]["calls"] for t in traces):
        problems.append("结果：这个请求只应当使用路由模型，不应当调用主力模型")
    outcome_ok = not any(p.startswith("结果") for p in problems)

    return {
        # 样本标注的是第一轮的意图；多轮样本各轮实际识别的结果记在 intents 里
        "id": sample["id"], "intent": sample["intent"], "predicted": traces[0]["intent"] if traces else "",
        "intents": [t["intent"] for t in traces],
        "intent_ok": bool(traces) and traces[0]["intent"] == sample["intent"],
        "trajectory_ok": trajectory_ok, "outcome_ok": outcome_ok, "passed": not problems,
        "problems": problems, "tools": names, "steps": sum(t["steps"] for t in traces),
        "seconds": round(seconds, 2), "cost": round(cost, 6),
        "worker_calls": sum(t["usage"]["worker"]["calls"] for t in traces),
        "worker_input": sum(t["usage"]["worker"]["input_tokens"] for t in traces),
        "worker_cached": sum(t["usage"]["worker"]["cached_tokens"] for t in traces),
        "worker_output": sum(t["usage"]["worker"]["output_tokens"] for t in traces),
        "answer": answer, "question": sample["turns"][-1],
        "hits": last.get("hits", []), "query": last.get("query", ""),
    }


def summarize(results: list[dict]) -> dict:
    def rate(items, key):
        return round(sum(r[key] for r in items) / len(items), 3) if items else 0.0

    def block(items):
        return {"samples": len(items), "intent": rate(items, "intent_ok"),
                "trajectory": rate(items, "trajectory_ok"), "outcome": rate(items, "outcome_ok"),
                "passed": rate(items, "passed"),
                "avg_seconds": round(statistics.mean(r["seconds"] for r in items), 2),
                "avg_cost": round(statistics.mean(r["cost"] for r in items), 5),
                "avg_worker_calls": round(statistics.mean(r["worker_calls"] for r in items), 2),
                "avg_worker_input": round(statistics.mean(r["worker_input"] for r in items)),
                "avg_worker_output": round(statistics.mean(r["worker_output"] for r in items)),
                # 接口模型的输入词元中，命中缓存的比例
                "cache_hit": round(sum(r["worker_cached"] for r in items)
                                   / max(1, sum(r["worker_input"] for r in items)), 3)}

    summary = {"overall": block(results)}
    for intent in ("knowledge", "personal", "action", "complex", "other"):
        items = [r for r in results if r["intent"] == intent]
        if items:
            summary[intent] = block(items)
    return summary


def print_summary(summary: dict) -> None:
    print(f"\n{'类别':<10}{'条数':>4}{'意图':>8}{'轨迹':>8}{'结果':>8}{'全部通过':>8}{'耗时(s)':>9}{'费用(元)':>10}"
          f"{'接口调用':>8}{'输入词元':>8}{'缓存命中':>8}")
    for name, b in summary.items():
        print(f"{name:<12}{b['samples']:>4}{b['intent']:>10.0%}{b['trajectory']:>10.0%}{b['outcome']:>10.0%}"
              f"{b['passed']:>10.0%}{b['avg_seconds']:>11.1f}{b['avg_cost']:>12.5f}{b['avg_worker_calls']:>10.1f}"
              f"{b['avg_worker_input']:>12}{b['cache_hit']:>11.0%}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="latest")
    parser.add_argument("--compare", help="与 reports/e2e-<这个标签>.json 逐条比较")
    parser.add_argument("--repeat", type=int, default=1, help="大于 1 时只运行标了 repeat 的样本，计算 pass^k")
    parser.add_argument("--only", help="只运行这些编号，用逗号分隔")
    args = parser.parse_args()

    samples = load("e2e.jsonl")
    if args.repeat > 1:
        samples = [s for s in samples if s.get("repeat")]
    if args.only:
        samples = [s for s in samples if s["id"] in args.only.split(",")]
    config.TRACE_PATH.unlink(missing_ok=True)

    runs = []
    async with ToolBox() as toolbox:
        assistant = Assistant(build_graph(toolbox, InMemorySaver()))
        for k in range(args.repeat):
            results = []
            for sample in samples:                         # 样本共用一个数据库，所以逐个运行
                result = await run_sample(assistant, sample, f"{args.label}-{k}")
                results.append(result)
                mark = "通过" if result["passed"] else "失败"
                print(f"[{k + 1}/{args.repeat}] {result['id']:<4}{mark}  {result['seconds']:>5.1f}s  "
                      f"{result['predicted']:<10}{' → '.join(result['tools'])}")
                for problem in result["problems"]:
                    print(f"        {problem}")
            runs.append(results)

    report = {"label": args.label, "samples": len(samples), "summary": summarize(runs[0]), "results": runs[0]}
    print_summary(report["summary"])
    if args.repeat > 1:
        always = [s["id"] for i, s in enumerate(samples) if all(run[i]["passed"] for run in runs)]
        once = [s["id"] for i, s in enumerate(samples) if runs[0][i]["passed"]]
        unstable = [s["id"] for i, s in enumerate(samples) if len({run[i]["passed"] for run in runs}) > 1]
        report["pass_k"] = {"k": args.repeat, "pass_1": round(len(once) / len(samples), 3),
                            "pass_k": round(len(always) / len(samples), 3), "unstable": unstable,
                            "per_run": [round(sum(r["passed"] for r in run) / len(run), 3) for run in runs]}
        print(f"\n单次通过率 {report['pass_k']['pass_1']:.0%}，pass^{args.repeat} {report['pass_k']['pass_k']:.0%}，"
              f"各次 {report['pass_k']['per_run']}，结果不稳定的样本 {unstable or '无'}")

    if args.compare:
        old = json.loads((HERE / "reports" / f"e2e-{args.compare}.json").read_text(encoding="utf-8"))
        old_passed = {r["id"]: r["passed"] for r in old["results"]}
        fixed = [r["id"] for r in runs[0] if r["passed"] and old_passed.get(r["id"]) is False]
        broken = [r["id"] for r in runs[0] if not r["passed"] and old_passed.get(r["id"]) is True]
        report["compare"] = {"with": args.compare, "fixed": fixed, "broken": broken}
        print(f"\n与 {args.compare} 比较：通过率 {old['summary']['overall']['passed']:.0%} → "
              f"{report['summary']['overall']['passed']:.0%}，修复 {fixed or '无'}，退步 {broken or '无'}")

    (HERE / "reports").mkdir(exist_ok=True)
    path = HERE / "reports" / f"e2e-{args.label}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n报告已保存：{path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
