"""可观测：每个请求写一条轨迹（JSONL），监控指标从轨迹中汇总。"""
import json
import statistics
import uuid
from collections import Counter
from datetime import datetime

from app import config


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def build_trace(trace_id: str, emp_id: str, session_id: str, question: str, state: dict,
                seconds: float, nodes: list[dict], error: str = "") -> dict:
    """把一次请求的最终状态整理成一条轨迹。"""
    usage = state.get("usage") or []
    by_role = {}
    for role in ("router", "worker"):
        records = [u for u in usage if u["role"] == role]
        by_role[role] = {"calls": len(records),
                         "input_tokens": sum(u["input_tokens"] for u in records),
                         "cached_tokens": sum(u.get("cached_tokens", 0) for u in records),
                         "output_tokens": sum(u["output_tokens"] for u in records),
                         "cost": round(sum(u["cost"] for u in records), 6)}
    messages = state.get("messages") or []
    answer = messages[-1].content if messages and messages[-1].type == "ai" else ""
    return {
        "trace_id": trace_id, "time": datetime.now().isoformat(timespec="seconds"),
        "emp_id": emp_id, "session_id": session_id, "question": question,
        "intent": state.get("intent", ""), "classifier": state.get("classifier", ""),
        "query": state.get("query", ""),
        "hits": [{"id": h["id"], "score": h["score"]} for h in state.get("hits") or []],
        "plan": state.get("plan") or [],
        "nodes": nodes,                                   # 经过的节点及各自的耗时
        "model_calls": usage,
        "tool_calls": state.get("tool_calls") or [],
        "steps": state.get("steps", 0),
        "answer": answer,
        "finish": "error" if error else state.get("finish", ""),
        "error": error,
        "seconds": round(seconds, 3),
        "usage": by_role,
        "cost": round(sum(r["cost"] for r in by_role.values()), 6),
    }


def write_trace(trace: dict) -> None:
    config.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    with config.TRACE_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(trace, ensure_ascii=False) + "\n")


def read_traces() -> list[dict]:
    if not config.TRACE_PATH.exists():
        return []
    with config.TRACE_PATH.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def percentile(values: list[float], q: float) -> float:
    """第 q 百分位数（0 到 100），用最近的名次取值。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q / 100 * len(ordered)) - 1))]


def summarize(traces: list[dict]) -> dict:
    """汇总监控指标。"""
    total = len(traces)
    if total == 0:
        return {"requests": 0}
    seconds = [t["seconds"] for t in traces]
    finish = Counter(t["finish"] for t in traces)
    agent_traces = [t for t in traces if t["steps"] > 0]
    tokens = {role: {"input": sum(t["usage"][role]["input_tokens"] for t in traces),
                     "cached": sum(t["usage"][role].get("cached_tokens", 0) for t in traces),
                     "output": sum(t["usage"][role]["output_tokens"] for t in traces)}
              for role in ("router", "worker")}
    by_intent = {}
    for intent in sorted({t["intent"] for t in traces}):
        group = [t for t in traces if t["intent"] == intent]
        by_intent[intent] = {"requests": len(group),
                             "avg_seconds": round(statistics.mean(t["seconds"] for t in group), 2),
                             "avg_cost": round(statistics.mean(t["cost"] for t in group), 5)}
    return {
        "requests": total,
        "latency_p50": round(percentile(seconds, 50), 2),
        "latency_p99": round(percentile(seconds, 99), 2),
        "by_intent": by_intent,
        "tool_calls": dict(Counter(c["name"] for t in traces for c in t["tool_calls"]).most_common()),
        "avg_agent_steps": round(statistics.mean(t["steps"] for t in agent_traces), 2) if agent_traces else 0,
        "error_rate": round(finish["error"] / total, 3),
        "no_answer_rate": round(finish["no_answer"] / total, 3),
        "max_steps_rate": round(finish["max_steps"] / total, 3),
        "router_fallback_rate": round(sum(t["classifier"] == "worker" for t in traces) / total, 3),
        # 因为连接中断而重试的模型调用次数；持续升高说明到模型服务的网络不稳定
        "model_retries": sum(c.get("retries", 0) for t in traces for c in t["model_calls"]),
        "tokens": tokens,
        "cost": round(sum(t["cost"] for t in traces), 4),
        "avg_cost": round(sum(t["cost"] for t in traces) / total, 5),
    }
