"""命令行里的对话，用来调试图的行为。会显示经过的节点、工具调用和用量。

用法（在 project 目录下）：python -m scripts.chat [工号] [问题 ...]
不带问题时进入交互模式；带问题时依次提问后退出，遇到确认一律同意。
"""
import asyncio
import sys

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app import config, db
from app.graph import build_graph, new_request
from app.tools import ToolBox

ANSWER_NODES = {"chitchat", "generate", "agent", "give_up"}


async def run_turn(graph, payload, thread: dict, ask_confirm) -> None:
    while True:
        pending = None
        async for mode, data in graph.astream(payload, thread, stream_mode=["updates", "messages"]):
            if mode == "messages":
                chunk, meta = data
                if meta["langgraph_node"] in ANSWER_NODES and chunk.content and chunk.type == "AIMessageChunk":
                    print(chunk.content, end="", flush=True)
                continue
            for node, update in data.items():
                if node == "__interrupt__":
                    pending = update[0].value
                    continue
                update = update or {}
                if node == "classify":
                    print(f"  [意图] {update['intent']}（由 {update['classifier']} 识别）")
                elif node == "rewrite":
                    print(f"  [检索问题] {update['query']}")
                elif node == "retrieve":
                    print("  [检索] " + "，".join(f"{h['id']}({h['score']})" for h in update["hits"]))
                elif node == "plan":
                    print("  [计划] " + " → ".join(update["plan"]))
                elif node == "tools":
                    for call in update["tool_calls"]:
                        print(f"  [工具] {call['name']} {call['args']} → {call['result'][:100]}")
                elif node in ("generate", "chitchat", "give_up") and not update.get("usage"):
                    print(update["messages"][-1].content, end="")     # 没有经过模型的固定回复
        if pending is None:
            break
        payload = Command(resume=ask_confirm(pending["summary"]))
    state = (await graph.aget_state(thread)).values
    usage = state.get("usage", [])
    # 每次模型调用：节点:输入+输出；输入中有缓存命中的部分时，写在括号里
    tokens = ", ".join(
        f"{u['node']}:{u['input_tokens']}"
        + (f"(缓存{u['cached_tokens']})" if u.get("cached_tokens") else "") + f"+{u['output_tokens']}"
        for u in usage)
    print(f"\n  [结束] {state.get('finish')}  用时 {sum(u['seconds'] for u in usage):.1f}s  "
          f"词元 {tokens}  费用 {sum(u['cost'] for u in usage):.5f} 元\n")


async def main() -> None:
    emp_id = sys.argv[1] if len(sys.argv) > 1 else "E1001"
    questions = sys.argv[2:]
    if not config.DB_PATH.exists():
        db.init_db()
    thread = {"configurable": {"thread_id": f"{emp_id}-cli"}}
    async with ToolBox() as toolbox:
        graph = build_graph(toolbox, InMemorySaver())
        if questions:
            for question in questions:
                print(f"> {question}")
                await run_turn(graph, new_request(question, emp_id), thread,
                               lambda summary: print(f"  [确认] {summary} → 同意") or True)
            return
        print(f"当前员工：{db.get_employee(emp_id)['name']}。输入问题，空行退出。")
        while question := input("> ").strip():
            await run_turn(graph, new_request(question, emp_id), thread,
                           lambda summary: input(f"  [确认] {summary}，确认吗？(y/n) ").lower().startswith("y"))


if __name__ == "__main__":
    asyncio.run(main())
