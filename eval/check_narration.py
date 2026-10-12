"""统计一个现象：模型在发起工具调用的同时，是否附带了文字（应当为空）。

对同一组请求，分别用带有和不带有第 8 条规则的系统提示词各运行一遍，比较出现的次数。
用法（在 project 目录下）：python -m eval.check_narration
"""
import asyncio

from langgraph.checkpoint.memory import InMemorySaver

from app import db, prompts
from app.graph import build_graph
from app.service import Assistant
from app.tools import ToolBox

QUESTIONS = ["帮我请这周五一天年假", "电脑在家连不上 VPN，报错 E-809，帮我报个修", "我还剩几天年假",
             "报销单 BX20260901 到哪一步了", "订明天上午十点到十一点的小会议室A",
             "我下周三去上海出差两天，帮我安排一下", "我上周报的修处理了吗",
             "帮我请周五的年假，再把周四下午两点到三点的小会议室A订了"]
RULE_8 = prompts.AGENT_PROMPT[prompts.AGENT_PROMPT.index("\n8. "):]


async def run(assistant: Assistant, tag: str) -> tuple[int, int, list[str]]:
    narrated, with_tools, examples = 0, 0, []
    for n, question in enumerate(QUESTIONS):
        db.init_db()
        session = f"{tag}{n}"
        events = [e async for e in assistant.chat("E1001", session, question)]
        while events[-1]["type"] == "confirm":
            events = [e async for e in assistant.confirm("E1001", session, True)]
        state = await assistant.graph.aget_state({"configurable": {"thread_id": f"E1001:{session}"}})
        for message in state.values["messages"]:
            if message.type == "ai" and message.tool_calls:
                with_tools += 1
                if message.content.strip():
                    narrated += 1
                    examples.append(message.content.strip()[:50])
    return narrated, with_tools, examples


async def main() -> None:
    original = prompts.AGENT_PROMPT
    async with ToolBox() as toolbox:
        assistant = Assistant(build_graph(toolbox, InMemorySaver()))
        for label, prompt in (("没有第 8 条", original.replace(RULE_8, "")), ("有第 8 条", original)):
            prompts.AGENT_PROMPT = prompt
            narrated, with_tools, examples = await run(assistant, "a" if prompt == original else "b")
            print(f"{label}：{with_tools} 条带工具调用的回复中，{narrated} 条附带了文字")
            for example in examples[:4]:
                print(f"    {example}")
    prompts.AGENT_PROMPT = original


if __name__ == "__main__":
    asyncio.run(main())
