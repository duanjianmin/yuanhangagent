"""测试用的替身：脚本化的模型和假的检索器。"""
import asyncio
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from app import llm
from app.graph import build_graph
from app.rag.ingest import Chunk
from app.rag.retriever import Hit
from app.service import Assistant
from app.tools import ToolBox


class ScriptedModel(FakeMessagesListChatModel):
    """按顺序返回预设的回复，用完之后从头循环。预设项是异常时则抛出。"""
    script: list[Any] = []                 # 预设的回复，可以是消息，也可以是异常
    prompts: list[Any] = []                # 记录每次调用收到的消息，供断言使用

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.prompts.append(messages)
        response = self.script[self.i % len(self.script)]
        self.i += 1
        if isinstance(response, Exception):
            raise response
        return ChatResult(generations=[ChatGeneration(message=response.model_copy(update={"id": None}))])


def scripted(*responses) -> ScriptedModel:
    return ScriptedModel(responses=[], prompts=[],
                         script=[AIMessage(r) if isinstance(r, str) else r for r in responses])


def tool_call(name: str, call_id: str = "c1", **args) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def calls(*tool_calls: dict, content: str = "") -> AIMessage:
    return AIMessage(content=content, tool_calls=list(tool_calls))


class FakeRetriever:
    def search(self, query, mode="hybrid_rerank", top_k=3):
        if "天气" in query:
            return [Hit(Chunk("员工手册#02", "员工手册", "年假", "无关的内容"), 0.001)]
        return [Hit(Chunk("员工手册#02", "员工手册", "年假", "满 1 年不满 10 年的，每年 5 天"), 0.9)]


def run_session(router: ScriptedModel, worker: ScriptedModel, steps):
    """启动助手，依次执行 steps，返回每一步产生的事件列表。

    steps 的每一项是 ("chat", 工号, 消息) 或 ("confirm", 工号, True/False)。
    """
    async def main():
        llm.set_model("router", router)
        llm.set_model("worker", worker)
        llm._semaphores.clear()            # 信号量绑定事件循环，每个测试新建
        try:
            async with ToolBox(FakeRetriever()) as toolbox:
                assistant = Assistant(build_graph(toolbox, InMemorySaver()))
                results = []
                for kind, emp_id, value in steps:
                    stream = (assistant.chat(emp_id, "s1", value) if kind == "chat"
                              else assistant.confirm(emp_id, "s1", value))
                    results.append([event async for event in stream])
                state = await assistant.graph.aget_state(
                    {"configurable": {"thread_id": assistant.thread_id(steps[0][1], "s1")}})
                return results, state.values
        finally:
            llm.set_model("router", None)
            llm.set_model("worker", None)
    return asyncio.run(main())


def types(events: list[dict]) -> list[str]:
    return [event["type"] for event in events]


def answer(events: list[dict]) -> str:
    return "".join(event["text"] for event in events if event["type"] == "token")
