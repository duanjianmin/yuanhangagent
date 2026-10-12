"""服务层：驱动图运行，把图的流式输出转换成一组事件，并在请求结束时写入轨迹。

接口（app/api.py）和评估脚本（eval/）都通过这里使用助手，两者看到的行为完全一致。

事件的类型：
  intent   识别出的意图            step     一个处理步骤（给用户看的进度）
  token    回答的正文片段          retract  撤回已经显示的片段（模型在调用工具之前说的话）
  confirm  等待用户确认的写入操作   done     本次请求结束      error  出错
"""
import asyncio
import time
from typing import AsyncIterator

from langgraph.types import Command

from app import config
from app.graph import new_request
from app.observability import build_trace, new_trace_id, write_trace

ANSWER_NODES = {"chitchat", "generate", "agent", "give_up"}


class Assistant:
    def __init__(self, graph):
        self.graph = graph
        self._turns: dict[str, dict] = {}            # 正在进行（包括等待确认）的请求
        self._locks: dict[str, asyncio.Lock] = {}    # 同一个会话同时只处理一个请求

    @staticmethod
    def thread_id(emp_id: str, session_id: str) -> str:
        # 会话的标识由服务端用工号拼出来，换一个工号就访问不到别人的会话
        return f"{emp_id}:{session_id}"

    async def chat(self, emp_id: str, session_id: str, message: str) -> AsyncIterator[dict]:
        thread_id = self.thread_id(emp_id, session_id)
        if await self._has_pending(thread_id):
            yield {"type": "error", "message": "还有一个操作在等待确认，请先确认或取消。"}
            return
        turn = {"trace_id": new_trace_id(), "emp_id": emp_id, "session_id": session_id,
                "question": message, "seconds": 0.0, "nodes": []}
        async for event in self._run(thread_id, new_request(message, emp_id), turn):
            yield event

    async def confirm(self, emp_id: str, session_id: str, approved: bool) -> AsyncIterator[dict]:
        thread_id = self.thread_id(emp_id, session_id)
        if not await self._has_pending(thread_id):
            yield {"type": "error", "message": "没有等待确认的操作。"}
            return
        # 服务重启之后，等待确认的状态仍然保存在检查点里，只是内存中的请求信息没有了
        turn = self._turns.pop(thread_id, None) or {
            "trace_id": new_trace_id(), "emp_id": emp_id, "session_id": session_id,
            "question": "", "seconds": 0.0, "nodes": []}
        async for event in self._run(thread_id, Command(resume=approved), turn):
            yield event

    async def _has_pending(self, thread_id: str) -> bool:
        state = await self.graph.aget_state({"configurable": {"thread_id": thread_id}})
        return bool(state.next)

    async def _run(self, thread_id: str, payload, turn: dict) -> AsyncIterator[dict]:
        lock = self._locks.setdefault(thread_id, asyncio.Lock())
        if lock.locked():
            yield {"type": "error", "message": "上一条消息还在处理中，请稍候。"}
            return
        run_config = {"configurable": {"thread_id": thread_id}, "run_name": "yuanhang-chat",
                      "metadata": {"trace_id": turn["trace_id"], "emp_id": turn["emp_id"]}}
        async with lock:
            started = node_started = time.perf_counter()
            pending, error = None, ""
            answering: dict[str, bool] = {}          # 消息编号 -> 是否已经输出过非空白的内容
            streaming = ""                           # 当前这个节点里，正在输出正文的那条消息的编号
            try:
                async with asyncio.timeout(config.REQUEST_TIMEOUT):
                    async for mode, data in self.graph.astream(
                            payload, run_config, stream_mode=["updates", "messages"]):
                        if mode == "messages":
                            chunk, meta = data
                            if (meta.get("langgraph_node") in ANSWER_NODES
                                    and chunk.type == "AIMessageChunk" and chunk.content):
                                text = chunk.content
                                if not answering.get(chunk.id):
                                    text = text.lstrip()          # 去掉回答开头的空行
                                if text:
                                    if streaming and streaming != chunk.id:
                                        # 同一个节点里换了一条消息：上一次调用中途断开，正在重试。
                                        # 已经显示出去的半截文字作废。
                                        yield {"type": "retract"}
                                    streaming = chunk.id
                                    answering[chunk.id] = True
                                    yield {"type": "token", "text": text}
                            continue
                        for node, update in data.items():
                            if node == "__interrupt__":
                                pending = update[0].value
                                continue
                            now = time.perf_counter()
                            turn["nodes"].append({"node": node, "seconds": round(now - node_started, 3)})
                            node_started = now
                            streaming = ""                   # 这个节点结束了
                            for event in self._events(node, update or {}):
                                yield event
            except TimeoutError:
                error = f"处理超过了 {config.REQUEST_TIMEOUT:.0f} 秒，已经中止。"
            except Exception as exception:
                # 连同引发它的底层异常一起记录。只记最外层的话，网络类的错误往往只有一句
                # “Connection error.”，看不出真正的原因。
                parts, current = [], exception
                while current is not None and len(parts) < 4:
                    parts.append(f"{type(current).__name__}: {current}")
                    current = current.__cause__ or current.__context__
                error = " <- ".join(parts)
            turn["seconds"] += time.perf_counter() - started

            if pending and not error:
                self._turns[thread_id] = turn        # 暂停，等待用户确认；确认之后继续计时
                yield {"type": "confirm", **pending}
                return
            state = (await self.graph.aget_state(run_config)).values
            trace = build_trace(turn["trace_id"], turn["emp_id"], turn["session_id"],
                                turn["question"], state, turn["seconds"], turn["nodes"], error)
            write_trace(trace)
            if error:
                yield {"type": "error", "message": error, "trace_id": trace["trace_id"]}
            else:
                yield {"type": "done", "trace_id": trace["trace_id"], "finish": trace["finish"],
                       "intent": trace["intent"], "seconds": trace["seconds"], "cost": trace["cost"],
                       "usage": trace["usage"], "answer": trace["answer"]}

    @staticmethod
    def _events(node: str, update: dict):
        """把一个节点的状态更新转换成给前端的事件。"""
        if node == "classify":
            yield {"type": "intent", "intent": update["intent"], "classifier": update["classifier"]}
        elif node == "rewrite" and update.get("usage"):
            yield {"type": "step", "text": f"把问题改写为：{update['query']}"}
        elif node == "retrieve":
            titles = "、".join(f"《{h['doc']}》{h['title']}" for h in update["hits"])
            yield {"type": "step", "text": f"检索制度：{titles or '没有结果'}"}
        elif node == "plan" and update.get("plan"):
            steps = "；".join(f"{n}. {s}" for n, s in enumerate(update["plan"], start=1))
            yield {"type": "step", "text": f"计划：{steps}"}
        elif node == "agent":
            message = update["messages"][-1]
            if message.tool_calls and message.content:
                yield {"type": "retract"}            # 调用工具之前说的话不算回答
        elif node == "tools":
            for call in update["tool_calls"]:
                status = "已取消" if call["confirmed"] is False else "完成"
                yield {"type": "step", "text": f"调用工具 {call['name']}：{status}"}
        if node in ("chitchat", "generate", "give_up") and not update.get("usage"):
            # 没有经过模型的固定回复，一次性发出
            yield {"type": "token", "text": update["messages"][-1].content}
