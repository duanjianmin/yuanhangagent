"""用 LangGraph 组装整个流程：意图识别之后分到四条分支。

START → classify ─┬─ other ──────→ chitchat ─────────────────────→ END
                  ├─ knowledge ──→ rewrite → retrieve → generate ─→ END
                  ├─ personal ─┐
                  ├─ action ───┴→ agent ⇄ tools（写入之前确认）──→ END
                  └─ complex ───→ plan → agent ⇄ tools ───────────→ END
"""
import asyncio
import json
import time
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from pydantic import BaseModel, Field, ValidationError

from app import config, db, llm, prompts
from app.tools import ToolBox, describe_action


def resettable(old: list | None, new: list | None) -> list:
    """可以清空的累加：节点返回列表时追加，新一轮请求传入 None 时清空。"""
    return [] if new is None else (old or []) + new


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]     # 整个会话的消息，跨请求保留
    emp_id: str
    # 以下字段只属于当前这一次请求，每次请求开始时重置
    intent: str
    classifier: str                             # 实际完成意图识别的角色：router 或 worker
    query: str                                  # 改写之后的检索问题
    hits: list[dict]                            # 检索到的块
    plan: list[str]
    steps: int                                  # Agent 已经调用模型的次数
    finish: str                                 # 结束的原因
    tool_calls: Annotated[list, resettable]     # 工具调用的记录
    usage: Annotated[list, resettable]          # 模型调用的用量记录


def new_request(message: str, emp_id: str) -> dict:
    """一次新请求的输入：追加用户消息，并重置只属于单次请求的字段。"""
    return {"messages": [HumanMessage(message)], "emp_id": emp_id, "intent": "", "classifier": "",
            "query": "", "hits": [], "plan": [], "steps": 0, "finish": "",
            "tool_calls": None, "usage": None}


class Plan(BaseModel):
    steps: list[str] = Field(min_length=1, max_length=config.MAX_PLAN_STEPS)


# ---------- 辅助函数 ----------

def last_question(state: State) -> str:
    return next(m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage))


def dialogue_text(messages: list, limit: int = 4) -> str:
    """最近几条有文字内容的对话，用于意图识别和查询改写。工具消息不包括在内。"""
    lines = []
    for message in messages:
        if isinstance(message, HumanMessage):
            lines.append(f"员工：{message.content}")
        elif isinstance(message, AIMessage) and message.content:
            lines.append(f"助手：{message.content[:200]}")
    return "\n".join(lines[-limit:])


def parse_intent(text: str) -> str:
    """从模型的输出中取出类别。输出不在五个类别之内时按 knowledge 处理：它只读，最安全。"""
    text = text.lower()
    found = [(text.rfind(name), name) for name in prompts.INTENTS if name in text]
    return max(found)[1] if found else "knowledge"


def strip_json(text: str) -> str:
    """去掉模型有时加在 JSON 外面的代码块标记。"""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    return text.strip()


# ---------- 图 ----------

def build_graph(toolbox: ToolBox, checkpointer=None):
    retriever = toolbox.retriever

    async def classify(state: State) -> dict:
        history = dialogue_text(state["messages"][:-1])
        prompt = prompts.INTENT_PROMPT.format(
            history=f"最近的对话：\n{history}\n\n" if history else "", message=last_question(state))
        classifier = "router"
        try:
            reply, record = await llm.call("router", [HumanMessage(prompt)], "classify")
        except Exception:
            # 路由模型不可用或者超时：降级为由主力模型识别意图。
            # 两个角色配置成同一个模型时，这相当于重试一次。
            classifier = "worker"
            reply, record = await llm.call("worker", [HumanMessage(prompt)], "classify")
        return {"intent": parse_intent(reply.content), "classifier": classifier, "usage": [record]}

    async def chitchat(state: State) -> dict:
        if state["classifier"] == "router":
            try:
                reply, record = await llm.call("router", [
                    SystemMessage(prompts.CHITCHAT_PROMPT), HumanMessage(last_question(state))], "chitchat")
                if reply.content.strip():
                    return {"messages": [AIMessage(reply.content.strip())], "usage": [record], "finish": "other"}
            except Exception:
                pass
        return {"messages": [AIMessage(prompts.CHITCHAT_FALLBACK)], "finish": "other"}

    async def rewrite(state: State) -> dict:
        question = last_question(state)
        history = dialogue_text(state["messages"][:-1])
        if not history:                                   # 第一轮对话不需要改写，省一次调用
            return {"query": question}
        prompt = prompts.REWRITE_PROMPT.format(history=history, message=question)
        reply, record = await llm.call("worker", [HumanMessage(prompt)], "rewrite")
        return {"query": reply.content.strip() or question, "usage": [record]}

    async def retrieve(state: State) -> dict:
        hits = await asyncio.to_thread(retriever.search, state["query"])
        return {"hits": [{"id": h.chunk.id, "doc": h.chunk.doc, "title": h.chunk.title,
                          "text": h.chunk.text, "score": round(h.score, 4)} for h in hits]}

    async def generate(state: State) -> dict:
        hits = state["hits"]
        if not hits or hits[0]["score"] < config.RERANK_MIN_SCORE:
            # 没有检索到相关的内容：直接拒答，不让模型凭印象回答，也省掉一次调用
            return {"messages": [AIMessage(prompts.NO_ANSWER)], "finish": "no_answer"}
        context = "\n\n".join(f"[{n}] 《{h['doc']}》{h['title']}\n{h['text']}"
                              for n, h in enumerate(hits, start=1))
        prompt = prompts.ANSWER_PROMPT.format(context=context, question=state["query"])
        reply, record = await llm.call("worker", [HumanMessage(prompt)], "generate")
        return {"messages": [reply], "usage": [record], "finish": "answered"}

    async def plan(state: State) -> dict:
        prompt = prompts.PLAN_PROMPT.format(
            max_steps=config.MAX_PLAN_STEPS, today=config.today(),
            calendar=prompts.calendar_text(), message=last_question(state))
        reply, record = await llm.call("worker", [HumanMessage(prompt)], "plan",
                                       response_format={"type": "json_object"})
        try:
            steps = Plan.model_validate_json(strip_json(reply.content)).steps
        except ValidationError:
            steps = []                                    # 规划失败时退回到普通的 Agent
        return {"plan": steps, "usage": [record]}

    async def agent(state: State) -> dict:
        employee = db.get_employee(state["emp_id"]) or {"name": "未知", "department": "未知"}
        system = prompts.AGENT_PROMPT.format(
            name=employee["name"], emp_id=state["emp_id"], department=employee["department"],
            today=config.today(), calendar=prompts.calendar_text())
        if state.get("plan"):
            numbered = "\n".join(f"{n}. {step}" for n, step in enumerate(state["plan"], start=1))
            system += prompts.PLAN_HINT.format(plan=numbered)
        reply, record = await llm.call("worker", [SystemMessage(system)] + state["messages"],
                                       "agent", tools=toolbox.schemas())
        update = {"messages": [reply], "usage": [record], "steps": state.get("steps", 0) + 1}
        if not reply.tool_calls:
            update["finish"] = "answered"
        return update

    async def tools(state: State) -> dict:
        calls = state["messages"][-1].tool_calls
        # 第一步：先为所有的写入操作取得用户的决定。
        # interrupt 恢复时，这个节点会从头重新执行，所以确认必须全部排在执行之前，
        # 否则先执行的写入会在恢复时被再执行一次。
        decisions = {}
        for call in calls:
            if toolbox.is_write(call["name"]):
                decisions[call["id"]] = bool(interrupt({
                    "tool": call["name"], "args": call["args"],
                    "summary": describe_action(call["name"], call["args"])}))
        # 第二步：执行
        messages, records = [], []
        for call in calls:
            started = time.perf_counter()
            confirmed = decisions.get(call["id"])          # None 表示只读工具，不需要确认
            if confirmed is False:
                result = json.dumps({"cancelled": "员工取消了这次操作，没有执行"}, ensure_ascii=False)
            else:
                result = await toolbox.call(call["name"], call["args"], state["emp_id"])
            messages.append(ToolMessage(result, tool_call_id=call["id"], name=call["name"]))
            records.append({"name": call["name"], "args": call["args"], "result": result[:300],
                            "confirmed": confirmed, "seconds": round(time.perf_counter() - started, 3)})
        return {"messages": messages, "tool_calls": records}

    async def give_up(state: State) -> dict:
        # 没有执行的工具调用也要补上结果，否则这段消息历史不合法，下一轮对话会报错
        pending = [ToolMessage(json.dumps({"error": "超过步数上限，没有执行"}, ensure_ascii=False),
                               tool_call_id=call["id"], name=call["name"])
                   for call in state["messages"][-1].tool_calls]
        return {"messages": pending + [AIMessage(prompts.MAX_STEPS_MESSAGE)], "finish": "max_steps"}

    def route_intent(state: State) -> str:
        return {"other": "chitchat", "knowledge": "rewrite", "complex": "plan"}.get(state["intent"], "agent")

    def route_agent(state: State) -> str:
        if not state["messages"][-1].tool_calls:
            return END
        limit = config.MAX_AGENT_STEPS + (config.MAX_PLAN_STEPS if state.get("plan") else 0)
        return "give_up" if state["steps"] >= limit else "tools"

    builder = StateGraph(State)
    for node in (classify, chitchat, rewrite, retrieve, generate, plan, agent, tools, give_up):
        builder.add_node(node.__name__, node)
    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_intent, ["chitchat", "rewrite", "plan", "agent"])
    builder.add_edge("chitchat", END)
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    builder.add_edge("plan", "agent")
    builder.add_conditional_edges("agent", route_agent, ["tools", "give_up", END])
    builder.add_edge("tools", "agent")
    builder.add_edge("give_up", END)
    return builder.compile(checkpointer=checkpointer)
