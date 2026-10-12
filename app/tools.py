"""宿主一侧的工具层。

本地工具和 MCP 工具在这里统一成 Tool 对象，Agent 不需要知道一个工具来自哪里。
两条安全规则也在这里落实：
1. 工号不交给模型填写。MCP 工具的 emp_id 参数会从给模型看的定义中去掉，调用时由宿主填入。
2. 哪些工具会修改数据（需要用户确认），由下面的 WRITE_TOOLS 决定，不由模型判断。
"""
import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Awaitable, Callable

from mcp import Client, StdioServerParameters
from pydantic import BaseModel, Field, ValidationError

from app import config, db
from app.rag.retriever import Retriever, format_context

WRITE_TOOLS = {"submit_leave_request", "submit_trip_request", "create_ticket", "book_meeting_room"}
MCP_SERVERS = {"hr": "app.mcp_servers.hr_server", "it": "app.mcp_servers.it_server"}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict                                   # 给模型看的参数定义（JSON Schema）
    source: str                                        # local、hr、it
    run: Callable[[str, dict], Awaitable[str]]         # (工号, 参数) -> 结果文本

    @property
    def is_write(self) -> bool:
        return self.name in WRITE_TOOLS

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}


def _error(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False)


# ---------- 本地工具 ----------

class SearchPolicyArgs(BaseModel):
    query: str = Field(description="要查询的制度问题，写成完整的一句话，例如“出差的住宿标准是多少”")


class RoomSlotArgs(BaseModel):
    day: str = Field(description="日期，格式 YYYY-MM-DD")
    start_hour: int = Field(description="开始的整点，24 小时制，例如 14 表示下午两点")
    end_hour: int = Field(description="结束的整点，24 小时制")


class BookRoomArgs(RoomSlotArgs):
    room: str = Field(description="会议室的名称，只能是 大会议室、小会议室A、小会议室B 之一")


def _local_tools(retriever: Retriever) -> list[Tool]:
    def search_policy(emp_id: str, args: SearchPolicyArgs) -> str:
        hits = retriever.search(args.query)
        if not hits or hits[0].score < config.RERANK_MIN_SCORE:
            return "制度中没有找到相关的内容。"
        return format_context(hits)

    def list_meeting_rooms(emp_id: str, args: RoomSlotArgs) -> str:
        return json.dumps(db.list_free_rooms(args.day, args.start_hour, args.end_hour),
                          ensure_ascii=False)

    def book_meeting_room(emp_id: str, args: BookRoomArgs) -> str:
        return json.dumps(db.book_room(emp_id, args.room, args.day, args.start_hour, args.end_hour),
                          ensure_ascii=False)

    specs = [
        (search_policy, SearchPolicyArgs,
         "检索公司的制度文档（员工手册、差旅与报销、IT 服务、入职指引、行政服务），返回相关的条款原文。"
         "凡是涉及制度规定、标准、流程的问题，都应当先用它查证，不要凭印象回答。"),
        (list_meeting_rooms, RoomSlotArgs, "查询某一天某个时段空闲的会议室及其可容纳的人数。"),
        (book_meeting_room, BookRoomArgs, "预订会议室。时段是否冲突由系统检查。"),
    ]

    def wrap(function, model):
        async def run(emp_id: str, args: dict) -> str:
            try:
                parsed = model.model_validate(args)          # 执行之前先校验参数
            except ValidationError as error:
                problems = "；".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
                return _error(f"参数不合法：{problems}")
            try:
                return await asyncio.to_thread(function, emp_id, parsed)
            except db.BusinessError as error:
                return _error(str(error))
        return run

    tools = []
    for function, model, description in specs:
        schema = model.model_json_schema()
        schema.pop("title", None)
        tools.append(Tool(function.__name__, description, schema, "local", wrap(function, model)))
    return tools


# ---------- MCP 工具 ----------

def _hide_emp_id(schema: dict) -> dict:
    """从参数定义中去掉 emp_id，模型既看不到它，也无法填写它。"""
    properties = {k: {kk: vv for kk, vv in v.items() if kk != "title"}
                  for k, v in schema.get("properties", {}).items() if k != "emp_id"}
    required = [name for name in schema.get("required", []) if name != "emp_id"]
    return {"type": "object", "properties": properties, "required": required}


def _mcp_tool(client: Client, source: str, spec) -> Tool:
    async def run(emp_id: str, args: dict) -> str:
        arguments = {**args, "emp_id": emp_id}             # 即使模型硬塞了 emp_id，也会被覆盖
        result = await client.call_tool(spec.name, arguments)
        text = result.content[0].text if result.content else ""
        return _error(text) if result.is_error else text
    return Tool(spec.name, spec.description or "", _hide_emp_id(spec.input_schema), source, run)


class ToolBox:
    """全部可用的工具。用 async with 管理 MCP 服务器子进程的启动和退出。"""

    def __init__(self, retriever: Retriever | None = None):
        self.retriever = retriever or Retriever()
        self.tools: dict[str, Tool] = {}
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> "ToolBox":
        for tool in _local_tools(self.retriever):
            self.tools[tool.name] = tool
        for source, module in MCP_SERVERS.items():
            params = StdioServerParameters(command=sys.executable, args=["-m", module],
                                           cwd=str(config.ROOT), env=dict(os.environ))
            client = await self._stack.enter_async_context(Client(params))
            for spec in (await client.list_tools()).tools:
                self.tools[spec.name] = _mcp_tool(client, source, spec)
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self._stack.aclose()

    def schemas(self, names: list[str] | None = None) -> list[dict]:
        return [t.schema() for t in self.tools.values() if names is None or t.name in names]

    def is_write(self, name: str) -> bool:
        return name in WRITE_TOOLS

    async def call(self, name: str, args: dict, emp_id: str) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return _error(f"没有名为 {name} 的工具，可用的工具有：{'、'.join(self.tools)}")
        try:
            return await tool.run(emp_id, args)
        except Exception as error:                          # 工具的故障作为观察结果返回给模型
            return _error(f"工具执行失败：{type(error).__name__}: {error}")


def describe_action(name: str, args: dict) -> str:
    """把一次写入操作写成用户看得懂的一句话，用于确认。"""
    a = {k: args.get(k, "") for k in ("leave_type", "start_date", "end_date", "reason", "destination",
                                      "category", "description", "room", "day", "start_hour", "end_hour")}
    if name == "submit_leave_request":
        return f"提交请假单：{a['leave_type']}，{a['start_date']} 至 {a['end_date']}"
    if name == "submit_trip_request":
        return f"提交出差申请：{a['destination']}，{a['start_date']} 至 {a['end_date']}"
    if name == "create_ticket":
        urgent = "（紧急）" if args.get("urgent") else ""
        return f"提交 IT 工单{urgent}：{a['category']}，{a['description']}"
    if name == "book_meeting_room":
        return f"预订{a['room']}：{a['day']} {a['start_hour']} 点至 {a['end_hour']} 点"
    return f"执行 {name}：{json.dumps(args, ensure_ascii=False)}"
