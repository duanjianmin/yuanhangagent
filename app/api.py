"""FastAPI 接口。启动（在 project 目录下）：uvicorn app.api:app --port 8080"""
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field

from app import config, db
from app.graph import build_graph
from app.observability import read_traces, summarize
from app.service import Assistant
from app.tools import ToolBox

MAX_IN_FLIGHT = 20                      # 同时处理的请求数上限，超过时直接拒绝而不是排队
state = {"assistant": None, "in_flight": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    if not config.DB_PATH.exists():
        db.init_db()
    # 两个 MCP 服务器子进程和检查点数据库的生命周期与应用一致
    async with ToolBox() as toolbox, \
            AsyncSqliteSaver.from_conn_string(str(config.CHECKPOINT_PATH)) as checkpointer:
        state["assistant"] = Assistant(build_graph(toolbox, checkpointer))
        yield


app = FastAPI(title="远航实业员工助手", lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    message: str = Field(min_length=1, max_length=1000)


class ConfirmRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    approved: bool


def current_employee(emp_id: str | None) -> str:
    """简化的登录：请求头里带工号。真实的系统应当在这里校验登录凭证。"""
    if not emp_id or not db.get_employee(emp_id):
        raise HTTPException(401, "请先选择一个员工账号")
    return emp_id


def sse(events) -> StreamingResponse:
    """把事件流包装成 SSE 响应：每个事件一行 data: {JSON}，以空行结束。"""
    if state["in_flight"] >= MAX_IN_FLIGHT:
        raise HTTPException(429, "当前请求过多，请稍后再试")

    async def body():
        state["in_flight"] += 1
        try:
            async for event in events:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            state["in_flight"] -= 1

    return StreamingResponse(body(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
async def index():
    return FileResponse(config.ROOT / "app" / "static" / "index.html")


@app.get("/api/employees")
async def employees():
    return [db.get_employee(emp_id) for emp_id in ("E1001", "E1002", "E1003")]


@app.post("/api/chat")
async def chat(request: ChatRequest, x_emp_id: str | None = Header(default=None)):
    emp_id = current_employee(x_emp_id)
    return sse(state["assistant"].chat(emp_id, request.session_id, request.message.strip()))


@app.post("/api/confirm")
async def confirm(request: ConfirmRequest, x_emp_id: str | None = Header(default=None)):
    emp_id = current_employee(x_emp_id)
    return sse(state["assistant"].confirm(emp_id, request.session_id, request.approved))


@app.get("/api/metrics")
async def metrics():
    return summarize(read_traces())


@app.get("/api/traces")
async def traces(limit: int = 20):
    """最近的若干条轨迹，用于排查问题。"""
    return read_traces()[-limit:][::-1]
