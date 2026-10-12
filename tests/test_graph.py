"""整张图的行为。模型用脚本化的替身，MCP 服务器真实启动，不需要密钥和 Docker。"""
import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app import config, db, prompts
from app.graph import parse_intent, strip_json
from app.observability import read_traces, summarize
from tests.fakes import answer, calls, run_session, scripted, tool_call, types

LEAVE = dict(leave_type="年假", start_date="2026-10-16", end_date="2026-10-16")


@pytest.fixture(autouse=True)
def clean_traces():
    config.TRACE_PATH.unlink(missing_ok=True)


@pytest.mark.parametrize("text, expected", [
    ("knowledge", "knowledge"), ("  Action\n", "action"), ("类别：personal", "personal"),
    ("不是 knowledge，应该是 complex", "complex"), ("我不确定", "knowledge"), ("", "knowledge"),
])
def test_parse_intent(text, expected):
    assert parse_intent(text) == expected


def test_strip_json_and_calendar():
    assert strip_json('```json\n{"steps": ["a"]}\n```') == '{"steps": ["a"]}'
    lines = prompts.calendar_text().splitlines()
    assert lines[0] == "2026-10-12 今天，本周星期一"
    assert "2026-10-16 本周星期五" in lines and "2026-10-21 下周星期三" in lines


def test_other_is_answered_by_router_only():
    router, worker = scripted("other", "你好，我是小远。"), scripted("不应当被调用")
    (events,), state = run_session(router, worker, [("chat", "E1001", "你好")])
    assert types(events)[0] == "intent" and types(events)[-1] == "done"
    assert state["messages"][-1].content == "你好，我是小远。"
    assert worker.i == 0                                   # 闲聊全程不调用接口模型
    assert events[-1]["cost"] == 0


def test_router_failure_falls_back_to_worker():
    router, worker = scripted(ConnectionError("路由模型连接不上")), scripted("other")
    (events,), state = run_session(router, worker, [("chat", "E1001", "你好")])
    assert events[0] == {"type": "intent", "intent": "other", "classifier": "worker"}
    assert answer(events) == prompts.CHITCHAT_FALLBACK


def test_knowledge_first_turn_skips_rewrite():
    router, worker = scripted("knowledge"), scripted("每年 5 天。[1]")
    (events,), state = run_session(router, worker, [("chat", "E1001", "年假有几天")])
    assert worker.i == 1                                   # 只有生成，没有改写
    assert state["query"] == "年假有几天" and state["finish"] == "answered"
    assert "[1] 《员工手册》年假" in worker.prompts[0][0].content
    assert state["tool_calls"] == []


def test_knowledge_follow_up_is_rewritten():
    router = scripted("knowledge")
    worker = scripted("每年 5 天。[1]", "病假需要什么证明", "需要诊断证明。[1]")
    _, state = run_session(router, worker, [("chat", "E1001", "年假有几天"),
                                            ("chat", "E1001", "那病假呢")])
    assert state["query"] == "病假需要什么证明"
    assert [u["node"] for u in state["usage"]] == ["classify", "rewrite", "generate"]   # 只含第二轮


def test_knowledge_without_hits_refuses_without_calling_model():
    router, worker = scripted("knowledge"), scripted("不应当被调用")
    (events,), state = run_session(router, worker, [("chat", "E1001", "今天天气怎么样")])
    assert answer(events) == prompts.NO_ANSWER and state["finish"] == "no_answer"
    assert worker.i == 0


def test_personal_query_uses_tool_without_confirmation():
    router = scripted("personal")
    worker = scripted(calls(tool_call("get_leave_balance")), "还剩 3 天。")
    (events,), state = run_session(router, worker, [("chat", "E1001", "我还剩几天年假")])
    assert "confirm" not in types(events)
    record = state["tool_calls"][0]
    assert record["name"] == "get_leave_balance" and record["confirmed"] is None
    assert '"remaining": 3' in record["result"]


def test_write_waits_for_confirmation():
    router = scripted("action")
    worker = scripted(calls(tool_call("submit_leave_request", **LEAVE)), "已提交，单号 QJ0003。")
    (first, second), state = run_session(router, worker, [
        ("chat", "E1001", "帮我请周五一天年假"), ("confirm", "E1001", True)])
    assert first[-1]["type"] == "confirm"
    assert first[-1]["summary"] == "提交请假单：年假，2026-10-16 至 2026-10-16"
    assert types(second)[-1] == "done"
    assert db.get_leave_balance("E1001")["remaining"] == 2
    assert state["tool_calls"][0]["confirmed"] is True
    traces = read_traces()
    assert len(traces) == 1 and traces[0]["question"] == "帮我请周五一天年假"   # 确认前后是同一条轨迹


def test_nothing_is_written_before_confirmation():
    router = scripted("action")
    worker = scripted(calls(tool_call("submit_leave_request", **LEAVE)))
    run_session(router, worker, [("chat", "E1001", "帮我请周五一天年假")])
    assert db.get_leave_balance("E1001")["remaining"] == 3
    assert len(db.list_leave_requests("E1001")) == 1


def test_rejected_write_is_not_executed():
    router = scripted("action")
    worker = scripted(calls(tool_call("submit_leave_request", **LEAVE)), "好的，没有提交。")
    (_, second), state = run_session(router, worker, [
        ("chat", "E1001", "帮我请周五一天年假"), ("confirm", "E1001", False)])
    assert db.get_leave_balance("E1001")["remaining"] == 3
    assert state["tool_calls"][0]["confirmed"] is False
    assert "取消" in state["messages"][-2].content        # 模型看到的工具结果


def test_two_writes_in_one_message_each_run_once():
    """恢复时节点会从头重跑：两个写入必须各执行一次，不能重复。"""
    router = scripted("complex")
    worker = scripted(
        '{"steps": ["报修", "订会议室"]}',
        calls(tool_call("create_ticket", "c1", category="硬件", description="电脑开机蓝屏"),
              tool_call("book_meeting_room", "c2", room="小会议室A", day="2026-10-15",
                        start_hour=14, end_hour=15)),
        "都办好了。")
    (first, second, third), state = run_session(router, worker, [
        ("chat", "E1001", "电脑蓝屏了帮我报修，再订周四下午两点的小会议室A"),
        ("confirm", "E1001", True), ("confirm", "E1001", True)])
    assert first[-1]["type"] == "confirm" and second[-1]["type"] == "confirm"
    assert second[-1]["tool"] == "book_meeting_room"
    assert types(third)[-1] == "done"
    assert len(db.list_tickets("E1001")) == 2              # 原有 1 张，新增 1 张
    assert len(db.list_room_bookings("E1001")) == 1
    assert all("error" not in record["result"] for record in state["tool_calls"])
    assert state["plan"] == ["报修", "订会议室"]


def test_new_message_is_refused_while_confirmation_is_pending():
    router = scripted("action")
    worker = scripted(calls(tool_call("submit_leave_request", **LEAVE)))
    (_, second), _ = run_session(router, worker, [
        ("chat", "E1001", "帮我请周五一天年假"), ("chat", "E1001", "算了，查一下年假")])
    assert types(second) == ["error"] and "等待确认" in second[0]["message"]


def test_step_limit_stops_the_loop_and_keeps_history_valid():
    router = scripted("personal")
    worker = scripted(calls(tool_call("get_leave_balance")))      # 永远只会调用工具
    (events,), state = run_session(router, worker, [("chat", "E1001", "我还剩几天年假")])
    assert state["finish"] == "max_steps" and state["steps"] == config.MAX_AGENT_STEPS
    assert answer(events) == prompts.MAX_STEPS_MESSAGE
    assert len(state["tool_calls"]) == config.MAX_AGENT_STEPS - 1
    # 每个工具调用都有对应的结果，消息历史可以继续用于下一轮
    pending = set()
    for message in state["messages"]:
        if isinstance(message, AIMessage):
            assert not pending
            pending = {call["id"] for call in message.tool_calls}
        elif isinstance(message, ToolMessage):
            pending.discard(message.tool_call_id)
    assert not pending


def test_invalid_plan_falls_back_to_plain_agent():
    router = scripted("complex")
    worker = scripted("这不是 JSON", "请问你要办理哪几件事？")
    _, state = run_session(router, worker, [("chat", "E1001", "帮我安排一下")])
    assert state["plan"] == [] and state["finish"] == "answered"


def test_agent_narration_before_tool_call_is_retracted():
    router = scripted("personal")
    worker = scripted(calls(tool_call("get_leave_balance"), content="Let me check."), "还剩 3 天。")
    (events,), _ = run_session(router, worker, [("chat", "E1001", "我还剩几天年假")])
    assert "retract" in types(events)


def test_model_call_is_retried_when_connection_drops():
    """流式响应传到一半被对方断开：重试之后请求照常完成，轨迹里记下重试的次数。"""
    dropped = httpx.RemoteProtocolError("peer closed connection without sending complete message body")
    router = scripted("personal")
    worker = scripted(calls(tool_call("get_leave_balance")), dropped, "还剩 3 天。")
    (events,), state = run_session(router, worker, [("chat", "E1001", "我还剩几天年假")])
    assert types(events)[-1] == "done" and state["messages"][-1].content == "还剩 3 天。"
    assert [u["retries"] for u in state["usage"]] == [0, 0, 1]
    assert len(state["tool_calls"]) == 1                    # 重试的是模型调用，工具没有被重复执行
    assert summarize(read_traces())["model_retries"] == 1


def test_connection_error_is_reported_after_retries_are_exhausted():
    dropped = httpx.RemoteProtocolError("peer closed connection")
    router, worker = scripted("personal"), scripted(dropped)        # 主力模型一直连不上
    (events,), _ = run_session(router, worker, [("chat", "E1001", "我还剩几天年假")])
    assert types(events)[-1] == "error" and "RemoteProtocolError" in events[-1]["message"]
    assert worker.i == 4                                            # 第一次加上三次重试
    assert read_traces()[-1]["finish"] == "error"


def test_other_errors_are_not_retried():
    router, worker = scripted("personal"), scripted(ValueError("提示词太长"))
    (events,), _ = run_session(router, worker, [("chat", "E1001", "我还剩几天年假")])
    assert types(events)[-1] == "error" and worker.i == 1


def test_trace_and_metrics():
    router = scripted("personal", "other", "你好。")
    worker = scripted(calls(tool_call("get_leave_balance")), "还剩 3 天。")
    run_session(router, worker, [("chat", "E1001", "我还剩几天年假"), ("chat", "E1001", "你好")])
    first, second = read_traces()
    assert first["intent"] == "personal" and first["steps"] == 2
    assert [n["node"] for n in first["nodes"]] == ["classify", "agent", "tools", "agent"]
    assert first["usage"]["worker"]["calls"] == 2 and first["answer"] == "还剩 3 天。"
    assert second["intent"] == "other" and second["tool_calls"] == []
    summary = summarize([first, second])
    assert summary["requests"] == 2 and summary["tool_calls"] == {"get_leave_balance": 1}
    assert summary["by_intent"]["personal"]["requests"] == 1 and summary["error_rate"] == 0
