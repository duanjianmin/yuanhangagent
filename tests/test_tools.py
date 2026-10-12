"""工具层：MCP 服务器真实启动（stdio 子进程），检索用假的替身，不需要模型和 Docker。"""
import asyncio
import json

from app import db
from app.rag.ingest import Chunk
from app.rag.retriever import Hit
from app.tools import ToolBox, describe_action


class FakeRetriever:
    def search(self, query, mode="hybrid_rerank", top_k=3):
        if "天气" in query:
            return [Hit(Chunk("x#01", "员工手册", "年假", "无关"), 0.001)]
        return [Hit(Chunk("员工手册#02", "员工手册", "年假", "满 1 年不满 10 年的，每年 5 天"), 0.9)]


def run(coroutine_function):
    async def main():
        async with ToolBox(FakeRetriever()) as toolbox:
            return await coroutine_function(toolbox)
    return asyncio.run(main())


def test_tools_are_merged_and_emp_id_is_hidden():
    async def check(toolbox):
        assert set(toolbox.tools) == {
            "search_policy", "list_meeting_rooms", "book_meeting_room",
            "get_leave_balance", "list_leave_requests", "submit_leave_request",
            "list_trip_requests", "submit_trip_request", "get_expense_status",
            "list_my_tickets", "create_ticket"}
        for schema in toolbox.schemas():
            parameters = schema["function"]["parameters"]
            assert "emp_id" not in parameters["properties"]
            assert "emp_id" not in parameters.get("required", [])
        leave = toolbox.tools["submit_leave_request"]
        assert leave.source == "hr" and leave.is_write
        assert "YYYY-MM-DD" in leave.parameters["properties"]["start_date"]["description"]
        assert not toolbox.tools["get_leave_balance"].is_write
    run(check)


def test_emp_id_is_bound_by_host():
    """模型即使在参数中写了别人的工号，也会被宿主覆盖。"""
    async def check(toolbox):
        mine = json.loads(await toolbox.call("get_leave_balance", {}, "E1001"))
        forged = json.loads(await toolbox.call("get_leave_balance", {"emp_id": "E1002"}, "E1001"))
        assert mine == forged and mine["remaining"] == 3
        other = json.loads(await toolbox.call("get_expense_status", {"expense_id": "BX20260820"}, "E1001"))
        assert other == []
    run(check)


def test_write_through_mcp_changes_database():
    async def check(toolbox):
        args = {"leave_type": "年假", "start_date": "2026-10-16", "end_date": "2026-10-16"}
        result = json.loads(await toolbox.call("submit_leave_request", args, "E1001"))
        assert result["status"] == "待审批"
        assert db.get_leave_balance("E1001")["remaining"] == 2
        ticket = json.loads(await toolbox.call(
            "create_ticket", {"category": "网络", "description": "VPN 报错 E-809"}, "E1001"))
        assert ticket["id"] == "IT0003"
    run(check)


def test_errors_are_returned_as_observations():
    async def check(toolbox):
        business = await toolbox.call("submit_leave_request", {
            "leave_type": "年假", "start_date": "2026-10-16", "end_date": "2026-10-16"}, "E1002")
        assert "余额不足" in business
        missing = await toolbox.call("submit_leave_request", {"leave_type": "年假"}, "E1001")
        assert "error" in missing and "start_date" in missing
        bad_type = await toolbox.call("list_meeting_rooms", {"day": "2026-10-13", "start_hour": "下午"}, "E1001")
        assert "参数不合法" in bad_type
        conflict = await toolbox.call("book_meeting_room", {
            "room": "大会议室", "day": "2026-10-13", "start_hour": 14, "end_hour": 15}, "E1001")
        assert "已经被预订" in conflict
        assert "没有名为" in await toolbox.call("delete_everything", {}, "E1001")
    run(check)


def test_search_policy():
    async def check(toolbox):
        found = await toolbox.call("search_policy", {"query": "年假有几天"}, "E1001")
        assert found.startswith("[1] 《员工手册》年假")
        assert "没有找到" in await toolbox.call("search_policy", {"query": "今天天气"}, "E1001")
    run(check)


def test_describe_action():
    text = describe_action("submit_leave_request", {
        "leave_type": "年假", "start_date": "2026-10-16", "end_date": "2026-10-16"})
    assert text == "提交请假单：年假，2026-10-16 至 2026-10-16"
    assert "紧急" in describe_action("create_ticket", {"category": "网络", "description": "断网", "urgent": True})
