"""数据层的规则与数据隔离。"""
import pytest

from app import db
from app.db import BusinessError


def test_leave_balance():
    assert db.get_leave_balance("E1001") == {"year": 2026, "total": 5, "used": 2, "remaining": 3}
    assert db.get_leave_balance("E1002")["remaining"] == 0


def test_submit_annual_leave_updates_balance():
    result = db.submit_leave_request("E1001", "年假", "2026-10-16", "2026-10-16", "家里有事")
    assert result["days"] == 1 and result["status"] == "待审批"
    assert db.get_leave_balance("E1001")["remaining"] == 2
    assert db.list_leave_requests("E1001")[-1]["id"] == result["id"]


def test_leave_counts_workdays_only():
    # 10 月 16 日是星期五，10 月 19 日是星期一，中间的周末不算
    assert db.submit_leave_request("E1001", "事假", "2026-10-16", "2026-10-19")["days"] == 2


@pytest.mark.parametrize("emp_id, args, message", [
    ("E1002", ("年假", "2026-10-16", "2026-10-16"), "余额不足"),
    ("E1001", ("年假", "2026-10-19", "2026-10-23"), "余额不足"),       # 5 天，只剩 3 天
    ("E1001", ("年假", "2026-10-01", "2026-10-01"), "不能早于今天"),
    ("E1001", ("年假", "2026-10-20", "2026-10-19"), "不能早于开始日期"),
    ("E1001", ("年假", "10月16日", "10月16日"), "日期格式"),
    ("E1001", ("探亲假", "2026-10-16", "2026-10-16"), "假期类型"),
    ("E1001", ("事假", "2026-10-17", "2026-10-18"), "休息日"),
])
def test_leave_rules(emp_id, args, message):
    with pytest.raises(BusinessError, match=message):
        db.submit_leave_request(emp_id, *args)
    assert db.get_leave_balance("E1001")["remaining"] == 3        # 失败的申请不改变余额


def test_leave_overlap():
    db.submit_leave_request("E1001", "事假", "2026-10-16", "2026-10-16")
    with pytest.raises(BusinessError, match="已经有请假单"):
        db.submit_leave_request("E1001", "年假", "2026-10-15", "2026-10-16")


def test_isolation():
    """张伟查不到王强的任何数据。"""
    assert [e["id"] for e in db.get_expense_status("E1001")] == ["BX20260901", "BX20260915"]
    assert db.get_expense_status("E1001", "BX20260820") == []      # 王强的单号
    assert db.get_expense_status("E1003", "bx20260820")[0]["status"] == "待付款"
    assert [t["id"] for t in db.list_tickets("E1001")] == ["IT0001"]
    assert all(r["id"] != "QJ0002" for r in db.list_leave_requests("E1001"))
    assert db.list_room_bookings("E1001") == []


def test_ticket():
    ticket = db.create_ticket("E1001", "网络", "在家连不上 VPN，报错 E-809", urgent=False)
    assert ticket["id"] == "IT0003"
    assert db.list_tickets("E1001")[0]["description"].endswith("E-809")
    with pytest.raises(BusinessError, match="工单类别"):
        db.create_ticket("E1001", "其他", "电脑开不了机")


def test_trip():
    trip = db.submit_trip_request("E1001", "上海", "2026-10-21", "2026-10-22", "拜访客户")
    assert trip["id"] == "CC0001"
    assert db.list_trip_requests("E1003") == []


def test_rooms():
    free = [r["room"] for r in db.list_free_rooms("2026-10-13", 15, 16)]
    assert free == ["小会议室A", "小会议室B"]                    # 大会议室 14 至 16 点已被预订
    with pytest.raises(BusinessError, match="已经被预订"):
        db.book_room("E1001", "大会议室", "2026-10-13", 15, 17)
    assert db.book_room("E1001", "大会议室", "2026-10-13", 16, 18)["id"] == "HY0002"


@pytest.mark.parametrize("day, start, end, message", [
    ("2026-10-13", 7, 9, "8 点至 20 点"),
    ("2026-10-13", 9, 14, "最长 4 小时"),
    ("2026-10-11", 9, 10, "过去的日期"),
    ("2026-10-27", 9, 10, "最多提前 14 天"),
])
def test_room_rules(day, start, end, message):
    with pytest.raises(BusinessError, match=message):
        db.book_room("E1001", "小会议室A", day, start, end)
