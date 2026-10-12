"""业务数据层。所有函数的第一个参数都是工号，查询只返回这名员工自己的数据；
写入之前的业务规则在这里用代码检查，不依赖模型。"""
import sqlite3
from contextlib import contextmanager
from datetime import date, timedelta

from app import config

LEAVE_TYPES = ("年假", "病假", "事假", "调休")
TICKET_CATEGORIES = ("硬件", "软件", "网络", "账号")
ROOMS = {"大会议室": 20, "小会议室A": 6, "小会议室B": 6}

SCHEMA = """
CREATE TABLE employees (emp_id TEXT PRIMARY KEY, name TEXT, department TEXT, hire_date TEXT);
CREATE TABLE leave_balances (emp_id TEXT, year INTEGER, total INTEGER, used INTEGER,
                             PRIMARY KEY (emp_id, year));
CREATE TABLE leave_requests (id TEXT PRIMARY KEY, emp_id TEXT, leave_type TEXT, start_date TEXT,
                             end_date TEXT, days INTEGER, reason TEXT, status TEXT);
CREATE TABLE trip_requests (id TEXT PRIMARY KEY, emp_id TEXT, destination TEXT, start_date TEXT,
                            end_date TEXT, reason TEXT, status TEXT);
CREATE TABLE expenses (id TEXT PRIMARY KEY, emp_id TEXT, amount REAL, category TEXT,
                       submitted_on TEXT, status TEXT);
CREATE TABLE tickets (id TEXT PRIMARY KEY, emp_id TEXT, category TEXT, description TEXT,
                      urgent INTEGER, created_on TEXT, status TEXT);
CREATE TABLE room_bookings (id TEXT PRIMARY KEY, room TEXT, day TEXT, start_hour INTEGER,
                            end_hour INTEGER, emp_id TEXT);
"""

SEED = {
    "employees": [
        ("E1001", "张伟", "销售部", "2019-03-01"),
        ("E1002", "李娜", "财务部", "2014-06-01"),
        ("E1003", "王强", "仓储部", "2022-08-15"),
    ],
    "leave_balances": [("E1001", 2026, 5, 2), ("E1002", 2026, 10, 10), ("E1003", 2026, 5, 1)],
    "leave_requests": [
        ("QJ0001", "E1001", "年假", "2026-07-06", "2026-07-07", 2, "家中有事", "已批准"),
        ("QJ0002", "E1003", "年假", "2026-05-20", "2026-05-20", 1, "个人事务", "已批准"),
    ],
    "trip_requests": [],
    "expenses": [
        ("BX20260901", "E1001", 1280.00, "差旅费", "2026-09-28", "财务审核中"),
        ("BX20260915", "E1001", 86.50, "快递费", "2026-09-16", "已付款"),
        ("BX20260820", "E1003", 2350.00, "差旅费", "2026-09-02", "待付款"),
        ("BX20260905", "E1002", 430.00, "业务招待费", "2026-09-08", "已退回"),
    ],
    "tickets": [
        ("IT0001", "E1001", "硬件", "工位的打印机无法连接", 0, "2026-10-08", "处理中"),
        ("IT0002", "E1003", "硬件", "仓库的扫码枪无法识别条码", 1, "2026-10-09", "待处理"),
    ],
    "room_bookings": [("HY0001", "大会议室", "2026-10-13", 14, 16, "E1003")],
}


class BusinessError(Exception):
    """业务规则不允许的操作。消息会原样返回给模型，再由模型转述给用户。"""


@contextmanager
def connect():
    connection = sqlite3.connect(config.DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db() -> None:
    """重建全部的表并写入演示数据。

    这里删除表而不是删除数据库文件：MCP 服务器是另外的进程，文件可能正被它们使用。
    """
    config.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    with connect() as db:
        for table in SEED:
            db.execute(f"DROP TABLE IF EXISTS {table}")
        db.executescript(SCHEMA)
        for table, rows in SEED.items():
            if rows:
                marks = ",".join("?" * len(rows[0]))
                db.executemany(f"INSERT INTO {table} VALUES ({marks})", rows)


def _rows(sql: str, *args) -> list[dict]:
    with connect() as db:
        return [dict(r) for r in db.execute(sql, args)]


def _next_id(db, table: str, prefix: str) -> str:
    count = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return f"{prefix}{count + 1:04d}"


def _parse_range(start_date: str, end_date: str) -> tuple[date, date]:
    try:
        start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    except ValueError:
        raise BusinessError("日期格式不正确，应为 YYYY-MM-DD")
    if end < start:
        raise BusinessError("结束日期不能早于开始日期")
    if start < config.today():
        raise BusinessError(f"开始日期不能早于今天（{config.today()}）")
    return start, end


def _workdays(start: date, end: date) -> int:
    days = (end - start).days + 1
    return sum((start + timedelta(n)).weekday() < 5 for n in range(days))


# ---------- 员工 ----------

def get_employee(emp_id: str) -> dict | None:
    rows = _rows("SELECT * FROM employees WHERE emp_id = ?", emp_id)
    return rows[0] if rows else None


# ---------- 假期 ----------

def get_leave_balance(emp_id: str) -> dict:
    year = config.today().year
    rows = _rows("SELECT total, used FROM leave_balances WHERE emp_id = ? AND year = ?", emp_id, year)
    total, used = (rows[0]["total"], rows[0]["used"]) if rows else (0, 0)
    return {"year": year, "total": total, "used": used, "remaining": total - used}


def list_leave_requests(emp_id: str) -> list[dict]:
    return _rows("SELECT id, leave_type, start_date, end_date, days, reason, status "
                 "FROM leave_requests WHERE emp_id = ? ORDER BY start_date", emp_id)


def submit_leave_request(emp_id: str, leave_type: str, start_date: str, end_date: str,
                         reason: str = "") -> dict:
    if leave_type not in LEAVE_TYPES:
        raise BusinessError(f"假期类型只能是：{'、'.join(LEAVE_TYPES)}")
    start, end = _parse_range(start_date, end_date)
    days = _workdays(start, end)
    if days == 0:
        raise BusinessError("所选的日期都是休息日，不需要请假")
    with connect() as db:
        overlap = db.execute(
            "SELECT id FROM leave_requests WHERE emp_id = ? AND start_date <= ? AND end_date >= ?",
            (emp_id, end_date, start_date)).fetchone()
        if overlap:
            raise BusinessError(f"这段时间已经有请假单 {overlap['id']}，不能重复申请")
        if leave_type == "年假":
            remaining = get_leave_balance(emp_id)["remaining"]
            if days > remaining:
                raise BusinessError(f"年假余额不足：申请 {days} 天，剩余 {remaining} 天")
            db.execute("UPDATE leave_balances SET used = used + ? WHERE emp_id = ? AND year = ?",
                       (days, emp_id, config.today().year))
        request_id = _next_id(db, "leave_requests", "QJ")
        db.execute("INSERT INTO leave_requests VALUES (?,?,?,?,?,?,?,?)",
                   (request_id, emp_id, leave_type, start_date, end_date, days, reason, "待审批"))
    return {"id": request_id, "leave_type": leave_type, "start_date": start_date,
            "end_date": end_date, "days": days, "status": "待审批"}


# ---------- 出差与报销 ----------

def list_trip_requests(emp_id: str) -> list[dict]:
    return _rows("SELECT id, destination, start_date, end_date, reason, status "
                 "FROM trip_requests WHERE emp_id = ? ORDER BY start_date", emp_id)


def submit_trip_request(emp_id: str, destination: str, start_date: str, end_date: str,
                        reason: str = "") -> dict:
    _parse_range(start_date, end_date)
    if not destination.strip():
        raise BusinessError("目的地不能为空")
    with connect() as db:
        request_id = _next_id(db, "trip_requests", "CC")
        db.execute("INSERT INTO trip_requests VALUES (?,?,?,?,?,?,?)",
                   (request_id, emp_id, destination, start_date, end_date, reason, "待审批"))
    return {"id": request_id, "destination": destination, "start_date": start_date,
            "end_date": end_date, "status": "待审批"}


def get_expense_status(emp_id: str, expense_id: str | None = None) -> list[dict]:
    """不传单号时返回本人的全部报销单。查询他人的单号与查询不存在的单号，结果相同。"""
    sql = "SELECT id, amount, category, submitted_on, status FROM expenses WHERE emp_id = ?"
    if expense_id:
        return _rows(sql + " AND id = ?", emp_id, expense_id.strip().upper())
    return _rows(sql + " ORDER BY submitted_on DESC", emp_id)


# ---------- IT 工单 ----------

def list_tickets(emp_id: str) -> list[dict]:
    return _rows("SELECT id, category, description, urgent, created_on, status "
                 "FROM tickets WHERE emp_id = ? ORDER BY created_on DESC", emp_id)


def create_ticket(emp_id: str, category: str, description: str, urgent: bool = False) -> dict:
    if category not in TICKET_CATEGORIES:
        raise BusinessError(f"工单类别只能是：{'、'.join(TICKET_CATEGORIES)}")
    if len(description.strip()) < 4:
        raise BusinessError("请把故障现象描述得具体一些")
    with connect() as db:
        ticket_id = _next_id(db, "tickets", "IT")
        db.execute("INSERT INTO tickets VALUES (?,?,?,?,?,?,?)",
                   (ticket_id, emp_id, category, description, int(urgent),
                    config.today().isoformat(), "待处理"))
    return {"id": ticket_id, "category": category, "urgent": urgent, "status": "待处理"}


# ---------- 会议室 ----------

def _check_slot(day: str, start_hour: int, end_hour: int) -> None:
    try:
        target = date.fromisoformat(day)
    except ValueError:
        raise BusinessError("日期格式不正确，应为 YYYY-MM-DD")
    if not 8 <= start_hour < end_hour <= 20:
        raise BusinessError("会议室可以预订的时间为 8 点至 20 点，结束时间要晚于开始时间")
    if end_hour - start_hour > 4:
        raise BusinessError("单次预订最长 4 小时")
    ahead = (target - config.today()).days
    if ahead < 0:
        raise BusinessError("不能预订过去的日期")
    if ahead > 14:
        raise BusinessError("最多提前 14 天预订")


def list_free_rooms(day: str, start_hour: int, end_hour: int) -> list[dict]:
    _check_slot(day, start_hour, end_hour)
    busy = {r["room"] for r in _rows(
        "SELECT room FROM room_bookings WHERE day = ? AND start_hour < ? AND end_hour > ?",
        day, end_hour, start_hour)}
    return [{"room": room, "capacity": capacity} for room, capacity in ROOMS.items()
            if room not in busy]


def list_room_bookings(emp_id: str) -> list[dict]:
    return _rows("SELECT id, room, day, start_hour, end_hour FROM room_bookings "
                 "WHERE emp_id = ? ORDER BY day, start_hour", emp_id)


def book_room(emp_id: str, room: str, day: str, start_hour: int, end_hour: int) -> dict:
    if room not in ROOMS:
        raise BusinessError(f"会议室只能是：{'、'.join(ROOMS)}")
    _check_slot(day, start_hour, end_hour)
    with connect() as db:
        conflict = db.execute(
            "SELECT 1 FROM room_bookings WHERE room = ? AND day = ? AND start_hour < ? AND end_hour > ?",
            (room, day, end_hour, start_hour)).fetchone()
        if conflict:
            raise BusinessError(f"{room}在这个时段已经被预订")
        booking_id = _next_id(db, "room_bookings", "HY")
        db.execute("INSERT INTO room_bookings VALUES (?,?,?,?,?,?)",
                   (booking_id, room, day, start_hour, end_hour, emp_id))
    return {"id": booking_id, "room": room, "day": day,
            "start_hour": start_hour, "end_hour": end_hour}
