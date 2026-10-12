"""人事系统的 MCP 服务器：假期、出差申请、报销进度。

独立的程序，由宿主以 stdio 方式启动：python -m app.mcp_servers.hr_server
每个工具都有 emp_id 参数。宿主不会把这个参数交给模型填写，而是在调用时填入当前登录的工号。
"""
import json
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from app import db

mcp = MCPServer("yuanhang-hr")


def _reply(function, *args, **kwargs) -> str:
    """执行数据层的函数，把结果或业务错误都转成 JSON 文本。"""
    try:
        return json.dumps(function(*args, **kwargs), ensure_ascii=False)
    except db.BusinessError as error:
        return json.dumps({"error": str(error)}, ensure_ascii=False)


@mcp.tool()
def get_leave_balance(emp_id: str) -> str:
    """查询员工本人今年的年假：总天数、已用天数、剩余天数。"""
    return _reply(db.get_leave_balance, emp_id)


@mcp.tool()
def list_leave_requests(emp_id: str) -> str:
    """列出员工本人提交过的全部请假单，包括类型、起止日期和审批状态。"""
    return _reply(db.list_leave_requests, emp_id)


@mcp.tool()
def submit_leave_request(
    emp_id: str,
    leave_type: Annotated[str, Field(description="假期类型，只能是 年假、病假、事假、调休 之一")],
    start_date: Annotated[str, Field(description="开始日期，格式 YYYY-MM-DD")],
    end_date: Annotated[str, Field(description="结束日期，格式 YYYY-MM-DD；只请一天时与开始日期相同")],
    reason: Annotated[str, Field(description="请假的原因，用户没有说明时留空")] = "",
) -> str:
    """提交一张请假单。年假余额是否足够由系统检查。"""
    return _reply(db.submit_leave_request, emp_id, leave_type, start_date, end_date, reason)


@mcp.tool()
def list_trip_requests(emp_id: str) -> str:
    """列出员工本人提交过的全部出差申请。"""
    return _reply(db.list_trip_requests, emp_id)


@mcp.tool()
def submit_trip_request(
    emp_id: str,
    destination: Annotated[str, Field(description="出差的城市，例如 上海")],
    start_date: Annotated[str, Field(description="出发日期，格式 YYYY-MM-DD")],
    end_date: Annotated[str, Field(description="返回日期，格式 YYYY-MM-DD")],
    reason: Annotated[str, Field(description="出差的事由，用户没有说明时留空")] = "",
) -> str:
    """提交一张出差申请。"""
    return _reply(db.submit_trip_request, emp_id, destination, start_date, end_date, reason)


@mcp.tool()
def get_expense_status(
    emp_id: str,
    expense_id: Annotated[str, Field(description="报销单号，例如 BX20260901；留空时返回本人的全部报销单")] = "",
) -> str:
    """查询员工本人报销单的金额、类别和处理进度。"""
    return _reply(db.get_expense_status, emp_id, expense_id or None)


@mcp.resource("hr://leave-types")
def leave_types() -> str:
    """公司的假期类型及一句话说明。"""
    return ("年假：按工龄享有，带薪。\n病假：因病无法上班，按基本工资的 80% 发放。\n"
            "事假：因私事请假，无薪。\n调休：用加班时长换取的休息。")


@mcp.prompt()
def leave_application(leave_type: str, start_date: str, end_date: str) -> str:
    """请假申请的模板：让模型按公司的格式起草一段请假说明。"""
    return (f"请替我起草一段{leave_type}申请的说明，时间是 {start_date} 至 {end_date}。"
            "要求：不超过 80 字，写明工作交接的安排，语气正式。")


if __name__ == "__main__":
    mcp.run(transport="stdio")
