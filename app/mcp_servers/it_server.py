"""IT 工单系统的 MCP 服务器。启动方式：python -m app.mcp_servers.it_server"""
import json
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from app import db

mcp = MCPServer("yuanhang-it")


@mcp.tool()
def list_my_tickets(emp_id: str) -> str:
    """列出员工本人提交过的 IT 工单及处理状态。"""
    return json.dumps(db.list_tickets(emp_id), ensure_ascii=False)


@mcp.tool()
def create_ticket(
    emp_id: str,
    category: Annotated[str, Field(description="工单类别，只能是 硬件、软件、网络、账号 之一")],
    description: Annotated[str, Field(description="故障现象，尽量包含设备、报错信息和出现的时间")],
    urgent: Annotated[bool, Field(description="是否紧急。只有影响多人或者完全无法工作时才为 true")] = False,
) -> str:
    """提交一张 IT 报修工单。"""
    try:
        return json.dumps(db.create_ticket(emp_id, category, description, urgent), ensure_ascii=False)
    except db.BusinessError as error:
        return json.dumps({"error": str(error)}, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
