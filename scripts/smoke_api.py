"""对正在运行的服务做一次冒烟检查：闲聊、制度问答、请假加确认、越权查询、指标。

用法：先启动服务，然后 python -m scripts.smoke_api [http://127.0.0.1:8080]
检查会真实地提交一张请假单；想恢复演示数据时，停掉服务后运行 python -m scripts.reset_data。
"""
import json
import sys
import time

import httpx

# 用 127.0.0.1 而不是 localhost：Windows 上 localhost 会先尝试 IPv6，每个请求白等两秒
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
SESSION = f"smoke{int(time.time())}"


def post(path: str, body: dict, emp_id: str = "E1001") -> list[dict]:
    """发送请求并读完 SSE，返回全部事件；同时打印每个事件到达的时间。"""
    events, started = [], time.perf_counter()
    with httpx.stream("POST", BASE + path, json={"session_id": SESSION, **body},
                      headers={"X-Emp-Id": emp_id}, timeout=120) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line.startswith("data: "):
                event = json.loads(line[6:])
                events.append({**event, "_at": round(time.perf_counter() - started, 2)})
    return events


def show(title: str, events: list[dict]) -> None:
    print(f"== {title}")
    tokens = [e for e in events if e["type"] == "token"]
    for event in events:
        if event["type"] == "token":
            continue
        brief = {k: v for k, v in event.items() if k not in ("type", "_at", "usage", "answer", "args")}
        print(f"  {event['_at']:>5}s {event['type']:<8} {json.dumps(brief, ensure_ascii=False)}")
    if tokens:
        print(f"  正文 {len(tokens)} 个片段，首个在 {tokens[0]['_at']}s，末个在 {tokens[-1]['_at']}s："
              f"{''.join(t['text'] for t in tokens)}")


if __name__ == "__main__":
    for attempt in range(60):                       # 等待服务启动
        try:
            httpx.get(BASE + "/api/employees", timeout=2).raise_for_status()
            break
        except httpx.HTTPError:
            time.sleep(1)
    print("没有工号：", httpx.post(BASE + "/api/chat", json={"session_id": "x", "message": "你好"}).status_code)
    show("闲聊", post("/api/chat", {"message": "你好"}))
    show("制度问答", post("/api/chat", {"message": "去上海出差住宿一晚能报多少"}))
    show("请假（等待确认）", post("/api/chat", {"message": "帮我请这周五一天年假"}))
    show("确认之前再发消息", post("/api/chat", {"message": "算了"}))
    show("确认", post("/api/confirm", {"approved": True}))
    show("越权查询", post("/api/chat", {"message": "帮我查一下工号E1003王强还剩几天年假"}))
    print("== 指标")
    print(json.dumps(httpx.get(BASE + "/api/metrics").json(), ensure_ascii=False, indent=1))
