"""两个模型角色的封装：router（路由模型，识别意图和回复闲聊）和 worker（主力模型，其余全部）。
两个角色默认是同一个模型，可以在配置里分别更换。

所有的模型调用都经过 call()，并发限制、超时、用量统计集中在这一处。
"""
import asyncio
import time

import httpx
import openai
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from app import config

_models: dict[str, BaseChatModel] = {}
_semaphores: dict[str, asyncio.Semaphore] = {}
_LIMITS = {"router": config.ROUTER_CONCURRENCY, "worker": config.WORKER_CONCURRENCY}
_TIMEOUTS = {"router": config.ROUTER_TIMEOUT, "worker": config.WORKER_TIMEOUT}
# 连接中断时重试的次数。路由模型失败之后有主力模型兜底，所以只重试一次。
_RETRIES = {"router": 1, "worker": 3}


def get_model(role: str) -> BaseChatModel:
    if role not in _models:
        if role == "router":
            _models[role] = ChatOpenAI(
                model=config.ROUTER_MODEL, base_url=config.ROUTER_BASE_URL,
                api_key=config.ROUTER_API_KEY, temperature=0, max_retries=0, stream_usage=True)
        else:
            _models[role] = ChatOpenAI(
                model=config.WORKER_MODEL, base_url=config.WORKER_BASE_URL,
                api_key=config.WORKER_API_KEY, temperature=0,
                max_retries=3,                 # 遇到限流和网络错误时退避重试
                stream_usage=True)             # 流式输出时也返回词元用量
    return _models[role]


def set_model(role: str, model: BaseChatModel | None) -> None:
    """替换某个角色的模型。测试时换成脚本化的假模型；传入 None 则恢复默认。"""
    if model is None:
        _models.pop(role, None)
    else:
        _models[role] = model


def _semaphore(role: str) -> asyncio.Semaphore:
    # 信号量要在事件循环里创建，所以延迟到第一次使用
    if role not in _semaphores:
        _semaphores[role] = asyncio.Semaphore(_LIMITS[role])
    return _semaphores[role]


def is_connection_error(error: BaseException) -> bool:
    """这个异常（或者引发它的底层异常）是不是连接层面的错误：连不上、连接被对方中途断开。"""
    seen = 0
    while error is not None and seen < 6:
        if isinstance(error, (httpx.TransportError, openai.APIConnectionError)):
            return not isinstance(error, (httpx.TimeoutException, openai.APITimeoutError))
        error, seen = error.__cause__ or error.__context__, seen + 1
    return False


def cost(role: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
    """按单价表折算费用，单位：元。输入中命中缓存的部分按缓存的价格计算。"""
    price = config.PRICES[role]
    return ((input_tokens - cached_tokens) * price["input"] + cached_tokens * price["cached"]
            + output_tokens * price["output"]) / 1_000_000


async def call(role: str, messages: list, node: str, tools: list[dict] | None = None,
               **kwargs) -> tuple[AIMessage, dict]:
    """调用一次模型，返回 (模型的回复, 用量记录)。

    用量记录的字段：role、node、input_tokens、cached_tokens、output_tokens、seconds、cost、retries。
    cached_tokens 是输入中命中服务端前缀缓存的部分，包含在 input_tokens 之内。
    retries 是因为连接中断而重试的次数；seconds 包含重试所花的时间。
    """
    model = get_model(role)
    if tools:
        model = model.bind_tools(tools)
    if kwargs:
        model = model.bind(**kwargs)
    async with _semaphore(role):
        started = time.perf_counter()
        retries = 0
        while True:
            try:
                reply = await asyncio.wait_for(model.ainvoke(messages), timeout=_TIMEOUTS[role])
                break
            except Exception as error:
                # 连接中断时重试。SDK 自带的重试（max_retries）只覆盖请求没有发出去的情况；
                # 流式响应传到一半被对方断开，要在这里处理。
                if retries >= _RETRIES[role] or not is_connection_error(error):
                    raise
                retries += 1
                await asyncio.sleep(0.5 * retries)
        seconds = time.perf_counter() - started
    usage = reply.usage_metadata or {}
    input_tokens, output_tokens = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    cached_tokens = (usage.get("input_token_details") or {}).get("cache_read") or 0
    record = {"role": role, "node": node, "input_tokens": input_tokens, "cached_tokens": cached_tokens,
              "output_tokens": output_tokens, "seconds": round(seconds, 3),
              "cost": round(cost(role, input_tokens, output_tokens, cached_tokens), 6),
              "retries": retries}
    return reply, record
