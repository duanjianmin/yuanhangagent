"""项目配置。所有可调的值集中在这里，都可以用环境变量覆盖。"""
import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = ROOT / "data" / "knowledge"
RUNTIME_DIR = Path(os.getenv("YUANHANG_RUNTIME_DIR", ROOT / "runtime"))
DB_PATH = RUNTIME_DIR / "business.db"
CHECKPOINT_PATH = RUNTIME_DIR / "checkpoints.db"
TRACE_PATH = RUNTIME_DIR / "traces.jsonl"


def today() -> date:
    """当前日期。评估和测试时用 YUANHANG_TODAY 固定，结果才可以复现。"""
    value = os.getenv("YUANHANG_TODAY")
    return date.fromisoformat(value) if value else date.today()


# ---- 模型角色 ----
# worker（主力模型）负责改写、生成、工具调用和规划；
# router（路由模型）只做意图识别和闲聊回复，这两件事简单、调用频繁，适合用更小更便宜的模型。
# 两个角色默认都是 deepseek-chat：实测它识别意图又快又准（见 eval/run_intent.py）。
# 保留两个角色是为了可以单独更换：给意图识别换一个模型时，只需要设置 ROUTER_* 这一组环境变量。
WORKER_BASE_URL = os.getenv("WORKER_BASE_URL", "https://api.deepseek.com")
WORKER_API_KEY = os.getenv("WORKER_API_KEY", os.getenv("DEEPSEEK_API_KEY", ""))
WORKER_MODEL = os.getenv("WORKER_MODEL", "deepseek-chat")
WORKER_TIMEOUT = float(os.getenv("WORKER_TIMEOUT", "60"))
WORKER_CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", "8"))

ROUTER_BASE_URL = os.getenv("ROUTER_BASE_URL", WORKER_BASE_URL)
ROUTER_API_KEY = os.getenv("ROUTER_API_KEY", WORKER_API_KEY)
ROUTER_MODEL = os.getenv("ROUTER_MODEL", WORKER_MODEL)
ROUTER_TIMEOUT = float(os.getenv("ROUTER_TIMEOUT", "20"))
ROUTER_CONCURRENCY = int(os.getenv("ROUTER_CONCURRENCY", "8"))


def distinct_roles() -> list[str]:
    """实际指向不同模型的角色。两个角色配置成同一个模型时只返回 router，评估时不必测两遍。"""
    same = (ROUTER_BASE_URL, ROUTER_MODEL) == (WORKER_BASE_URL, WORKER_MODEL)
    return ["router"] if same else ["router", "worker"]


# 单价：元 / 百万词元。
# 默认值是 deepseek-flash（deepseek-chat 是它的别名）2026 年 10 月的高峰时段价格；
# 空闲时段的价格是它的一半。输入分为“缓存命中”和“缓存未命中”两种价格，相差 50 倍。
# 路由模型的单价默认与主力模型相同，可以用三个 ROUTER_PRICE_* 单独设置。
_WORKER_PRICE = {
    "input": float(os.getenv("WORKER_PRICE_INPUT", "2")),        # 缓存未命中的输入
    "cached": float(os.getenv("WORKER_PRICE_CACHED", "0.04")),   # 缓存命中的输入
    "output": float(os.getenv("WORKER_PRICE_OUTPUT", "8")),
}
PRICES = {
    "worker": _WORKER_PRICE,
    "router": {
        "input": float(os.getenv("ROUTER_PRICE_INPUT", _WORKER_PRICE["input"])),
        "cached": float(os.getenv("ROUTER_PRICE_CACHED", _WORKER_PRICE["cached"])),
        "output": float(os.getenv("ROUTER_PRICE_OUTPUT", _WORKER_PRICE["output"])),
    },
}

# ---- 检索相关的服务 ----
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "yuanhang_policy")
EMBED_URL = os.getenv("EMBED_URL", "http://localhost:8082")
RERANK_URL = os.getenv("RERANK_URL", "http://localhost:8081")

# ---- 检索参数 ----
CHUNK_MAX_CHARS = 400      # 超过这个长度的块再按句子切分
RECALL_K = 10              # 向量检索和 BM25 各取多少条
RERANK_TOP_K = 3           # 重排序之后保留多少条
RERANK_MIN_SCORE = 0.05    # 最高分低于这个值时，认为制度中没有相关内容

# ---- Agent 的限制 ----
MAX_AGENT_STEPS = 6
MAX_PLAN_STEPS = 5
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "90"))
