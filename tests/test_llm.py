"""费用的折算。"""
import pytest

from app import config, llm


def test_cost_separates_cached_input(monkeypatch):
    monkeypatch.setitem(config.PRICES, "worker", {"input": 2.0, "cached": 0.04, "output": 8.0})
    # 2000 个输入词元中 1500 个命中缓存，输出 100 个
    expected = (500 * 2.0 + 1500 * 0.04 + 100 * 8.0) / 1_000_000
    assert llm.cost("worker", 2000, 100, 1500) == pytest.approx(expected)
    assert llm.cost("worker", 2000, 100) == pytest.approx((2000 * 2.0 + 100 * 8.0) / 1_000_000)
    # 路由模型的单价单独配置；设为 0 时不产生费用
    monkeypatch.setitem(config.PRICES, "router", {"input": 0.0, "cached": 0.0, "output": 0.0})
    assert llm.cost("router", 2000, 100, 1500) == 0
