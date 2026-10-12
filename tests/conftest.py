"""测试的公共设置：固定“今天”，使用临时目录中的数据库，每个测试之前重建数据。"""
import os
import tempfile

# 必须在导入 app.config 之前设置
os.environ["YUANHANG_TODAY"] = "2026-10-12"          # 星期一
os.environ["YUANHANG_RUNTIME_DIR"] = tempfile.mkdtemp(prefix="yuanhang-test-")

import pytest

from app import db


@pytest.fixture(autouse=True)
def fresh_db():
    db.init_db()
