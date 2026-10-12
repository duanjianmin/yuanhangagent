"""评估脚本。导入这个包时固定“今天”并使用独立的运行目录，评估结果才可以复现，
也不会改动正在运行的服务所使用的数据。"""
import os
from pathlib import Path

os.environ.setdefault("YUANHANG_TODAY", "2026-10-12")          # 星期一；数据集中的日期都以它为准
os.environ.setdefault("YUANHANG_RUNTIME_DIR", str(Path(__file__).resolve().parent.parent / "runtime" / "eval"))
