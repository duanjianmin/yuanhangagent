"""把演示数据恢复到初始状态：重建业务表，清空对话检查点和轨迹。向量库不受影响。

用法（在 project 目录下，先停掉应用）：python -m scripts.reset_data
"""
from app import config, db


def main() -> None:
    # 先删除检查点：应用还在运行时它被占用，删除会失败，此时什么都还没有改动
    targets = [config.CHECKPOINT_PATH, *config.RUNTIME_DIR.glob(config.CHECKPOINT_PATH.name + "-*"),
               config.TRACE_PATH]
    removed = []
    for path in targets:
        try:
            path.unlink()
            removed.append(path.name)
        except FileNotFoundError:
            pass
        except PermissionError:
            raise SystemExit(f"{path.name} 正在被使用，请先停掉应用再运行。")
    db.init_db()
    print(f"业务数据已重建：{config.DB_PATH}")
    print(f"已删除：{'、'.join(removed) or '无'}")


if __name__ == "__main__":
    main()
