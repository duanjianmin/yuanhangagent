"""下载 Embedding 和重排序两个模型的文件，只下载项目用得到的那几个。

默认从国内可以访问的镜像站 https://hf-mirror.com 下载；能直接访问官网的读者，
可以把环境变量 HF_ENDPOINT 设置为 https://huggingface.co。
支持断点续传：中途断开之后重新运行即可，已经下载完的文件会跳过。

用法（在 project 目录下）：python -m scripts.download_models [保存的目录，默认 D:/work/model]
"""
import os
import sys
import time
from pathlib import Path

import httpx

ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
TARGET = Path(sys.argv[1] if len(sys.argv) > 1 else os.getenv("MODEL_DIR", "D:/work/model"))

# 每个仓库里同一份权重有几种格式，这里各取一种：
# Embedding 模型的仓库没有 ONNX，用 model.safetensors；重排序模型的仓库有，用 onnx/model.onnx。
MODELS = {
    "BAAI/bge-small-zh-v1.5": [
        "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt",
        "modules.json", "config_sentence_transformers.json", "sentence_bert_config.json",
        "1_Pooling/config.json", "model.safetensors",
    ],
    "BAAI/bge-reranker-base": [
        "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "sentencepiece.bpe.model", "onnx/model.onnx",
    ],
}


def remote_size(client: httpx.Client, url: str) -> int:
    """文件的总大小。用只取第一个字节的请求来问，响应头 Content-Range 的末尾就是总大小。"""
    response = client.get(url, headers={"Range": "bytes=0-0"})
    response.raise_for_status()
    content_range = response.headers.get("content-range", "")
    return int(content_range.rsplit("/", 1)[-1]) if "/" in content_range else len(response.content)


def download(client: httpx.Client, url: str, path: Path) -> str:
    total = remote_size(client, url)
    done = path.stat().st_size if path.exists() else 0
    if done == total:
        return "已存在，跳过"
    if done > total:                               # 本地的文件比远端的大，说明不是同一个文件，重新下载
        done = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    # 大文件经常传到一半就被服务端断开（实测镜像站每个连接只传 1 到 2 MB）。
    # 所以这里是一个循环：断开之后从已经写入的位置接着请求，只要还有进展就继续；
    # 连续 5 次没有任何进展才放弃。
    stalled = 0
    with path.open("ab" if done else "wb") as file:
        while done < total:
            before = done
            try:
                with client.stream("GET", url, headers={"Range": f"bytes={done}-"}) as response:
                    response.raise_for_status()
                    for chunk in response.iter_bytes(1024 * 1024):
                        file.write(chunk)
                        done += len(chunk)
                        if total > 5_000_000:
                            print(f"\r    {done / 1e6:8.1f} / {total / 1e6:.1f} MB", end="", flush=True)
            except httpx.HTTPError as error:
                problem = f"{type(error).__name__}: {error}"
            else:
                problem = "连接正常结束，但是没有收到数据"
            if done > before:
                stalled = 0
            else:
                stalled += 1
                if stalled >= 5:
                    raise IOError(f"连续 5 次没有进展，停在 {done} / {total} 字节。最后一次：{problem}")
                time.sleep(2 * stalled)
    if total > 5_000_000:
        print()
    return "完成"


def main() -> None:
    print(f"下载地址：{ENDPOINT}\n保存目录：{TARGET}\n")
    failed = []
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30, read=120)) as client:
        for repo, files in MODELS.items():
            folder = TARGET / repo.split("/")[1]
            print(repo)
            for name in files:
                url = f"{ENDPOINT}/{repo}/resolve/main/{name}"
                try:
                    print(f"  {name}：{download(client, url, folder / name)}")
                except (httpx.HTTPError, OSError) as error:
                    failed.append(f"{repo}/{name}")
                    print(f"  {name}：失败（{type(error).__name__}: {error}）")
    if failed:
        raise SystemExit(f"\n有 {len(failed)} 个文件没有下载成功，重新运行本脚本会从断开的地方继续。")
    print("\n全部完成。")


if __name__ == "__main__":
    main()
