"""文档处理：读取、清洗、分块。不依赖任何外部服务，可以单独测试。"""
import re
from dataclasses import dataclass
from pathlib import Path

from app import config

HEADER = re.compile(r"^.*内部资料.*请勿外传\s*$")
FOOTER = re.compile(r"^\s*第\s*\d+\s*页\s*/\s*共\s*\d+\s*页\s*$")
SENTENCE_END = re.compile(r"(?<=[。！？；])")


@dataclass
class Chunk:
    id: str        # 例如 "员工手册#02"
    doc: str       # 来源文档
    title: str     # 所在的二级标题
    text: str      # 正文，不含标题

    @property
    def indexed_text(self) -> str:
        """用于向量化和关键词检索的文本：把文档名和标题拼在正文之前。"""
        return f"{self.doc} {self.title}\n{self.text}"


def clean(text: str) -> str:
    """去掉页眉、页脚、乱码行和多余的空白。"""
    lines = []
    for line in text.splitlines():
        line = line.rstrip()
        if HEADER.match(line) or FOOTER.match(line):
            continue
        if "�" in line:                       # 解码失败留下的替换字符
            continue
        line = re.sub(r"[ \t　]{2,}", " ", line)   # 行内连续的空白
        lines.append(line)
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split_long(text: str, max_chars: int) -> list[str]:
    """把过长的正文按句子切成若干段，每段不超过 max_chars。"""
    if len(text) <= max_chars:
        return [text]
    pieces, current = [], ""
    for sentence in SENTENCE_END.split(text):
        if current and len(current) + len(sentence) > max_chars:
            pieces.append(current.strip())
            current = ""
        current += sentence
    if current.strip():
        pieces.append(current.strip())
    return pieces


def split_by_heading(doc: str, text: str, max_chars: int = config.CHUNK_MAX_CHARS) -> list[Chunk]:
    """按二级标题分块；一级标题和它下面的引言不单独成块。"""
    chunks, title, body = [], None, []

    def flush():
        if title is None:
            return
        content = "\n".join(body).strip()
        for piece in split_long(content, max_chars):
            chunks.append(Chunk(f"{doc}#{len(chunks) + 1:02d}", doc, title, piece))

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            title, body = line[3:].strip(), []
        elif not line.startswith("# "):
            body.append(line)
    flush()
    return chunks


def load_chunks(directory: Path = config.KNOWLEDGE_DIR) -> list[Chunk]:
    """读取目录下的全部文档，返回清洗和分块之后的结果。"""
    chunks = []
    for path in sorted(directory.glob("*.md")):
        text = clean(path.read_text(encoding="utf-8"))
        chunks.extend(split_by_heading(path.stem, text))
    return chunks
