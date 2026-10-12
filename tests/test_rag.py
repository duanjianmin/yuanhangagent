"""文档处理和检索中不依赖外部服务的部分。"""
from app.rag.ingest import clean, load_chunks, split_by_heading, split_long
from app.rag.retriever import BM25, rrf, tokenize

DIRTY = """远航实业有限公司 · 内部资料 · 请勿外传
# IT 服务指南



## VPN 远程接入

需要先连接 VPN。   客户端名称为“远航通”。
第 1 页 / 共 4 页
���
## 邮箱
容量为 20 GB。
"""


def test_clean_removes_header_footer_and_garbage():
    text = clean(DIRTY)
    assert "内部资料" not in text and "页 /" not in text and "�" not in text
    assert "\n\n\n" not in text and "VPN。 客户端" in text          # 多余的空行和连续空格被压缩
    assert text.startswith("# IT 服务指南")


def test_split_by_heading():
    chunks = split_by_heading("IT服务指南", clean(DIRTY))
    assert [(c.id, c.title) for c in chunks] == [("IT服务指南#01", "VPN 远程接入"), ("IT服务指南#02", "邮箱")]
    assert chunks[1].text == "容量为 20 GB。"
    assert chunks[1].indexed_text.startswith("IT服务指南 邮箱\n")


def test_split_long_respects_sentences():
    text = "第一句话比较长一些。第二句话也不短！第三句呢？最后一句；"
    pieces = split_long(text, 12)
    assert pieces == ["第一句话比较长一些。", "第二句话也不短！", "第三句呢？最后一句；"]
    assert all(p[-1] in "。！？；" for p in pieces)
    assert split_long("短句。", 12) == ["短句。"]


def test_knowledge_base_is_clean():
    chunks = load_chunks()
    assert len(chunks) == 34 and len({c.id for c in chunks}) == 34
    assert all(c.text and "内部资料" not in c.text and "�" not in c.text for c in chunks)


def test_tokenize_keeps_codes_whole():
    assert tokenize("VPN 报错 E-809") == ["vpn", "报错", "e-809"]
    assert tokenize("年假怎么算") == ["年假", "假怎", "怎么", "么算"]


def test_bm25_finds_exact_code():
    chunks = load_chunks()
    scores = BM25([c.indexed_text for c in chunks]).scores("报错 E-809")
    assert chunks[scores.index(max(scores))].id == "IT服务指南#01"
    assert sum(score > 0 for score in scores) == 1                 # 只有一个块包含这个编号


def test_rrf():
    assert rrf([["a", "b", "c"], ["b", "a", "d"]])[:2] in (["a", "b"], ["b", "a"])
    assert rrf([["a", "b"], ["c", "b"]])[0] == "b"                  # 两路都靠前的排在第一
