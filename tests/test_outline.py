from __future__ import annotations

from pathlib import Path

from pipeline.preprocess.models import Block, Table
from pipeline.preprocess.outline import build_outline, node_path, read_outline, write_outline


def test_outline_tree_and_heading_paths(tmp_path: Path):
    blocks = [
        Block(block_id="d:b0", doc_id="d", kind="paragraph", text="Intro before any heading", order=0),
        Block(block_id="d:b1", doc_id="d", kind="heading", text="1 Context", order=1, page=1, style_info={"heading_level": 1}),
        Block(block_id="d:b2", doc_id="d", kind="paragraph", text="Background.", order=2, page=1),
        Block(block_id="d:b3", doc_id="d", kind="heading", text="1.1 Needs", order=3, page=2, style_info={"heading_level": 2}),
        Block(block_id="d:b4", doc_id="d", kind="table", text="Table", order=4, page=2, style_info={"table_id": "d:t000"}, is_logframe=True),
        Block(block_id="d:b5", doc_id="d", kind="heading", text="2 Logframe", order=5, page=3, style_info={"heading_level": 1}),
        Block(block_id="d:b6", doc_id="d", kind="paragraph", text="Indicators.", order=6, page=3),
    ]
    tables = [Table(table_id="d:t000", doc_id="d", source="pdf", header=["a"], rows=[["1"]], n_rows=1, n_cols=1)]
    nodes = build_outline(blocks, tables, "d")
    assert [(n.title, n.level, n.parent_id) for n in nodes] == [("(document)", 0, None), ("1 Context", 1, "d:n0000"), ("1.1 Needs", 2, "d:n0001"), ("2 Logframe", 1, "d:n0000")]
    assert nodes[0].block_ids == ["d:b0"]
    assert nodes[1].block_ids == ["d:b1", "d:b2"] and nodes[2].block_ids == ["d:b3", "d:b4"] and nodes[2].table_ids == ["d:t000"]
    assert blocks[4].heading_path == ["1 Context", "1.1 Needs"] and tables[0].heading_path == ["1 Context", "1.1 Needs"] and tables[0].is_logframe
    assert blocks[3].heading_path == ["1 Context"]  # a heading's own path is its ancestors
    assert blocks[6].heading_path == ["2 Logframe"]
    by_id = {n.node_id: n for n in nodes}
    assert node_path(nodes[2], by_id) == ["1 Context", "1.1 Needs"]
    write_outline(nodes, tmp_path / "outline.json")
    assert read_outline(tmp_path / "outline.json") == nodes
