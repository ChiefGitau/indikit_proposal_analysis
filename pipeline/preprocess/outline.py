"""Document outline (paper Section 3.3.2 "outline knowledge base"): a heading tree built from the heading
blocks, with every block and table attached to its section. Written to ``outline.json``."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pipeline.preprocess.models import Block, OutlineNode, Table


def build_outline(blocks: list[Block], tables: list[Table], doc_id: str) -> list[OutlineNode]:
    """Heading tree in reading order. Non-heading blocks and tables are assigned to the innermost open heading
    (a synthetic root node collects content before the first heading). Also fills ``heading_path`` where empty."""
    root = OutlineNode(node_id=f"{doc_id}:n0000", title="(document)", level=0)
    nodes: list[OutlineNode] = [root]
    stack: list[OutlineNode] = [root]
    tables_by_id = {t.table_id: t for t in tables}
    for b in blocks:
        if b.kind == "heading":
            level = max(1, int(b.style_info.get("heading_level", 1) or 1))
            while len(stack) > 1 and stack[-1].level >= level:
                stack.pop()
            node = OutlineNode(node_id=f"{doc_id}:n{len(nodes):04d}", title=b.text, level=level, page=b.page, parent_id=stack[-1].node_id, is_logframe=b.is_logframe)
            node.block_ids.append(b.block_id)
            nodes.append(node)
            stack.append(node)
            b.heading_path = [n.title for n in stack[1:-1]]
            continue
        b.heading_path = [n.title for n in stack[1:]]
        stack[-1].block_ids.append(b.block_id)
        if b.kind == "table":
            tid = b.style_info.get("table_id")
            if tid in tables_by_id:
                stack[-1].table_ids.append(tid)
                tables_by_id[tid].heading_path = list(b.heading_path)
                if b.is_logframe:
                    tables_by_id[tid].is_logframe = True
    return nodes


def node_path(node: OutlineNode, nodes_by_id: dict[str, OutlineNode]) -> list[str]:
    path: list[str] = []
    cur: Optional[OutlineNode] = node
    while cur is not None and cur.level > 0:
        path.append(cur.title)
        cur = nodes_by_id.get(cur.parent_id) if cur.parent_id else None
    return list(reversed(path))


def write_outline(nodes: list[OutlineNode], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([n.model_dump(mode="json") for n in nodes], ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")


def read_outline(path: Path) -> list[OutlineNode]:
    return [OutlineNode.model_validate(n) for n in json.loads(Path(path).read_text(encoding="utf-8"))]
