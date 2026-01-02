"""
Extract per-note structural and categorical metrics from a linked
category snapshot JSON and export them as a CSV table.

Each row represents a note. Metrics describe content density,
formatting, metadata richness, and derived categorical flags.
"""

import json
import re
import pandas as pd
from pathlib import Path


# === HELPERS ===

def markdown_to_text(md: str) -> str:
    # Remove fenced code blocks
    md = re.sub(r"```.*?```", " ", md, flags=re.DOTALL)

    # Remove inline code
    md = re.sub(r"`[^`]+`", " ", md)

    # Remove images
    md = re.sub(r"!\[.*?\]\(.*?\)|!\[\[.*?\]\]", " ", md)

    # Remove links but keep text
    md = re.sub(r"\[\[([^\]|]+)(\|[^\]]+)?\]\]", r"\1", md)
    md = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", md)

    # Remove markdown symbols
    md = re.sub(r"[#>*_\-]+", " ", md)

    return md


def extract_structural_metrics(record: dict) -> dict:
    text = record.get("text", "")
    plain_text = markdown_to_text(text)
    lines = text.splitlines()

    code_blocks = re.findall(r"```", text)
    code_block_count = len(code_blocks) // 2

    code_block_line_count = sum(
        block.count("\n") + 1
        for block in re.findall(r"```.*?\n(.*?)```", text, re.DOTALL)
    )

    image_count = len(re.findall(r"!\[.*?\]\(.*?\)|!\[\[.*?\]\]", text))
    table_count = len(re.findall(r"\|.*?\|", text))
    quote_block_count = len(re.findall(r"^>\s", text, re.MULTILINE))
    list_item_count = len(re.findall(r"^[-*+]\s", text, re.MULTILINE))

    headings = re.findall(r"^(#{1,6})\s+", text, re.MULTILINE)

    return {
        "note_id": record["normalized_filename"],
        "filename": record["filename"],
        "word_count": len(plain_text.split()),
        "line_count": len([l for l in lines if l.strip()]),
        "section_count": len(headings),
        "max_heading_depth": max((len(h) for h in headings), default=0),
        "list_item_count": list_item_count,
        "code_block_count": code_block_count,
        "code_block_line_count": code_block_line_count,
        "image_count": image_count,
        "table_count": table_count,
        "quote_block_count": quote_block_count,
        "tag_count": len(record.get("tags", [])),
        "has_code": code_block_count > 0,
        "has_images": image_count > 0,
        "has_table": table_count > 0,
        "has_quotes": quote_block_count > 0,
        "has_lists": list_item_count > 0,
        "is_empty": len(plain_text.split()) < 10,
        "has_math": bool(re.search(r"\$\$.*?\$\$", text, re.DOTALL))
                    or bool(re.search(r"(?<!\\)\$(?!\$).*?(?<!\\)\$", text)),
    }


# === MAIN ===

def build_structural_csv(input_path: Path):
    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    rows = [extract_structural_metrics(rec) for rec in records]
    df = pd.DataFrame(rows)

    output_path = input_path.parent / "metrics_structural.csv"
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    input_file = Path("..") / "data" / "1-categories_snapshot_linked.json"
    build_structural_csv(input_file)
