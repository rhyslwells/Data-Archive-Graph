"""
Build bidirectional link metadata for a category snapshot JSON.

This script reads a category snapshot exported from the Data Archive
(e.g. `0-categories_snapshot.json`), treats each entry as a node, and
derives link relationships from Obsidian-style wiki links (`[[...]]`)
found in the `text` field.

Each entry is augmented with:
- normalized_filename: canonical identifier derived from filename
- outlinks: normalized filenames this entry links to
- inlinks: normalized filenames that link to this entry

All existing fields are preserved unchanged. Link resolution is done
strictly via normalized filenames to ensure consistency.
"""

import json
import re
from collections import defaultdict
from pathlib import Path


# === HELPERS ===

def extract_links(text):
    """
    Extract Obsidian-style wiki links from text.

    Supports:
      [[target]]
      [[target|display]]

    Returns the raw target string before any normalization.
    """
    return [m.split("|")[0] for m in re.findall(r"\[\[([^\]]+)\]\]", text or "")]


def normalize_title(value):
    """
    Normalize a filename or link target into a canonical identifier.
    """
    return value.strip().lower().replace(" ", "_")


# === MAIN ===

def link_category_snapshot(input_path: Path, output_path: Path):
    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Attach normalized filenames
    for rec in records:
        rec["normalized_filename"] = normalize_title(rec["filename"])

    # Map normalized_filename -> record
    record_map = {
        rec["normalized_filename"]: rec
        for rec in records
    }

    outlink_map = defaultdict(list)

    # First pass: extract outlinks
    for rec in records:
        source_id = rec["normalized_filename"]
        raw_links = extract_links(rec.get("text", ""))
        outlinks = [normalize_title(l) for l in raw_links]

        rec["outlinks"] = list(set(outlinks))
        rec["inlinks"] = []

        for target_id in rec["outlinks"]:
            outlink_map[target_id].append(source_id)

    # Second pass: populate inlinks
    for target_id, sources in outlink_map.items():
        if target_id in record_map:
            record_map[target_id]["inlinks"] = sorted(set(sources))

    # Write enriched snapshot
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    input_file = Path("..") / "data" / "0-categories_snapshot.json"
    output_file = Path("..") / "data" / "1-categories_snapshot_linked.json"

    link_category_snapshot(input_file, output_file)
