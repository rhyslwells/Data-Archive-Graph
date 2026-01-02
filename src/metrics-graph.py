"""
Compute graph-level metrics from a linked category snapshot JSON.

The graph is directed and defined by outlinks between normalized
filenames. Each row represents a node (note).
"""

import json
import pandas as pd
import networkx as nx
from pathlib import Path


# === MAIN ===

def build_graph_metrics_csv(input_path: Path):
    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    G = nx.DiGraph()

    # Add nodes
    for rec in records:
        G.add_node(rec["normalized_filename"])

    # Add edges
    for rec in records:
        source = rec["normalized_filename"]
        for target in rec.get("outlinks", []):
            G.add_edge(source, target)

    rows = []
    for node in G.nodes:
        inlinks = G.in_degree(node)
        outlinks = G.out_degree(node)

        rows.append({
            "note_id": node,
            "outlink_count": outlinks,
            "inlink_count": inlinks,
            "total_degree": inlinks + outlinks,
            "is_orphan": inlinks == 0,
        })

    df = pd.DataFrame(rows)
    output_path = input_path.parent / "metrics_graph.csv"
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    input_file = Path("..") / "data" / "1-categories_snapshot_linked.json"

    build_graph_metrics_csv(input_file)
