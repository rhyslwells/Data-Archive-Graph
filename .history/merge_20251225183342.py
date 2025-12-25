"""
merge.py

Purpose
-------
This script merges multiple markdown or text documents from a single
category directory into one consolidated document.

It is designed to support downstream workflows such as:
- NLP processing
- Knowledge graph construction
- Topic analysis
- Versioned snapshots of a living knowledge base

Input
-----
Source directory:
C:\\Users\\RhysL\\Desktop\\Data-Archive\\content\\categories\\natural-language

All readable text-based files in this directory will be merged.

Output
------
Generated file:
C:\\Users\\RhysL\\Desktop\\Data-Archive-Graph\\merged_natural_language.md

The output file contains:
- A generated header
- Clear separators between source documents
- Source filenames preserved for traceability

Outcome
-------
The result is a single, structured document that represents the
current state of the "natural-language" category, suitable for
graph ingestion or NLP pipelines.
"""

from pathlib import Path
from datetime import datetime

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

SOURCE_DIR = Path(
    r"C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language"
)

OUTPUT_FILE = Path(
    r"C:\Users\RhysL\Desktop\Data-Archive-Graph\merged_natural_language.md"
)

ALLOWED_EXTENSIONS = {".md", ".txt"}

# -------------------------------------------------------------------
# Merge Logic
# -------------------------------------------------------------------

def merge_documents(source_dir: Path, output_file: Path) -> None:
    """
    Merge all text-based documents in a directory into a single file.

    Parameters
    ----------
    source_dir : Path
        Directory containing source documents to merge.
    output_file : Path
        Destination file for the merged output.
    """

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    files = sorted(
        f for f in source_dir.iterdir()
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS
    )

    with output_file.open("w", encoding="utf-8") as out:
        # Write global header
        out.write("# Merged Natural Language Archive\n\n")
        out.write(f"Generated on: {datetime.utcnow().isoformat()} UTC\n\n")
        out.write("---\n\n")

        for file in files:
            out.write(f"## Source: {file.name}\n\n")
            out.write(f"_Path_: `{file}`\n\n")
            out.write("---\n\n")

            content = file.read_text(encoding="utf-8")
            out.write(content.strip())
            out.write("\n\n---\n\n")

    print(f"Merged {len(files)} files into {output_file}")

# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------

if __name__ == "__main__":
    merge_documents(SOURCE_DIR, OUTPUT_FILE)
