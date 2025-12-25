"""
Purpose
-------
This script snapshots the contents of the `content/categories` folder from the
Data-Archive GitHub repository into a single JSON file.

It recursively traverses all category subfolders (one level deep),
downloads Markdown files, extracts YAML front matter (if present),
promotes YAML keys to top-level JSON fields, and stores the remaining
Markdown body as clean text.

The output JSON is designed to be:
- Stable across manual re-runs (Option 1: polling)
- Suitable for downstream NLP, embedding, and graph construction
- Diff-friendly using GitHub SHA hashes

This script intentionally performs no NLP processing.
It defines a clean ingestion boundary only.

Manual polling script to snapshot the contents of
`content/categories` from the Data-Archive GitHub repository.

Adds explicit progress logging and network safeguards so execution
is observable and safe to interrupt.
"""


"""
Purpose
-------
Manual polling script to snapshot the contents of
`content/categories` from the Data-Archive GitHub repository.

Adds explicit progress logging and network safeguards so execution
is observable and safe to interrupt.
"""

import requests
import json
from pathlib import Path
import re
import yaml
import time

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

OWNER = "rhyslwells"
REPO = "Data-Archive"
ROOT_PATH = "content/categories"
OUTPUT_FILE = "categories_snapshot.json"

API_ROOT = f"https://api.github.com/repos/{OWNER}/{REPO}/contents"

HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    # Strongly recommended to avoid rate limits
    # "Authorization": "Bearer YOUR_GITHUB_TOKEN"
}

REQUEST_TIMEOUT = 15  # seconds
SLEEP_BETWEEN_REQUESTS = 0.2  # be polite to GitHub

YAML_PATTERN = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n(.*)",
    re.DOTALL
)

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def fetch_contents(path):
    print(f"[DIR] Scanning: {path}")
    response = requests.get(
        f"{API_ROOT}/{path}",
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    return response.json()

def fetch_text(download_url):
    response = requests.get(
        download_url,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    return response.text

def split_yaml_and_text(raw_text):
    match = YAML_PATTERN.match(raw_text)

    if not match:
        return {}, raw_text.strip()

    yaml_block, body = match.groups()
    try:
        yaml_data = yaml.safe_load(yaml_block) or {}
    except yaml.YAMLError:
        yaml_data = {}

    return yaml_data, body.strip()

# -------------------------------------------------------------------
# Traversal
# -------------------------------------------------------------------

def walk_folder(path, records, stats):
    items = fetch_contents(path)

    for item in items:
        if item["type"] == "dir":
            walk_folder(item["path"], records, stats)

        elif item["type"] == "file":
            stats["seen"] += 1
            print(f"  [FILE {stats['seen']}] {item['path']}")

            try:
                raw_text = fetch_text(item["download_url"])
            except requests.RequestException as e:
                print(f"    [WARN] Failed to fetch file: {e}")
                stats["failed"] += 1
                continue

            yaml_data, text = split_yaml_and_text(raw_text)

            record = {
                "category": Path(item["path"]).parts[2],
                "filename": Path(item["name"]).stem,
                "sha": item["sha"],
                "url": item["html_url"],
                "text": text
            }

            record.update(yaml_data)

            records.append(record)
            stats["saved"] += 1

            time.sleep(SLEEP_BETWEEN_REQUESTS)

# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

def main():
    records = []
    stats = {"seen": 0, "saved": 0, "failed": 0}

    print("=== GitHub Categories Snapshot ===")
    walk_folder(ROOT_PATH, records, stats)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print("\n=== Summary ===")
    print(f"Files seen:    {stats['seen']}")
    print(f"Files saved:  {stats['saved']}")
    print(f"Files failed: {stats['failed']}")
    print(f"Output file:  {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
