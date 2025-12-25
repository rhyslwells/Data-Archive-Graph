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
"""
Purpose
-------
Fast manual polling snapshot of `content/categories` from GitHub
using anonymous access (no token).

Downloads files in parallel with conservative limits to avoid
rate limiting or authorization errors.
"""

import requests
import json
from pathlib import Path
import re
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

OWNER = "rhyslwells"
REPO = "Data-Archive"
ROOT_PATH = "content/categories"
OUTPUT_FILE = "categories_snapshot.json"

API_ROOT = f"https://api.github.com/repos/{OWNER}/{REPO}/contents"

HEADERS = {
    "Accept": "application/vnd.github.v3+json"
}

REQUEST_TIMEOUT = 15
MAX_WORKERS = 4  # conservative for anonymous GitHub access

YAML_PATTERN = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n(.*)",
    re.DOTALL
)

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def fetch_contents(path):
    response = requests.get(
        f"{API_ROOT}/{path}",
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    return response.json()

def fetch_file(item):
    response = requests.get(
        item["download_url"],
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()

    raw_text = response.text
    match = YAML_PATTERN.match(raw_text)

    if match:
        yaml_block, body = match.groups()
        try:
            yaml_data = yaml.safe_load(yaml_block) or {}
        except yaml.YAMLError:
            yaml_data = {}
        text = body.strip()
    else:
        yaml_data = {}
        text = raw_text.strip()

    record = {
        "category": Path(item["path"]).parts[2],
        "filename": Path(item["name"]).stem,
        "sha": item["sha"],
        "url": item["html_url"],
        "text": text
    }

    record.update(yaml_data)
    return record

# -------------------------------------------------------------------
# Discovery
# -------------------------------------------------------------------

def collect_files(path, files):
    items = fetch_contents(path)

    for item in items:
        if item["type"] == "dir":
            collect_files(item["path"], files)
        elif item["type"] == "file":
            files.append(item)

# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

def main():
    print("Discovering files...")
    files = []
    collect_files(ROOT_PATH, files)
    print(f"Files discovered: {len(files)}")

    records = []

    print("Downloading files...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(fetch_file, f) for f in files]

        for i, future in enumerate(as_completed(futures), 1):
            try:
                records.append(future.result())
                print(f"[{i}/{len(files)}] downloaded")
            except Exception as e:
                print(f"[WARN] download failed: {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\nSnapshot written: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

# - [ ] category specific to be passed to it, then saves snapshot as this.