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
"""

import requests
import json
from pathlib import Path
import re
import yaml

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
    # Add a token if you encounter rate limits
    # "Authorization": "Bearer YOUR_GITHUB_TOKEN"
}

# Regex to capture YAML front matter at the top of Markdown files
YAML_PATTERN = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n(.*)",
    re.DOTALL
)

# -------------------------------------------------------------------
# GitHub API helpers
# -------------------------------------------------------------------

def fetch_contents(path):
    """Fetch directory contents from the GitHub Contents API."""
    response = requests.get(f"{API_ROOT}/{path}", headers=HEADERS)
    response.raise_for_status()
    return response.json()

def fetch_text(download_url):
    """Download raw file text from GitHub."""
    response = requests.get(download_url, headers=HEADERS)
    response.raise_for_status()
    return response.text

# -------------------------------------------------------------------
# Markdown parsing
# -------------------------------------------------------------------

def split_yaml_and_text(raw_text):
    """
    Split YAML front matter from Markdown body.

    Returns
    -------
    yaml_data : dict
        Parsed YAML front matter (empty if none or malformed)
    body_text : str
        Markdown content with YAML removed
    """
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
# Recursive folder traversal
# -------------------------------------------------------------------

def walk_folder(path, records):
    """
    Recursively traverse a GitHub folder and collect file records.
    """
    items = fetch_contents(path)

    for item in items:
        if item["type"] == "dir":
            walk_folder(item["path"], records)

        elif item["type"] == "file":
            raw_text = fetch_text(item["download_url"])
            yaml_data, text = split_yaml_and_text(raw_text)

            # Category is always the first folder under `categories`
            category = Path(item["path"]).parts[2]

            record = {
                "category": category,
                "filename": Path(item["name"]).stem,
                "sha": item["sha"],
                "url": item["html_url"],
                "text": text
            }

            # Promote YAML keys to top-level fields
            # Missing YAML simply results in no additional keys
            record.update(yaml_data)

            records.append(record)

# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

def main():
    records = []
    walk_folder(ROOT_PATH, records)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Snapshot written to: {OUTPUT_FILE}")
    print(f"Total files captured: {len(records)}")

if __name__ == "__main__":
    main()
