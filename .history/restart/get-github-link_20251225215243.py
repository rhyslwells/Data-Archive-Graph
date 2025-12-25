import requests
import json
from pathlib import Path
import re
import yaml

OWNER = "rhyslwells"
REPO = "Data-Archive"
ROOT_PATH = "content/categories"
OUTPUT_FILE = "categories_snapshot.json"

API_ROOT = f"https://api.github.com/repos/{OWNER}/{REPO}/contents"

HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    # Optional:
    # "Authorization": "Bearer YOUR_GITHUB_TOKEN"
}

YAML_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)", re.DOTALL)

def fetch_contents(path):
    response = requests.get(f"{API_ROOT}/{path}", headers=HEADERS)
    response.raise_for_status()
    return response.json()

def fetch_text(download_url):
    response = requests.get(download_url, headers=HEADERS)
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

def walk_folder(path, records):
    items = fetch_contents(path)

    for item in items:
        if item["type"] == "dir":
            walk_folder(item["path"], records)

        elif item["type"] == "file":
            raw_text = fetch_text(item["download_url"])
            yaml_data, text = split_yaml_and_text(raw_text)

            records.append({
                "category": Path(item["path"]).parts[2],
                "filename": Path(item["name"]).stem,
                "sha": item["sha"],
                "url": item["html_url"],
                "yaml": yaml_data,
                "text": text
            })

def main():
    records = []
    walk_folder(ROOT_PATH, records)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Snapshot written: {OUTPUT_FILE}")
    print(f"Files captured: {len(records)}")

if __name__ == "__main__":
    main()
