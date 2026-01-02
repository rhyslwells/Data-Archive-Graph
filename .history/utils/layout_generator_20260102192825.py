from pathlib import Path

def generate_tree(root: Path, prefix=""):
    """Recursively generates a tree-like string of files and folders."""
    lines = []
    entries = sorted(root.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
    for i, entry in enumerate(entries):
        connector = "└─ " if i == len(entries) - 1 else "├─ "
        lines.append(f"{prefix}{connector}{entry.name}")
        if entry.is_dir():
            extension = "    " if i == len(entries) - 1 else "│   "
            lines.extend(generate_tree(entry, prefix + extension))
    return lines

# === CONFIGURATION ===
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  # utils/ -> project root
folders_to_include = ["docs", "notebooks", "src", "utils"]

all_lines = []
for folder in folders_to_include:
    path = project_root / folder
    if path.exists():
        all_lines.append(f"{folder}/")
        all_lines.extend(generate_tree(path, "│   "))

# Write to ../docs/repo-layout.md
output_file = project_root / "docs" / "repo-layout.md"
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(all_lines))

print(f"Repository layout written to {output_file}")
