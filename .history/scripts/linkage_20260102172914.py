
Using the following as inspiration give me a script that

Takes ..\data\0-categories_snapshot.json

which contains :


[  {
    "category": "OTHER",
    "filename": "Bandit example output",
    "sha": "7b9c76403b39c5b2ec839414d9024d557dc09d3a",
    "url": "https://github.com/rhyslwells/Data-Archive/blob/main/content/categories/OTHER/Bandit%20example%20output.md",
    "text": "## Complicated example output of bandit\nRunning bandit on [[ML_Tools]] file [[Bandit_Example_Nonfixed.py]] gives. Fixing this gives [[Bandit_Example_Fixed.py]]\n\n```\n[main]  INFO    profile include tests: None\n[main]  INFO    profile exclude tests: None\n[main]  INFO    cli include tests: None\n[main]  INFO    cli exclude tests: None\n[main]  INFO    running on Python 3.10.8\nRun started:2025-01-11 17:19:41.806346\n\nTest results:\n>> Issue: [B404:blacklist] Consider possible security implications associated with the subprocess module.\n   Severity: Low   Confidence: High\n   CWE: CWE-78 (https://cwe.mitre.org/data/definitions/78.html)\n   More Info: https://bandit.readthedocs.io/en/1.8.0/blacklists/blacklist_imports.html#b404-import-subprocess\n   Location: .\\Bandit_Example.py:1:0\n1       import subprocess\n2       import os\n3       import pickle\n\n--------------------------------------------------\n>> Issue: [B403:blacklist] Consider possible security implications associated with pickle module.\n   Severity: Low   Confidence: High\n   CWE: CWE-502 (https://cwe.mitre.org/data/definitions/502.html)\n   More Info: https://bandit.readthedocs.io/en/1.8.0/blacklists/blacklist_imports.html#b403-import-pickle\n   Location: .\\Bandit_Example.py:3:0\n2       import os\n3       import pickle\n4\n\n--------------------------------------------------\n>> Issue: [B602:subprocess_popen_with_shell_equals_true] subprocess call with shell=True identified, security issue.\n   Severity: High   Confidence: High\n   CWE: CWE-78 (https://cwe.mitre.org/data/definitions/78.html)\n   More Info: https://bandit.readthedocs.io/en/1.8.0/plugins/b602_subprocess_popen_with_shell_equals_true.html\n   Location: .\\Bandit_Example.py:16:4\n15          \"\"\"\n16          subprocess.call(f\"cmd /c echo {user_input}\", shell=True)\n17\n\n--------------------------------------------------\n>> Issue: [B105:hardcoded_password_string] Possible hardcoded password: 'SuperSecret123'\n   Severity: Low   Confidence: Medium\n   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)\n   More Info: https://bandit.readthedocs.io/en/1.8.0/plugins/b105_hardcoded_password_string.html\n   Location: .\\Bandit_Example.py:28:15\n27          \"\"\"\n28          password = \"SuperSecret123\"  # Example of hardcoded sensitive information\n29          print(password)\n\n--------------------------------------------------\n   CWE: CWE-78 (https://cwe.mitre.org/data/definitions/78.html)\n   More Info: https://bandit.readthedocs.io/en/1.8.0/blacklists/blacklist_calls.html#b307-eval\n   Location: .\\Bandit_Example.py:40:17\n39          try:\n40              result = eval(user_input)  # Evaluate the input\n41              print(f\"Result of eval: {result}\")  # Print the result\n\n--------------------------------------------------\n>> Issue: [B301:blacklist] Pickle and modules that wrap it can be unsafe when used to deserialize untrusted data, possible security issue.\n   Severity: Medium   Confidence: High\n   CWE: CWE-502 (https://cwe.mitre.org/data/definitions/502.html)\n   More Info: https://bandit.readthedocs.io/en/1.8.0/blacklists/blacklist_calls.html#b301-pickle\n   Location: .\\Bandit_Example.py:53:11\n52          \"\"\"\n53          return pickle.loads(data)  # If data is malicious, it can execute arbitrary code.\n54\n\n--------------------------------------------------\n\nCode scanned:\n        Total lines of code: 77\n        Total lines skipped (#nosec): 0\n\nRun metrics:\n        Total issues (by severity):\n                Undefined: 0\n                Low: 3\n                Medium: 2\n                High: 1\n        Total issues (by confidence):\n                Undefined: 0\n                Low: 0\n                Medium: 1\n                High: 5\nFiles skipped (0):\n```\n## Simple example of bandit output\nWhen i run bandit on the following code.\n\n```python\nimport subprocess\n\nuser_input = input(\"Enter your name: \")\nsubprocess.call(f\"echo {user_input}\", shell=True)\n```\n\nit gives:\n\n```\nmain]  INFO    profile include tests: None\n[main]  INFO    profile exclude tests: None\n[main]  INFO    cli include tests: None\n[main]  INFO    cli exclude tests: None\n[main]  INFO    running on Python 3.10.8\nRun started:2025-01-11 16:56:32.096644\n\nTest results:\n>> Issue: [B404:blacklist] Consider possible security implications associated with the subprocess motettetted with the subprocess module.\n   Severity: Low   Confidence: High\n   CWE: CWE-78 (https://cwe.mitre.org/data/definitions/78.html)\n   More Info: https://bandit.readthedocs.io/en/1.8.0/blacklists/blacklist_imports.html#b404-import-subprocess\n   Location: .\\Bandit_ex1.py:1:0\n1       import subprocess\n2\n3       user_input = input(\"Enter your name: \")\n\n--------------------------------------------------\n>> Issue: [B602:subprocess_popen_with_shell_equals_true] subprocess call with shell=True identified, security issue.\n   Severity: High   Confidence: High\n   CWE: CWE-78 (https://cwe.mitre.org/data/definitions/78.html)\n   More Info: https://bandit.readthedocs.io/en/1.8.0/plugins/b602_subprocess_popen_with_shell_equals_true.html\n   Location: .\\Bandit_ex1.py:4:0\n3       user_input = input(\"Enter your name: \")\n4       subprocess.call(f\"echo {user_input}\", shell=True)\n\n--------------------------------------------------\n\nCode scanned:\n        Total lines of code: 3\n        Total lines skipped (#nosec): 0\n\nRun metrics:\n        Total issues (by severity):\n                Undefined: 0\n                Low: 1\n                Medium: 0\n                High: 1\n        Total issues (by confidence):\n                Undefined: 0\n                Low: 0\n                Medium: 0\n                High: 2\nFiles skipped (0):\n```",
    "aliases": [],
    "date modified": "27-09-2025",
    "tags": [
      "test"
    ]
  }]

saving it as
..\data\1-categories_snapshot_linked.json


using something like this...


# === HELPERS ===
def extract_frontmatter(md_text):
    """
    Extract YAML frontmatter and content from markdown text.
    """
    match = re.match(r'^---\n(.*?)\n---\n(.*)', md_text, re.DOTALL)
    if match:
        frontmatter = yaml.safe_load(match.group(1))
        content = match.group(2)
    else:
        frontmatter = {}
        content = md_text
    return frontmatter, content

def extract_links(content):
    """
    Extract links in the format [[link|display_name]] from markdown content.
    Returns only the link part before the pipe character.
    """
    return [match.split('|')[0] for match in re.findall(r'\[\[([^\]]+)\]\]', content)]

def markdown_to_text(md_content):
    """
    Convert markdown content to plain text by first converting it to HTML and then extracting the text.
    """
    html = markdown(md_content)
    soup = BeautifulSoup(html, features="html.parser")
    return soup.get_text()

def normalize_title(title):
    """
    Normalize title by converting to lowercase and replacing spaces with underscores.
    """
    try:
        return title.lower().replace(" ", "_")
    except AttributeError:
        print(f"Error normalizing title: {title}")
        return title

def summarise_text(text, word_limit=100):
    """
    Summarize text by limiting it to a specified word count.
    """
    words = text.strip().split()
    return " ".join(words[:word_limit]) + ("..." if len(words) > word_limit else "")

# === MAIN FUNCTION ===
def index_vault(vault_path):
    """
    Index all markdown files in the vault, extracting metadata, links, and generating summaries.
    This function now also restructures the vault_index.
    """
    vault_index = {}
    outlink_map = defaultdict(list)

    for note_path in vault_path.rglob("*.md"):
        with open(note_path, 'r', encoding='utf-8') as f:
            raw_md = f.read()
        
        # Extract frontmatter and content
        frontmatter, content = extract_frontmatter(raw_md)
        plain_text = markdown_to_text(content)

        # Get title and normalize it
        # raw_title = frontmatter.get("title")
        # title = raw_title if raw_title else note_path.stem
        title=note_path.stem
        # note_id = normalize_title(title)
        note_id = normalize_title(note_path.stem)  # Use filename (not title) for ID


        # Extract tags, aliases, and outlinks
        tags = frontmatter.get("tags", [])
        aliases = frontmatter.get("aliases", [])
        outlinks_raw = extract_links(content)
        outlinks = [normalize_title(link) for link in outlinks_raw]

        # Store note metadata in vault_index
        vault_index[note_id] = {
            "title": title,
            "tags": tags,
            "aliases": aliases,
            "outlinks": outlinks,
            "inlinks": [],  # Will be filled later
            "summary": summarise_text(plain_text, word_limit=25)
        }

        # Add the note's ID to the outlink_map for each link it references
        for link_id in outlinks:
            outlink_map[link_id].append(note_id)

    # Now update inlinks for each target note, ensuring no repeats
    for target_id, sources in outlink_map.items():
        if target_id in vault_index:
            # Remove duplicates by converting to a set, then back to a list
            vault_index[target_id]["inlinks"] = list(set(sources))

    # Restructure vault_index: remove duplicate outlinks and create final structure
    new_vault_index = {}
    for note_id, note_data in vault_index.items():
        # Create a new structure where the note_id is the key, excluding the redundant note_id key
        new_vault_index[note_id] = {
            "title": note_data["title"],
            "tags": note_data["tags"],
            "aliases": note_data["aliases"],
            "outlinks": list(set(note_data["outlinks"])),  # Remove duplicates from outlinks
            "inlinks": note_data["inlinks"],  # Inlinks are already handled for uniqueness
            "summary": note_data["summary"]
        }

    return new_vault_index
