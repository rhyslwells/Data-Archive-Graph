# It's it possible given a GitHub link repo folder to pull the content of the files from it , and then process them nlp style? The repo is regularly update hence why I want to do this?

import requests

repo_owner = "username"
repo_name = "repository"
path = "folder/subfolder"
url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{path}"

response = requests.get(url)
files = response.json()

for file in files:
    if file["type"] == "file":
        content = requests.get(file["download_url"]).text
        print(file["name"], len(content))
