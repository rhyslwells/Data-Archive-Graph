# Overview

1) Get a JSON of the metadata of the Data Archive.
2) Using the JSON explore graph relationships and metadata.

# Explorations

Use Notebook LLM to query NLP documents.

## explore_vault_index.ipynb

## build_vault_index.ipynb
We build the vault_index.json file from the metadata of the Data Archive. The metadata is stored in a JSON file, which contains information about each note in the vault, including its title, tags, and links to other notes.

## TFIDF_vault_index.ipynb (Done for now)

- **TF-IDF for Single Note:** Reads and preprocesses a Markdown note (removing YAML and converting to plain text), then computes TF-IDF scores for its terms using `CountVectorizer` and `TfidfTransformer`.

- **Vault-Wide Processing:** Loads a list of Markdown files and computes TF-IDF scores for each, extracting the top 10 terms per document.

- **Enhance JSON Index:** Loads `vault_index.json`, matches filenames to entries, and injects each document's top TF-IDF scores under a new `"TFIDF_Score"` key.

- **Output Enhanced Index:** Writes the modified data structure to `enhanced_vault_index.json`.

- **Explores TF-IDF Features:** Loads the enhanced JSON, builds a document-term matrix from the TF-IDF scores, and does analysis. We do search for does with  TFIDF keys, Hierarchical Clustering
and retrieves docs in a cluster. Vector embedding and search for similar docs using Spacy. Documenet querying and wordclouds.

## LDA_vault_index.ipynb (LATER)
Further enhancement of vault_index.json by adding topics as values using Latent Dirichlet Allocation (LDA) similar to how TFIDF worked. #### 

# Files

## vault_index.json

## enhanced_vault_index.json


