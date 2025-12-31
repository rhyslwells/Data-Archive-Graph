# Data-Archive-Explorer — Feature Roadmap (Ordered)

## Phase 0 — Foundations (Required for Everything Else)

#retrieval #ml_process

* **Parse Markdown Vault**

  * Read `.md` files
  * Extract title, headings, body text, tags, links, timestamps

* **Chunk Notes**

  * Split notes into semantically meaningful chunks (heading / paragraph level)
  * Preserve note + heading hierarchy metadata

* **Build Embedding Index**

  * Embed chunks using a sentence-level embedding model
  * Store embeddings in FAISS or Chroma
  * Persist chunk → note → metadata mapping

* **Metadata Store**

  * Maintain lookup for tags, titles, aliases, links
  * Normalise casing and formats

---

## Phase 1 — Minimal Useful Product (MVP)

#retrieval #semantic_search

* **Semantic Querying**

  * Free-text query → top-$k$ relevant chunks
  * Return chunk text + source note + heading path

* **Context Scoping**

  * Allow query constraints by:

    * tags
    * folders
    * time range (optional)

* **Simple Front-End**

  * Streamlit UI with:

    * query input
    * context selectors
    * ranked results list
    * Suggested query examples

This phase should already answer:

> “Have I written about this before, and where?”

---

## Phase 2 — Context Reconstruction

#context #knowledge_navigation

* **Adjacent Context Expansion**

  * Show preceding and following chunks
  * Show parent heading content

* **Backlink Surfacing**

  * Show other notes linking to retrieved note
  * Show notes sharing high semantic similarity

* **Result Quantification**

  * Display number of related chunks per query
  * Indicate confidence / density of matches

---

## Phase 3 — Query Guidance & Modes

Query constructor

* **Query Mode Templates**

  * Concept: “What have I written about $X$?”
  * Recall: “Have I thought about this before?”
  * Comparison: “Where do I relate $X$ and $Y$?”
  * Process: “How do I usually approach $X$?”

* **Composable Query Builder**

  * Clickable query examples
  * Optional filters layered onto free-text queries

This addresses:

> “How do I articulate what I want?”

---

## Phase 4 — Lightweight Knowledge Graph

* **Note Graph Construction**
  * Nodes: notes
  * Edges: wikilinks, shared tags, metadata relations

* **Subgraph Extraction**

  * From retrieved notes:
    * include $1$–$2$ hop neighbours
    * include semantically close nodes

* **Graph Visualisation**

  * Interactive graph using PyVis or Cytoscape
  * Highlight query-relevant nodes

---

## Phase 5 — Graph-Aware Retrieval (Graph-RAG Lite)

#rag #knowledge_graph

* **Graph-Scoped Retrieval**

  * Run semantic search only within selected subgraph
  * Reduce noise for focused exploration

* **Subgraph-as-Context**

  * Treat selected subgraph as “working memory”
  * Re-rank results within this scope

This enables:

> “Explore this theme deeply, not the whole vault.”

---

## Phase 6 — Theme & Structure Discovery

#clustering #topic_modeling

* **Concept Clustering**

  * Cluster embeddings to surface latent themes
  * Expose clusters as navigable topics

* **Theme Evolution Tracking**

  * Show when notes in a cluster were created or revised
  * Identify long-running vs abandoned themes

* **Link Density Analysis**

  * Identify:

    * isolated notes
    * overly dense clusters

(LDA/TF-IDF optional here; embeddings likely sufficient.)

---

## Phase 7 — Reflective Prompts (Non–Gen-AI)

#evaluation #ml_process

* **Overlap Detection**

  * Flag notes with high semantic overlap
  * Suggest merge or refactor candidates

* **Gap Detection**

  * Identify clusters with many notes but no synthesis
  * Surface “missing summary” signals

* **Reflective Questions (Rule-Based)**

  * “You have $n$ notes touching $X$ — no synthesis exists.”
  * “These notes overlap but are unlinked.”

No generative text required.

---

## Phase 8 — Optional Enhancements

#optimisation #ux

* **Voice-to-Text Query Input**

  * Whisper-based transcription

* **Note Consolidation Export**

  * Combine selected notes into a single downloadable markdown file
  * Intended for NotebookLM or long-form writing

* **Interaction Feedback Loop**

  * Track clicked notes
  * Adjust retrieval weighting over time

* Use subgraph context to be fed into chain-of-thought prompting.
---

## Guiding Constraint (Keep This Explicit)

Every feature should answer at least one of:

* Did this help me **find** something I forgot?
* Did this help me **see a connection** I missed?
* Did this reduce **friction when thinking or writing**?

---

## Suggested Immediate Next Step

Define **one concrete opening question** the MVP must answer, for example:

> “What ideas related to uncertainty have I written about but never developed?”

That question should drive Phase 1–3 design decisions.
