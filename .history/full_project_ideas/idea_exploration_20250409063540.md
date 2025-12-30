Use FAISS or Qdrant to do approximate nearest-neighbor (ANN) search over this vector index.

#### **Mid-Term Goals:**
2. **Leverage a Language Model (LLM):**
   - Interpret vague or general user inputs to provide meaningful insights.
   - Automatically identify relevant notes, tags, and links (subgraphs) based on the query.
   - Generate questions or summaries derived from the selected subgraph to aid in deeper understanding or content development.

3. **Feed Selected Notes into Downstream Prompts:**
   - Use the extracted content for:
     - Content generation.
     - Idea development.
     - Formulating questions or hypotheses for further exploration.

#### **Optional Experiments:**
- **Interactive Exploration** in a **Jupyter notebook**, focusing on:
   - Visualizing note relationships and connections.
   - Performing semantic search to find related notes.
   - Composing prompts for the LLM based on note content.

**Prompt Tuning:** Fine-tune LLM prompts using the summaries of notes to increase the relevance and quality of generated content.

#### **Future Goals:**
4. **Advanced Querying and Subgraph Extraction:**
   - Develop the ability to query and extract relevant subgraphs from your vault based on vague, natural language prompts.
   - Enhance the exploration experience by dynamically adjusting the scope of subgraphs for more targeted insights.