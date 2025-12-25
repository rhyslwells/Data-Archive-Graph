# Merged Natural Language Archive

Generated on: 2025-12-25T18:33:46.274422 UTC

---

## Source: AI Agents Memory.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\AI Agents Memory.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- evaluation
- language_models
- NLP
- optimisation
---
How to give memory to long-running AI agents. That is how do we enable agents to **maintain**, **retrieve**, and **update** salient conversation memories over extended conversations.

Context:
* LLMs tend to forget information within and across sessions due to fixed context windows.
* Forgetting over sessions can cause user frustration.
* Large contexts increase both **cost** and **latency**.
* Memory systems should not depend solely on the size of the input prompt.

### Challenges

* **Context overflow**: exceeding the token window.
* **Irrelevant history**: retaining unused or low-value information.
* **Memory management**: removing stale or outdated memories without losing critical data.

### Use Cases

* Personalised learning assistants.
* Customer support bots.
* Financial advisory agents.
### Architecture Overview

1. **Store** memories in a database.
2. **Retrieve** relevant memories via semantic \[\[similarity search]].
3. **Update** process: add, update, delete, or NOOP (no operation).
4. Optional **graph memory** layer

Metrics to evaluate:
  * Memory quality.
  * F1 score.
  * BLEU-1 score.
  * Token consumption analysis.

Key Takeaways
* Persistent long-term memory improves **performance**, reduces **cost**, and increases **speed**.
* Reasoning is essential during the memory update phase to maintain accuracy and coherence.

### Example Engineering Questions

#### Architecture & Scale

* What are the biggest engineering challenges in maintaining long-term memory at scale, especially with respect to latency, consistency, and cost?
* How should **mem0** be integrated into production with a self-hosted setup (e.g., Docker with MCP)?
* Which API calls are required for summarisation and retrieval, and at which stages?

#### Cost & Retrieval

* How can cost be controlled while maintaining contextual memory?
* What trade-offs exist between retrieval speed and memory depth?
* Does marking graph relationships as invalid (instead of deleting them) cause memory bloat and higher cost?

#### Memory Quality & Versioning

* How is memory versioned as the agent or model changes?
* Are there mechanisms for detecting **memory drift** or **bias accumulation**?

#### API Behaviour & Retrieval Logic

* How many API calls are required for create/update and retrieval?
* Does retrieval fetch top-N similar memories (e.g., top 10 via \[\[cosine similarity]])?
* Can the number of historical messages retrieved for fact generation be configured?
* Differences between open-source and enterprise versions of mem0, especially in fact generation.

#### Candidate Facts & Domain Adaptation

* Difference between *candidate facts* and *summary*.
* Must developers define candidate facts per domain?
* How are nodes selected?
* Should mem0 or mem0g be chosen per domain, or can mem0g be applied universally?

#### Evaluation

* Evaluation metrics considered: F1 score, BLEU-1, memory quality.
* How does Mem0 quantitatively evaluate long-term memory effectiveness in retrieval accuracy, relevance, and downstream task performance over time?
* For fine-tuning memory quality for specific applications, what parameters can be adjusted?

### Resources
Paper: Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory
Related: [[LLM Memory]]

---

## Source: Attention mechanism.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Attention mechanism.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- language_models
---
The attention mechanism is inspired by how humans read: we don’t give equal focus to every word-we ==concentrate on those most relevant to understanding the context==. Neural networks apply the same principle, dynamically weighting parts of the input sequence based on relevance.

Originally introduced to overcome the limitations of models like [[Recurrent Neural Networks]]s and [[LSTM]]s, attention mechanisms significantly improve the handling of long-range dependencies in sequence tasks. They are now central to many modern [[NLP]] applications, including machine translation, text generation, and language understanding.
### Why Attention Matters

In traditional sequence models, all information must be compressed into a single fixed-size vector, which leads to loss of context-especially for long inputs. Attention allows the model to:

* Focus selectively on relevant input tokens
* Dynamically adjust what it "attends" to at each prediction step
* Better capture dependencies across distant positions in a sequence

### How Attention Works (Simplified)

1. Score Calculation: Compute how relevant each token is to a given query token (e.g., using dot product).
2. Weighting: Apply softmax to get attention weights (a probability distribution).
3. Context Vector: Take the weighted sum of value vectors to produce a context-specific representation.

This mechanism enables the model to emphasize important tokens and de-emphasize irrelevant ones during prediction.
### See Also

* [[Self-Attention]]
* [[Self-Attention vs Multi-Head Attention]]
* [[Key Components of Attention and Formula]]
* [[Transformer]]

---

## Source: Bag of Words.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Bag of Words.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
  - NLP
  - ML_Tools
---
In the context of natural language processing (NLP), the Bag of Words (BoW) model is a simple and commonly used ==method for text representation==. It converts text data into numerical form by treating each ==document as a collection of individual words, disregarding grammar and word order==. Here's how it works:

1. Vocabulary Creation: A vocabulary is created from the entire corpus, which is a list of all unique words appearing in the documents.

2. Vector Representation: Each document is represented as a vector, where each element corresponds to a word in the vocabulary. The value of each element is typically the count of occurrences of the word in the document.

3. Simplicity and Limitations: While BoW is easy to implement and useful for tasks like text classification, it has limitations. It ignores word order and context, and can result in large, sparse vectors for large vocabularies.

Despite its simplicity, BoW can be effective for certain NLP tasks, especially when combined with other techniques like [[TF-IDF]] to weigh the importance of words.

Takes key terms of a text in normalised ==unordered== form.

`CountVectorizer` from scikit-learn to convert a collection of text documents into a matrix of token counts.

```python
#Need normalize_document
from sklearn.feature_extraction.text import CountVectorizer

# Using CountVectorizer with the custom tokenizer
bow = CountVectorizer(tokenizer=normalize_document)
bow.fit(corpus)  # Fitting text to this model
print(bow.get_feature_names_out())  # Key terms
```

Represent each sentence by a vector of length determined by get_feature_names_out. representing the tokens contained.

---

## Source: BERT.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\BERT.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- deep_learning
- language_models
- NLP
---
### Overview
* [[BERT]] (Bidirectional Encoder Representations from [[Transformer]]) is a [[Transformer]]-based model developed by [[Google]] in 2018 (transformers are often better than traditional [[NLP]] methods.
* It is built on the [[Transformer]] architecture and uses a bidirectional context representation—capturing the meaning of words based on both their left and right context.
* Introduced in the paper: "[[BERT Pretraining of Deep Bidirectional Transformers for Language Understanding]]".

### Pretraining and [[Transfer Learning]]

Pre-trained on large corpora using two main objectives:
  * Masked Language Modeling (MLM): Predict randomly masked words.
  * Next Sentence Prediction (NSP): Predict whether one sentence follows another.
Enables [[Transfer Learning]] through task-specific fine-tuning.
### Input [[Vector Embedding|Embeddings]]

* [[Tokenisation]]: Representation of each word or token. Then embeds these tokens as [[Vector Embedding]].
* Sentence Embeddings/ [[Sentence Transformers]]: Capture relationships between entire sentences.
* [[Positional Encoding]]: Adds information about the position of words to handle order.

### Applications of BERT
1. Text Classification – Sentiment analysis, topic classification.
2. [[Named Entity Recognition|NER]] – Extraction of entities like names, places, etc.
3. Question Answering – Find answers based on a passage.
4. Text [[Summarisation]] – Create concise summaries of documents.
5. Language Translation – Assist with machine translation.
6. [[Sentence Similarity]] – Evaluate semantic similarity between sentences.

### Limitations of BERT with Large Datasets

BERT generates contextual embeddings for each word in a sentence, which are typically pooled—using methods like mean pooling—to form a single sentence embedding (see [[Sentence Transformers]]). However, such pooling treats all words equally, regardless of their importance to the sentence’s overall meaning. This limits BERT’s ability to capture fine-grained semantic relationships.

While fine-tuning BERT on sentence pairs can help produce embeddings that better reflect relational meaning, this process is computationally intensive and does not scale well to large datasets.
### Resources
* [What is BERT and how does it work? | A Quick Review](https://www.youtube.com/watch?v=6ahxPTLZxU8&list=PLcWfeUsAys2my8yUlOa6jEWB1-QbkNSUl&index=12)
### Exploratory Questions

* [ ] What does [[BERT]] learn about syntax vs semantics? ⏬
* [ ] How do [[Attention mechanism]] heads contribute to sentence meaning?
* [ ] What are the limitations of BERT for [[Sentence Similarity]] and sentence clustering?

### Variants
* BERT-base: 12 layers, 110M parameters.
* BERT-large: 24 layers, 340M parameters.
* Optimized alternatives for specific tasks:
	* [[Sentence Similarity]]: Use [[Sentence-BERT]] instead of BERT for better performance on semantic similarity.

### Related
- [[Word2vec]]

---

## Source: BERTScore.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\BERTScore.md`

---

---
aliases:
- null
category: LANG
date modified: 27-09-2025
tags:
- language_models
---

---

## Source: Chain of thought.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Chain of thought.md`

---

---
aliases:
- null
category: LANG
date modified: 27-09-2025
tags:
- language_models
---
**Chain of Thought (CoT) reasoning**

Asking sequenced questions that guide someone (or yourself) through a reasoning path is a core technique in problem-solving and teaching. Examples:

- "What is the known information?"
- "What is being asked?"
- "What patterns can we observe?"
- "What similar problems have we solved before?"

Used in in AI systems is a cognitive-inspired framework that improves the performance of large [[Language Models]] (LLMs) by explicitly guiding the AI through intermediate reasoning steps.

Advantages of Chain of Thought:
- **Improved [[Interpretability]]**: Since the model outputs intermediate steps, it's easier for humans to understand how the final answer was reached.
- **Better Performance on Complex Tasks**: CoT allows the model to handle multi-step reasoning more effectively.
- **Easier Debugging**: If there's an error in reasoning, it can be spotted at a specific step in the chain, which aids in model fine-tuning and debugging.

Related to:
- [[Model Ensemble]]

---

## Source: ChatGPT.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\ChatGPT.md`

---

---
aliases:
- null
category: LANG
date modified: 27-09-2025
tags:
- language_models
---

---

## Source: Claude.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Claude.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- GenAI
---
Claude is better for code and uses Artifact for tracking code changes.

Claude is crazy see: https://youtu.be/RudrWy9uPZE?t=473

Artefacts exist

---

## Source: Comparing LLMs.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Comparing LLMs.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- language_models
---
Use lmarena.ai as a bench marking tool. 
web dev arena
text to image leader board

Related:
[[LLM]]
[[Hugging Face]]

---

## Source: Distillation.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Distillation.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- GenAI
- language_models
- ml
---
Training smaller models with larger.

[[Transfer Learning]]
[[Small Language Models]]

![[Pasted image 20250130074219.png]]

---

## Source: ElasticSearch.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\ElasticSearch.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- NLP
- tool
---
Elasticsearch is an open source distributed [[Search]] and analytics engine, often used to store and [[Search]] through text data (e.g., logs, documents, articles). It's commonly integrated with NLP workflows for:

 - Storing extracted named entities or keywords
 - Enabling full-text search over processed corpora
 - Ranking documents based on custom scoring

Use Cases:

 - Search systems over preprocessed corpora
 - Document similarity lookup
 - Named entity indexing

Integration Example:

 - Use [[spaCy]] to extract keywords or metadata
 - Store results in Elasticsearch index
 - Use query interface to retrieve matching or related docs

Exploratory Questions:

 - How does spaCy output map to ElasticSearch indexing fields?
 - Can entity relationships or dependency trees be indexed effectively?
 - How can [[TF-IDF]] or vector search (e.g., via Elastic’s k-NN or OpenSearch) be layered in?

---

## Source: Embedded Methods.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Embedded Methods.md`

---

---
aliases:
- null
category: LANG
date modified: 27-09-2025
tags:
- selection
---
Embedded methods for [[Feature Selection]] ==integrate [[Feature Selection]] directly into the model training process.==

Embedded methods provide a convenient and efficient approach to feature selection by integrating it into the model training process, ultimately leading to models that are more parsimonious and potentially more interpretable.

1. Incorporated into Model Training: Unlike [[Filter Methods]] and [[Wrapper Methods]], which involve feature selection as a separate step from model training, embedded methods perform feature selection simultaneously with model training. This means that feature importance or relevance is determined within the context of the model itself.

2. Regularization Techniques: Embedded methods commonly use [[Regularisation]] techniques to penalize the inclusion of unnecessary features during model training. 

3. Automatic Feature Selection: Embedded methods automatically select the most relevant features by learning feature importance during the training process. The model adjusts the importance of features iteratively based on their contribution to minimizing the [[Loss function]].

4. Examples of Embedded Methods:
   - [[L1 Regularisation]] (L1 Regularization):
   - [[Elastic Net]]: Elastic Net combines L1 ([[L1 Regularisation]]) and L2 ([[Ridge]]) regularization .
   - Tree-based Methods: [[Decision Tree]] and ensemble methods like [[Random Forest]] and [[Gradient Boosting]] inherently perform feature selection during training by selecting the most informative features at each split node of the tree.
   - [[CART]]

5. Advantages:
   - Simplicity: Embedded methods simplify the feature selection process by integrating it into model training, reducing the need for additional preprocessing steps.
   - Efficiency: Because feature selection is performed during model training, embedded methods can be more computationally efficient compared to wrapper methods, which require training multiple models.

6. Considerations:
   - [[Hyperparameter Tuning]]: Tuning regularization parameters or other model-specific parameters may be necessary to optimize feature selection performance.
   - Model [[Interpretability]]: While embedded methods can automatically select features, interpreting the resulting model may be challenging, especially for complex models like ensemble methods.

---

## Source: embeddings for OOV words.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\embeddings for OOV words.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- NLP
- optimisation
---
Can you find words in a [[Vector Embedding|word embedding]] that where not used to creates the embedding? These are [[OOV words]].

Yes, but with important caveats. If a word is not in the [[spaCy]] model’s vocabulary with a vector, then:

### What you can do

#### Option 1: Filter out words without vectors (what you're doing now)
This is the cleanest option:
```python
if token.has_vector:
    embeddings.append(token.vector)
    valid_words.append(word)
```

#### Option 2: Fallback to character-level embeddings (optional)
If you're using `en_core_web_lg`, spaCy sometimes provides approximate vectors for out-of-vocabulary (OOV) words using subword features. But with `en_core_web_md`, OOV words truly lack vector meaning.

#### Option 3: Use a different embedding model
Use FastText or transformer-based models (e.g., Sentence Transformers), which can produce [[embeddings for OOV words]] based on subword information or context.

Example with [[FastText]] (using gensim):
```python
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format("cc.en.300.vec")  # or download from FastText
embedding = model.get_vector("unseenword")  # FastText will synthesize it
```

### Summary

| Approach                     | Handles OOV? | Notes |
|-----------------------------|--------------|-------|
| spaCy `en_core_web_md`      | ❌            | Skips words without vectors (recommended) |
| spaCy `en_core_web_lg`      | ⚠️ Sometimes  | May infer vectors using subword info |
| FastText / GloVe            | ✅            | Good for unseen words |
| Sentence Transformers (BERT)| ✅            | Contextualized, ideal for phrases/sentences |

---

## Source: Evaluate Embedding Methods.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Evaluate Embedding Methods.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- analysis
- evaluation
- nlp
---
#### How to Evaluate Embedding Methods

##### Semantic Relationship Fidelity

A good [[Vector Embedding|embedding]] should place semantically similar sentences or words closer together in the embedding space. You can test this using:

 a) [[Cosine Similarity]]

* Compute the [[Cosine Similarity]] between embedding vectors.
* Higher similarity between semantically related pairs (e.g., *"Paris is the capital of France"* vs *"France’s capital is Paris"*) indicates better embedding quality.
* Compare scores across methods:

  ```python
  from sklearn.metrics.pairwise import cosine_similarity
  similarity = cosine_similarity([vec1], [vec2])
  ```

 b) Analogy or Word Arithmetic

* Test whether embeddings support compositional reasoning.
* Example:
  $\text{Embedding}(\text{king}) - \text{Embedding}(\text{man}) + \text{Embedding}(\text{woman}) \approx \text{queen}$
* This shows if semantic and syntactic dimensions are meaningfully encoded ([[syntactic relationships]]).

 c) Clustering Consistency

* Cluster the embeddings (e.g. via k-means) and evaluate whether related texts group together.
* Measure cluster cohesion and separation (e.g. using Silhouette Score).
##### Information Content & Sparsity

 a) Use [[TF-IDF]] as Baseline

* TF-IDF scores highlight the most important words in a text.
* Evaluate how well dense embeddings retain the importance structure identified by TF-IDF.
* For example, check whether high TF-IDF words receive higher attention in models like BERT (via attention weights) or influence sentence embedding directions.

##### Downstream Task Performance

Train simple classifiers (e.g. [[Logistic Regression]]) on embeddings to predict:
  * Sentiment
  * Topic
  * Semantic similarity class (entailment, contradiction, etc.)

Better embeddings typically yield better accuracy/F1 on such tasks.

##### Visual Inspection

 a) [[Dimensionality Reduction]]
* Use [[Principal Component Analysis|PCA]], [[t-SNE]], or [[UMAP]] to project embeddings into 2D.
* Visually inspect whether semantically similar items form coherent clusters.

#### Guiding Questions

* Do the embeddings distinguish fine-grained semantic shifts (e.g., “bank” as a financial institution vs riverbank)?
* Are word or sentence embeddings stable across paraphrased or reordered text?
* Do similar sentences result in embeddings with high cosine similarity?
* How well do embeddings handle [[OOV words]] or rare terms?

---

## Source: Fuzzywuzzy.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Fuzzywuzzy.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- NLP
---
Tool used for correcting spelling with [[Pandas]].

[[Data Cleansing]]

---

## Source: Generative AI From Theory to Practice.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Generative AI From Theory to Practice.md`

---

---
aliases:
- null
category: LANG
date modified: 27-09-2025
tags:
- GenAI
---
### Objective:

 How do LLMs work and operate.  
 Enabling [[LLM]]'s at scale:
 Explore recent AI and [[Generative AI]] language models 


### Steps

Math on words: Turn words into coordinates.
Statistics on words: Given context what is the probability of whats next.
Vectors on words. [[Cosine Similarity]]
How train: Use [[Markov chain]] for prediction of the next [[Tokenisation]]


Tokeniser: map from token to number

1. Pre-training: tokenise input using [[NLP]] techqinues
2. [[LLM]] looks at context: nearby tokens, in order to predict

different implmentationg for differnet languages. Differnet tokenisers or translating after.

Journey to scale:

1. Demos, POC (plan to scale): understand limitations
2. Beyond experiments and before production: 
3. Enterprise level: translate terms so they can use governess techniques.

Building:

![[Pasted image 20240524130607.png]]

### [[Software Development Life Cycle]]

For GenAI: Building an applicaiton with GenAi features

1. Plan: use case: prompts : archtecture: cloud or on site
2. Build: vector database
3. Test: Quality and responsible ai. 

### [[call summarisation]]

take transcript - > summariser -> summarise

Source: human labeled transcripts to check summariser. 

![[Pasted image 20240524131311.png|500]]

[[Ngrams]] analysis - when specific words realy matter


### [[RAG]]

Use relvant data to make response better:

![[Pasted image 20240524131603.png]]

## [[GAN]]

For image models.

Examples: midjourney,stable diffusion,dall-e 3

image model techniques:
- text to image
- image to image
## Notes: 

Use [[LLM]]'s to get short info, then cluster.
Going round training data : called a Epochs

---

## Source: Generative AI.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Generative AI.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- GenAI
---
Tools that generate content.

Mainly:
- Text
- Images

---

## Source: Grammar method.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Grammar method.md`

---

---
aliases:
- null
category: LANG
date modified: 27-09-2025
tags:
- NLP
---
Can understand the Grammar as a method for acceptable sentences.

---

## Source: Guardrails.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Guardrails.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- business
- GenAI
---
Controlling a [[Generative AI]] in business through the use of [[Guardrails]] ensures that the AI remains aligned with specific business goals and avoids unintended or harmful outputs. Guardrails are essential for maintaining security, compliance, and reliability in AI systems. Here's an outline based on your notes:

### 1. Input Guardrails

   - Prompt Injection Control: [[Prompts]] To prevent users from prompting the AI in ways that could result in harmful or inappropriate responses, filtering or validating inputs can be essential. This reduces the risk of the model being "jailbroken" (i.e., forced to generate outputs outside its intended use case).
   - Topic Restriction: Limit the AI’s inputs to specific business-relevant topics. For instance, if the AI is designed for customer support, it should ignore inputs about unrelated topics (e.g., entertainment or politics).
   - User Authentication: Depending on business needs, certain input guardrails can restrict access to specific features or sensitive information based on user credentials or roles.

### 2. Output Guardrails

   - Content Moderation: Post-processing can be applied to outputs to ensure they align with business values, compliance regulations, or safety standards. For example, any harmful or offensive language can be filtered out.
   - Pre-defined Boundaries: Limit the AI’s responses to fall within specific domains. For instance, when the AI is asked questions outside its scope, it can respond with a predefined message, such as "I am not programmed to handle that topic."
   - Compliance and Ethical Constraints: Outputs can be regulated to ensure the model adheres to legal, ethical, and regulatory constraints, which is especially important in industries like finance or healthcare.

### 3. Jailbreaking Concerns

   - Jailbreaking occurs when a user manipulates the system to bypass these guardrails, leading to undesirable outputs. This depends on the business context—some may tolerate more flexible AI behavior, while others, like legal or healthcare firms, need strict controls.

### 4. Business-Specific Use Cases

   - Tailor the AI to address specific business needs. For example, a generative AI for a legal firm should stick to legal advice and documentation, whereas a customer service chatbot should handle predefined topics like returns and product support.
   - [[Data Observability|monitoring]] / Monitoring and Logging: Keep track of input and output interactions to ensure that the AI’s performance remains within its intended boundaries.

---

## Source: How businesses use Gen AI.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\How businesses use Gen AI.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- business
- GenAI
---
Businesses leverage [[Generative AI]] to transform various operations, using models like OpenAI, Gemini (Google Cloud), Anthropic, and Meta models. These models provide services through cloud providers, making them accessible via APIs. Key use cases include:

1. **Content Creation**: Generative AI can produce text, images, code, and even videos, enhancing marketing, design, and communication efforts.
2. **Customer Support**: AI chatbots and assistants automate customer interactions, reducing response times and improving service quality.
3. **Data Analysis & Insights**: Models help businesses analyze large datasets, enabling predictive analytics and trend forecasting.
4. **Customization**: Personalization of products and services, such as tailored recommendations or [[transactional journeys]]/customer experiences, is powered by generative AI.
5. **Multi-Model Access**: Enterprises use AI gateways to integrate multiple generative models, allowing them to choose the best model for specific tasks based on performance or cost efficiency.

Cloud providers like **Google Cloud (Gemini)** or **Microsoft Azure (OpenAI)** offer easy integration of these models into business workflows through APIs, streamlining deployment for large-scale applications

## AI Gateway?

An AI Gateway is a middleware platform that simplifies and secures interactions between AI models and applications. In this context, businesses use AI gateways to streamline the integration, management, and deployment of generative AI models like those provided by OpenAI, Google (Gemini), and Anthropic. AI gateways provide the following key benefits:

1. **Model Access and Management**: They centralize access to multiple AI models via APIs, making it easier for businesses to switch between or utilize multiple AI models for different tasks.
2. **Security and Governance**: AI gateways add layers of security, enabling compliance with regulations and protecting proprietary data when using external AI services [1] . [2]
3. **Performance Optimization**: By handling the AI model interactions efficiently, gateways can reduce latency and improve [[model performance]] in business applications [3]
## 🌐 Sources
1. [konghq.com - What is an AI Gateway? Concepts and Examples](https://konghq.com/blog/enterprise/what-is-an-ai-gateway)
2. [ibm.com - How an AI Gateway provides leaders with greater control](https://www.ibm.com/blog/announcement/how-an-ai-gateway-provides-greater-control-and-visibility-into-ai-services/)
3. [traefik.io - AI Gateway: What Is It? How Is It Different From API Gateway?](https://traefik.io/glossary/ai-gateway/)

---

## Source: How LLMs store facts.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\How LLMs store facts.md`

---

---
aliases:
  - 
category: LANG
date modified: 27-09-2025
tags:
  - agents
  - NLP
---
[How might LLMs store facts](https://www.youtube.com/watch?v=9-Jl0dxWQs8&list=PLZx_FHIHR8AwKD9csfl6Sl_pgCXX19eer&index=6)

Not solved

How do [[Multilayer Perceptrons]] store facts?

Different directions encode information in [[Vector Embedding]] space.

MLP's are blocks of vectors, these are acted on my the context matrix 

[[Johnson–Lindenstrauss lemma]]

Sparse Autoencoder - used in [[Interpretability]] of [[LLM]] responses

See [[Anthropic]] posts
- https://transformer-circuits.pub/2022/toy_model/index.html#adversarial
- https://transformer-circuits.pub/2023/monosemantic-features

---

## Source: How to reduce the need for Gen AI responses.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\How to reduce the need for Gen AI responses.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- business
- GenAI
---
Reducing the need for frequent [[Generative AI]] (Gen AI) responses can be done by leveraging techniques such as [[caching]] and setting up predefined [[transactional journeys]]. Here's a breakdown:

1. **Caching AI Responses**: Caching allows storing frequently requested AI responses and reusing them. This reduces the number of queries to the AI model, thus lowering both response time and cost. For example, common queries like "How do I reset my password?" can be cached for quick reuse without engaging the AI model each time (1).

2. **Predefined Transactional Journeys**: For repetitive tasks (e.g., "I want to close my account"), predefined ==workflows== or "journeys" can be set up. These automate processes without requiring AI interaction. This is ideal for tasks like bill payments, account management, or order cancellations, where responses can be scripted or handled by traditional logic, bypassing AI.

### Examples of User Journeys:
- **Account Closure**: Guiding users through the steps to close an account without involving AI.
- **Password Reset**: Automating the reset process with predefined steps.
- **Order Tracking**: Providing real-time updates using existing tracking systems.
## 🌐 Sources
1. [medium.com - How Cache Helps in Generative AI Response and Cost Optimization](https://medium.com/@punya8147_26846/how-cache-helps-in-generative-ai-response-and-cost-optimization-9a6c9be058bb)
2. [medium.com - Slash Your AI Costs by 80%](https://medium.com/gptalk/slash-ai-costs-by-80-the-game-changing-power-of-prompt-caching-d44bcaa2e772)
3. [botpress.com - How to Optimize AI Spend Cost in Botpress](https://botpress.com/blog/how-to-optimize-ai-spend-cost-in-botpress)

---

## Source: How would you decide between using TF-IDF and Word2Vec for text vectorization.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\How would you decide between using TF-IDF and Word2Vec for text vectorization.md`

---

---
aliases:
- null
category: LANG
date modified: 27-09-2025
tags:
- NLP
---

---

## Source: In NER how would you handle ambiguous entities.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\In NER how would you handle ambiguous entities.md`

---

---
aliases:
- null
category: LANG
date modified: 27-09-2025
tags:
- NLP
---
Handling ambiguous entities in [[Named Entity Recognition]] (NER) can be quite challenging. Here are some strategies that can be employed:

1. **Contextual Analysis**: Utilize the surrounding ==context== of the ambiguous entity to determine its correct classification. For example, the word "Apple" could refer to the fruit or the company, but the context in which it appears can help disambiguate its meaning.

2. **Disambiguation Models**: Implement additional models specifically designed for entity disambiguation. These models can leverage knowledge bases or ontologies to determine the most likely entity based on context.

3. **Multi-label Classification**: Instead of forcing a single label, allow for multiple possible labels for ambiguous entities. This can be useful in cases where an entity might belong to more than one category.

4. **Training Data**: Ensure that the training dataset includes examples of ambiguous entities in various contexts. This can help the model learn to recognize and differentiate between them.

5. **User Feedback**: Incorporate user feedback mechanisms to refine the model's predictions. If users can correct or confirm entity classifications, this can improve the model over time.

---

## Source: Key Components of Attention and Formula.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Key Components of Attention and Formula.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- math
- NLP
---
### Key Components of Attention and Formula

1. Query: Represents the current word or position that requires attention: $Q$
2. Key: Represents each word in the input sequence: $K$
3. Value: Represents the actual content or information in the input sequence: $V$
4. Attention Scores: The attention mechanism computes the relevance between the query and each key by computing a similarity score (such as dot-product or other scoring methods).
5. Softmax: These scores are then passed through a softmax function to form a probability distribution, which gives us the attention weights.
6. Context Vector: A weighted sum of the values ($V$), using the attention weights, is computed. This context vector is what the model uses to generate the output token.

Given a query matrix, key matrix, and value matrix, attention is calculated as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q$,$K$, and $V$are matrices of query, key, and value vectors.
- $d_k$ is the dimension of the keys.
- The softmax is applied row-wise to produce attention weights.

---

## Source: Knowledge graph vs RAG setup.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Knowledge graph vs RAG setup.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- data_structure
- GenAI
- memory_management
---
### Comparison: Knowledge Graph vs. RAG Setup

- ==**Knowledge Graphs** are structured representations of entities and their relationships, designed primarily for querying, reasoning, and storing factual information.==
- ==**RAG setups** enhance generative models by retrieving external knowledge (from unstructured or semi-structured data) and integrating it into the generation process.==

While not the same, these two concepts can be used together to build systems that combine structured knowledge retrieval with the natural language generation capabilities of RAG models.

While **knowledge graphs** and **RAG** are distinct, they can be integrated to improve certain systems:
- ==A **RAG model** could use a **knowledge graph** as the retrieval source.== Instead of retrieving unstructured text documents, the RAG model could retrieve structured, factual triples from a knowledge graph and incorporate this into the generation process. This would improve the accuracy of fact-based questions and answers.

A [[Knowledge Graph]] and a **Retrieval-Augmented Generation ([[RAG]])** setup are related but distinct concepts, particularly in how they handle knowledge representation and retrieval. While they can complement each other in certain applications, they serve different purposes and operate in different ways.

| Aspect                      | Knowledge Graph                                                                         | RAG Setup                                                                             |
| --------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **Purpose**                 | Stores and organizes knowledge for querying and reasoning                               | Combines retrieval of external information with ==text generation==                   |
| **Data Structure**          | ==Highly structured== (graph with nodes and edges)                                      | Unstructured or semi-structured (documents, text snippets)                            |
| **Retrieval Mechanism**     | Queries are made through graph traversal or SPARQL-like languages                       | Information is retrieved via search mechanisms (e.g., dense embeddings)               |
| **Usage**                   | Often used for querying factual data, answering structured queries, [[Semantic Relationships]] | Used to enhance the factual accuracy of generative models by retrieving external data |
| **Reasoning and Inference** | Capable of logical reasoning based on relationships                                     | Does not perform reasoning; it retrieves and integrates relevant text                 |
| **Scalability**             | Requires careful design to manage large, complex graphs                                 | Can handle large text corpora, but retrieval quality affects the final generation     |
| **Generative Capabilities** | Not generative (focused on querying existing knowledge)                                 | [[Generative]] (synthesizes and generates natural language responses)                 |

---

## Source: Language Model Output Optimisation.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Language Model Output Optimisation.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- language_models
- optimisation
---
What techniques from [[information theory]] can be used to measure and optimize the amount of information conveyed by an language model?

In information theory, several techniques can be applied to measure and optimize the amount of information conveyed by an [[Language Models]].

1. Entropy: Entropy measures the uncertainty or unpredictability of a random variable. In the context of language models, it can be used to quantify the uncertainty in predicting the next word in a sequence. Lower entropy indicates more predictable and informative outputs.

2. [[Cross Entropy]]: This measures the difference between two probability distributions. For language models, cross-entropy can be used to evaluate how well the predicted distribution of words matches the actual distribution in the data. Minimizing cross-entropy during training helps optimize the model's predictions.

3. Perplexity: Perplexity is a common metric for evaluating language models. It is the exponentiation of the cross-entropy and represents the model's uncertainty in predicting the next word. Lower perplexity indicates a better-performing model.

4. Mutual Information: This measures the amount of information shared between two variables. In language models, it can be used to assess how much information about the input is retained in the output, helping to optimize the model's ability to convey relevant information.

5. KL Divergence: Kullback-Leibler divergence measures how one probability distribution diverges from a second, expected probability distribution. It can be used to optimize language models by minimizing the divergence between the predicted and true distributions.

6. Information Bottleneck: This technique involves finding a balance between compressing the input data and preserving relevant information for the task. It can be used to optimize models by focusing on the most informative features.

7. Rate-Distortion Theory: This involves finding the trade-off between the fidelity of the information representation and the amount of compression. It can be applied to optimize language models by balancing the complexity of the model with the quality of the information conveyed.

8. [[Attention mechanism]]: While not strictly an information theory concept, attention mechanisms in neural networks can be seen as a way to dynamically allocate information processing resources, focusing on the most informative parts of the input.

---

## Source: Language Models Large (LLMs) vs Small (SLMs).md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Language Models Large (LLMs) vs Small (SLMs).md`

---

---
aliases:
- null
category: LANG
date modified: 27-09-2025
tags:
- language_models
---
### Overview
Language models can be categorized into large language models ([[LLM]]) and small language models ([[SLM]]). While LLMs boast extensive general-purpose knowledge and capabilities, SLMs offer distinct advantages in certain scenarios, particularly when it comes to efficiency, resource constraints, and task-specific environments.

### Key Differences

| Aspect             | LLMs                                              | SLMs                                                 |
|--------------------|---------------------------------------------------|------------------------------------------------------|
| Accuracy        | Higher accuracy across broad tasks due to large datasets and extensive training. | Comparable performance in domain-specific tasks after fine-tuning. |
| Efficiency      | Computationally expensive; requires significant resources for training and inference. | More resource-efficient; suited for edge devices and real-time applications. |
| [[Interpretability]]| Often a "black box"; difficult to explain decision-making. | More interpretable due to simpler architecture. |
| Generality      | General-purpose; capable of handling a wide range of tasks. | Task-specific; excels in specific domains and structured data. |

---

## Source: Language Models.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Language Models.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- portal
---
A language model is a machine learning model that is designed to understand, generate, and predict human language. 

It does this by analyzing large amounts of text data to learn the patterns, structures, and relationships between words and phrases. 

They work by assigning probabilities to sequences of words, allowing them to predict the next word in a sentence or generate coherent text based on a given prompt.

Related to:
[[LLM]]
[[Small Language Models|SLM]]

---

## Source: lemmatization.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\lemmatization.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- NLP
---
Lemmatization is the process of ==reducing a word to its base or root== form, known as the "lemma." 

Unlike stemming, which simply cuts off word endings, lemmatization considers the context and morphological analysis of the words. 

It ensures that the root word is a valid word in the language. ==For example, the words "running," "ran," and "runs" would all be lemmatized to "run."== 

This process helps in normalizing text data for natural language processing tasks by grouping together different forms of a word.

---

## Source: LLM Memory.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\LLM Memory.md`

---

---
aliases:
- context
- What is LLM memory
category: LANG
date modified: 27-09-2025
tags:
- language_models
- NLP
---
Memory in large [[Language Models]] (LLMs) involves managing context windows to enhance reasoning capabilities without the high costs associated with traditional training methods. The goal of [[LLM Memory]] is to address challenges like "forgetting," where LLMs struggle to retain context across interactions.
## Key Concepts:

Forgetting Context:
Understanding how and why LLMs lose context, especially in multi-turn dialogues, and its impact on response accuracy. Forgetting occurs due to the limitations of fixed context windows, manifesting differently in single-turn (immediate forgetting) versus multi-turn interactions (progressive loss of context).

Prioritization of Context:
Techniques for determining which parts of the context are most relevant and need to be retained, optimizing memory usage.

Time Length of Memory:
Balancing how long memory should be maintained to ensure it remains useful and relevant over time.

Dynamic Memory Management:
Adapting memory structures in real-time to accommodate evolving knowledge and interactions.

In-Context Memory:
Memory tied to specific interactions, making it more relevant and easier to apply in particular scenarios.

Multi-turn Interactions:
Addressing context retention across multiple interactions, emphasizing the importance of maintaining coherence over extended conversations.
## Types of Memory:

Semantic Memory:
Focuses on the meaning and [[Semantic Relationships]] between concepts, which is crucial for improving LLM reasoning and context understanding.

Hierarchical Memory:
Balances immediate retrieval with long-term storage of information, enabling better performance in various applications.
Supports evolving and persistent memory systems tailored to specific tasks.

---

## Source: LLM.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\LLM.md`

---

---
aliases:
- Large Language Models
- LLMs
category: LANG
date modified: 27-09-2025
tags:
- language_models
---
A Large Language Model (LLM) is a type of language model designed for language understanding and generation. They can perform a variety of tasks, including:

- Text generation
- Machine translation
- Summary writing
- Image generation from text
- Machine coding
- Chatbots or Conversational AI
# Questions

- [[How do we evaluate of LLM Outputs]]
- [[LLM Memory|What is LLM memory]]
- [[Relationships in memory|Managing LLM memory]]
- [[Mixture of Experts]]: having multiple experts instead of one big model.
- [[Distillation]]
- [[Mathematics]] on the parameter usage [[Attention mechanism]]
- Use of [[Reinforcement learning]] in training [[Chain of thought]] methods in LLM's (deepseek)

## How do Large Language Models (LLMs) Work?

Large [[Language Models]] (LLMs) are a type of artificial intelligence model that is designed to understand and generate human language. Key aspects of how they work include:

- Word Vectors: LLMs represent words as long lists of numbers, known as word vectors ([[Vector Embedding|word embedding]]).
- Neural Network Architecture: They are built on a neural network architecture known as the [[Transformer]]. This architecture enables the model to identify relationships between words in a sentence, irrespective of their position in the sequence.
- [[Transfer Learning]]: LLMs are trained using a technique known as transfer learning, where a pre-trained model is adapted to a specific task.

## Characteristics of LLMs

- ==Non-Deterministic:== LLMs are non-deterministic, meaning the types of problems they can be applied to are of a probabilistic nature (==temperature==).
- Data Dependency: The performance and behaviour of LLMs are heavily influenced by the data they are trained on.

---

## Source: Local LLM use cases.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Local LLM use cases.md`

---

---
aliases:
- null
category: LANG
date modified: 27-09-2025
tags:
- language_models
---
- [ ] Can you load a local model onto a SD card and run it on a raspberry pi?

Why small models work:
- They’re faster, cheaper, and can run on consumer hardware (laptops, even Raspberry Pi-level devices with optimisations) while giving acceptable quality for narrow, well-defined tasks. Pairing them with a vector store (e.g., Chroma, [[Weaviate]], Milvus) for [[RAG]] can dramatically boost usefulness without increasing model size.
### Use cases

#### Text Processing & Automation
* Template filling – e.g., generating structured responses, filling in report fields.
* Summarisation – condensing meeting transcripts or local documents without sending data to the cloud.
* Classification – tagging or categorising requests, tickets, or files.
* Text cleaning – grammar correction, standardising language for logs or reports.

#### Domain-Specific Models
* Fine-tune a small LLM for:
  * Industry jargon translation (e.g., maintenance logs → plain English).
  * Technical troubleshooting guides.
  * Incident classification in operations or engineering.
* Works well when paired with RAG (Retrieval-Augmented Generation) from a local knowledge base.
#### Edge & Offline Scenarios
* Field work in remote areas (e.g., utilities, scientific expeditions).
* IoT devices with natural language interfaces.
* Portable knowledge assistants for technicians, inspectors, or surveyors.
#### Educational & Training Tools
* Interactive Q\&A tutors for company onboarding.
* Scenario-based training simulations where the model plays a role.

#### Related:
- [[Small Language Models]]

---

## Source: Mathematical Reasoning in Transformers.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Mathematical Reasoning in Transformers.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- '#question'
---
**transformer-based models** that address mathematical reasoning either through pretraining, hybrid systems, or fine-tuning on specific mathematical tasks

- **Challenges**: General-purpose transformers [[Transformer|Transformer]] are trained primarily on large corpora of text, which include mathematical problems but lack systematic and rigorous math-specific training. This results in limited capabilities for handling complex calculations or abstract algebraic problems.

- **Grokking in Mathematical Reasoning**: This is an area of research where models are trained on small datasets of synthetic math problems to encourage **grokking**, a phenomenon where the model suddenly achieves near-perfect performance after extended training. Researchers are interested in how transformers might be able to **"[[grok]]"** math concepts after seeing many examples.

math data sets: MATH dataset,Aristo

Pretrained transformers on math specific data.

[[GPT-f]] represents a significant advancement in the use of **transformer-based models for mathematical reasoning**,

---

## Source: Mixture of Experts.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Mixture of Experts.md`

---

---
aliases:
- null
category: LANG
date modified: 27-09-2025
tags:
- language_models
---
Different parts of the network focusing on parts of the questions

Routing, distribution

activating

---

## Source: Model Cascading.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Model Cascading.md`

---

---
aliases:
- null
category: LANG
date modified: 27-09-2025
tags:
- language_models
---

---

## Source: Multi-head attention.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Multi-head attention.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- deep_learning
- NLP
---
Multi-head attention extends the standard [[Attention mechanism]] by enabling the model to attend to different parts of an input sequence simultaneously, capturing diverse relationships-both local and global.

#### Why Use Multi-Head Attention?

* Multiple Focus Areas: Each head attends to different parts of the sequence. Some capture short-range (syntactic) relationships, others long-range (semantic) dependencies.
* Diverse Representations: Each head operates in a distinct learned subspace, allowing the model to represent the same input in multiple ways.
* Richer Contextual Understanding: By aggregating these views, the model gains a more expressive and nuanced understanding of the input.

#### How It Works (Simplified Steps)

1. Linear Projections: Input tokens are projected into queries ($Q$), keys ($K$), and values ($V$) separately for each head.
2. Independent Attention: Each head computes attention scores and outputs a context vector.
3. Concatenation: Outputs from all heads are concatenated.
4. Final Projection: A linear transformation combines the multi-head output into a single vector.

#### Example Applications

In language translation, heads might focus on:
  * Aligning subject-verb structures
  * Resolving pronoun references
  * Handling grammatical reordering between source and target languages

In semantic tasks, they can disambiguate words (e.g., “bank” as riverbank or financial institution) by attending to context.

---

## Source: Named Entity Recognition.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Named Entity Recognition.md`

---

---
aliases:
- Entity Recognition
- NER
category: LANG
date modified: 27-09-2025
tags:
- modeling
- NLP
---
Named Entity Recognition (NER) is a subtask of [[NLP|Natural Language Processing]] (NLP) that involves identifying and classifying key entities in text into predefined categories such as names, organizations, locations.

The process typically employs algorithms like Conditional Random Fields (CRFs) or deep learning models such as Bi-directional [[LSTM]] (Long Short-Term Memory) networks.

Mathematically, NER can be framed as a sequence labeling problem where the goal is to assign a label $y_i$ to each token $x_i$ in a sentence. The model learns from annotated datasets, optimizing parameters to maximize the likelihood $P(y|x)$ using techniques like [[Backpropagation]].

NER has significant implications in information extraction, search engines, and automated customer support systems.

### Important
 - NER transforms unstructured text into [[structured data]] for analysis.
 - The choice of model significantly impacts the accuracy of entity recognition.

### Example
 An example of NER is identifying "Apple Inc." as an organization in the sentence: "Apple Inc. released a new product."

### Follow up questions
 - [[How does the choice of training data affect the performance of NER models]]
 - [[What are the challenges of NER in multilingual contexts]]
 - [[Why is named entity recognition (NER) a challenging task]]
 - [[In NER how would you handle ambiguous entities]]
 - [[NER Implementation]]

### Related Topics
 - Text classification in [[NLP]]  
 - Information extraction techniques

---

## Source: NER Implementation.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\NER Implementation.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
  - NLP
---
```python
import spacy
# Load spaCy model for NER
!python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")
# ===============================================
# 6. EXTRACT COMPANY NAMES (NER)
# ===============================================
def extract_companies(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == 'ORG']

df['companies'] = df['headline'].apply(extract_companies)
```

---

## Source: Ngrams.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Ngrams.md`

---

---
aliases:
- null
category: LANG
date modified: 27-09-2025
tags:
- NLP
---
N-grams are used in NLP that allow for the analysis of text data by breaking it down into smaller, manageable sequences. 

An **N-gram** is a contiguous sequence of *n* items (or tokens) from a given sample of text or speech. In the context of natural language processing ([[NLP]]) and text analysis, these items are typically words or characters. 

N-grams are used to analyze and ==model the structure of language==, and they can help in various tasks such as [[Text Classification]].
### Types of N-grams
- **Unigram**: An N-gram where *n = 1*. It represents individual words or tokens. For example, in the sentence "I love AI", the unigrams are ["I", "love", "AI"].

- **Bigram**: An N-gram where *n = 2*. It represents pairs of consecutive words. For the same sentence, the bigrams would be ["I love", "love AI"].

- **Higher-order N-grams**: These can go beyond three words, such as 4-grams (quadgrams) or 5-grams, and so on.
### Code implementations:

This can be does through kwargs in CountVectorizer.

---

## Source: NLP Portal.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\NLP Portal.md`

---

---
aliases: []
category: LANG
date modified: 28-09-2025
tags:
  - portal
---

---

## Source: NLP.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\NLP.md`

---

---
aliases:
- Natural Language Processing
category: LANG
date modified: 27-09-2025
tags:
- NLP
---
Natural Language Processing (NLP) involves the interaction between computers and humans using natural language. It encompasses various techniques and models to process and analyze large amounts of natural language data.

## Key Concepts

### [[Preprocessing]]
- [[Normalisation of Text]]: The process of converting text into a standard format, which may include lowercasing, removing punctuation, and stemming or [[lemmatization]].
- [[Part of speech tagging]]: Assigning a specific part-of-speech category (such as noun, verb, adjective, etc.) to each word in a text.
- Understanding a sentence: Participants and actions.

### Models
- [[Bag of Words]]: Represents text data by counting the occurrence of each word in a document, ignoring grammar and word order. It takes key terms of a text in normalized unordered form.
- [[TF-IDF]]: Stands for Term Frequency-Inverse Document Frequency. It improves on Bag of Words by considering the importance of a word in a document relative to its frequency across multiple documents.
- Vectorization: Converting text into numerical vectors. Techniques like Bag of Words, TF-IDF, or [[Vector Embedding]] (e.g., Word2Vec, GloVe) are used to represent text data numerically.

### Analysis
- [[One-hot encoding]]: Converts categorical data into a binary vector representation, indicating the presence or absence of a word from a list in the given text.

### Methods
- [[Ngrams]]: Creates tokens from groupings of words, not just single words. Useful for capturing context and meaning in text data.
- [[Grammar method]]: Involves analyzing the grammatical structure of sentences to extract meaning and relationships between words.

### Actions
- [[Summarisation]]: The process of distilling the most important information from a text to produce a concise version.

## Tools and Libraries

### General Imports

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
```

- [[nltk]]: A leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources.
  - punkt: An unsupervised trainable model for tokenizing text into sentences and words.
  - [[stopwords]]: Commonly used words (such as "the", "is", "in") that are often removed from text data because they do not carry significant meaning.
  - wordnet: A lexical database for the English language that groups words into sets of synonyms and provides short definitions and usage examples.
  - re: Regular expressions for pattern matching and text manipulation.

---

## Source: nltk.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\nltk.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- NLP
---
A package for natural language processing toolkit.

NLTK (Natural Language Toolkit) is a Python library for working with human language data. It provides tools for text processing, linguistic analysis, and building natural language processing ([[NLP]]) models. 

NLTK is an accessible toolkit for classical NLP tasks. While more modern libraries like [[spaCy]] or [[Transformer]]s are preferred for production systems, NLTK remains valuable for learning, prototyping, and linguistic exploration.

### Key Features:
- [[Tokenisation]]: breaking text into words or sentences.
- Stopwords removal: filtering out common non-informative words.
- [[Stemming]] and [[lemmatization]]: reducing words to base/root forms.
- [[Part of speech tagging]]: identifying parts of speech (e.g., noun, verb).
- [[Named Entity Recognition]] (NER)
- Parsing and Treebanks
- Access to many corpora (e.g., Gutenberg texts, [[WordNet]])

---

## Source: Non-negative Matrix Factorization.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Non-negative Matrix Factorization.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- NLP
- selection
---
Non-negative Matrix Factorization (NMF) is a matrix decomposition technique that factors a non-negative matrix $V$ into two non-negative matrices $W$ and $H$ such that:

$$
V \approx W \cdot H
$$

In [[NLP]], NMF is often applied to document-term matrices for [[topic modeling]] and [[Feature Extraction]].

### How NMF Works in NLP

1. Construct a document-term matrix $V$:

   * Rows = documents
   * Columns = terms/words
   * Entries = term frequency (TF) or TF-IDF.

1. Decompose $V$ into:

   * $W$ (document-topic matrix): Each row represents the distribution of topics for a document.
   * $H$ (topic-term matrix): Each row represents the distribution of terms for a topic.

3. Interpret topics:

   * Each topic is represented by a set of high-weight words from $H$.
   * Each document is represented by a mixture of topics from $W$.

### Application to Topic Importance Indicators

* Topic Importance for Documents: Look at $W$ to see which topics dominate a document.
* Key Words for Topics: Look at $H$ to find top terms per topic, which serve as indicators of the topic’s content or importance.
* Ranking Features: Terms with higher weights in $H$ are more important for defining a topic.
### Benefits

* Produces interpretable topics because all entries are non-negative.
* Works well with sparse and high-dimensional NLP data.
* Can complement feature importance analysis in text classification and clustering.
### Example (Python, using TF-IDF)

```python
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

documents = ["I love NLP", "Machine learning is fun", "NLP and ML are related"]
vectorizer = TfidfVectorizer()
V = vectorizer.fit_transform(documents)

nmf = NMF(n_components=2, random_state=42)
W = nmf.fit_transform(V)  # document-topic matrix
H = nmf.components_       # topic-term matrix
```

* Rows of $H$ → important words per topic (topic indicators)
* Rows of $W$ → importance of topics per document

### Image

![[Pasted image 20250823094439.png]]

---

## Source: NotebookLM.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\NotebookLM.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- tool
---
https://www.youtube.com/watch?v=EOmgC3-hznM

key topics 

chat interface takes into account resources.

save to note- to dave. 

how to select and folders - from obsidian [[Data Archive]] for this ? A getter of some kind

can convert muiltple notes into a single note.

Can add website as source.

project context - similar projects notes

Focus knowledge retrieval
- get info from sources (folders)

FAQ 

Note: can help with file extraction rem utils function (for [[NotebookLM]])

---

## Source: OOV words.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\OOV words.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- NLP
---
In [[NLP|Natural Language Processing]] (NLP), OOV words refers to:

> OOV words are terms not seen during training or not present in a model’s vocabulary. They pose challenges for text understanding, especially in traditional NLP approaches. Modern tokenization strategies (e.g., subwords) greatly reduce—but do not entirely eliminate—the issue.
### Modern NLP Models

Modern transformer-based models (e.g., BERT, GPT) almost eliminate the concept of OOV by using subword [[Tokenisation]]:
- Any word, even if unseen, can be broken into known subword units.
- However, this still has semantic implications: e.g., rare words may be split into unintuitive or ambiguous fragments.

### Context and Meaning

Many NLP models (e.g., classical models like [[Bag of Words]], or early word embeddings like [[Word2vec]]) rely on a fixed vocabulary that was built from a training corpus. Any word not seen during training is considered out-of-vocabulary.
### Why OOV Words Matter

1. Loss of information: If a model cannot represent or process a word (e.g., "microservices" or a new slang term), it cannot reason about its meaning.
2. Performance degradation: In text classification, machine translation, or entity recognition tasks, frequent OOV words reduce the model’s accuracy.
3. Domain adaptation challenges: OOV words often appear in domain-specific or noisy data (e.g., medical, legal, user-generated content).
### Strategies to Handle OOV Words

| Strategy                   | Description                                                                        | Common Usage                         |
| -------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------ |
| UNK token                  | Map all unknown words to a special token like `<UNK>`                              | Basic [[Recurrent Neural Networks|RNN]] models, early NLP          |
| Subword tokenization       | Break OOV words into smaller known parts (e.g., BPE, ==WordPiece==, SentencePiece) | Used in BERT, GPT, RoBERTa           |
| Character-level models     | Process input character-by-character, avoiding fixed vocabulary                    | Useful in noisy or multilingual text |
| Dynamic vocabulary updates | Re-train or extend embeddings on new corpora                                       | Custom applications                  |
### Example

Assume a vocabulary contains:

```python
["cat", "dog", "runs", "the", "fast"]
```

Now, the sentence:

```text
"The cheetah runs fast"
```

 “cheetah” is not in the vocabulary → it's an OOV word.
 
 A model may:

   - Replace it with `<UNK>`: `"the <UNK> runs fast"`
   - Use subwords: `"the chee ##tah runs fast"` (WordPiece-style)

---

## Source: Pandas Dataframe Agent.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Pandas Dataframe Agent.md`

---

---
aliases:
- null
category: LANG
date modified: 27-09-2025
tags:
- agents
---
Example:
https://github.com/AssemblyAI/youtube-tutorials/tree/main/pandas-dataframe-agent

Follow:

https://www.youtube.com/watch?v=ZIfzpmO8MdA&list=PLcWfeUsAys2kC31F4_ED1JXlkdmu6tlrm&index=7

Can as pandas questions to a dataframe. 

Types of questions:
- what is the max value of "col1"

---

## Source: Part of speech tagging.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Part of speech tagging.md`

---

---
aliases:
- null
category: LANG
date modified: 27-09-2025
tags:
- NLP
---
Part of speech tagging : assigning a specific part-of-speech category (such as noun, verb, adjective, etc.) to each word in a text

Part-of-speech tagging involves assigning a specific part-of-speech category (such as noun, verb, adjective, etc.) to each word in a text
```python
from nltk import pos_tag
pos_tag(temp[:20])
```
will get outputs such as [('history', 'NN'), ('poland', 'NN'), ('roots', 'NNS'), ('early', 'JJ').

---

## Source: Prompt Engineering.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Prompt Engineering.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- language_models
- NLP
---
Prompt engineering is a technique in the field of natural language processing (NLP), particularly when working with [[LLM|large language models]] (LLMs). 

It involves designing and optimizing input [[Prompts]] to get the most relevant and accurate responses from these models. 

Techniques like [[prompt retrievers]], which include systems like UPRISE and DaSLaM, enhance the ability to retrieve and generate contextually appropriate prompts.

Prompt engineering aims to ==guide LLMs toward producing desired outputs while minimizing ambiguity.== 

### Key Takeaways

- Prompt engineering optimizes input to improve LLM responses.
- Techniques like prompt retrievers (e.g., UPRISE, DaSLaM) enhance prompt effectiveness.
- Quality prompts reduce ambiguity and guide model outputs.
- Applications span multiple industries, enhancing user interaction and content generation.

### Key Components Breakdown

Methods: 
  - Prompt Design: Crafting specific, clear prompts to guide model responses.
  - Prompt Retrieval: Utilizing systems like UPRISE and DaSLaM to find effective prompts based on context.
  
Concepts:
  - Contextualization: Understanding the context in which prompts are used to improve relevance.
  - Iterative Testing: Continuously refining prompts based on model performance.

Algorithms:
  - Retrieval-Augmented Generation (RAG): Combines retrieval of relevant documents with generative responses.
  - Few-Shot Learning: Providing examples within prompts to guide model behavior.

### Concerns, Limitations, or Challenges
- Ambiguity: Poorly designed prompts can lead to vague or irrelevant responses.
- Dependence on Training Data: LLMs may produce biased or inaccurate outputs based on their training data.
- Complexity: Designing effective prompts requires a deep understanding of both the model and the task.

### Example
For instance, if a user wants to generate a summary of a scientific article, a poorly constructed prompt like "Summarize this" may yield unsatisfactory results. In contrast, a well-engineered prompt such as "Provide a concise summary of the key findings and implications of the following article on climate change" is likely to produce a more relevant and informative response.

### Follow-Up Questions
1. [[Evaluating the effectiveness of prompts]]
2. [[How can prompt engineering be integrated into existing NLP workflows to enhance performance]]

---

## Source: prompt retrievers.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\prompt retrievers.md`

---

---
aliases:
- null
category: LANG
date modified: 27-09-2025
tags:
- language_models
---

---

## Source: Prompts.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Prompts.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- prompt
---
Pre set prompts are most useful when they are easily accessible.

In obsidian copilot can select prompts with "/".
#### How to write prompts

[Link](https://www.youtube.com/watch?v=jC4v5AS4RIM)

Use a formula to design a prompt.

1) Persona
2) Context
3) Task
4) Format

![[Pasted image 20240910072458.png| 500]]

### Prompts to Ask Better Questions

- What are the underlying assumptions here?
- Is there another way to frame this?
- What’s missing from this picture?
- If this were false, what else would be true?
- What would X say about this (e.g., an expert, a critic, a novice)?
- What’s the smallest next step to test this idea?

### Related
- [[Asking questions]]

---

## Source: Pyright.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Pyright.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- prompt
---
Pyright is a ==static type checker== for Python that enhances code reliability by enforcing type constraints ==at compile-time.==

It utilizes type hints to identify potential errors, such as type mismatches, before runtime, thereby improving code robustness. 

Pyright significantly reduces runtime errors by enforcing type constraints at compile-time.

The use of type hints in Pyright improves code readability and [[Maintainability]], serving as [[Documentation & Meetings]] for function signatures.

### Related Topics

- Type inference in programming languages
- The role of type systems in [[functional programming]]
- [[Debugging]]
- [[Maintainable Code]]
- [[type checking]]

### Follow up questions

- How does the inclusion of Pyright impact the performance of large-scale Python applications?
- What are the trade-offs between using Pyright and other static type checkers in terms of accuracy and speed?

---

## Source: RAG.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\RAG.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
resources: https://www.youtube.com/watch?v=T-D1OfcDW1M
tags:
- language_models
---
Rag is a framework the help [[LLM]] be more up to date.

RAG grounds the Gen AI in external data.

Given a question sometimes the answer given is wrong, issue with [[LLM]] is no source of data and is out of date. RAG is a specific architecture used in natural language processing ([[NLP]]), where a retrieval mechanism is combined with a generative model ([[Generative]]) (often a [[Transformer]] like GPT). RAG systems are designed to ==enhance the ability of a generative model to answer questions or generate content by incorporating factual knowledge retrieved from external data sources== (such as documents, databases, or knowledge repositories). RAG is the connection of [[LLM]]'s with external databases. 

 Example of a RAG System:
 - A user asks: *"What is the capital of France?"*
 - The retrieval module fetches a relevant document (e.g., from Wikipedia) that contains the information about France’s capital.
 - The generation module synthesizes the response: "The capital of France is Paris."

### [[LLM]] Challenges
- Responses are sometimes no sources and out of date.
- LLM's are trained on some store of data (static). We want this store to be updated.
### Key characteristics of RAG:

![[Pasted image 20240928194559.png|500]]

Based on a [[Prompts]].

1. Retrieval Component:
   - This module fetches relevant documents (and up to date) or information from an external corpus based on the query or input. It may use traditional search methods like dense vector retrieval (e.g., using embeddings) or keyword-based retrieval.
   - Retriever should be good enough to give the most truthful information based on the store
   
1. Generative Component:
   - After retrieving relevant documents, the [[Generative]] model (such as GPT or [[BERT]]-based models) synthesizes the final response, integrating both the input query and the retrieved information to generate more accurate and contextually informed outputs.
   
1. Augmentation with External Knowledge:
   - Instead of solely relying on pre-trained internal knowledge (as in traditional language models), RAG setups use the external knowledge source for augmenting generation, thus improving factual accuracy and reducing the risk of hallucinations (incorrect or fabricated responses).

Model should be able to saay "I dont know" instead of [[hallucinating]]

### Resources

Problems
1. LLMs struggle with memorization > "LLMs may struggle with
tasks that require domain-specific expertise or up-to-date
information.
2. LLMs struggle with generating factually inaccurate content
(hallucinations)
Solution
3. A lightweight [[retriever]] (SM) to extract relevant document
fragments from external knowledge bases, document collections,
or other tools

Types of retrievers: 
 Different RaG techniques:
 Sparse retrievers, 
 BM25, 
 dense retrievers.
Use Bert for similarity matching:

---

## Source: Scaling Agentic Systems.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Scaling Agentic Systems.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- GenAI
- language_models
---
[[Agentic Solutions]] propose an improvement over traditional Large Language Model ([[LLM]]) usage by employing networks of Small Language Models (SLMs). These systems aim to strike a balance between scalability, control, and performance, addressing specific tasks with precision while maintaining overall system adaptability.

Ideas from MLOPs talk by MaltedAI.

Agentic solutions represent a pragmatic approach to AI systems by focusing on modularity, task-specific efficiency, and the thoughtful integration of human expertise. These architectures show promise for enhancing scalability, control, and adaptability in real-world applications.
## Contrasting SLMs and LLMs

[[Small Language Models|SLM]] (Small Language Models):
    - Intent-based conversations and decision trees.
    - Controlled systems, harder to build features but easier to execute.
    - Task-specific and efficient in offline environments.

LLMs (Large Language Models):
    - Flexible and natural in handling diverse queries.
    - Suitable for general-purpose scenarios and exploratory tasks.
    - High computational and scaling costs.

### Combined Approach:

- Use [[Small Language Models|SLM]] for inference and LLMs for training.
- Shift the focus from making models larger to solving real-world problems effectively.
## Key Concepts in Agentic Solutions

1. Neural Decision Parser:
    - Acts as the "brain" of the system, determining the appropriate action given user input.
    - SLMs interpret user utterances to express code aligned with system intent.

1. Phased Policy:
    - Distinguishes between contextual and general-purpose questions.
    - Ensures deliberate task execution in stages for clarity and efficiency.

1. Knowledge Graphs and Interaction Models:
    - Complex graph structures enable intelligent conversations between models.
    - RAG setups leverage teacher-student frameworks for effective task distribution.

1. [[Distillation]] Networks of SLMs:
    - SMEs (Subject Matter Experts) guide teacher models that distill their expertise into student SLMs.
    - Enhances task scalability while controlling costs.

1. Scaling with Distillation:
    - Leverage teacher-student frameworks for high-quality, scalable data.
    - Allow teacher models to handle hard-to-scale aspects.

1. Knowledge Discovery:
    - Extract SME knowledge effectively while filtering irrelevant data.
    - Build high-quality datasets for task-specific applications.

## Applications of SLM Networks

1. Task-Specific Systems:
    - Offline processing, task search, and targeted QA.
    - Optimized embedding spaces for domain-specific applications.

1. Swarm Intelligence:
    - Decision-making through deliberation among SLMs.
    - Aggregates diverse inputs (HR, tech, CEO) for robust conclusions.

1. Business Process Models:
    
    - Search and page ranking systems.
    - Smaller, focused systems tailored to specific business needs.



## Designing Agentic Solutions

1. Role of SMEs:
    
    - Define tasks and input structures.
    - Guide model development with domain knowledge.
2. Data Preparation:
    
    - Comprehensive sampling of the problem space ensures generalization.
    - Data variability is critical for robust models.
3. Evaluation and Responsiveness:
    
    - Measure system performance to enable continuous improvement.
    - Focus on responsive, real-time processing.
4. Tool Integration:
    
    - Use LLMs with Python engines or computational tools like Wolfram for data analysis and complex interactions.



## Advantages of SLM Networks

- Precision: Models perform only what they are designed for.
- Efficiency: Smaller models are scalable and cost-effective.
- Focused Applications: Avoids the complexity of embedding spaces for entire businesses.


## Future Directions

---

## Source: Self attention vs multi-head attention.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Self attention vs multi-head attention.md`

---

---
aliases:
- null
category: LANG
date modified: 27-09-2025
tags:
- NLP
---
https://www.youtube.com/shorts/Muvjex0nkes

[[Self-Attention]]: Take every word pays attention to every other word to capture context by:
1. take input word vectors,
2. break words into Q,K,V vectors,
3. compute attention matrix
4. generate final word vectors.

[[Multi-head attention]]: perform self attention in parallel.
1. take word vectors,
2. break words into Q,K,V vectors,
	1. Break each Q,K,V vector into the number of heads parts
3. compute attention matrix for each head
4. generate final word vectors for each head
5. Combine back together

These have better understanding of the context.

---

## Source: Self-Attention.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Self-Attention.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- architecture
- devops
- NLP
---
In this mechanism, the model applies attention to itself. This means each word in the input sequence attends to all other words in the sequence, including itself. Self-attention is used in models like [[Transformer]] to capture dependencies within a sentence.

[[Self-Attention]]
* Each token in a sequence considers all others when computing its representation.
* This enables rich, context-aware embeddings, even for long inputs.
* Unlike [[Recurrent Neural Networks]], Transformers allow parallel processing, making them more efficient and scalable.

Self-attention is the core of models like [[BERT]]& [[GPT]]

---

## Source: Semantic Relationships.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Semantic Relationships.md`

---

---
aliases:
- semantic similarity
category: LANG
date modified: 27-09-2025
tags:
- language_models
- NLP
---
Semantic relationships (Semantic Similarity) refer to the connections and associations between words and concepts based on their meanings. 

Understanding these relationships can enhance various natural language processing tasks, such as information retrieval, text analysis, and sentiment analysis.

### Lexical Resources like [[WordNet]]

One of the key resources for exploring semantic relationships is [[WordNet]]

You can use WordNet to find synonyms or related concepts for important words (those with high [[TF-IDF]] scores) in your documents. If different documents contain synonyms or words related in the WordNet hierarchy, this may indicate a semantic relationship between them, even if the exact words differ.

WordNet also provides measures of semantic similarity between **synsets** based on their paths in the hypernym hierarchy. These measures can be explored to quantify the semantic relatedness of key terms in your documents. The Natural Language Toolkit ([[nltk]]) offers an interface to access WordNet.

### Sentiment Analysis with SentiWordNet

Another resource is SentiWordNet, which extends WordNet by **assigning sentiment scores** (positive, negative, objective) to different senses of words. While your primary goal may be to explore semantic relationships, analyzing the sentiment expressed in your documents based on important words can provide an additional layer of understanding. 

Documents discussing similar topics might also share similar sentiments, strengthening the case for a semantic link. NLTK provides access to SentiWordNet, allowing you to incorporate sentiment analysis into your exploration of semantic relationships.

### Related
- [[Sentence Similarity]]
- [[Smart Connections]]
-

---

## Source: Semantic search.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Semantic search.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- GenAI
---
[[Semantic Relationships]]

---

## Source: Sentence Similarity.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Sentence Similarity.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- NLP
---
Sentence similarity refers to the degree to which two sentences are alike in meaning. It is a crucial concept in natural language processing ([[NLP]]) tasks such as information retrieval, text summarization, and paraphrase detection. Measuring sentence similarity involves comparing the semantic content of sentences to determine how closely they relate to each other.

There are several methods to measure sentence similarity:

1. **Lexical Similarity**: This involves comparing the words in the sentences directly. Common techniques include:
   - **Jaccard Similarity**: Measures the overlap of words between two sentences.
   - **[[Cosine Similarity]]**: Represents sentences as vectors (e.g., using [[TF-IDF]]) and measures the cosine of the angle between them.

2. **Syntactic Similarity**: This considers the structure of the sentences, using techniques like:
   - **Parse Trees**: Comparing the syntactic trees of sentences to see how similar their structures are.

3. **Semantic Similarity**: This goes beyond surface-level word matching to understand the meaning of sentences:
   - **Word Embeddings** ([[Vector Embedding]]): Using models like [[Word2vec]], GloVe, or FastText to represent words in a continuous vector space, then averaging these vectors to represent sentences.
   - **Sentence Embeddings**: Using models like Universal Sentence Encoder, BERT, or Sentence-[[BERT]] to directly obtain embeddings for entire sentences, which can then be compared using [[Cosine Similarity]] or other distance metrics.

4. **Neural Network Models**: Advanced models like BERT, RoBERTa, or GPT can be fine-tuned on specific tasks to directly predict similarity scores between sentence pairs.



Each method has its strengths and weaknesses, and the choice of method often depends on the specific requirements of the task and the available computational resources.

---

## Source: Sentence Transformer Workflow.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Sentence Transformer Workflow.md`

---

---
aliases:
- null
category: LANG
date modified: 27-09-2025
tags:
- NLP
- process
---
### Sentence Transformer Workflow 

#### Step 1: Input Sentence Pair

* Input consists of two sentences: A and B.
* Both are processed independently using the same [[BERT]] model (a twin/siamese network).

#### Step 2: [[Vector Embedding|Embedding]] Extraction

* Sentences A and B are passed separately through BERT.
* Each yields a fixed-size embedding vector: $a = \text{Embed}(A), b = \text{Embed}(B)$.

#### Step 3: Compute Difference and Combine

* Compute absolute difference: |a - b|.
* Form a combined vector: $\[a; b; |a - b|].

#### Step 4: Feedforward Neural Network ([[Feed Forward Neural Network|FFNN]])

* Pass the combined vector through a two-layer FFNN.
* Output is a set of raw logits (real-valued scores for each class).

#### Step 5: [[Classification]] via Softmax

* Apply softmax to logits to get class probabilities.
* The class with the highest probability is selected.

---

## Source: Similarity Search.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Similarity Search.md`

---

---
aliases:
- null
category: LANG
date modified: 27-09-2025
tags:
- language_models
- NLP
---
Given an input text, [[Search]] a [[Vector Embedding]] space to related text.

**Similarity search** retrieves items that are **close in meaning, content, or structure** to a given [[Querying|query]], typically using a **vector space model**. It is a foundation of modern search, recommendation, and information retrieval systems.

Query: "How to schedule a task?"
Match: "Creating [[Cron jobs]] in Linux"
Similar in meaning, found via vector similarity

---

## Source: Small Language Models.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Small Language Models.md`

---

---
aliases:
- SLM
category: LANG
date modified: 27-09-2025
resources: https://arxiv.org/pdf/2409.06857
tags:
- language_models
- NLP
---
[[LLM|LLMs]] dominate many general-purpose NLP tasks, small [[Language Models]] have their own place in specialized tasks, where they excel due to computational efficiency, [[Interpretability]], and task-specific fine-tuning. 

SLMs remain highly relevant for [[Edge ML]] and edge computing, ==domain-specific tasks==, and applications requiring [[Interpretability]], making them a crucial tool in the NLP landscape.

### Use Cases for Small Language Models (SLMs)

- [[Contrastive Decoding]]: Improve the quality of generated content by filtering out low-quality outputs, by having a SLM guide and critique a LLM or other way ([[inference]])
	- Mitigate hallucinations
	- Augmented Reasoning
	  
- [[Distillation]]: Transfer the knowledge from a larger model to a smaller one, retaining performance but reducing computational requirements (see [[BERT]] Teacher model).
  
- Data Synthesis: Generate or augment datasets in scenarios with limited data.
  
- [[Model Cascading]]: Use a combination of smaller models and larger models in a cascading architecture, where simpler tasks are handled by SLMs and more complex ones by LLMs. Model cascading and routing allow SMs to handle simpler tasks, reducing computational overhead. Or the other way first do a general search with a LLM then refine to domain specific small model which is more [[Interpretability|interpretable]] and specific.
  
- Domain specific & Limited Data Availability: SMs, however, can be ==effectively fine-tuned== on smaller, ==domain-specific datasets== and outperform general LLMs in tasks with limited data availability.
  
- [[RAG]] (Retrieval Augmented Generation): Lightweight ==retrievers== (SMs) can support LLMs in finding relevant external information.

### Advantages of SLMs

- Require less computational power and are faster in [[inference]] compared to LLMs.
- [[Interpretability]]
- Accessible for those without resources in power and data

---

## Source: spaCy.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\spaCy.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- NLP
- python
---
#### What is spaCy and how is it best used within Python? 

spaCy is a fast, production-ready [[NLP]] library in Python, commonly used for tasks such as:

 - Tokenization
 - [[Named Entity Recognition]] (NER)
 - Part-of-speech tagging
 - Dependency parsing
 - Sentence segmentation
 - [[lemmatization]]

It is designed to work efficiently on large volumes of text and offers:

 - Pretrained pipelines (e.g., `en_core_web_sm`, `en_core_web_trf`)
 - Seamless integration with deep learning frameworks (e.g., [[PyTorch]], Transformers via `spacy-transformers`)

Best practices for using spaCy:

 - Process documents as streams (e.g., use `nlp.pipe` with generators)
 - Avoid processing documents one-by-one unless debugging
 - Use spaCy's `DocBin` for serialized storage of processed data
 - Combine with custom pipelines (e.g., text cleaning → spaCy → downstream classification)

Example:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
texts = ["This is sentence one.", "Here is another sentence."]

# Efficient [[batch processing]] using nlp.pipe
for doc in nlp.pipe(texts):
    print([ent.text for ent in doc.ents])
```

Exploratory Questions:

 - How does spaCy performance scale with document size and number of texts?
 - How can we integrate spaCy with a [[Data Pipeline]] (e.g., stream from disk/database)?
 - What are use cases where rule-based patterns outperform pretrained models?

#### Using Generators with spaCy

spaCy is designed to work well with generators—especially in `nlp.pipe`, which supports any iterable:

```python
def text_stream(file_path):
    with open(file_path, "r") as f:
        for line in f:
            yield line.strip()

texts = text_stream("documents.txt")
for doc in nlp.pipe(texts, batch_size=32):
    yield doc  # or process and write to file
```

This approach is significantly more efficient than looping over `nlp(text)` one-by-one.

Exploratory Questions:
 - What are good strategies to pair `nlp.pipe()` with result-saving (e.g., JSONL or database)?
 - How do you monitor progress over long-running generators?
 - How do `batch_size` and `n_process` affect spaCy throughput?

---

## Source: Stemming.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Stemming.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- NLP
---
Shorting words to the key term.

---

## Source: stopwords.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\stopwords.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
  - NLP
---

---

## Source: Summarisation.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Summarisation.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- NLP
---
## Summarization in NLP

Summarization in natural language processing (NLP) is the process of condensing a text document into a shorter version while retaining its main ideas and key information. There are two primary forms of summarization:

The unsupervised summarization process involves ==splitting text, tokenizing sentences, assigning scores based on importance, and selecting top sentences==. Effective scoring methods include calculating sentence ==similarity== and analyzing ==word frequencies== to ensure that the summary captures the essence of the original text.

[[Extraction]]:
- This method involves selecting specific words or sentences directly from the original text to create a summary. It focuses on identifying and pulling out the most important parts of the text without altering the original wording.
[[Abstraction]]:
- The abstraction method generates a summary that may include new words and phrases not present in the original text. This approach is more complex as it requires understanding the content and rephrasing it, often using techniques like paraphrasing.

### Unsupervised Summarization Process

The basic idea behind unsupervised summarization involves the following steps:

1. **Split Text into Sentences**: The text is divided into individual sentences for analysis.
  
2. **Tokenize Sentences**: Each sentence is tokenized into separate words, allowing for detailed examination of word usage.

3. **Assign Scores to Sentences**: Sentences are evaluated based on their importance, which is a crucial step in the summarization process.

4. **Select Top Sentences**: The highest-scoring sentences are selected and displayed in their original order to form the summary.

### Methods for Assigning Scores

The main point of summarization is effectively assigning scores to sentences. Here are some common methods for doing this:

- **==Similarity== Calculation**: Calculate the similarity between each pair of sentences and select those that are most similar to the majority of sentences. This helps identify sentences that capture the central themes of the text.

- **Word Frequencies**: Analyze word frequencies to identify the most common words in the text. Sentences that contain a higher number of these frequent words are then selected for the summary.

---

## Source: syntactic relationships.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\syntactic relationships.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- language_models
---
Syntactic relationships refer to the structural connections between words or phrases in a sentence, focusing on grammar and the arrangement of words. They determine how words combine to form phrases, clauses, and sentences, following the rules of syntax.

[[Semantic Relationships]], on the other hand, deal with the meaning and interpretation of words and phrases. They focus on how words relate to each other in terms of meaning, such as synonyms, antonyms, and hierarchical relationships like hypernyms and hyponyms.

The key difference is that syntactic relationships are concerned with the form and structure of language, while semantic relationships are concerned with meaning and interpretation.

---

## Source: Text2Cypher.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Text2Cypher.md`

---

---
aliases:
- null
category: LANG
date modified: 27-09-2025
tags:
- NLP
- querying
---
Text2Cypher is a concept that allows users to convert natural language queries into Cypher queries, which are used to interact with [[GraphRAG|graph database]] like [[neo4j]]. This functionality enables users to ask questions in a more intuitive/[[Interpretability|interpretable]], conversational manner, rather than needing to know the specific syntax of [[Cypher]].

Allows the user to ask vague questions.
Allows for multihop queries on the graph

Overall, Text2Cypher aims to simplify the interaction with graph databases, making it accessible to users who may not be familiar with query languages.
### Key Features of Text2Cypher:

1. **Natural Language Processing**: It utilizes natural language processing (NLP) techniques to understand user queries and translate them into structured Cypher queries.

2. **Flexibility**: Users can ask vague or complex questions that may not directly relate to the underlying data structure, making it easier to retrieve information from a graph database.

3. **Traversal Queries**: Text2Cypher can generate traversal queries that navigate through the graph, allowing for multi-hop queries that explore relationships between entities.

4. **Explainability**: By converting natural language into Cypher, it helps provide a clearer understanding of how the data is structured and how the queries are executed, enhancing interpretability.

---

## Source: TF-IDF Implementation.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\TF-IDF Implementation.md`

---

---
aliases:
- null
category: LANG
date modified: 27-09-2025
tags:
- NLP
---
## TF-IDF Implementation 

### Python Script (scikit-learn version)

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Step 1: Tokenize and vectorize using Bag of Words
bow = CountVectorizer(tokenizer=normalize_document)
X_counts = bow.fit_transform(corpus)

# Step 2: Apply TF-IDF transformation
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)

# Optional: View TF-IDF scores per document
for doc_id in range(len(corpus)):
    print(f"Document {doc_id}: {corpus[doc_id]}")
    print("TF-IDF values:")
    tfidf_vector = X_tfidf[doc_id].T.toarray()
    for term, score in zip(bow.get_feature_names_out(), tfidf_vector):
        if score > 0:
            print(f"{term.rjust(10)} : {score[0]:.4f}")
```

### Python Script (custom TF-IDF implementation)

```python
import math
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.util import bigrams, trigrams

stop_words = stopwords.words('english')
tokenizer = RegexpTokenizer(r'\w+')

def tokenize(text):
    tokens = tokenizer.tokenize(text.lower())
    tokens = [t for t in tokens if len(t) > 2 and t not in stop_words]
    return tokens + [' '.join(b) for b in bigrams(tokens)] + [' '.join(t) for t in trigrams(tokens)]

def tf(term, doc_tokens):
    return doc_tokens.count(term) / len(doc_tokens)

def idf(term, docs_tokens):
    doc_count = sum(1 for doc in docs_tokens if term in doc)
    return math.log(len(docs_tokens) / (1 + doc_count))

def compute_tfidf(docs):
    docs_tokens = [tokenize(doc) for doc in docs]
    all_terms = set(term for doc in docs_tokens for term in doc)
    tfidf_scores = []
    for tokens in docs_tokens:
        tfidf = {}
        for term in all_terms:
            if term in tokens:
                tfidf[term] = tf(term, tokens) * idf(term, docs_tokens)
        tfidf_scores.append(tfidf)
    return tfidf_scores
```

---

## Source: TF-IDF.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\TF-IDF.md`

---

---
aliases:
- TFIDF
category: LANG
date modified: 27-09-2025
tags:
- code_snippet
- NLP
- preprocessing
---
TF-IDF is a statistical technique used in text analysis to determine the importance of a word in a document relative to a collection of documents (corpus). It balances two ideas:

- Term Frequency (TF): Captures how often a term occurs in a document.
- Inverse Document Frequency (IDF): Discounts terms that appear in many documents.

High TF-IDF scores indicate terms that are frequent in a document but rare in the corpus, making them useful for distinguishing between documents in tasks such as information retrieval, document classification, and recommendation.

TF-IDF combines local and global term [[Statistics]]:
- TF gives high scores to frequent terms in a document
- IDF reduces the weight of common terms across documents
- TF-IDF identifies terms that are both frequent and distinctive

Can be used to give an initial snapshot of a notes themes and topic.
### Equations

#### Term Frequency

$TF(t, d)$ measures how often a term $t$ appears in a document $d$, normalized by the total number of terms in $d$:

$$
TF(t, d) = \frac{f_{t,d}}{\sum_k f_{k,d}}
$$

Where:
- $f_{t,d}$ is the raw count of term $t$ in document $d$  
- $\sum_k f_{k,d}$ is the total number of terms in $d$ (i.e. the document length)

#### Inverse Document Frequency

IDF assigns lower weights to frequent terms:

$$
IDF(t, D) = \log \left( \frac{N}{1 + |\{d \in D : t \in d\}|} \right)
$$

Where:
- $N$ is the number of documents in the corpus $D$  
- $|\{d \in D : t \in d\}|$ is the number of documents containing term $t$  
- Adding 1 to the denominator avoids division by zero

#### TF-IDF Score

The final score is:

$$
TF\text{-}IDF(t, d, D) = TF(t, d) \times IDF(t, D)
$$

### Related Notes

- [[Bag of Words]]
- [[Tokenisation]]
- [[Clustering]]
- [[Search]]
- [[Recommender systems]]
- [[nltk]]
- [[TF-IDF Implementation]] <-

### Exploratory Ideas
- Can track TF-IDF over time (e.g., note evolution)
- Can cluster or classify the documents using TF-IDF?

---

## Source: Tokenisation.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Tokenisation.md`

---

---
aliases:
- tokenization
category: LANG
date modified: 27-09-2025
tags:
- code_snippet
- NLP
- preprocessing
---
Tokenisation is a core step in natural language processing ([[NLP]]) where raw text is split into smaller units known as tokens. These tokens can be words, sentences, or subwords, depending on the level of analysis. Tokenisation prepares text for downstream tasks like embedding, classification, or parsing.

There are different kinds: [[spaCy]] or [[Hugging Face]] tokenisers.

### Types of Tokenisation

#### 1. Word Tokenisation

Splits text into individual words and retains punctuation.

```python
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text_original)
print(tokens)
```

#### 2. Sentence Tokenisation

Segments text into full sentences.

```python
from nltk.tokenize import sent_tokenize
sentences = sent_tokenize(text_original)
print(sentences)
```

#### 3. Custom Tokenisation with Cleaning

For cleaner preprocessing (e.g. lowercasing, removing non-alphanumerics and stopwords):

```python
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

temp = text_original.lower()
temp = re.sub(r"[^a-zA-Z0-9]", " ", temp)      # keep only letters and numbers
temp = re.sub(r"[[0-9]+\]", "", temp)         # remove bracketed numbers
tokens = word_tokenize(temp)
tokens_clean = [t for t in tokens if t not in stopwords.words("english")]
print(tokens_clean)
```

### Special Tokens in [[Transformer]] Models

Modern models like [[BERT]] use special tokens during tokenisation to preserve input structure:
* [CLS]: Marks the start of the input sequence; used for classification tasks.
* [SEP]: Separates different segments (e.g. question from context in QA tasks).

### Related:
- [[nltk]]

---

## Source: topic modeling.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\topic modeling.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- language_models
- NLP
---

---

## Source: Vectorisation.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Vectorisation.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- code_snippet
- software
---
### Vectorisation in [[Python]]

Vectorisation refers to the practice of replacing explicit loops with array operations, typically using libraries like [[Numpy]]. This leads to faster and more efficient code execution.

#### Why is NumPy vectorisation faster than a `for` loop?

* NumPy operations like `np.dot()` are implemented in compiled C and optimised for parallel execution.
* They utilise SIMD (Single Instruction, Multiple Data) and can leverage multi-threading and GPU acceleration (with appropriate backends).
* In contrast, `for` loops in Python are interpreted sequentially, adding overhead and limiting performance.
#### Example: Dot Product

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Vectorised
np.dot(a, b)

# Manual loop
sum([x*y for x, y in zip(a, b)])
```

> Vectorised code runs simultaneously across elements, whereas loops run sequentially.
### Resources

Related:
- [[Numpy]]
- [[Pandas]]

[Link](https://www.youtube.com/watch?v=uvTL1N02f04&list=PLkDaE6sCZn6FNC6YRfRQc_FbeQrF8BwGI&index=24)

![[Pasted image 20241217204829.png|500]]

---

## Source: Why is named entity recognition (NER) a challenging task.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Why is named entity recognition (NER) a challenging task.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- language_models
---
Named Entity Recognition (NER) is considered a challenging task for several reasons:

1. **Ambiguity**: Entities can be ambiguous, meaning the same word or phrase can refer to different entities depending on the context. For example, "Washington" could refer to a city, a state, or a person. Disambiguating these entities requires a deep understanding of context.

2. **Variability in Language**: Natural language is highly variable and can include slang, idioms, and different syntactic structures. This variability makes it difficult for NER models to consistently identify entities across different texts.

3. **Named Entity Diversity**: Entities can take many forms, including names, organizations, locations, dates, and more. Each type may have different characteristics, requiring the model to adapt to various patterns.

4. **Lack of Annotated Data**: High-quality annotated datasets are crucial for training NER models. However, creating such datasets can be time-consuming and expensive, leading to limited training data for certain domains or languages.

5. **Multilingual Challenges**: NER systems often struggle with multilingual texts, where the same entity may be represented differently in different languages. This adds complexity to the recognition process.

6. **Nested Entities**: In some cases, entities can be nested within each other (e.g., "The University of California, Berkeley"). Recognizing such nested structures can be particularly challenging for NER systems.

7. **Domain-Specific Language**: Different domains (e.g., medical, legal, technical) may have specific terminologies and entities that general NER models may not recognize effectively without domain-specific training.

---

## Source: Word2vec.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\Word2vec.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
  - NLP
  - python
  - ML_Tools
---
Word2Vec is a technique for generating vector representations of words. Developed by researchers at Google, it uses a shallow [[Neural network]] to produce [[Vector Embedding|word embedding]] that capture [[Semantic Relationships]] and [[syntactic relationships]]. Word2Vec has two main architectures:

1. CBOW (Continuous [[Bag of Words]]):
    - Predicts a target word given its context (neighboring words).
    - Efficient for smaller datasets.
      
2. Skip-Gram:
    - Predicts the context words given a target word.
    - Performs better on larger datasets.

Word2Vec generates dense, continuous vector representations where words with similar meanings are close to each other in the embedding space. For example:

- `vector("king") - vector("man") + vector("woman") ≈ vector("queen")`

Uses Negative [[Resampling]].

---

## Source: WordNet.md

_Path_: `C:\Users\RhysL\Desktop\Data-Archive\content\categories\natural-language\WordNet.md`

---

---
aliases: []
category: LANG
date modified: 27-09-2025
tags:
- NLP
---
WordNet, a lexical database that groups words into sets of **cognitive synonym**s called synsets. These synsets are linked together in a hierarchy based on semantic relations, including:

- Hypernymy: Represents an "is-a" relationship (e.g., "dog" is a hypernym of "beagle").
- Hyponymy: Represents a more specific type (e.g., "beagle" is a hyponym of "dog").

---

