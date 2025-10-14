
# 🧠 RAG Hackathon Project — Advanced Retrieval-Augmented Generation Pipeline

## 🚀 Overview
This project demonstrates a **robust, hybrid Retrieval-Augmented Generation (RAG) pipeline** that integrates multiple retrieval and reranking strategies for improved contextual understanding, scalability, and answer confidence.  
It provides both **command-line (main.py)** and **interactive UI (streamline.py)** modes.

---

## 🧩 Architecture Diagram

```mermaid
flowchart TD
    A[Input Query] --> B[Query Expansion (LLM Reformulation)]
    B --> C[Hybrid Retrieval: TF-IDF + BM25 + Vector Search]
    C --> D[Top-K Relevant Chunks]
    D --> E[Reranking Stage]
    E -->|LLM-Based| F[Gemini Relevance Scoring]
    E -->|Cohere Rerank| G[Cohere Cross-Doc Ranking]
    E -->|CrossEncoder| H[Fine-Grained Semantic Scoring]
    F --> I[Fused Rank Aggregation]
    G --> I
    H --> I
    I --> J[Answer Synthesis via Gemini LLM]
    J --> K[Confidence Calculation & Output]
```

---

## 📂 Project Structure

```
code/
├── src/
│   ├── preprocess.py
│   ├── embedder.py
│   ├── vector_store.py
│   ├── main.py
│   ├── streamline.py
│   ├── data/
│   │   ├── chromadb/        # Vector database files
│   │   ├── input.json       # Input KB/FAQ dataset
│   └── __init__.py
├── tests/
│   ├── test_chunking.py
│   ├── test_embedding.py
│   ├── test_vectorstore.py
│   └── test_main.py
├── requirements.txt
└── README.md
```

---

## 🧱 Chunking Strategy

We use a **semantic chunking approach** that balances **context preservation** and **retrieval granularity**.  
Each document is split into chunks of ~400 tokens with **10–20% overlap** to ensure contextual continuity.  
Metadata such as document titles and sections are preserved for meaningful retrieval.  

This helps maintain coherence during retrieval and ensures no context boundaries are lost.

---

## 🔍 Embedding Logic

We employ the **Gemini Embedding API** to transform text chunks into high-dimensional vectors.  
Each embedding captures semantic meaning, enabling **contextual similarity search** instead of keyword-based matching.

Embeddings are batched, cached, and checkpointed to reduce API calls and cost.

---

## 🧠 Hybrid Retrieval

To improve retrieval precision and recall, a **hybrid strategy** combines:
- **TF-IDF** for lexical matching (exact word overlap)
- **BM25** for ranking based on term frequency relevance
- **Vector similarity (ChromaDB)** for semantic proximity

Additionally, **Query Expansion** is performed using an LLM to reformulate user queries semantically — improving recall for unseen or paraphrased questions.

---

## 🔁 Reranking Strategies

After hybrid retrieval, reranking ensures that the most relevant context is prioritized using:
1. **LLM-Based Reranking** — Gemini evaluates the contextual alignment between query and retrieved chunks.
2. **Cohere Rerank API** — Uses a cross-encoder model to re-score documents for relevance.
3. **CrossEncoder Model** — Provides fine-grained semantic comparison using sentence-transformer architecture.

These scores are fused through **weighted rank aggregation** for final ordering before passing context to the answer generation model.

---

## 🧮 Confidence Calculation

Confidence is computed based on:
- Average similarity of top-K retrieved results.
- Re-ranking score normalization.
- Contextual agreement between top chunks.

A composite confidence score is displayed along with token count, retrieval latency, and answer quality metrics.

---

## 💾 Vector Storage (ChromaDB)

We use **ChromaDB** for vector storage and similarity search.  
It provides an efficient **in-memory HNSW index**, supporting metadata, persistence, and scalability.  
It enables fast cosine similarity computation and supports hybrid filtering based on metadata.

---

## ⚙️ Running the Project

### 1️⃣ CLI Mode
Run the RAG pipeline directly:
```bash
python src/main.py
```

### 2️⃣ Streamlit UI Mode
Launch the interactive RAG interface:
```bash
streamlit run src/streamline.py
```

---

## 🧪 Testing & Coverage

All modules (preprocessing, embedding, retrieval, UI) are covered under `tests/`.  
To run full test coverage:
```bash
pytest --cov=src --cov-report=term-missing
```

---

## 🧭 Design Philosophy

A focused retrieval pipeline that is **readable, reproducible, and justified**.  
The design emphasizes:

- 📘 **Understanding of RAG** — Retrieval impacts answer quality.
- 🧪 **Experimental Rigor** — Each strategy measured for latency, accuracy, and cost.
- 🧰 **Engineering Discipline** — Logging, testing, and modular architecture.
- ⚡ **Efficiency** — Caching, batching, and smart API usage.
- 🧾 **Real-World Design** — Handles API failures, data missing cases, and optimizes for cost.

---

## 🏁 Summary

Our approach integrates hybrid retrieval, semantic reranking, and confidence-based evaluation to deliver reliable, scalable, and cost-efficient RAG pipelines for enterprise use.

