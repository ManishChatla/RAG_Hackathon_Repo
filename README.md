# 🚀 RAG Hackathon Project

This project implements an **Advanced Retrieval-Augmented Generation (RAG)** pipeline powered by **Gemini LLM** and **ChromaDB**. It supports both CLI-based execution and an interactive Streamlit UI for experimentation, visualization, and evaluation.

---

## 🧩 Project Overview

This repository demonstrates an **end-to-end RAG framework** with modular components, including:

- **Preprocessing & Chunking**
- **Embedding Generation (Gemini)**
- **Hybrid Retrieval (TF-IDF, BM25, ChromaDB)**
- **Reranking using LLM, Cohere, and CrossEncoder**
- **Confidence and Latency Evaluation**
- **Interactive Streamlit Interface**

---

## 📂 Folder Structure

```
RAG_Hackathon_Repo/
│
├── code/
│   ├── src/
│   │   ├── data/
│   │   │   ├── input_data.json         # Input corpus for processing
│   │   │   ├── chroma_db/              # ChromaDB vector store
│   │   ├── main.py                     # CLI-based execution entry point
│   │   ├── streamline_app.py           # Streamlit UI for interactive testing
│   │   ├── preprocess.py               # Text loading, cleaning, and chunking logic
│   │   ├── embedder.py                 # Gemini embedding generation
│   │   ├── vector_store.py             # ChromaDB integration for storage & retrieval
│   │   ├── reranker.py                 # Reranking methods (LLM, Cohere, CrossEncoder)
│   │   ├── config.py                   # Configuration constants
│   │   └── prompts.py                  # RAG-specific prompt formatting
│   │
│   ├── tests/                          # Unit tests for all modules
│   │   ├── test_chunking.py
│   │   ├── test_embedding.py
│   │   ├── test_vectorstore.py
│   │   ├── test_reranker.py
│   │   └── test_main.py
│   │
│   └── requirements.txt
│
└── README.md
```

---

## ⚙️ Running the Project

### ▶️ Option 1: Command-line Execution
```bash
cd code/src
python main.py
```

### 💻 Option 2: Streamlit UI
```bash
cd code/src
streamlit run streamline_app.py
```

---

## 🧠 Chunking Strategy

The text corpus is segmented into **semantic chunks** using a **hybrid chunking algorithm** that combines:

- **Sentence boundary detection**
- **Token-based windowing (e.g., 512–1024 tokens)**
- **Overlap context preservation (typically 10–20%)**

This ensures both **context continuity** and **retrieval efficiency**.

---

## 🔍 Embedding Generation

We use **Gemini’s embedding model** (`models/embedding-001`) to generate high-dimensional vectors. These vectors are stored in **ChromaDB**, which supports efficient retrieval using **HNSW indexing** (Hierarchical Navigable Small World graphs).

---

## 🧮 Hybrid Retrieval Approach

The retrieval pipeline combines **semantic similarity** and **lexical relevance** using:

- **TF-IDF** – For fast keyword relevance scoring  
- **BM25** – For improved term weighting and ranking  
- **ChromaDB (Vector-based)** – For semantic retrieval using cosine similarity  
- **Query Expansion** – For reformulating user queries to enhance recall

The final retrieval results are **merged and deduplicated** for better coverage.

---

## 🔁 Reranking Techniques

To improve contextual accuracy, retrieved documents are reranked using **three distinct approaches**:

1. **LLM-based Reranker** – Uses Gemini to analyze contextual match with the query.  
2. **Cohere Reranker** – Leverages Cohere’s `rerank-english-v2.0` model for relevance scoring.  
3. **CrossEncoder Reranker** – Uses transformer-based pair scoring (query, document) similarity.

The combined reranking score enhances the precision of the final retrieval set.

---

## 💾 Vector Storage (ChromaDB)

We use **ChromaDB** for efficient similarity search. It provides:

- Persistent **vector store**
- Optimized **HNSW graph indexing**
- Metadata-based **filtering & retrieval**

Each chunk is indexed with metadata like `doc_id`, `chunk_id`, and `source_file`.

---

## 📊 Evaluation Metrics

During retrieval and reranking, the following metrics are calculated:

| Metric | Description |
|--------|--------------|
| **Retrieval Confidence** | Average cosine similarity of top matches |
| **Answer Confidence** | LLM-based estimate of response certainty |
| **Token Count** | Number of tokens used in generation |
| **Retrieval Latency** | Time taken for embedding & retrieval |

---

## ✅ Testing & Coverage

All major modules have Pytest-based unit tests.  
To run tests with coverage:

```bash
pytest --maxfail=1 --disable-warnings -q
pytest --cov=src --cov-report=term-missing
```

---

## 🧠 Summary

This RAG framework brings together **semantic, lexical, and contextual intelligence**. By integrating **Gemini embeddings**, **hybrid retrieval**, and **multi-model reranking**, it achieves a **balanced blend of recall and precision**—making it ideal for enterprise-scale retrieval applications.

---

**Authors:** Manish & Team  
**Hackathon:** Advanced RAG Challenge  
**Tech Stack:** Python, Streamlit, ChromaDB, Gemini, Cohere, HuggingFace Transformers
