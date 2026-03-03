# Design Report: Financial RAG System

This report outlines the design choices and architectural decisions made for the SEC 10-K Retrieval-Augmented Generation (RAG) system.

## 1. Chunking Strategy
The system employs a **Sliding Window Chunking** approach combined with **Structural Metadata Tracking**:
- **Window Size**: 650 words per chunk.
- **Overlap**: 200 words to ensure semantic continuity across boundaries.
- **Metadata Preservation**: Each chunk is tagged with its source document name, the specific SEC "Item" (e.g., Item 1A - Risk Factors), and the page number. This allows for the high-precision sourcing required by the assignment.
- **Table Handling**: Tables are extracted using PyMuPDF’s `find_tables` and converted into Markdown format. These are treated as independent chunks to prevent the LLM from losing structural alignment during processing.

## 2. LLM Choice
The system utilizes **Qwen2.5-3B-Instruct** for the following reasons:
- **Local Execution**: It is an open-access model that runs efficiently on consumer-grade hardware (3B parameters) without requiring external APIs (GPT-4/Claude).
- **Instruction Following**: It demonstrates superior performance in following strict output formats (JSON/Source Arrays) compared to other models in its size class.
- **Context Window**: It comfortably handles the 7 reranked chunks provided in the prompt while maintaining reasoning capabilities.

## 3. Retrieval Pipeline
A **Hybrid Retrieval + Reranking** architecture was chosen to maximize accuracy:
- **Phase 1 (Dense)**: FAISS (IndexFlatIP) with `BAAI/bge-m3` embeddings captures semantic meaning.
- **Phase 2 (Sparse)**: BM25Okapi captures exact keyword matches (e.g., specific dollar amounts or dates like "September 28, 2024").
- **Phase 3 (Reranking)**: A Cross-Encoder (`BAAI/bge-reranker-v2-m3`) re-scores the top 25 candidates to select the final 7. This drastically reduces "hallucinations" by ensuring the LLM only sees the most relevant context.

## 4. Out-of-Scope Handling
To prevent the model from generating speculative or external knowledge, the system uses **Strict Prompt Constraints**:
- **Explicit Fallbacks**: The system message defines two mandatory response strings:
  1. *"Not specified in the document."* (For missing internal data).
  2. *"This question cannot be answered based on the provided documents."* (For future predictions or subjective questions like stock forecasts).
- **Temperature Setting**: The LLM is set to a temperature of 0.1 to ensure deterministic, fact-based responses rather than creative ones.

---
