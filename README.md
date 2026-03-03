# Financial RAG System: SEC 10-K Analysis

This repository contains a Retrieval-Augmented Generation (RAG) system designed to perform complex financial and legal analysis on SEC filings. Specifically, it processes **Apple’s 2024 10-K** and **Tesla’s 2023 10-K** to provide accurate, well-sourced answers using a local, open-access LLM pipeline.

## 🚀 Features

- **Hybrid Retrieval**: Combines dense vector search (**FAISS**) with sparse keyword search (**BM25**) for robust document fetching.
- **Advanced Reranking**: Utilizes a Cross-Encoder reranker to ensure the most relevant context is passed to the LLM.
- **Table Support**: Automatically extracts PDF tables and converts them to Markdown format to preserve structural data for the LLM.
- **Local LLM Integration**: Powered by `Qwen2.5-3B-Instruct` for private, high-performance inference without external API dependencies.
- **Strict Sourcing**: Every answer includes a source array (Document, Item, Page) to ensure auditability.

## 🛠️ Tech Stack

- **Embeddings**: `BAAI/bge-m3`
- **Reranker**: `BAAI/bge-reranker-v2-m3`
- **LLM**: `Qwen/Qwen2.5-3B-Instruct`
- **Vector DB**: FAISS (IndexFlatIP)
- **PDF Parsing**: PyMuPDF (fitz) + Custom Table Extractor
- **Frameworks**: Transformers, Sentence-Transformers, Rank-BM25

## 📋 Installation

1. **Clone the repository**:
      git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   
2. **Install dependencies**:
      pip install torch transformers sentence-transformers faiss-cpu pymupdf pandas rank_bm25 tabulate
   
3. **Prepare Documents**:
   Ensure the following PDFs are located in your `./Downloads/` directory:
   - `10-Q4-2024-As-Filed.pdf` (Apple)
   - `tsla-20231231-gen.pdf` (Tesla)

## 💻 Usage

Run the main script to initialize the pipeline and execute the evaluation suite:

python main.py

### Required Interface
The system exposes a primary function for querying:

result = pipeline.answer_question("What was Apples total revenue for the fiscal year ended September 28, 2024?")
print(result)
# Output: {"answer": "USD 391,036 million", "sources": ["Apple 10-K", "Item 8", "p. 282"]}

## 🧠 Pipeline Architecture

1. **Ingestion**: PDFs are parsed, and SEC "Items" are tracked dynamically via regex to maintain context.
2. **Chunking**: A sliding window approach (650 words with 200-word overlap) ensures no context is lost at boundaries.
3. **Retrieval**: 
   - **Step A**: Retrieve top 25 candidates using Hybrid Search (FAISS + BM25).
   - **Step B**: Rerank the 25 candidates down to the top 7 most relevant chunks using a Cross-Encoder.
4. **Generation**: The LLM is prompted with a strict system instruction to only use provided context and handle out-of-scope questions with predefined fallback phrases.

## 📄 Documentation

- [Design Report](https://github.com/tusg/rag_assignment/blob/main/design_report.md): Detailed justification for model choices, chunking strategies, and out-of-scope handling.
- [Kaggle Notebook](https://www.kaggle.com/code/tgupta3/llm-rag-assignment): Link to the runnable Kaggle notebook. Please ensure GPU is enabled for faster retrievals

## ⚖️ Evaluation
The system is tested against 13 specific financial questions, including edge cases (e.g., stock price forecasts and non-existent data) to ensure it refuses out-of-scope queries as required by the assignment constraints.
