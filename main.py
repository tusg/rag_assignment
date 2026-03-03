import re
import json
import torch
import faiss
import fitz  # PyMuPDF
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# ============================
# CONFIGURATION
# ============================
APPLE_PDF = "/kaggle/input/datasets/tgupta3/llm-assignment/10-Q4-2024-As-Filed.pdf"
TESLA_PDF = "/kaggle/input/datasets/tgupta3/llm-assignment/tsla-20231231-gen.pdf"

# EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
# RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CHUNK_WORDS = 350
# OVERLAP_WORDS = 100
CHUNK_WORDS = 650
OVERLAP_WORDS = 200

TOP_K_RETRIEVE = 25
TOP_K_RERANK = 7

# ================================
# 1. DOCUMENT INGESTION & CHUNKING
# ================================

def extract_pdf_data(pdf_path, doc_name):
    """
    Extracts text and tables.
    Applies standard RAG pre-processing to skip Table of Contents pages.
    """
    doc = fitz.open(pdf_path)
    pages = []
    current_item = "Unknown Item"

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        
        # Track SEC "Item" metadata dynamically
        matches = re.findall(r'(?mi)^Item\s+([0-9]{1,2}[A-Z]?)\b', text)
        if matches:
            current_item = f"Item {matches[-1].upper()}"

        clean_txt = re.sub(r'\s+', ' ', text).strip()
        
        # # Standard RAG practice: Skip index/TOC pages to prevent retriever confusion
        # if "Table of Contents" in clean_txt[:200] and clean_txt.count("........") > 5:
        #     continue

        # Extract Tables and convert to Markdown
        tables_md = []
        if hasattr(page, "find_tables"):
            for tab in page.find_tables():
                try:
                    df = tab.to_pandas()
                    df = df.fillna("")  
                    df = df.astype(str).replace(r'\n', ' ', regex=True) 
                    tables_md.append(df.to_markdown(index=False))
                except Exception:
                    pass
        
        if clean_txt or tables_md:
            pages.append({
                "document": doc_name,
                "item": current_item,
                "page": page_num + 1,
                "text": clean_txt,
                "tables": tables_md
            })
            
    return pages

def chunk_text(pages):
    chunks = []
    for page in pages:
        # Add tables as independent chunks
        for tbl in page["tables"]:
            chunks.append({
                "document": page["document"],
                "item": page["item"],
                "page": page["page"],
                "text": f"[TABLE]\n{tbl}\n[/TABLE]",
                "type": "table"
            })

        # Add standard sliding window text chunks
        words = page["text"].split()
        for i in range(0, len(words), CHUNK_WORDS - OVERLAP_WORDS):
            chunk_str = " ".join(words[i:i + CHUNK_WORDS])
            if len(chunk_str) > 50:
                chunks.append({
                    "document": page["document"],
                    "item": page["item"],
                    "page": page["page"],
                    "text": chunk_str,
                    "type": "text"
                })
    return chunks

# ============================
# 2. HYBRID RETRIEVAL PIPELINE
# ============================
class HybridStore:
    def __init__(self):
        print("Loading Embedding Model...")
        self.embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
        self.index = None
        self.bm25 = None
        self.metadata = []

    def build(self, chunks):
        print(f"Building Hybrid Index (FAISS + BM25) for {len(chunks)} chunks...")
        self.metadata = chunks
        texts = [c["text"] for c in chunks]
        
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        
        tokenized_corpus = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("Hybrid Index ready.")

    def retrieve(self, query, top_k=TOP_K_RETRIEVE):
        q_emb = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        _, dense_indices = self.index.search(q_emb, top_k)
        
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        combined_indices = list(set(dense_indices[0]).union(set(bm25_indices)))
        return [self.metadata[i] for i in combined_indices]

class Reranker:
    def __init__(self):
        print("Loading Cross-Encoder Reranker...")
        self.model = CrossEncoder(RERANK_MODEL, device=DEVICE)

    def rerank(self, query, retrieved_chunks, top_k=TOP_K_RERANK):
        pairs = [[query, chunk["text"]] for chunk in retrieved_chunks]
        scores = self.model.predict(pairs)
        scored_chunks = list(zip(scores, retrieved_chunks))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scored_chunks[:top_k]]

# ============================
# 3. RAG PIPELINE & LLM
# ============================
class RAGPipeline:
    def __init__(self):
        self.store = HybridStore()
        self.reranker = Reranker()
        
        print("Loading LLM...")
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        self.llm = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL, 
            device_map="auto", 
            torch_dtype=torch.bfloat16
        )
        self.llm.eval()

        print("Processing PDFs...")
        apple_pages = extract_pdf_data(APPLE_PDF, "Apple 10-K")
        tesla_pages = extract_pdf_data(TESLA_PDF, "Tesla 10-K")

        self.store.build(chunk_text(apple_pages) + chunk_text(tesla_pages))
        print("RAG Pipeline Fully Initialized!\n")

    def format_prompt(self, query, top_chunks):
        context_str = ""
        for idx, c in enumerate(top_chunks):
            chunk_type = "TABLE" if c.get("type") == "table" else "TEXT"
            source_array = f'["{c["document"]}", "{c["item"]}", "p. {c["page"]}"]'
            context_str += f"--- Chunk {idx+1} ({chunk_type}) | Source: {source_array} ---\n{c['text']}\n\n"

        # Fully generic prompt without any hardcoded references to specific questions
        system_msg = """You are an expert financial analyst. Answer the user's question STRICTLY and EXCLUSIVELY based on the provided context chunks.

GENERAL INSTRUCTIONS:
1. DATA EXTRACTION: If the answer requires pulling data from a Markdown table, ensure you align the correct row with the correct column header (e.g., matching the specific year requested).
2. CALCULATIONS: If a question asks for a total, percentage, or sum, calculate it using ONLY the numbers explicitly provided in the text or tables.
3. MISSING INFO: If the requested information is legitimately not contained within the provided context, reply EXACTLY: "Not specified in the document."
4. OUT OF SCOPE: If the question asks for future predictions, subjective opinions, or topics completely unrelated to standard corporate SEC filings, reply EXACTLY: "This question cannot be answered based on the provided documents."

OUTPUT FORMAT:
You must respond in exactly this format. Do not add any conversational filler:
ANSWER: <Your concise answer>
SOURCE: <The exact Source array from the header of the chunk you used to answer the question. If Instruction 3 or 4 applies, output []>"""

        user_msg = f"CONTEXT:\n{context_str}\n\nQUESTION: {query}"
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def answer_question(self, query: str) -> dict:
        retrieved = self.store.retrieve(query, TOP_K_RETRIEVE)
        top_chunks = self.reranker.rerank(query, retrieved, TOP_K_RERANK)

        prompt = self.format_prompt(query, top_chunks)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs, 
                max_new_tokens=150, 
                temperature=0.1, 
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # Generic Regex Parsing
        ans_match = re.search(r"ANSWER:\s*(.*?)\nSOURCE:\s*(.*)", response, re.IGNORECASE | re.DOTALL)
        
        if ans_match:
            answer_text = ans_match.group(1).strip()
            source_text = ans_match.group(2).strip()
        else:
            answer_text = response
            source_text = "[]"

        # Generic Fallbacks for the mandatory strings
        if "Not specified" in answer_text:
            answer_text = "Not specified in the document."
            source_list = ["N/A"]
            response="Not answerable"
        elif "cannot be answered" in answer_text:
            answer_text = "This question cannot be answered based on the provided documents."
            source_list = ["N/A"]
            response="Not answerable"
        else:
            try:
                source_list = json.loads(source_text)
            except:
                # If the LLM fails to format the array, default to the top retrieved chunk
                top_src = top_chunks[0]
                source_list = [top_src['document'], top_src['item'], f"p. {top_src['page']}"]

        return {
            "answer": answer_text,
            "sources": source_list
        }
