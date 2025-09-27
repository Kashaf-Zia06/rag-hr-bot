<<<<<<< HEAD
# rag-hr-bot
=======
# RAG-HR (Minimal, Local-First)

A lightweight Retrieval-Augmented Generation (RAG) demo for your HR/Employee Services dataset.

## Features
- Ingests Markdown (policies), CSVs (system/data), and YAML into a unified vector store (FAISS).
- Chunking tailored for policies and tables.
- Simple CLI for Q&A with source citations (filenames + row/section ids).
- Pluggable LLM:
  - **Option A (default)**: OpenAI (set `OPENAI_API_KEY`)
  - **Option B**: Local Ollama (e.g., `llama3.1`), set `USE_OLLAMA=1`.

## Quickstart

1) Create & activate environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2) Put your dataset inside `rag_hr/data/` (or keep path as-is and point to the zip/folder).
   - For the sample you downloaded: unzip `rag_seed_data.zip` and move the `rag_seed_data/` folder here,
     or keep it elsewhere and pass `--data-path` to `ingest.py`.

3) Configure LLM:
- Copy `.env.example` to `.env` and set values.
  - For OpenAI: `OPENAI_API_KEY=...`
  - For Ollama: install Ollama, `USE_OLLAMA=1`, `OLLAMA_MODEL=llama3.1`

4) Build the index
```bash
python rag_hr/ingest.py --data-path ./rag_hr/data/rag_seed_data --index-path ./rag_hr/vectorstore/index.faiss
```

5) Ask questions
```bash
python rag_hr/query.py --index-path ./rag_hr/vectorstore/index.faiss   --question "How many annual leave days do I have and do I need a medical certificate?"
```

## Examples
```bash
python rag_hr/query.py --index-path ./rag_hr/vectorstore/index.faiss   --question "Who approves expenses above PKR 50,000 and what's the SLA?"
```
```bash
python rag_hr/query.py --index-path ./rag_hr/vectorstore/index.faiss   --question "Show the approval chain for a resignation and any notice period rules."
```

## Notes
- This is a minimal educational scaffold. For production, add reranking, caching, evals, auth, and guardrails.
>>>>>>> c01b6306 (initial phase for rag completed)
