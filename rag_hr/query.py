import os, argparse, pickle, faiss, numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from tabulate import tabulate
import json
import textwrap

def call_llm(prompt: str):
    # Select LLM backend using env flags
    import os, json
    from dotenv import load_dotenv
    load_dotenv()
    if os.getenv("USE_GROQ", "0") == "1":
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            return "[LLM ERROR] Please set a valid GROQ_API_KEY in your .env file or disable USE_GROQ."
        try:
            client = Groq(api_key=api_key)
            model = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You answer using ONLY the provided context. If unknown, say you don't know."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"[LLM ERROR] API call failed: {str(e)}. Please check your API key and try again."
    
    # Fallback response when no LLM is configured
    return "I can help you with HR questions, but I need a valid API key to provide detailed answers. Please configure your GROQ_API_KEY in the .env file."

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-path", required=True)
    ap.add_argument("--meta-path", default=None)
    ap.add_argument("--question", required=True)
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    # Load index + metadata
    index = faiss.read_index(args.index_path)
    meta_path = args.meta_path or (args.index_path + ".meta.pkl")
    with open(meta_path, "rb") as f:
        docs = pickle.load(f)  # list of {text, meta}
    texts = [d["text"] for d in docs]

    # Embed query
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    qemb = model.encode([args.question], convert_to_numpy=True)
    faiss.normalize_L2(qemb)
    D, I = index.search(qemb, args.k)
    I = I[0]

    # Build context with lightweight dedup by source
    seen = set()
    context_blocks = []
    citations = []
    for idx in I:
        if idx < 0: 
            continue
        src = docs[idx]["meta"].get("source","unknown")
        key = (src,)
        if key in seen: continue
        seen.add(key)
        context_blocks.append(f"[Source: {src}]\n{docs[idx]['text']}")
        citations.append(src)

    context = "\n\n---\n\n".join(context_blocks[:args.k])

    prompt = f"""You are an HR assistant for an internal system.
Answer the user's question using ONLY the following context. Be specific and extract exact details from the context.
Do NOT include citations in brackets within your answer. The sources will be listed separately.
If the answer isn't contained in the context, say "I don't know based on the indexed policies/data."

Question: {args.question}

Context:
{context}

Instructions:
- Extract specific numbers, dates, and details from the context
- Provide direct answers with exact information when available
- Don't say "I don't have information" if the details are clearly in the context
- Write a clean answer without any [source: filename] citations in the text
"""

    answer = call_llm(prompt)

    print("\n" + "="*80)
    print("QUESTION:")
    print(args.question)
    print("="*80)
    print("ANSWER:")
    print(textwrap.fill(answer, width=100))
    print("="*80)
    print("CITED SOURCES (filenames):")
    print(", ".join(citations))

def retrieve_hr_answer(question, k=6):
    import faiss, pickle
    index_path = 'utils/vectorstore/index.faiss'
    meta_path = 'utils/vectorstore/index.faiss.meta.pkl'
    index = faiss.read_index(index_path)
    with open(meta_path, 'rb') as f:
        docs = pickle.load(f)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    qemb = model.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(qemb)
    D, I = index.search(qemb, k)
    I = I[0]
    seen = set()
    context_blocks = []
    citations = []
    for idx in I:
        if idx < 0:
            continue
        src = docs[idx]['meta'].get('source', 'unknown')
        key = (src,)
        if key in seen:
            continue
        seen.add(key)
        context_blocks.append(f"[Source: {src}]\n{docs[idx]['text']}")
        citations.append(src)
    context = "\n\n---\n\n".join(context_blocks[:k])
    prompt = f"""You are an HR assistant for an internal system.
Answer the user's question using ONLY the following context. Be specific and extract exact details from the context.
Do NOT include citations in brackets within your answer. The sources will be listed separately.
If the answer isn't contained in the context, say "I don't know based on the indexed policies/data."

Question: {question}

Context:
{context}

Instructions:
- Extract specific numbers, dates, and details from the context
- Provide direct answers with exact information when available
- Don't say "I don't have information" if the details are clearly in the context
- Write a clean answer without any [source: filename] citations in the text
"""
    answer = call_llm(prompt)
    return answer, citations

if __name__ == "__main__":
    main()
