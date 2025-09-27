import os, argparse, glob, pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from langchain.docstore.document import Document
from utils.loaders import make_chunks_for_file

def iter_files(root):
    for ext in ("**/*.md","**/*.txt","**/*.yaml","**/*.yml","**/*.csv"):
        for p in glob.glob(os.path.join(root, ext), recursive=True):
            yield p

def build_corpus(data_path: str):
    docs = []
    for p in iter_files(data_path):
        chunks = make_chunks_for_file(p)
        for ch in chunks:
            docs.append(Document(page_content=ch["text"], metadata={"source": ch["source"]}))
    return docs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True, help="Folder with your dataset (e.g., rag_seed_data)")
    ap.add_argument("--index-path", required=True, help="Where to save FAISS index file")
    ap.add_argument("--meta-path", default=None, help="Where to save metadata pickle (defaults to index-path + .meta.pkl)")
    args = ap.parse_args()

    print(f"[Ingest] Scanning: {args.data_path}")
    docs = build_corpus(args.data_path)
    print(f"[Ingest] Chunks: {len(docs)}")

    # Embeddings
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [d.page_content for d in docs]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    # Build FAISS
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # Save index and metadata
    faiss.write_index(index, args.index_path)
    meta_path = args.meta_path or (args.index_path + ".meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump([{"text": d.page_content, "meta": d.metadata} for d in docs], f)
    print(f"[Ingest] Saved index to {args.index_path}")
    print(f"[Ingest] Saved metadata to {meta_path}")

if __name__ == "__main__":
    main()
