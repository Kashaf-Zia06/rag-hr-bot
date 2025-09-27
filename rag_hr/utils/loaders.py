import os, csv, json, yaml, pandas as pd
from typing import List, Dict
from .chunker import chunk_markdown, chunk_table_row

def load_markdown(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()  # treat as text but could also parse

def load_csv_rows(path: str, max_rows: int = None):
    df = pd.read_csv(path)
    rows = []
    for i, row in df.iterrows():
        if max_rows and i >= max_rows: break
        # Create a semantic text view of the row
        kv = [f"{col}: {row[col]}" for col in df.columns]
        text = f"TABLE ROW from {os.path.basename(path)} | " + " | ".join(kv)
        rows.append(text)
    return rows

def make_chunks_for_file(path: str):
    fname = os.path.basename(path)
    ext = os.path.splitext(fname)[1].lower()
    chunks = []
    if ext in [".md", ".txt"]:
        text = load_markdown(path)
        chunks = chunk_markdown(text, source=fname)
    elif ext in [".yaml", ".yml"]:
        text = load_yaml(path)
        chunks = chunk_markdown(text, source=fname)
    elif ext in [".csv"]:
        for row_text in load_csv_rows(path):
            chunks.append(chunk_table_row(row_text, source=fname))
    else:
        # ignore unknown file types
        pass
    return chunks
