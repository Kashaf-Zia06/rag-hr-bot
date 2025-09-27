import re
from typing import List, Dict

def chunk_markdown(text: str, source: str, max_chars: int = 1200, overlap: int = 150):
    # Split by headings first
    parts = re.split(r"(?m)^#{1,6}\s", text)
    chunks = []
    for part in parts:
        if not part.strip():
            continue
        # secondary split into paragraphs
        pgs = re.split(r"\n\s*\n", part)
        buff = ""
        for p in pgs:
            if len(buff) + len(p) + 2 <= max_chars:
                buff += ("\n\n" if buff else "") + p
            else:
                if buff:
                    chunks.append({"text": buff.strip(), "source": source})
                # start next chunk; include overlap
                prev = buff[-overlap:] if overlap and len(buff) > overlap else ""
                buff = (prev + "\n\n" + p).strip()
        if buff:
            chunks.append({"text": buff.strip(), "source": source})
    return chunks

def chunk_table_row(row_text: str, source: str, max_chars: int = 1000):
    # For CSV rows we keep row per chunk, but truncate if extremely long
    t = row_text if len(row_text) <= max_chars else row_text[:max_chars] + " ..."
    return {"text": t, "source": source}
