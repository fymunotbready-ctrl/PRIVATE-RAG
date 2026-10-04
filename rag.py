"""Private RAG engine: hybrid search (BM25 + embeddings) + grounded answers.
Everything runs locally. Usage:
  python rag.py ingest ./docs
  python rag.py ask "What is the termination notice period?"
"""
import sys, re, json, pickle
from pathlib import Path
import numpy as np
import requests
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

INDEX = Path("index.pkl")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "llama3.1"            # any model pulled in Ollama
import os
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/generate"
CHUNK_WORDS, OVERLAP = 220, 40
MIN_SCORE = 0.30                   # below this similarity -> refuse to answer

_embedder = None
def embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder

def tokenize(t):
    return re.findall(r"\w+", t.lower())

def read_file(p: Path):
    """Yield (page_label, text) for each page/section."""
    if p.suffix.lower() == ".pdf":
        for i, page in enumerate(PdfReader(str(p)).pages, 1):
            yield f"p.{i}", page.extract_text() or ""
    elif p.suffix.lower() in {".txt", ".md"}:
        yield "full", p.read_text(errors="ignore")

def chunk(text):
    words = text.split()
    step = CHUNK_WORDS - OVERLAP
    for i in range(0, max(len(words), 1), step):
        piece = " ".join(words[i:i + CHUNK_WORDS])
        if len(piece.split()) > 15:
            yield piece

def ingest(folder, index=INDEX):
    chunks = []
    for p in sorted(Path(folder).rglob("*")):
        for label, text in read_file(p):
            for c in chunk(text):
                chunks.append({"source": p.name, "loc": label, "text": c})
    if not chunks:
        Path(index).unlink(missing_ok=True)
        return 0
    vecs = embedder().encode([c["text"] for c in chunks], normalize_embeddings=True)
    pickle.dump({"chunks": chunks, "vecs": vecs}, Path(index).open("wb"))
    print(f"Indexed {len(chunks)} chunks from {folder}")
    return len(chunks)

def search(query, k=5, index=INDEX):
    db = pickle.load(Path(index).open("rb"))
    chunks, vecs = db["chunks"], db["vecs"]
    # semantic ranking
    qv = embedder().encode([query], normalize_embeddings=True)[0]
    sims = vecs @ qv
    sem_rank = np.argsort(-sims)
    # keyword ranking
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    kw_rank = np.argsort(-bm25.get_scores(tokenize(query)))
    # reciprocal rank fusion
    fused = {}
    for ranking in (sem_rank[:30], kw_rank[:30]):
        for r, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0) + 1 / (60 + r)
    top = sorted(fused, key=fused.get, reverse=True)[:k]
    return [(chunks[i], float(sims[i])) for i in top]

PROMPT = """You are a compliance assistant. Answer ONLY from the numbered sources.
Rules:
- Cite every claim like [1] or [2].
- If the sources do not contain the answer, reply exactly: "I can't find this in the provided documents."
- Never guess or use outside knowledge.

Sources:
{sources}

Question: {q}
Answer:"""

def ask(q, index=INDEX):
    if not Path(index).exists():
        return "No documents uploaded yet.", []
    hits = search(q, index=index)
    if not hits or max(s for _, s in hits) < MIN_SCORE:
        return "I can't find this in the provided documents.", []
    sources = "\n\n".join(f"[{i}] ({c['source']}, {c['loc']})\n{c['text']}"
                          for i, (c, _) in enumerate(hits, 1))
    r = requests.post(OLLAMA_URL, json={
        "model": LLM_MODEL, "stream": False,
        "prompt": PROMPT.format(sources=sources, q=q),
        "options": {"temperature": 0},
    }, timeout=300)
    r.raise_for_status()
    return r.json()["response"].strip(), [(c["source"], c["loc"]) for c, _ in hits]

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    cmd, arg = sys.argv[1], " ".join(sys.argv[2:])
    if cmd == "ingest":
        ingest(arg)
    elif cmd == "ask":
        answer, refs = ask(arg)
        print(answer)
        for i, (s, l) in enumerate(refs, 1):
            print(f"  [{i}] {s} {l}")
