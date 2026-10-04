"""Dashboard server. Run: uvicorn server:app --host 0.0.0.0 --port 8000
Clients live in clients.json: {"acme": "long-secret-key", "beta": "another-key"}
Each client gets isolated storage in data/<client>/.
"""
import json, re, shutil, hmac
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import rag

app = FastAPI()
DATA = Path("data")
CLIENTS = json.loads(Path("clients.json").read_text())
ALLOWED = {".pdf", ".txt", ".md"}
MAX_MB = 50

def client_for(key: str) -> str:
    for name, secret in CLIENTS.items():
        if key and hmac.compare_digest(secret, key):
            return name
    raise HTTPException(401, "Invalid access key")

def paths(name):
    docs = DATA / name / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    return docs, DATA / name / "index.pkl"

class Q(BaseModel):
    question: str

@app.get("/")
def home():
    return FileResponse("index.html")

@app.get("/api/docs")
def list_docs(x_key: str = Header(None)):
    docs, _ = paths(client_for(x_key))
    return sorted(p.name for p in docs.iterdir())

@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...), x_key: str = Header(None)):
    docs, index = paths(client_for(x_key))
    for f in files:
        safe = re.sub(r"[^\w.\- ]", "_", Path(f.filename).name)
        if Path(safe).suffix.lower() not in ALLOWED:
            raise HTTPException(400, f"Unsupported file type: {safe}")
        data = await f.read()
        if len(data) > MAX_MB * 1024 * 1024:
            raise HTTPException(400, f"{safe} is over {MAX_MB} MB")
        (docs / safe).write_bytes(data)
    n = rag.ingest(docs, index)
    return {"chunks": n}

@app.delete("/api/docs/{name}")
def delete_doc(name: str, x_key: str = Header(None)):
    docs, index = paths(client_for(x_key))
    target = docs / Path(name).name
    if not target.exists():
        raise HTTPException(404)
    target.unlink()
    rag.ingest(docs, index)
    return {"ok": True}

@app.post("/api/ask")
def ask(q: Q, x_key: str = Header(None)):
    _, index = paths(client_for(x_key))
    answer, refs = rag.ask(q.question, index)
    return {"answer": answer, "sources": [{"file": s, "loc": l} for s, l in refs]}
