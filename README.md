# Private RAG v2 (dashboard)

## Run locally
```
pip install -r requirements.txt
ollama pull llama3.1
# edit clients.json: one long random key per client
uvicorn server:app --port 8000
```
Open http://localhost:8000, enter the client's key, upload documents, ask.

## Run in Docker (client's own server)
```
docker build -t private-rag .
docker run -p 8000:8000 -v $(pwd)/data:/app/data -v $(pwd)/clients.json:/app/clients.json private-rag
```
(Ollama runs on the host; on Linux add `--add-host=host.docker.internal:host-gateway`.)

## Before selling
- Put it behind HTTPS (Caddy or nginx), never plain HTTP.
- Test with 50 real questions from the client's documents and record accuracy.
- Scanned PDFs need OCR (not included yet).
