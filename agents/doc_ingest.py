import os
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# Simple embeddings + vectorstore
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# -------- CONFIG --------
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")  # or "gemma:2b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

DATA_DIR = Path("data")

# --------- helpers ----------
def read_text_files(folder: Path) -> List[Dict]:
    """Read all .txt files into a list of dicts {filename, text}"""
    docs = []
    for p in folder.glob("*.txt"):
        text = p.read_text(encoding="utf-8")
        docs.append({"filename": p.name, "text": text})
    return docs

# --------- Ingest Agent ----------
def ingest_agent(docs: List[Dict], embed_model_name="all-MiniLM-L6-v2"):
    """
    Lightweight ingestion:
    - chunking is simple: split on paragraphs
    - compute embeddings with sentence-transformers
    - build FAISS index (in-memory) and return embeddings/ids/index + docs
    """
    print("[ingest] loading embedding model...")
    embedder = SentenceTransformer(embed_model_name)

    chunks = []
    for d in docs:
        paragraphs = [p.strip() for p in d["text"].split("\n\n") if p.strip()]
        for i, p in enumerate(paragraphs):
            chunks.append({
                "source": d["filename"],
                "chunk_id": f"{d['filename']}_p{i}",
                "text": p
            })
    texts = [c["text"] for c in chunks]
    print(f"[ingest] computing embeddings for {len(texts)} chunks...")
    embs = embedder.encode(texts, show_progress_bar=False)
    embs = embs.astype("float32")

    # build faiss index
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)

    # store in-memory mapping
    id_to_chunk = {i: chunks[i] for i in range(len(chunks))}

    print("[ingest] completed. Returning index and metadata.")
    return {
        "faiss_index": index,
        "id_to_chunk": id_to_chunk,
        "embedder": embedder
    }
