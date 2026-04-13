from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from contextlib import asynccontextmanager
import os

data = None
df = None
embedding_matrix = None

def load_index():
    global data, df, embedding_matrix
    index_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "lecture_index.joblib")
    data = joblib.load(index_path)
    df = data["df"]
    embedding_matrix = data["embedding_matrix"]
    df = df.sort_values(["lecture", "start"]).reset_index(drop=True)

def create_embeddings(texts):
    provider = os.getenv("LLM_PROVIDER", "ollama")
    
    if provider == "ollama":
        r = requests.post(
            f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/embed",
            json={"model": os.getenv("EMBEDDING_MODEL", "bge-m3"), "input": texts},
            timeout=120
        )
        r.raise_for_status()
        return r.json()["embeddings"]
    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {
            "model": "llm-embeddings",  # Will need fallback for embeddings
            "input": texts
        }
        # Groq doesn't have free embeddings - use a workaround
        raise HTTPException(status_code=501, detail="Groq embeddings not available - use Ollama or another provider")
    else:
        raise HTTPException(status_code=501, detail=f"Unknown LLM provider: {provider}")

def expand_with_neighbors(df, indices, window=1):
    expanded = set()
    for idx in indices:
        for i in range(max(0, idx-window), min(len(df), idx+window+1)):
            if df.iloc[i].lecture == df.iloc[idx].lecture:
                expanded.add(i)
    return df.iloc[sorted(expanded)]

def merge_segments(df):
    merged, current = [], None
    for row in df.itertuples():
        if current is None:
            current = {
                "lecture": row.lecture,
                "start": row.start,
                "end": row.end,
                "text": row.text
            }
        elif row.lecture == current["lecture"] and row.start <= current["end"] + 1.0:
            current["end"] = row.end
            current["text"] += " " + row.text
        else:
            merged.append(current)
            current = {
                "lecture": row.lecture,
                "start": row.start,
                "end": row.end,
                "text": row.text
            }
    if current:
        merged.append(current)
    return merged

def call_llm(prompt):
    provider = os.getenv("LLM_PROVIDER", "ollama")
    
    if provider == "ollama":
        r = requests.post(
            f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/generate",
            json={"model": os.getenv("LLM_MODEL", "llama3.1:8b"), "prompt": prompt, "stream": False},
            timeout=300
        )
        r.raise_for_status()
        return r.json()["response"]
    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}]
        }
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data, timeout=300)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    else:
        raise HTTPException(status_code=501, detail=f"Unknown LLM provider: {provider}")

def build_prompt(incoming_query, contexts):
    blocks = []
    for i, ctx in enumerate(contexts, start=1):
        blocks.append(
            f"[{i}] Lecture {ctx['lecture']} "
            f"({ctx['start']:.1f}s–{ctx['end']:.1f}s)\n{ctx['text']}"
        )
    return f"""You are a teaching assistant answering strictly from the provided lecture excerpts.
If the answer is not contained in them, say:
"I don't know based on the provided lectures."

Question:
{incoming_query}

Lecture excerpts:
{chr(10).join(blocks)}

Answer:
""".strip()

class QueryRequest(BaseModel):
    question: str

class ContextSnippet(BaseModel):
    lecture: str
    start: float
    end: float
    text: str

class QueryResponse(BaseModel):
    answer: str
    contexts: list[ContextSnippet]

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_index()
    yield

app = FastAPI(title="Lecture RAG API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        q_emb = create_embeddings([request.question])[0]
        sims = cosine_similarity(embedding_matrix, [q_emb]).flatten()

        if sims.max() < 0.25:
            return QueryResponse(
                answer="I don't know based on the provided lectures.",
                contexts=[]
            )

        top_idx = sims.argsort()[::-1][:5]
        expanded = expand_with_neighbors(df, top_idx, window=1)
        contexts = merge_segments(expanded)
        contexts = contexts[:5]

        prompt = build_prompt(request.question, contexts)
        answer = call_llm(prompt)

        return QueryResponse(
            answer=answer,
            contexts=[ContextSnippet(**ctx) for ctx in contexts]
        )
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Ollama service not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
