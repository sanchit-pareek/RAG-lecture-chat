from contextlib import asynccontextmanager
from time import perf_counter
import os
import hashlib
from pathlib import Path
from typing import Literal
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
try:
    from .rag import load_index, retrieve, build_prompt, call_llm_details, model_digest, generation_options
except ImportError:
    from rag import load_index, retrieve, build_prompt, call_llm_details, model_digest, generation_options

df = embedding_matrix = metadata = None
RUNTIME_CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes() +
    Path(__file__).with_name('rag.py').read_bytes()).hexdigest()

@asynccontextmanager
async def lifespan(app):
    global df, embedding_matrix, metadata
    df, embedding_matrix, metadata = load_index()
    if metadata.get('model_digest') != model_digest():
        raise RuntimeError('Ollama embedding model changed; rebuild index with --force')
    yield

app = FastAPI(title='Lecture RAG API', version='1.2.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
    allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])

class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    neighbor_window: Literal[0, 1] = 1

@app.get('/health')
def health():
    return dict(status='healthy', segments=len(df), embedding_shape=list(embedding_matrix.shape),
                embedding_model=metadata['embedding_model'], corpus_sha256=metadata['corpus_sha256'],
                api_version=app.version, pid=os.getpid(),
                code_sha256=RUNTIME_CODE_SHA256,
                embedding_model_digest=metadata.get('model_digest'),
                embedding_num_gpu=os.getenv('EMBEDDING_NUM_GPU', 'automatic'),
                generation_model=os.getenv('LLM_MODEL', 'llama3.1:8b'),
                generation_options=generation_options())

@app.post('/query')
def query(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail='Question must not be blank')
    start = perf_counter()
    try:
        result = retrieve(df, embedding_matrix, question, window=request.neighbor_window)
        generation_start = perf_counter()
        generated = (call_llm_details(build_prompt(question, result['contexts'])) if result['contexts']
                     else dict(answer="I don't know based on the provided lectures.", metrics=None, options=None))
        result['timings_ms']['generation'] = (perf_counter()-generation_start)*1000
        result['timings_ms']['total'] = (perf_counter()-start)*1000
        return dict(answer=generated['answer'], contexts=result['contexts'], timings_ms=result['timings_ms'],
                    generation_metrics=generated['metrics'], generation_options=generated['options'],
                    neighbor_window=request.neighbor_window,
                    top5=[dict(lecture=df.iloc[i].lecture, segment=int(df.iloc[i].local_segment_id),
                               score=float(result['scores'][i])) for i in result['indices']],
                    delivered_segments=[dict(lecture=r.lecture, segment=int(r.local_segment_id))
                                        for r in result['selected'].itertuples()] if result['contexts'] else [])
    except requests.RequestException as exc:
        raise HTTPException(status_code=503, detail=f'Ollama request failed: {exc}') from exc

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
