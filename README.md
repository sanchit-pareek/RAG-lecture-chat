# RAG Teaching Assistant

A local lecture question-answering application built with **FastAPI, React, BGE-M3, and Llama 3.1 8B**. Ask questions about a probability course, retrieve matching transcript passages, and inspect the lecture timestamps behind each answer.

The included corpus contains **25 lecture transcripts, 21.1 hours of content, and 23,912 timestamped segments**. Models run locally through Ollama.

## Features

- Semantic search over 1,024-dimensional BGE-M3 embeddings with top-five cosine retrieval.
- Adjacent-segment expansion within lecture boundaries and merging of contiguous excerpts.
- Llama 3.1 8B answers, numbered references, and timestamped source passages in a React interface.
- Resumable embedding batches with corpus/model fingerprints and atomic index installation.
- Shared retrieval and generation code across the API, CLI, and evaluation tools.
- Component timing, model-residency diagnostics, and paired full-response HTTP evaluation.

The prompt requests grounded answers and abstention when evidence is missing. This behavior is not guaranteed: evaluation found incomplete evidence, wrong answers, and unsupported citations.

## Measured results

Local Windows measurements used CPU query embeddings and Llama 3.1 8B generation, with an NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM), an 8,192-token context, and a 512-token output cap.

| Measurement | Result | Scope |
| --- | --- | --- |
| Corpus | 25 lectures; 21.1 hours; 23,912 segments | Duration from transcript metadata |
| CPU retrieval latency | 256 ms median; 321 ms p95 | 30 distinct questions, one measured pass |
| Placement consistency | Exact top-five order preserved on 30/30 questions | Against saved automatic-placement rankings |
| Full-response HTTP latency, raw segments | 2.81 s median; 10.26 s p95 | Eight-question pilot, eight responses |
| Full-response HTTP latency, expanded context | 3.58 s median; 7.85 s p95 | Same questions, eight responses |

The response pilot excluded one warmup and randomized paired configuration order. It deliberately included known retrieval misses; caching, model residency, and output length affect results. Eight samples per condition do not establish general tail latency or throughput. Answer quality was mixed, so no overall improvement from expansion is claimed. Relevance labels require independent validation; accuracy percentages remain exploratory.

See [evaluation/RESULTS.md](evaluation/RESULTS.md) for findings and evidence, and [EVALUATION.md](EVALUATION.md) for detailed protocols and earlier diagnostics.

## Run locally on Windows

Install Python, Node.js/npm, and Ollama. The measured Python environment was 3.14.3; other versions have not been independently verified here. Models and the generated index require local disk space. FFmpeg and Whisper are only needed for new recordings.

From PowerShell in the repository root:

```powershell
.\start-local.ps1 -CpuEmbeddings
```

This creates or repairs `.venv`, installs backend dependencies, starts Ollama when needed, downloads missing models, builds or validates the index, builds React, and runs a real API smoke query. Initial indexing embeds all 23,912 segments; interrupted batches resume from `.embedding-cache/`.

- App: <http://127.0.0.1:5173>
- API documentation: <http://127.0.0.1:8000/docs>
- Health: <http://127.0.0.1:8000/health>

`-CpuEmbeddings` places BGE-M3 query embeddings on the CPU. In local diagnostics this avoided repeated embedding/generation model switching. Placement is opt-in. Stop an older backend before starting one with changed settings. Use `-RebuildIndex` when the corpus or embedding model changes. Load only trusted joblib files.

### Manual startup

With Ollama running:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
ollama pull bge-m3
ollama pull llama3.1:8b
$env:EMBEDDING_NUM_GPU = '0'
.\.venv\Scripts\python.exe preprocessjson.py
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
cd frontend
npm ci
npm run dev -- --host 127.0.0.1
```

The CLI is available as `.\.venv\Scripts\python.exe llmquering.py`.

## API example

```powershell
$body = @{ question = 'What is a random variable?'; neighbor_window = 1 } | ConvertTo-Json
Invoke-RestMethod 'http://127.0.0.1:8000/query' -Method Post -ContentType 'application/json' -Body $body
```

The response includes the answer, timestamped contexts, raw top-five hits, delivered segment IDs, component timings, and generation counters. `neighbor_window=0` uses raw segments; `1` adds neighbors. `/health` reports corpus/model identity, placement, generation settings, and runtime code identity.

## Evaluation and tests

```powershell
# Corpus statistics and CPU retrieval
.\.venv\Scripts\python.exe evaluate.py stats
.\.venv\Scripts\python.exe evaluate.py benchmark --embedding-num-gpu 0 --repeats 1 --output evaluation/cpu_new.json

# Eight-question paired HTTP pilot; backend and Ollama must already be running
.\run-evaluation.ps1

# All 29 included questions and three expanded passes
.\run-evaluation.ps1 -Full -Repeats 3

# Regression checks
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The larger run generates 116 measured answers plus a warmup and can take substantially longer. The runner saves completed responses, checks ranking consistency before changing backend placement, and produces independent label-review and shuffled A/B answer packets. Twelve regression tests passed in the checked environment. Review remains necessary before using accuracy claims.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama service |
| `EMBEDDING_MODEL` | `bge-m3` | Query/index embeddings |
| `LLM_MODEL` | `llama3.1:8b` | Answer model |
| `EMBEDDING_NUM_GPU` | Automatic placement | Set `0` for CPU embeddings |
| `LLM_NUM_CTX` | `8192` | Generation context budget |
| `LLM_NUM_PREDICT` | `512` | Output token cap |
| `INDEX_PATH` | Project-root `lecture_index.joblib` | Generated index |
| `RAG_ROOT` | Project root | Corpus/artifact root |

Generation uses temperature 0 and seed 42. The frontend defaults to same-origin `/api`, forwarded by Vite or NGINX to FastAPI; `VITE_API_URL` can override the frontend base URL.

## Repository guide

| Path | Purpose |
| --- | --- |
| `backend/main.py` | API routes and runtime health |
| `backend/rag.py` | Shared loading, retrieval, prompts, generation |
| `frontend/` | React/Vite interface and proxy configuration |
| `transcripts/` | Included timestamped lecture JSON |
| `preprocessjson.py` | Resumable index builder |
| `evaluate.py`, `evaluation_suite.py` | Retrieval and paired HTTP evaluation |
| `profile_generation.py` | Loading, residency, and decoding diagnostics |
| `tests/` | Regression tests |
| `start-local.ps1`, `run-evaluation.ps1` | Local setup and evaluation |

## Optional recording ingestion

For new recordings, install FFmpeg and root `requirements.txt`, create `videos/` and `audios/`, then run `process_videos.py` and `transcription.py`. Transcription uses faster-whisper `large-v3` with CUDA/FP16. Adapt filename conventions and device settings for your recordings, then rebuild the index. The included corpus needs no retranscription.

## Limitations and next work

- ASR segments average 3.15 seconds and often end mid-sentence. Neighbor expansion can still miss the explanation.
- Displayed sources do not guarantee supported citations. The pilot observed both unsupported correct answers and incorrect answers.
- The similarity threshold is not calibrated as a reliable out-of-domain detector.
- Independent relevance labels, held-out student questions, and concurrent-load measurements remain future work.
- Docker/NGINX configuration is included; Docker and Railway deployment have not been verified. The existing deployment guide contains historical instructions and needs review against current configuration.

The interface identifies the material as MIT OpenCourseWare probability lectures by John Tsitsiklis. Course content retains its original attribution and applicable terms. No standalone code-license file is currently included; the earlier README's MIT license assertion has been removed pending an explicit license choice.
