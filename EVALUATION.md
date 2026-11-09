# RAG Teaching Assistant: verified metrics and evaluation

> Latest status, 9 November 2026: the varied-question CPU test and paired HTTP pilot have now completed. CPU placement preserved every top-five order on 30 questions and was verified in the measured backend. Median full-response HTTP time was 2.81 s for raw segments and 3.58 s for expanded context across eight questions. Quality was mixed; independent relevance review remains pending. See [evaluation/RESULTS.md](evaluation/RESULTS.md) and [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md). Sections below retain earlier measurements and diagnostic history; statements about the broader run being pending are historical.

## Current result (9 November 2026)

| Measurement | Verified value |
| --- | ---: |
| Lecture transcript files | 25 |
| Transcript segments | 23,912 |
| Transcript bytes | 2,875,537 (2.742 MiB) |
| Total duration from transcript metadata | 21.1163 hours |
| Average segment duration | 3.1456 seconds |
| Embedding matrix dimensions | 23,912 x 1,024 |
| Hit@1 (draft benchmark) | 33.3% |
| Hit@5 (draft benchmark) | 66.7% |
| MRR@5 (draft benchmark) | 0.4339 |
| Retrieval latency, p50 / p95 (window 0) | 205.2 / 280.3 ms |
| Expanded retrieval latency, p50 / p95 | 205.3 / 280.3 ms |
| Single smoke-query server response time | 67.69 seconds |

Duration is the sum of the JSON `duration` values; source videos were not independently timed. Segment duration is `end - start`, averaged over all segments. Data is already checked in: downloading videos, running FFmpeg, and transcribing again are unnecessary for running this corpus.

The local setup has now completed: `lecture_index.joblib` exists (99,791,459 bytes), the real model-backed API smoke query succeeded, and the user confirmed the app is running. Both model manifests and referenced blobs were verified present. The benchmark records Python 3.14.3 and an NVIDIA GeForce RTX 4050 Laptop GPU with 6,141 MiB VRAM (driver 592.82). The initial `evaluation/environment_check.json` setup-blocker status is historical; the generated results supersede it.

`evaluation/benchmark_results.json` contains 90 retrieval measurements from 30 unique draft questions, repeated three times. Dataset and corpus hashes match the current files. Hit@1 is 33.3%, Hit@5 is 66.7%, and MRR@5 is 0.4339. These are exploratory results on assistant-authored, source-derived questions with unreviewed, non-exhaustive labels; they are not yet suitable as an independently validated resume accuracy claim.

Query embedding p50/p95 was 72.3/131.8 ms; cosine search including ranking was 136.1/187.8 ms. Component percentiles do not add to combined percentiles. Expanded context coverage of labeled support windows increased from 6.89% to 18.21%, while mean context size increased from 41.4 to 112.3 words. This does not demonstrate improved answer quality; no controlled generation comparison has run.

The saved smoke query ("What is a random variable?") returned an answer and timestamped sources. Server-side retrieval took 350 ms, generation 67.34 seconds, and total processing 67.69 seconds. This is one sample, with unknown warm/cold model residency, not a generation-latency benchmark or a client HTTP latency measurement. Investigate generation and review source sufficiency before calling the assistant fast overall.

Six regression tests passed, including a direct API route check with mocked model calls. Python compilation, PowerShell syntax, and Git whitespace checks passed. Earlier sandbox connection/install restrictions prevented execution inside the agent session; the newly generated local artifacts establish the subsequent successful run. Docker remains unverified.

## CPU placement diagnostic completed

The CPU-embedding trial kept both models resident and reduced repeated-question diagnostic pipeline times from 15.48–15.81 s to 1.69–1.72 s. These cached, single-question samples are not general API latency or a broad speedup benchmark. See [CPU embedding results](evaluation/CPU_EMBEDDING_RESULTS.md). The app defaults and running backend were not changed.

## Follow-up label audit and generation diagnosis

See [evaluation/REVIEW.md](evaluation/REVIEW.md) for the source-reviewed labels, a matched-subset rescoring, the generation profiler command, and updated resume bullets. The reviewed Hit@5 of 75.86% is post-hoc passage localization, not an independent accuracy result or a retriever improvement. The first generation profile has completed: two embedding-then-generation trials took 15.48 and 15.81 seconds, with repeated model reloading. See [generation findings](evaluation/GENERATION_FINDINGS.md). These are diagnostic samples, not an end-to-end benchmark.

## Finish setup and run locally

From a normal PowerShell terminal in this repository:

```powershell
.\start-local.ps1 -Evaluate
```

If your PowerShell policy blocks local scripts, invoke this for this process only:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start-local.ps1 -Evaluate
```

The script repairs/creates `.venv`, installs backend dependencies, starts Ollama if needed, pulls either model only when missing, creates or validates `lecture_index.joblib`, builds React, starts FastAPI, sends a real model-backed smoke query, benchmarks retrieval, and starts the frontend. Initial indexing embeds 23,912 short segments in batches of 32; progress is printed. Interrupted batches resume from `.embedding-cache/`. A complete index is installed atomically only after all batches finish. Do not close the terminal while setup/indexing is active.

Open <http://127.0.0.1:5173>. API documentation: <http://127.0.0.1:8000/docs>. Background logs are `backend-error.log`, `frontend-error.log`, and `ollama-error.log`. The script leaves the services running. Stop the corresponding Python/Node processes when finished; Ollama is shared with other projects.

Use `-RebuildIndex` when transcript data, the embedding model, or model digest changes. Cached batches are keyed by corpus and model fingerprint, so changed data cannot silently reuse old embeddings. Defaults can be overridden using `OLLAMA_BASE_URL`, `EMBEDDING_MODEL`, and `LLM_MODEL`. The index is local generated data and is excluded from Git. Load only your own trusted joblib files.

If an existing backend is running when you change code, stop that process before restarting. Optional audio-transcription dependencies in root `requirements.txt` are only needed when processing new recordings; the app uses `backend/requirements.txt`.

## Reproduce measurements

```powershell
.\.venv\Scripts\python.exe evaluate.py stats
.\.venv\Scripts\python.exe evaluate.py benchmark --repeats 3
# Slower: generate baseline and expanded-context answers for manual quality review.
.\.venv\Scripts\python.exe evaluate.py benchmark --repeats 1 --generate --output evaluation/answer_comparison.json
# Run with the API already started to measure real HTTP response latency too.
.\.venv\Scripts\python.exe evaluate.py benchmark --repeats 3 --api-url http://127.0.0.1:8000 --output evaluation/api_results.json
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`evaluation/questions.json` contains 30 assistant-authored, source-derived draft questions with evidence and zero-based original transcript segment positions. The labels are supporting windows, not exhaustive relevance judgments. They require human review: remove ambiguous or weak questions, narrow/expand support labels, include alternate valid passages across lectures, and preferably replace some with independently written student questions. Keep a separate held-out set when tuning. Results from this draft must be described as exploratory rather than independent benchmark accuracy.

Definitions and controls:

- Hit@1 / Hit@5 and MRR@5 use the raw top five cosine-ranked segments before threshold rejection. Missing a relevant segment in the first five gives reciprocal rank zero. This is truncated MRR, not full-corpus MRR.
- A similarity threshold of 0.25 controls whether any context is sent to generation. Acceptance is recorded separately. This threshold has not been calibrated as a reliable out-of-domain detector.
- One retrieval warm-up is excluded. Subsequent measurements include query embedding HTTP time and cosine search over the entire index. Results report p50 and p95 with sample counts; three repeats of 30 questions give 90 samples but only 30 unique questions. Sequential local measurements do not establish concurrent throughput.
- Baseline and expansion use exactly the same retrieved top five. Window 0 selects raw hits; window 1 adds adjacent same-lecture segments. Both merge contiguous excerpts. No selected excerpts are discarded. Record the changed context size alongside coverage and latency.
- Relevant-context coverage is the fraction of labeled support segments actually delivered. Adding neighbors mechanically increases the opportunity for coverage; it does not prove better answers.
- `--generate` stores both configurations' answers and source text with fixed generation settings. Evaluate correctness, completeness, grounding and citations manually and blindly before claiming an answer-quality benefit. Pair order is baseline then expansion, so generation timings may be affected by warm-up/model residency. Use a separate controlled experiment for strong causal latency claims.
- `--api-url` measures complete client HTTP time, including embedding, search and LLM work. API timings within the response separate generation and retrieval. Timeouts/errors fail the run instead of being silently excluded. `benchmark_partial.json` preserves completed measurements if interrupted.
- Reports record corpus/dataset hashes, model digest, matrix dimensions, hardware, Python/package versions, model names and generation settings. Model residency and thermal/load conditions also affect latency. Record those conditions when publishing results.

## Implementation fixes

The old API and CLI sorted transcript rows without moving their embedding rows. Shared loading now reorders both together and rejects stale or incompatible indexes. Index creation is deterministic and resumable. API/CLI/evaluation share the same retrieval and generation functions. The API no longer truncates chronologically sorted contexts to five after expansion, which could discard top-ranked evidence. Blocking model requests use synchronous FastAPI routes so they run in its worker thread pool. Empty questions are rejected.

The frontend uses same-origin `/api`; Vite and NGINX strip that prefix when forwarding to FastAPI. NGINX permits long local generation requests. Docker now uses the Ollama service hostname and explicit index/corpus mounts. Docker was not tested in this session. To use it, first build the local index, start the Ollama container, pull the two matching models inside that container, then start backend/frontend. Container model digests must match the index. See `docker-compose.yml`.

## Defensible resume wording now

- Built a lecture-grounded RAG teaching assistant over 25 transcripts spanning 21.1 hours and 23,912 timestamped segments, using BGE-M3 embeddings and cosine-similarity retrieval.
- Developed a FastAPI and React question-answering pipeline with Llama 3.1 8B, timestamped sources, neighboring-segment context expansion, and a resumable embedding-index builder.

These describe implementation and corpus scale. Do not add accuracy, latency, improved answer quality, production readiness, or a successful Docker deployment until actual measurements support those claims. The first bullet can later include validated Hit@5 and measured median retrieval latency, with the labeled-query count and benchmark conditions documented here.
